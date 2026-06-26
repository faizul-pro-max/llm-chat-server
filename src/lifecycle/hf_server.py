"""Naive HuggingFace Transformers inference server (the un-optimized baseline).

Exposes a subset of the OpenAI / vLLM HTTP API on the same host:port that vLLM
would use, so the rest of the orchestrator (warmup, health checks, the Node
client) talks to it without changes:

    GET  /health               — 200 once the model is loaded
    GET  /v1/models            — the loaded model id
    POST /v1/completions       — text completion
    POST /v1/chat/completions  — chat completion (supports stream=true)

Deliberately naive: it loads the model with `AutoModelForCausalLM` and runs
`model.generate()` for one request at a time behind a lock. There is no
continuous batching and no paged attention — that is the whole point. Benchmark
this against a vLLM scenario to see what vLLM's optimizations are worth.

Run standalone (this is what `BaseScenario.build_hf_command()` launches):
    python -m src.lifecycle.hf_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Globals populated at startup by main() ───────────────────────────────────
_MODEL = None
_TOKENIZER = None
_MODEL_ID: str = ""
_MAX_MODEL_LEN: int = 4096
_DEVICE: str = "cpu"
_API_KEY: str = ""   # set in main() from --api-key; empty disables auth


async def _require_bearer(authorization: str = Header(None)) -> None:
    """Mirror vLLM's auth: API routes need `Authorization: Bearer <key>`.

    No-op when no key was configured, so the un-optimized baseline can still run
    standalone without a token (e.g. local debugging).
    """
    if not _API_KEY:
        return
    if authorization != f"Bearer {_API_KEY}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

# Serialise generation: the naive baseline handles one request at a time.
_GEN_LOCK = threading.Lock()

app = FastAPI(title="HF Transformers Baseline Server", version="1.0.0")


# ── Request schemas (loose — we only read the fields we use) ─────────────────
class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str = ""
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = []
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


# ── Generation helpers ───────────────────────────────────────────────────────
def _dtype_from_str(name: str):
    import torch

    return {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }.get(name, torch.float16)


def _gen_kwargs(max_tokens: int, temperature: float) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max(1, int(max_tokens)),
        "pad_token_id": _TOKENIZER.pad_token_id or _TOKENIZER.eos_token_id,
    }
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature))
    else:
        kwargs.update(do_sample=False)
    return kwargs


def _generate(prompt: str, max_tokens: int, temperature: float) -> Dict[str, int | str]:
    """Blocking, single-request generation. Returns text + token counts."""
    import torch

    inputs = _TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_MODEL_LEN,
    ).to(_DEVICE)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    with _GEN_LOCK, torch.no_grad():
        out = _MODEL.generate(**inputs, **_gen_kwargs(max_tokens, temperature))

    gen_ids = out[0][prompt_tokens:]
    text = _TOKENIZER.decode(gen_ids, skip_special_tokens=True)
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(gen_ids.shape[-1]),
    }


def _stream_tokens(prompt: str, max_tokens: int, temperature: float):
    """Yield generated text chunks as they are produced (for SSE streaming)."""
    import torch
    from transformers import TextIteratorStreamer

    inputs = _TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_MODEL_LEN,
    ).to(_DEVICE)

    streamer = TextIteratorStreamer(
        _TOKENIZER, skip_prompt=True, skip_special_tokens=True
    )

    def _run():
        with _GEN_LOCK, torch.no_grad():
            _MODEL.generate(**inputs, **_gen_kwargs(max_tokens, temperature), streamer=streamer)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for chunk in streamer:
        if chunk:
            yield chunk
    thread.join()


def _render_chat(messages: List[ChatMessage]) -> str:
    """Apply the model's chat template, falling back to a plain concatenation."""
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    try:
        return _TOKENIZER.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        body = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return body + "\nassistant:"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> Dict[str, Any]:
    ready = _MODEL is not None
    return {"status": "ok" if ready else "loading", "ts": time.time()}


@app.get("/v1/models", dependencies=[Depends(_require_bearer)])
async def models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": _MODEL_ID, "object": "model", "owned_by": "huggingface"}],
    }


@app.post("/v1/completions", dependencies=[Depends(_require_bearer)])
async def completions(req: CompletionRequest):
    if req.stream:
        return StreamingResponse(
            _completion_sse(req.prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream",
        )
    result = await asyncio.to_thread(_generate, req.prompt, req.max_tokens, req.temperature)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": _MODEL_ID,
        "choices": [{"index": 0, "text": result["text"], "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_require_bearer)])
async def chat_completions(req: ChatCompletionRequest):
    prompt = _render_chat(req.messages)
    if req.stream:
        return StreamingResponse(
            _chat_sse(prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream",
        )
    result = await asyncio.to_thread(_generate, prompt, req.max_tokens, req.temperature)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


def _completion_sse(prompt: str, max_tokens: int, temperature: float):
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    for chunk in _stream_tokens(prompt, max_tokens, temperature):
        payload = {
            "id": cmpl_id, "object": "text_completion", "created": created,
            "model": _MODEL_ID,
            "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


def _chat_sse(prompt: str, max_tokens: int, temperature: float):
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    for chunk in _stream_tokens(prompt, max_tokens, temperature):
        payload = {
            "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
            "model": _MODEL_ID,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    done = {
        "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
        "model": _MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


# ── Startup ──────────────────────────────────────────────────────────────────
def _load_model(model_id: str, dtype: str) -> None:
    global _MODEL, _TOKENIZER, _MODEL_ID, _DEVICE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hf_server] loading {model_id} (dtype={dtype}) on {_DEVICE} ...", flush=True)

    _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_dtype_from_str(dtype),
        device_map=_DEVICE if _DEVICE == "cuda" else None,
    )
    if _DEVICE == "cpu":
        _MODEL = _MODEL.to(_DEVICE)
    _MODEL.eval()
    _MODEL_ID = model_id
    print(f"[hf_server] model ready: {model_id}", flush=True)


def main() -> None:
    global _MAX_MODEL_LEN, _API_KEY
    parser = argparse.ArgumentParser(description="Naive HF Transformers baseline server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--api-key", default="",
                        help="Require 'Authorization: Bearer <key>' on /v1/* routes")
    args = parser.parse_args()

    _MAX_MODEL_LEN = args.max_model_len
    _API_KEY = args.api_key
    _load_model(args.model, args.dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
