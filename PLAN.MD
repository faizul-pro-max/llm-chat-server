# GPU Server Orchestrator — Detailed Plan

> Python orchestrator that runs on every Vast.ai GPU instance.
> Manages model loading, vLLM lifecycle, metrics agent, warmup, and health checks.
> CLI-driven with one make command to start everything.

---

## Design Principles

1. **Easy to read and debug** — no magic, no metaclasses, no decorators stacked 5 deep. Flat functions, clear names, explicit print statements.
2. **Fail fast and visibly** — every step prints what it's doing and exits cleanly on error. Never silent failures.
3. **Doctor first** — `make doctor` runs before any expensive operation (model download, GPU rental). Catches problems before they cost money.
4. **tmux for service management** — vLLM and gpu_agent run inside named tmux sessions. Easy to attach, see logs, detach without killing.
5. **Scenario as code** — each experiment is a Python file in `src/scenarios/`. You read the file, you know exactly what config will be used. No hidden YAML inheritance.
6. **One command to run** — `make start SCENARIO=baseline` launches everything (doctor → download → warmup → ready).

---

## Folder Structure

```
gpu-server/
├── Makefile                          ← one-command entry point
├── README.md                         ← copy-paste setup instructions
├── requirements.txt                  ← pinned Python deps
├── .env.example                      ← env vars template
│
├── src/
│   ├── cli.py                        ← Click CLI entry — `start`, `stop`, `doctor`, `status`, `logs`
│   ├── orchestrator.py               ← main lifecycle: doctor → download → start → warmup → ready
│   │
│   ├── doctor/                       ← pre-flight checks (run before anything else)
│   │   ├── __init__.py
│   │   ├── runner.py                 ← runs all checks, formats output
│   │   ├── check_network.py          ← speed test (HF Hub download throughput)
│   │   ├── check_disk.py             ← free disk + VRAM check
│   │   ├── check_cuda.py             ← driver, CUDA, Python, PyTorch versions
│   │   ├── check_ports.py            ← :8000 and :9100 reachability
│   │   ├── check_hf.py               ← HF Hub access + token validity
│   │   └── check_cache.py            ← models already downloaded
│   │
│   ├── observer-agent/               ← GPU metrics streaming agent
│   │   ├── __init__.py
│   │   ├── server.py                 ← FastAPI on :9100 — /gpu, /health
│   │   ├── nvml_reader.py            ← pynvml wrapper (single GPU + multi-GPU support)
│   │   └── auth.py                   ← x-api-key header check
│   │
│   ├── models/                       ← scenario configs (one file per experiment)
│   │   ├── __init__.py
│   │   ├── _base.py                  ← BaseScenario class — common fields, validation
│   │   ├── baseline.py               ← Scenario 1: no optimizations
│   │   ├── prefix_cache.py           ← Scenario 2: --enable-prefix-caching
│   │   ├── chunked_prefill.py        ← Scenario 3: --enable-chunked-prefill
│   │   ├── awq_quant.py              ← Scenario 4: AWQ INT4 model
│   │   ├── spec_decode.py            ← Scenario 5: speculative decoding
│   │   └── dual_model.py             ← Scenario 7: 2× vLLM instances
│   │
│   ├── lifecycle/                    ← lifecycle steps (orchestrator calls these)
│   │   ├── __init__.py
│   │   ├── download.py               ← HF Hub model download with progress
│   │   ├── start_vllm.py             ← launches vLLM in tmux session
│   │   ├── start_agent.py            ← launches observer agent in tmux session
│   │   ├── warmup.py                 ← runs 20 warmup requests
│   │   └── health.py                 ← /health endpoint aggregator
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tmux.py                   ← tmux session helpers (create, attach, kill, is_running)
│       ├── logging.py                ← coloured console output (rich library)
│       ├── env.py                    ← env var loading + validation
│       └── http.py                   ← shared httpx client with retries
│
├── scripts/                          ← bash helpers
│   ├── install.sh                    ← apt install tmux + pip install requirements
│   └── tunnel.sh                     ← optional: Tailscale setup
│
└── logs/                             ← tmux pipe output (created at runtime)
    ├── vllm.log
    └── agent.log
```

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| python | 3.10+ | Vast.ai default |
| click | ^8.1 | CLI framework (better than argparse) |
| rich | ^13 | Coloured output, progress bars, tables |
| httpx | ^0.27 | Async HTTP client (calls to vLLM, HF) |
| fastapi | ^0.115 | observer-agent web server |
| uvicorn | ^0.32 | ASGI server for FastAPI |
| pynvml | ^11.5 | NVIDIA driver bindings (GPU metrics) |
| huggingface_hub | ^0.26 | Model download with progress |
| pydantic | ^2.9 | Scenario config validation |
| python-dotenv | ^1.0 | .env loading |
| vllm | latest | (separately installed — heavy, scenario-specific) |
| libtmux | ^0.39 | Python tmux control |

---

## Makefile — single entry point

```makefile
# Default scenario if not specified
SCENARIO ?= baseline

.PHONY: help install doctor start stop status logs attach clean

help:
	@echo "GPU Server Orchestrator — make targets:"
	@echo ""
	@echo "  make install              Install system + Python deps"
	@echo "  make doctor               Run pre-flight checks (free, no GPU cost)"
	@echo "  make start SCENARIO=name  Full lifecycle: doctor → download → start → warmup"
	@echo "  make stop                 Stop all services (vLLM + agent)"
	@echo "  make status               Show service status"
	@echo "  make logs                 Tail combined logs"
	@echo "  make attach SVC=vllm      Attach to tmux session (vllm or agent)"
	@echo "  make clean                Stop services + clean tmp files (keeps models)"
	@echo ""
	@echo "Scenarios available: $$(ls src/models/*.py | xargs -n1 basename | sed 's/\.py//' | grep -v '^_')"

install:
	bash scripts/install.sh

doctor:
	python -m src.cli doctor

start:
	python -m src.cli start --scenario $(SCENARIO)

stop:
	python -m src.cli stop

status:
	python -m src.cli status

logs:
	python -m src.cli logs

attach:
	tmux attach -t $(SVC)

clean:
	python -m src.cli stop
	rm -rf logs/*.log
```

---

## CLI Commands (src/cli.py)

```bash
# Pre-flight (free — run before renting GPU or downloading models)
python -m src.cli doctor [--skip-network]

# Full startup
python -m src.cli start --scenario baseline [--no-warmup] [--no-download]

# Individual lifecycle steps (for debugging)
python -m src.cli download --scenario baseline
python -m src.cli warmup [--count 20]

# Service management
python -m src.cli stop [--service vllm|agent|all]
python -m src.cli status
python -m src.cli logs [--service vllm|agent] [--tail 50]

# Scenario info
python -m src.cli scenarios               # list all
python -m src.cli scenarios show baseline # print config
```

---

## Doctor Command — Detailed

**Goal:** detect all problems BEFORE you spend money on GPU time, model downloads, or debugging time.

### Checks performed (in order, fail-fast)

| # | Check | What it verifies | Pass criteria |
|---|---|---|---|
| 1 | CUDA + driver | `nvidia-smi` works, driver version | Driver ≥ 525, CUDA ≥ 12.1 |
| 2 | Python + PyTorch | Python version, torch.cuda.is_available() | Python ≥ 3.10, PyTorch CUDA |
| 3 | Disk space | Free space on /root or /home | ≥ 50 GB free (model + KV swap) |
| 4 | VRAM | Total + free per GPU | ≥ model_size × 1.5 |
| 5 | Network speed | Download 100MB from HF mirror | ≥ 50 MB/s (warns < 10 MB/s) |
| 6 | HF Hub access | Auth + reachability | 200 OK on /api/models |
| 7 | Ports free | :8000 and :9100 not in use | Both unbound |
| 8 | Cached models | List already-downloaded models | Informational only |
| 9 | tmux installed | `tmux -V` returns version | Any version |
| 10 | Existing tmux sessions | Check for orphaned vllm/agent sessions | Warn if found |

### Output format

```
$ make doctor

🩺 LLM Bench Doctor — Pre-flight Checks
═══════════════════════════════════════════════════════════════

[1/10] CUDA + Driver ........................... ✓ 535.183.01 / CUDA 12.3
[2/10] Python + PyTorch ........................ ✓ Python 3.10.12 / torch 2.4.0+cu121
[3/10] Disk Space .............................. ✓ 412 GB free on /root
[4/10] VRAM .................................... ✓ 1× RTX PRO 6000 / 96 GB free
[5/10] Network Speed ........................... ⏳ Testing... 84.3 MB/s ✓
[6/10] HuggingFace Hub Access .................. ✓ Authenticated as user@org
[7/10] Ports 8000 + 9100 Free .................. ✓ Both available
[8/10] Cached Models ........................... ⓘ Qwen2.5-7B-Instruct (14.2 GB)
[9/10] tmux ..................................... ✓ tmux 3.2a
[10/10] Existing Sessions ...................... ✓ None running

═══════════════════════════════════════════════════════════════
✓ ALL CHECKS PASSED — ready to start scenarios
═══════════════════════════════════════════════════════════════

Estimated cost to download Qwen2.5-7B at 84.3 MB/s: ~2.8 minutes
Estimated GPU cost during download: $0.10
```

### Failure mode output

```
[5/10] Network Speed ........................... ⏳ Testing... 4.1 MB/s ✗

═══════════════════════════════════════════════════════════════
✗ NETWORK TOO SLOW — 4.1 MB/s
═══════════════════════════════════════════════════════════════

Downloading Qwen2.5-7B (14 GB) at this speed will take ~58 minutes.
Cost on this A100 ($1.07/hr) = ~$1.04 just for download.

OPTIONS:
  1. Pick a different host with better network (--skip-network to bypass)
  2. Use a smaller model (Phi-3.5-mini = 7.6 GB)
  3. Use a pre-downloaded snapshot from HF Hub mirror

Aborting to save money. Re-run with --skip-network if you want to proceed anyway.
```

This single feature can save $5–30 per session. The doctor is the most important piece of the project.

---

## Scenario Config (src/models/baseline.py)

```python
"""Baseline scenario — no optimizations. The zero line."""
from src.models._base import BaseScenario

class BaselineScenario(BaseScenario):
    name = "baseline"
    description = "Plain vLLM, FP16, no optimizations"

    # Model
    model = "Qwen/Qwen2.5-7B-Instruct"
    dtype = "float16"
    quantization = None

    # vLLM server config
    host = "0.0.0.0"
    port = 8000
    gpu_memory_utilization = 0.90
    max_model_len = 4096
    max_num_seqs = 256

    # Optimizations (all OFF for baseline)
    enable_prefix_caching = False
    enable_chunked_prefill = False
    speculative_model = None
    num_speculative_tokens = 0

    # Warmup
    warmup_requests = 20
    warmup_prompts = [
        "Short prompt.",
        "Explain how transformers work in detail.",
        "Write a long essay on attention mechanisms in neural networks " * 5,
    ]

    def build_vllm_command(self) -> list[str]:
        """Return list of args for `vllm serve ...`"""
        cmd = [
            "vllm", "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--disable-log-requests",
        ]
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if self.speculative_model:
            cmd.extend(["--speculative-model", self.speculative_model])
            cmd.extend(["--num-speculative-tokens", str(self.num_speculative_tokens)])
        if self.quantization:
            cmd.extend(["--quantization", self.quantization])
        return cmd

    def estimated_vram_gb(self) -> float:
        """Used by doctor to verify VRAM fits."""
        return 14.0  # FP16 7B weights

scenario = BaselineScenario()
```

Other scenarios (`prefix_cache.py`, `awq_quant.py`, etc.) inherit and override only what changes. Easy to read, easy to diff.

---

## Orchestrator Flow (src/orchestrator.py)

```python
"""Main lifecycle. Called by `make start SCENARIO=name`."""

def start(scenario_name: str, skip_warmup: bool = False, skip_download: bool = False):
    """Full lifecycle. Each step prints clearly, fails fast on error."""

    # Step 1 — Load scenario config
    log.section("Loading scenario")
    scenario = load_scenario(scenario_name)
    log.info(f"Scenario: {scenario.name}")
    log.info(f"Model:    {scenario.model}")
    log.info(f"Config:   {scenario.summary()}")

    # Step 2 — Doctor (mandatory)
    log.section("Pre-flight checks")
    doctor_result = doctor.run_all(scenario)
    if not doctor_result.passed:
        log.error("Doctor failed — aborting")
        sys.exit(1)

    # Step 3 — Download model (idempotent — skips if cached)
    if not skip_download:
        log.section("Model download")
        download.download_model(scenario.model, show_progress=True)

    # Step 4 — Start observer agent (FAST — start first)
    log.section("Observer agent")
    start_agent.start_in_tmux(session_name="agent", port=9100)
    health.wait_for_agent(timeout=15)
    log.success("Agent ready at :9100")

    # Step 5 — Start vLLM (SLOW — 60–90s)
    log.section("vLLM server")
    cmd = scenario.build_vllm_command()
    log.info(f"Command: {' '.join(cmd)}")
    start_vllm.start_in_tmux(session_name="vllm", command=cmd)
    health.wait_for_vllm(timeout=180, show_log_tail=True)
    log.success("vLLM ready at :8000")

    # Step 6 — Warmup
    if not skip_warmup:
        log.section("Warmup")
        warmup.run(scenario.warmup_prompts, count=scenario.warmup_requests)
        log.success(f"{scenario.warmup_requests} warmup requests complete — system stabilised")

    # Step 7 — Final status
    log.section("Ready")
    print_connection_info(scenario)
```

### Final status output

```
═══════════════════════════════════════════════════════════════
✓ READY FOR BENCHMARKS
═══════════════════════════════════════════════════════════════

Scenario:     baseline
Model:        Qwen/Qwen2.5-7B-Instruct
GPU:          NVIDIA RTX PRO 6000 (96 GB)
Public IP:    45.32.11.8

Endpoints:
  vLLM:       http://45.32.11.8:12453      (internal :8000)
  Agent:      http://45.32.11.8:12454      (internal :9100)

For your app server .env:
  GPU_SERVER_IP=45.32.11.8
  VLLM_PORT=12453
  GPU_AGENT_PORT=12454

Status:
  vLLM:       running   (tmux session: vllm)
  Agent:      running   (tmux session: agent)
  VRAM:       14.2 / 96 GB
  Warmup:     20 requests · TTFT stabilised at 87ms

Useful commands:
  make logs                 — tail combined logs
  make attach SVC=vllm      — see vLLM live output
  make attach SVC=agent     — see agent live output
  make stop                 — stop all services
```

---

## Observer Agent (src/observer-agent/server.py)

FastAPI server on :9100. Endpoints:

- `GET /gpu` — current GPU metrics JSON (requires `x-api-key`)
- `GET /health` — health check (no auth)
- `GET /system` — host info, CPU, RAM (requires auth)
- `GET /stream` — Server-Sent Events stream of metrics at 500ms (optional, for direct frontend connection)

GPU metrics shape (matches the contract in your CLAUDE.md):

```json
{
  "ts": 1748612345.123,
  "gpu_util": 87,
  "vram_used_mb": 14200,
  "vram_total_mb": 98304,
  "power_w": 287.4,
  "temp_c": 71,
  "gpu_name": "NVIDIA RTX PRO 6000",
  "hostname": "gpu-host-4a21"
}
```

---

## tmux Session Management (src/utils/tmux.py)

Each service runs in a named tmux session. You can attach and detach without killing the process.

```python
# Create session and run command
tmux.create_session("vllm", command="vllm serve ...")

# Check if session exists
tmux.is_running("vllm")  # → True

# Read recent output (last N lines)
tmux.capture_pane("vllm", lines=50)

# Kill session
tmux.kill_session("vllm")
```

User can attach manually:
```bash
tmux attach -t vllm     # see vLLM live
# Press Ctrl-B then D to detach without killing
```

---

## Doctor — Network Speed Test Details

Most important check on Vast.ai (where this is unpredictable).

```python
# src/doctor/check_network.py

def check_network(model_size_gb: float):
    """Download a 100MB test file from HF to measure throughput."""
    url = "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval.tar.gz"
    # Actually it's smaller — use any 100MB-ish public asset
    # OR: use HF Hub's CDN test endpoint

    start = time.time()
    with httpx.stream("GET", url) as r:
        bytes_downloaded = 0
        for chunk in r.iter_bytes(chunk_size=1_000_000):
            bytes_downloaded += len(chunk)
            if bytes_downloaded > 100_000_000:  # 100MB
                break
    elapsed = time.time() - start
    speed_mbps = bytes_downloaded / elapsed / 1_000_000

    # Estimate model download time
    estimated_seconds = (model_size_gb * 1000) / speed_mbps
    estimated_cost = (estimated_seconds / 3600) * 1.07  # A100 hourly rate

    return CheckResult(
        passed=speed_mbps >= 10,  # warn below 10, fail below 5
        speed_mbps=speed_mbps,
        estimated_download_seconds=estimated_seconds,
        estimated_download_cost=estimated_cost,
    )
```

The doctor compares to the scenario's model size and tells you upfront how much the download will cost.

---

## Failure & Recovery

| Failure | Doctor catches it? | Recovery |
|---|---|---|
| Slow network (<10 MB/s) | ✓ Step 5 | Choose different host before paying |
| Insufficient VRAM | ✓ Step 4 | Scenario won't start — no GPU cost wasted |
| CUDA driver mismatch | ✓ Step 1 | Switch host or update driver |
| HF token expired | ✓ Step 6 | Update token before download |
| Ports already in use | ✓ Step 7 | Run `make stop` to kill old sessions |
| Model download interrupted | After Step 5 | `huggingface_hub` resumes from partial — re-run safely |
| vLLM crash on startup | After Step 5 | `make attach SVC=vllm` to see error |
| OOM during warmup | After Step 6 | Lower `gpu_memory_utilization` in scenario |

---

## Cost-Saving Features Summary

| Feature | Saves |
|---|---|
| Network speed test in doctor | $1–5 per session (skips bad hosts) |
| Disk space check before download | Avoids failed downloads on full disks |
| Cached model detection | Skips re-download if model already there |
| Fail-fast on all errors | No GPU runs while debugging Python errors |
| One-command startup | No accidental commands during expensive GPU time |
| Persistent volume hint in doctor | Reminds user to attach volume for model cache |

---

## Build Order

1. **Week 1** — Doctor + CLI skeleton + tmux helpers (no GPU needed)
2. **Week 2** — Observer agent + scenarios + orchestrator (test with mock vLLM)
3. **Week 3** — First real GPU rental + baseline scenario end-to-end
4. **Week 4** — Add 5 remaining scenarios + warmup polish