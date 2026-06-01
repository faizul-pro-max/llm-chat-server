# GPU Server Orchestrator

Python CLI that runs on a Vast.ai GPU instance. Manages the full lifecycle of a vLLM inference server — model download, startup, warmup, health checks, and a live GPU metrics agent.

One command starts everything:

```bash
make start SCENARIO=baseline
```

---

## Prerequisites

### Hardware

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA (any CUDA-capable) | RTX 3090 / A100 / RTX PRO 6000 |
| VRAM | 16 GB (FP16 7B model) | 24 GB+ |
| Disk | 50 GB free | 200 GB+ (for multiple models) |
| RAM | 16 GB | 32 GB |

> For the AWQ quantized scenario (`awq_quant`) VRAM can be as low as **8 GB**.

### Software (on the GPU host)

| Software | Version | Notes |
|---|---|---|
| Ubuntu | 20.04 / 22.04 | Vast.ai default images work |
| Python | 3.10+ | `python3 --version` |
| NVIDIA Driver | ≥ 525 | `nvidia-smi` |
| CUDA | ≥ 12.1 | `nvcc --version` |
| tmux | any | `apt install tmux` |
| pip | latest | `pip install --upgrade pip` |

### Python dependencies

All pinned in [requirements.txt](requirements.txt). Installed by `make install`.

```
click, rich, httpx, fastapi, uvicorn, pynvml,
huggingface_hub, pydantic, python-dotenv, libtmux
```

> **vLLM is NOT in requirements.txt** — it is heavy and scenario-specific. Install it separately:
> ```bash
> pip install vllm
> ```

### Accounts / tokens

| Service | Required? | Notes |
|---|---|---|
| HuggingFace | Optional | Required for gated models (e.g. Llama). `HF_TOKEN` in `.env` |
| Vast.ai | Yes | Where you rent the GPU instance |

---

## Setup

### 1. Clone the repo onto the GPU instance

```bash
git clone <repo-url> gpu-server
cd gpu-server
```

### 2. Install dependencies

```bash
make install
# or manually:
bash scripts/install.sh
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```bash
VLLM_API_KEY=your_vllm_secret       # any string — used as Bearer token
AGENT_SECRET=your_agent_secret      # any string — used as x-api-key header
HF_TOKEN=hf_xxxxxxxxxxxx            # HuggingFace token (skip if using public models)
HF_HOME=/root/.cache/huggingface    # where models are stored — point to persistent volume
LOG_DIR=./logs
```

### 4. Install vLLM

```bash
pip install vllm
```

---

## Running the project

### Quick start

```bash
# Run pre-flight checks first (free — no GPU cost)
make doctor

# Start everything for the baseline scenario
make start SCENARIO=baseline
```

### All make targets

```bash
make help              # show all available targets

make install           # install system + Python dependencies
make doctor            # run 10 pre-flight checks
make start SCENARIO=baseline      # full lifecycle start
make stop              # stop vLLM + agent
make status            # show service status + HTTP health
make logs              # tail combined log output
make attach SVC=vllm   # attach to vLLM tmux session (Ctrl-B D to detach)
make attach SVC=agent  # attach to agent tmux session
make clean             # stop services + delete log files (keeps models)
```

### Available scenarios

| Scenario | What it tests | VRAM needed |
|---|---|---|
| `baseline` | Plain FP16, no optimizations — the zero line | ~14 GB |
| `prefix_cache` | `--enable-prefix-caching` | ~14 GB |
| `chunked_prefill` | `--enable-chunked-prefill` | ~14 GB |
| `awq_quant` | AWQ INT4 quantized model | ~7 GB |
| `spec_decode` | Speculative decoding (7B target + 0.5B draft) | ~16 GB |
| `dual_model` | Two vLLM instances on :8000 and :8001 | ~21 GB |

```bash
# List scenarios
python -m src.cli scenarios list

# Show full config of a scenario
python -m src.cli scenarios show baseline

# Start a specific scenario
make start SCENARIO=prefix_cache
```

---

## How it works

### Startup flow

```
make start SCENARIO=baseline
        │
        ▼
1. Load scenario config   (src/models/baseline.py)
        │
        ▼
2. Doctor checks          (10 pre-flight checks — fails fast if issues found)
        │
        ▼
3. Download model         (HF Hub → HF_HOME, idempotent)
        │
        ▼
4. Start observer agent   (FastAPI on :9100, tmux session: agent)
        │
        ▼
5. Start vLLM server      (OpenAI-compatible on :8000, tmux session: vllm)
        │
        ▼
6. Warmup                 (20 requests to stabilise KV cache + TTFT)
        │
        ▼
7. Print connection info  (public IP, endpoints, .env snippet)
```

### Doctor checks (run before any GPU cost)

The doctor runs 10 checks and aborts with actionable advice on failure:

```
[1/10] CUDA + Driver
[2/10] Disk space           (≥50 GB free)
[3/10] VRAM                 (≥ model size × 1.1)
[4/10] Network speed        (measures HF download throughput, estimates cost)
[5/10] HuggingFace Hub      (reachability + token validity)
[6/10] Ports 8000 + 9100    (both must be free)
[7/10] Cached models        (informational)
[8/10] tmux                 (must be installed)
[9/10] Existing sessions    (warns if orphaned vllm/agent sessions found)
```

Skip the network test if you know the connection is fast:

```bash
python -m src.cli doctor --skip-network
```

### Observer agent endpoints

The agent runs on `:9100` and streams GPU metrics. All endpoints except `/health` require the `x-api-key` header set to `AGENT_SECRET`.

| Endpoint | Auth | Description |
|---|---|---|
| `GET /health` | No | Health check |
| `GET /gpu?index=0` | Yes | Current GPU metrics JSON |
| `GET /gpus` | Yes | All GPUs |
| `GET /system` | Yes | CPU / RAM / hostname |
| `GET /stream?x_api_key=…` | Query param | SSE stream at 500 ms |

```bash
# Example (replace with your AGENT_SECRET)
curl -H "x-api-key: your_agent_secret" http://localhost:9100/gpu
```

Response shape:

```json
{
  "ts": 1748612345.123,
  "gpu_index": 0,
  "gpu_name": "NVIDIA RTX PRO 6000",
  "gpu_util": 87,
  "vram_used_mb": 14200,
  "vram_total_mb": 98304,
  "power_w": 287.4,
  "temp_c": 71,
  "hostname": "gpu-host-4a21"
}
```

---

## Testing without a GPU

Some parts can be tested on a CPU-only machine (the doctor will fail the CUDA check, but the code structure can be verified):

```bash
pip install -r requirements.txt

# Verify CLI loads
python -m src.cli --help
python -m src.cli scenarios list
python -m src.cli scenarios show baseline

# Run doctor (will fail CUDA + VRAM — that's expected)
python -m src.cli doctor --skip-network
```

To test the observer agent in isolation (requires `pynvml` and a GPU):

```bash
uvicorn src.observer_agent.server:app --host 0.0.0.0 --port 9100
curl http://localhost:9100/health
```

---

## Logs and debugging

```bash
# Tail live logs from both services
make logs

# Attach to a session to see raw output
make attach SVC=vllm
make attach SVC=agent
# Press Ctrl-B then D to detach without killing

# Check individual log files
tail -f logs/vllm.log
tail -f logs/agent.log
```

### Common problems

| Symptom | Cause | Fix |
|---|---|---|
| Doctor fails: "CUDA + Driver" | No GPU / wrong driver | Check `nvidia-smi` |
| Doctor fails: "VRAM" | Model too large for GPU | Use `awq_quant` scenario |
| Doctor fails: "Network speed" | Slow host | Pick a different Vast.ai host, or `--skip-network` |
| `make start` hangs at "Waiting for vLLM" | vLLM crashed on load | `make attach SVC=vllm` to see the error |
| Port 8000 already in use | Previous run not cleaned | `make stop` |
| 401 on `/gpu` endpoint | Wrong `AGENT_SECRET` | Check `.env` |

---

## Project structure

```
.
├── Makefile                    ← one-command entry point
├── requirements.txt
├── .env.example
├── scripts/
│   ├── install.sh
│   └── tunnel.sh               ← optional Tailscale VPN setup
├── src/
│   ├── cli.py                  ← Click CLI entry point
│   ├── orchestrator.py         ← lifecycle chain
│   ├── doctor/                 ← 9 pre-flight check modules
│   ├── observer_agent/         ← FastAPI metrics server on :9100
│   ├── models/                 ← scenario configs (one .py per scenario)
│   ├── lifecycle/              ← discrete steps: download, start, warmup, health
│   └── utils/                  ← tmux, logging, env, http helpers
└── logs/                       ← tmux pipe output (created at runtime)
```

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Doctor checks failed |
| 2 | Service failed to start or become healthy |
| 3 | User error (bad scenario name, missing env var) |
