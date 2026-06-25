# GPU Server Orchestrator

Python CLI that runs on a cloud GPU instance. Manages the full lifecycle of a vLLM inference server — model download, startup, warmup, health checks, and a live GPU metrics agent.

One command starts everything:

```bash
make start SCENARIO=baseline
```

When startup is complete, the orchestrator prints the public IP, mapped ports, and a ready-to-paste `.env` snippet for your Node.js app server.

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
| Ubuntu | 20.04 / 22.04 | Most cloud GPU provider default images work |
| Python | 3.10+ | `python3 --version` |
| NVIDIA Driver | ≥ 525 | `nvidia-smi` |
| CUDA | ≥ 12.1 | `nvcc --version` |
| tmux | any | installed by `make install` |
| speedtest | Ookla CLI | installed by `make install` |
| ngrok | any | installed by `make install` |

### Python dependencies

All pinned in [requirements.txt](requirements.txt). Installed by `make install`.

```
click, rich, httpx, fastapi, uvicorn, pynvml,
huggingface_hub, pydantic, python-dotenv, libtmux
```

> **vLLM is in a separate [requirements-gpu.txt](requirements-gpu.txt)** — it is heavy and only installs on Linux + CUDA, so it is kept out of `requirements.txt` to keep the no-GPU test path usable. `make install` installs it into `.venv` automatically on the GPU host.

### Accounts / tokens

| Service | Required? | Notes |
|---|---|---|
| Cloud GPU provider | Yes | Where you rent the GPU instance |
| HuggingFace | Optional | Required only for gated models (e.g. Llama). Set `HF_TOKEN` in `.env` |
| ngrok | Optional | Required only for `make tunnel`. Free account at [ngrok.com](https://ngrok.com) |

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
```

This runs [scripts/install.sh](scripts/install.sh) which installs:
- System packages: `tmux`, `curl`
- Ookla speedtest CLI (via packagecloud.io apt repo)
- ngrok (via ngrok apt repo)
- Python packages from `requirements.txt` into `.venv`
- GPU packages from `requirements-gpu.txt` (vLLM) into `.venv`

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```bash
VLLM_API_KEY=your_vllm_secret        # Bearer token for vLLM — same value in your Node.js .env
AGENT_SECRET=your_agent_secret       # x-api-key header for the observer agent
HF_TOKEN=hf_xxxxxxxxxxxx             # HuggingFace token (skip if using public models)
HF_HOME=/root/.cache/huggingface     # where models are stored — point to a persistent volume
LOG_DIR=./logs
NGROK_AUTHTOKEN=your_ngrok_token     # only needed for `make tunnel`
```

> Get your ngrok auth token at [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)

> vLLM is installed into `.venv` by `make install` (step 2) via [requirements-gpu.txt](requirements-gpu.txt) — no separate install step needed. To install it manually, use the venv pip: `.venv/bin/pip install -r requirements-gpu.txt`.

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
make help                        # show all available targets

make install                     # install system + Python deps (tmux, speedtest, ngrok, pip)
make start SCENARIO=name         # full lifecycle: doctor → download → start → warmup → ready
make stop                        # stop vLLM + agent tmux sessions
make status                      # show service status + HTTP health endpoints
make info                        # reprint connection banner (public IP, mapped ports, .env snippet)
make info SCENARIO=name          # use the scenario you started with (defaults to baseline)
make logs                        # tail combined log output
make attach SVC=vllm             # attach to vLLM tmux session (Ctrl-B D to detach)
make attach SVC=agent            # attach to agent tmux session
make tunnel                      # create ngrok HTTPS tunnels (run after make start)
make clean                       # stop services + delete log files (keeps models)
make teardown                    # undo make install: kill sessions, delete .venv, logs, .env
make teardown-full               # teardown + delete model cache + remove apt packages

# Doctor — all variants
make doctor                      # all 11 checks, rich table output
make doctor CHECK=network        # run one check only (see check names below)
make doctor SIMPLE=1             # plain dot-format output instead of table
make doctor CHECK=cuda SIMPLE=1  # combine: single check + plain output
make doctor-network              # shorthand — any check name works after the dash
make doctor-cuda
make doctor-ram
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

## Running with Docker

The stack can run as containers instead of bare-VM tmux sessions. Two services,
both with GPU access, sharing a model-cache volume:

| Container | Contents | Ports |
|---|---|---|
| `inference-server` | vLLM **+** observer/metrics agent, supervised by supervisord | 8000, 9100 |
| `log-tailer` | tmux service that follows every per-service log in split panes | — |

`doctor` and the orchestrator (warmup + connection banner) stay **host
commands** — they drive the stack from outside, they are not containerized.

### Prerequisites

- NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host
- Docker + Docker Compose v2
- Host Python deps for the orchestrator step: `pip install -r requirements.txt` (no GPU/vLLM needed on the host)

### Quick start

```bash
cp .env.example .env          # fill in secrets (VLLM_API_KEY, AGENT_SECRET, HF_TOKEN)

# Optional pre-flight on the host GPU (free, no container)
make doctor

# Build, start the stack, wait for health, warm up, print the .env banner
make docker-start SCENARIO=baseline
```

`make docker-start` runs `docker compose up -d --build`, then runs the host-side
orchestrator (`src.cli await-ready`) which polls both `/health` endpoints, sends
warmup requests, and prints the same connection banner as the bare-VM flow.

### Docker make targets

```bash
make docker-start SCENARIO=name   # build + up + wait-ready -> warmup -> banner
make docker-up    SCENARIO=name   # start detached only (skip warmup/banner)
make docker-logs                  # attach to the tmux log-tailer (all services, split panes)
make docker-tail                  # stream combined container logs (no tmux)
make docker-status                # container + health status
make docker-stop                  # stop containers (keep volumes/models)
make docker-down                  # stop + remove containers (keeps model cache volume)
make docker-clean                 # docker-down + delete the model cache volume
```

> **Logs:** each inner service writes its own file to `./logs/` on the host
> (`vllm.log`, `agent.log`) via the bind-mounted logs volume. `make docker-logs`
> attaches to the tmux session that tails them (`Ctrl-B` then `D` to detach).

> **Model cache** lives on the `inference-server` in the named `hf-cache` volume
> (`HF_HOME=/root/.cache/huggingface`). It survives `docker-down`; only
> `docker-clean` deletes it. The first start downloads the model — allow several
> minutes (the healthcheck `start_period` is 10 min).

> **Scenarios:** `SCENARIO=<name>` selects the model + vLLM flags; the
> `inference-server` entrypoint runs `src.cli serve --scenario $SCENARIO`, which
> exec()s vLLM (or the HF baseline server) directly so supervisord supervises it.
> The `dual_model` scenario needs a second vLLM program — publish `8001:8001` and
> add a `serve --instance secondary` program to `docker/supervisord.conf`.

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
2. Doctor checks          (11 checks — all run, rich table shown at the end)
        │
        ▼
3. Download model         (HF Hub → HF_HOME, idempotent — skips if already cached)
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
7. Print connection info  (public IP, provider-mapped ports, .env snippet)
```

### Connection info output

At the end of `make start`, the orchestrator prints the actual external address to connect from your Node.js server:

```
✓  READY FOR BENCHMARKS

  Scenario:          baseline
  Model:             Qwen/Qwen2.5-7B-Instruct
  Public IP:         45.32.11.8

  Endpoints:
  vLLM:              http://45.32.11.8:12453  (internal :8000)
  Agent:             http://45.32.11.8:12454  (internal :9100)

  Paste into Node.js .env:
  GPU_SERVER_IP=45.32.11.8
  VLLM_PORT=12453
  GPU_AGENT_PORT=12454
```

> The cloud GPU provider maps internal ports to external ports automatically. The orchestrator reads the provider-injected `VAST_TCP_PORT_8000` / `VAST_TCP_PORT_9100` env vars to resolve the real external ports.

### Doctor checks

The doctor runs 11 checks. **All checks always run** — results are shown together at the end so you can see the full picture even when multiple things fail.

#### Output formats

**Default — rich table (great for screenshots):**

```
╭──────────────────────────────────────────────────────────────╮
│        🩺  GPU Server Orchestrator — Pre-flight Doctor        │
╰──────────────────────────────────────────────────────────────╯

╭────┬──────────────────────┬────────┬──────────────────────────────────────╮
│  # │ Check                │ Status │ Result                               │
├────┼──────────────────────┼────────┼──────────────────────────────────────┤
│  1 │ CUDA + Driver        │   ✓    │ 535.18 / CUDA 12.3 / torch 2.4.0    │
│  2 │ Disk space           │   ✓    │ 412 GB free on /root                 │
│  3 │ VRAM                 │   ✓    │ 1× RTX PRO 6000 / 96 GB free         │
│  4 │ CPU cores            │   ✓    │ 16 physical cores, 32 threads        │
│  5 │ RAM                  │   ✓    │ 28.3 GB free / 32.0 GB total (11%)   │
│  6 │ Network speed        │   ✓    │ ↓ 8.4 MB/s  ↑ 6.2 MB/s (~27 min)   │
│  7 │ HuggingFace Hub      │   ✓    │ Authenticated as user@org            │
│  8 │ Ports 8000 + 9100    │   ✓    │ Both available                       │
│  9 │ Cached models        │   ⓘ    │ Qwen2.5-7B-Instruct (14.2 GB)       │
│ 10 │ tmux                 │   ✓    │ tmux 3.2a                            │
│ 11 │ Existing sessions    │   ✓    │ None running                         │
╰────┴──────────────────────┴────────┴──────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────╮
│   ✓  ALL CHECKS PASSED — ready to start scenarios            │
╰──────────────────────────────────────────────────────────────╯
```

**Plain dot-format (`SIMPLE=1`) — compact, CI-friendly:**

```
[1/11] CUDA + Driver .............. ✓ 535.18 / CUDA 12.3
[2/11] Disk space ................. ✓ 412 GB free on /root
...
```

#### Check reference

| Key | Check | Pass criteria |
|---|---|---|
| `cuda` | CUDA + Driver | nvidia-smi works, driver ≥ 525, CUDA ≥ 12.1 |
| `disk` | Disk space | ≥ 50 GB free on model cache path |
| `vram` | VRAM | ≥ model size × 1.1 GB free |
| `cpu` | CPU cores | ≥ 4 physical cores |
| `ram` | RAM | ≥ 16 GB required, ≥ 32 GB recommended |
| `network` | Network speed | ↓ ↑ via Ookla speedtest CLI (HTTP fallback) |
| `hf` | HuggingFace Hub | reachable + token valid |
| `ports` | Ports 8000 + 9100 | both unbound |
| `cache` | Cached models | informational only |
| `tmux` | tmux | installed |
| `sessions` | Existing sessions | no orphaned vllm/agent sessions |

#### Running individual checks

```bash
# Single check — table output
make doctor CHECK=network
make doctor CHECK=cuda
make doctor CHECK=ram

# Shorthand — append check name after the dash
make doctor-network
make doctor-cuda
make doctor-vram
make doctor-cpu
make doctor-ram
make doctor-hf
make doctor-ports
make doctor-cache
make doctor-tmux
make doctor-sessions

# Single check + plain output
make doctor CHECK=network SIMPLE=1

# Skip network check when running all
python -m src.cli doctor --skip-network
```

---

## ngrok tunnel

If the provider's port mapping is blocked or you want a stable HTTPS URL, use the ngrok tunnel:

```bash
# After make start, in a second terminal:
make tunnel
```

Output:

```
════════════════════════════════════════════════════════════
  ngrok tunnels active
════════════════════════════════════════════════════════════

  ✓ vLLM:   https://a1b2c3d4.ngrok-free.app
  ✓ Agent:  https://e5f6g7h8.ngrok-free.app

  Add to your Node.js app server .env:
  VLLM_BASE_URL=https://a1b2c3d4.ngrok-free.app
  GPU_AGENT_URL=https://e5f6g7h8.ngrok-free.app
```

> **Free ngrok tier** supports 1 simultaneous tunnel. Upgrade to a paid plan to run both vLLM and agent tunnels at the same time.

The ngrok dashboard is available at `http://localhost:4040` on the GPU box while the tunnel is active.

To stop the tunnel:

```bash
pkill -f ngrok
```

---

## Observer agent endpoints

The agent runs on `:9100` and streams GPU metrics. All endpoints except `/health` require the `x-api-key` header set to `AGENT_SECRET`.

| Endpoint | Auth | Description |
|---|---|---|
| `GET /health` | No | Health check |
| `GET /gpu?index=0` | Yes | Current GPU metrics JSON |
| `GET /gpus` | Yes | All GPUs |
| `GET /system` | Yes | CPU / RAM / hostname |
| `GET /stream?x_api_key=…` | Query param | SSE stream at 500 ms intervals |

```bash
# Example (replace with your AGENT_SECRET and actual port)
curl -H "x-api-key: your_agent_secret" http://45.32.11.8:12454/gpu
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

## Teardown

To undo everything set up by `make install` and `make start`:

```bash
# Safe teardown — keeps model cache
make teardown

# Full teardown — also deletes model cache and removes apt packages
make teardown-full
```

What each removes:

| Step | `make teardown` | `make teardown-full` |
|---|---|---|
| Kill vllm + agent tmux sessions | Yes | Yes |
| Delete `logs/` | Yes | Yes |
| Delete `.env` | Yes | Yes |
| Delete `.venv/` | Yes | Yes |
| Remove `__pycache__` / `.pyc` | Yes | Yes |
| Delete HF model cache (`HF_HOME`) | No | Yes |
| `apt remove tmux curl` | No | Yes |

---

## Testing without a GPU

The CLI and doctor can be tested on a CPU-only machine (your MacBook, a CI runner, etc.). The CUDA check will fail — that is expected.

```bash
pip install -r requirements.txt

# Verify CLI loads
python -m src.cli --help
python -m src.cli scenarios list
python -m src.cli scenarios show baseline

# Run doctor (CUDA + VRAM will fail — everything else runs)
python -m src.cli doctor --skip-network
```

To test the observer agent in isolation (requires a GPU with pynvml):

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
tail -f logs/ngrok.log
```

### Common problems

| Symptom | Cause | Fix |
|---|---|---|
| Doctor: "CUDA + Driver" fails | No GPU / wrong driver | `nvidia-smi` to diagnose |
| Doctor: "VRAM" fails | Model too large for GPU | Use `awq_quant` scenario |
| Doctor: "Network speed" warns | Slow connection | Pick a faster GPU host, or `--skip-network` |
| `make start` hangs at "Waiting for vLLM" | vLLM crashed on load | `make attach SVC=vllm` to see the error |
| Port 8000 already in use | Previous run not cleaned | `make stop` |
| 401 on `/gpu` endpoint | Wrong `AGENT_SECRET` | Check `.env` |
| ngrok shows only 1 tunnel | Free tier limit | Upgrade ngrok plan or use the provider's port mapping |

---

## Project structure

```
.
├── Makefile                    ← one-command entry point
├── requirements.txt
├── .env.example
├── scripts/
│   ├── install.sh              ← installs tmux, speedtest CLI, ngrok, Python deps
│   ├── tunnel.sh               ← ngrok tunnel for vLLM + agent
│   └── teardown.sh             ← undo all install steps
├── src/
│   ├── cli.py                  ← Click CLI entry point
│   ├── orchestrator.py         ← lifecycle chain + connection info output
│   ├── doctor/                 ← 9 pre-flight check modules
│   ├── observer_agent/         ← FastAPI GPU metrics server on :9100
│   ├── models/                 ← scenario configs (one .py per scenario)
│   ├── lifecycle/              ← discrete steps: download, start, warmup, health
│   └── utils/                  ← tmux, logging, env, http helpers
└── logs/                       ← tmux + ngrok output (created at runtime)
```

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Doctor checks failed |
| 2 | Service failed to start or become healthy |
| 3 | User error (bad scenario name, missing env var) |
