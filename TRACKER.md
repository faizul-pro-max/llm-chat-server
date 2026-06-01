# GPU Server Orchestrator — Implementation Tracker

> Tracks build progress against the plan in PLAN.MD.
> Build order follows CLAUDE.ME § "Build Order".

---

## Phase 1 — Project Scaffolding
| File | Status | Notes |
|---|---|---|
| `Makefile` | ✅ Done | All make targets wired |
| `requirements.txt` | ✅ Done | Pinned versions per CLAUDE.ME |
| `.env.example` | ✅ Done | All required env vars |
| `scripts/install.sh` | ✅ Done | apt + pip install |
| `scripts/tunnel.sh` | ✅ Done | Tailscale optional setup |
| `logs/` directory | ✅ Done | Created at runtime |

## Phase 2 — Utils
| File | Status | Notes |
|---|---|---|
| `src/utils/__init__.py` | ✅ Done | |
| `src/utils/tmux.py` | ✅ Done | libtmux wrappers: create/kill/capture/is_running |
| `src/utils/logging.py` | ✅ Done | rich Console, section/info/success/error/warning |
| `src/utils/env.py` | ✅ Done | .env loading + required-var validation |
| `src/utils/http.py` | ✅ Done | shared httpx client with retries |

## Phase 3 — CLI Skeleton
| File | Status | Notes |
|---|---|---|
| `src/cli.py` | ✅ Done | All commands: doctor, start, stop, status, logs, scenarios |
| `src/__init__.py` | ✅ Done | |

## Phase 4 — Doctor (10 pre-flight checks)
| File | Status | Notes |
|---|---|---|
| `src/doctor/__init__.py` | ✅ Done | |
| `src/doctor/runner.py` | ✅ Done | CheckResult, run_all, formatted output |
| `src/doctor/check_cuda.py` | ✅ Done | nvidia-smi, driver ≥525, CUDA ≥12.1, PyTorch |
| `src/doctor/check_disk.py` | ✅ Done | Free disk ≥50 GB + VRAM check |
| `src/doctor/check_network.py` | ✅ Done | 100 MB HF download speed test |
| `src/doctor/check_ports.py` | ✅ Done | :8000 and :9100 free |
| `src/doctor/check_hf.py` | ✅ Done | HF Hub access + token validity |
| `src/doctor/check_cache.py` | ✅ Done | Already-downloaded model detection |

## Phase 5 — Observer Agent
| File | Status | Notes |
|---|---|---|
| `src/observer-agent/__init__.py` | ✅ Done | |
| `src/observer-agent/auth.py` | ✅ Done | x-api-key header check |
| `src/observer-agent/nvml_reader.py` | ✅ Done | pynvml: single + multi-GPU |
| `src/observer-agent/server.py` | ✅ Done | FastAPI: /gpu, /health, /system, /stream |

## Phase 6 — Scenario Configs
| File | Status | Notes |
|---|---|---|
| `src/models/__init__.py` | ✅ Done | |
| `src/models/_base.py` | ✅ Done | BaseScenario with pydantic validation |
| `src/models/baseline.py` | ✅ Done | FP16, no optimizations |
| `src/models/prefix_cache.py` | ✅ Done | --enable-prefix-caching |
| `src/models/chunked_prefill.py` | ✅ Done | --enable-chunked-prefill |
| `src/models/awq_quant.py` | ✅ Done | AWQ INT4 quantization |
| `src/models/spec_decode.py` | ✅ Done | Speculative decoding |
| `src/models/dual_model.py` | ✅ Done | 2× vLLM instances |

## Phase 7 — Lifecycle Steps
| File | Status | Notes |
|---|---|---|
| `src/lifecycle/__init__.py` | ✅ Done | |
| `src/lifecycle/download.py` | ✅ Done | HF Hub download with progress |
| `src/lifecycle/start_vllm.py` | ✅ Done | Launch vLLM in tmux |
| `src/lifecycle/start_agent.py` | ✅ Done | Launch observer agent in tmux |
| `src/lifecycle/warmup.py` | ✅ Done | 20 warmup requests |
| `src/lifecycle/health.py` | ✅ Done | wait_for_vllm, wait_for_agent |

## Phase 8 — Orchestrator
| File | Status | Notes |
|---|---|---|
| `src/orchestrator.py` | ✅ Done | Full lifecycle: doctor→download→start→warmup→ready |

---

## Summary

| Phase | Files | Status |
|---|---|---|
| 1 — Scaffolding | 6 | ✅ Complete |
| 2 — Utils | 5 | ✅ Complete |
| 3 — CLI | 2 | ✅ Complete |
| 4 — Doctor | 8 | ✅ Complete |
| 5 — Observer Agent | 4 | ✅ Complete |
| 6 — Scenarios | 8 | ✅ Complete |
| 7 — Lifecycle | 6 | ✅ Complete |
| 8 — Orchestrator | 1 | ✅ Complete |
| **Total** | **40** | **✅ All done** |

---

## Test Plan (run on real GPU box)

- [ ] `make doctor` — all 10 checks pass on clean Vast.ai instance
- [ ] `make doctor --skip-network` — skips network test
- [ ] `make start SCENARIO=baseline` — full lifecycle completes, vLLM serves on :8000
- [ ] `make status` — shows both services running
- [ ] `make logs` — tails combined logs
- [ ] `make attach SVC=vllm` — attaches to vLLM tmux session
- [ ] `make stop` — kills both sessions cleanly
- [ ] `make start SCENARIO=prefix_cache` — prefix caching enabled
- [ ] `make start SCENARIO=awq_quant` — AWQ model loads
- [ ] Observer agent `/gpu` endpoint returns valid JSON
- [ ] Observer agent `/stream` SSE stream sends at 500ms

---

## Known Gaps / Future Work

- `dual_model.py` scenario runs two vLLM instances — needs port management for second instance (`:8001`)
- `scripts/tunnel.sh` is a stub — Tailscale setup depends on user's account
- No integration tests yet (require real GPU hardware)
