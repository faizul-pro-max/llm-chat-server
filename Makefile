SCENARIO     ?= baseline
SVC          ?= vllm
CHECK        ?=
SIMPLE       ?=
NO_DOWNLOAD  ?=
NO_WARMUP    ?=
PYTHON       ?= .venv/bin/python3
COMPOSE      ?= docker compose

.PHONY: help install doctor start stop status info logs attach clean teardown tunnel \
        docker-build docker-up docker-start docker-logs docker-tail docker-status \
        docker-stop docker-down docker-clean

help:
	@echo "GPU Server Orchestrator — make targets:"
	@echo ""
	@echo "  make install              Install system + Python deps"
	@echo "  make doctor               Run all pre-flight checks (table view)"
	@echo "  make doctor CHECK=network Run a single check  (cuda|vllm|disk|vram|cpu|ram|network|hf|ports|cache|tmux|sessions)"
	@echo "  make doctor SIMPLE=1      Plain dot-format output instead of table"
	@echo "  make doctor-network       Shorthand for CHECK=network (any check name works)"
	@echo "  make start SCENARIO=name                    Full lifecycle: doctor -> download -> start -> warmup"
	@echo "  make start SCENARIO=name NO_DOWNLOAD=1      Skip model download (already cached)"
	@echo "  make start SCENARIO=name NO_WARMUP=1        Skip warmup requests"
	@echo "  make stop                 Stop all services (vLLM + agent)"
	@echo "  make stop-agent           Stop ONLY the agent (vLLM keeps running); also make stop-vllm"
	@echo "  make status               Show service status"
	@echo "  make info                 Reprint connection banner (public IP, ports, .env snippet)"
	@echo "  make logs                 Tail combined logs"
	@echo "  make attach SVC=vllm      Attach to tmux session (vllm or agent)"
	@echo "  make clean                Stop services + clean log files (keeps models)"
	@echo "  make teardown             Undo make install: kill sessions, delete .venv, logs, .env"
	@echo "  make teardown-full        teardown + delete model cache + remove apt packages"
	@echo ""
	@echo "Docker (containerized stack — inference-server + log-tailer):"
	@echo "  make docker-start SCENARIO=name   Build + up + wait-ready -> warmup -> banner"
	@echo "  make docker-up SCENARIO=name      Start detached (skip warmup/banner)"
	@echo "  make docker-logs                  Attach to tmux log-tailer (all services)"
	@echo "  make docker-status                Container + health status"
	@echo "  make docker-down                  Stop + remove containers (keep models)"
	@echo "  make docker-clean                 docker-down + delete model cache volume"
	@echo ""
	@echo "Scenarios available: $$(ls src/models/*.py 2>/dev/null | xargs -n1 basename | sed 's/\.py//' | grep -v '^_' | tr '\n' ' ')"

install:
	bash scripts/install.sh

doctor:
	$(PYTHON) -m src.cli doctor \
	  $(if $(CHECK),--check $(CHECK),) \
	  $(if $(SIMPLE),--simple,)

# make doctor-network / make doctor-cuda / make doctor-ram ...
doctor-%:
	$(PYTHON) -m src.cli doctor --check $*

start:
	$(PYTHON) -m src.cli start --scenario $(SCENARIO) \
	  $(if $(NO_DOWNLOAD),--no-download,) \
	  $(if $(NO_WARMUP),--no-warmup,)

stop:
	$(PYTHON) -m src.cli stop

# make stop-agent / make stop-vllm — stop a single service, leave the other running
stop-%:
	$(PYTHON) -m src.cli stop --service $*

status:
	$(PYTHON) -m src.cli status

info:
	$(PYTHON) -m src.cli info --scenario $(SCENARIO)

logs:
	$(PYTHON) -m src.cli logs

attach:
	tmux attach -t $(SVC)

tunnel:
	bash scripts/tunnel.sh

clean:
	$(PYTHON) -m src.cli stop
	rm -f logs/*.log

teardown:
	bash scripts/teardown.sh

teardown-full:
	bash scripts/teardown.sh --purge-models --full

# ── Docker ───────────────────────────────────────────────────────────────────
# inference-server (vLLM + agent) + log-tailer (tmux) run in containers; doctor
# and the orchestrator (warmup + banner) stay host commands.

docker-build:                                  ## Build both images
	$(COMPOSE) build

docker-up:                                     ## Start the stack detached (no warmup/banner)
	SCENARIO=$(SCENARIO) $(COMPOSE) up -d

docker-start:                                  ## Build + up, then wait-ready -> warmup -> banner
	SCENARIO=$(SCENARIO) $(COMPOSE) up -d --build
	$(PYTHON) -m src.cli await-ready --scenario $(SCENARIO) $(if $(NO_WARMUP),--no-warmup,)

docker-logs:                                   ## Attach to the tmux log-tailer (Ctrl-B D to detach)
	docker exec -it llm-log-tailer tmux attach -t logs

docker-tail:                                   ## Stream combined container logs (no tmux)
	$(COMPOSE) logs -f inference-server

docker-status:                                 ## Show container + health status
	$(COMPOSE) ps
	-$(PYTHON) -m src.cli status

docker-stop:                                   ## Stop containers (keep volumes/models)
	$(COMPOSE) stop

docker-down:                                   ## Stop + remove containers (keep model cache volume)
	$(COMPOSE) down

docker-clean:                                  ## down + delete the model cache volume
	$(COMPOSE) down -v
