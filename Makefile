SCENARIO ?= baseline
SVC      ?= vllm
CHECK    ?=
SIMPLE   ?=
PYTHON   ?= .venv/bin/python3

.PHONY: help install doctor start stop status logs attach clean teardown tunnel

help:
	@echo "GPU Server Orchestrator — make targets:"
	@echo ""
	@echo "  make install              Install system + Python deps"
	@echo "  make doctor               Run all pre-flight checks (table view)"
	@echo "  make doctor CHECK=network Run a single check  (cuda|disk|vram|cpu|ram|network|hf|ports|cache|tmux|sessions)"
	@echo "  make doctor SIMPLE=1      Plain dot-format output instead of table"
	@echo "  make doctor-network       Shorthand for CHECK=network (any check name works)"
	@echo "  make start SCENARIO=name  Full lifecycle: doctor → download → start → warmup"
	@echo "  make stop                 Stop all services (vLLM + agent)"
	@echo "  make status               Show service status"
	@echo "  make logs                 Tail combined logs"
	@echo "  make attach SVC=vllm      Attach to tmux session (vllm or agent)"
	@echo "  make clean                Stop services + clean log files (keeps models)"
	@echo "  make teardown             Undo make install: kill sessions, delete .venv, logs, .env"
	@echo "  make teardown-full        teardown + delete model cache + remove apt packages"
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
	$(PYTHON) -m src.cli start --scenario $(SCENARIO)

stop:
	$(PYTHON) -m src.cli stop

status:
	$(PYTHON) -m src.cli status

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
