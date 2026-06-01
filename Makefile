SCENARIO ?= baseline
SVC ?= vllm
PYTHON ?= .venv/bin/python3

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
	@echo "  make clean                Stop services + clean log files (keeps models)"
	@echo ""
	@echo "Scenarios available: $$(ls src/models/*.py 2>/dev/null | xargs -n1 basename | sed 's/\.py//' | grep -v '^_' | tr '\n' ' ')"

install:
	bash scripts/install.sh

doctor:
	$(PYTHON) -m src.cli doctor

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

clean:
	$(PYTHON) -m src.cli stop
	rm -f logs/*.log
