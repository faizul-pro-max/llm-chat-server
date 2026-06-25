#!/usr/bin/env bash
# Async-tail every service logfile in a tmux session (one pane per file), and
# also stream a combined tail to stdout so `docker compose logs log-tailer`
# works without attaching.
set -euo pipefail

LOG_DIR="${LOG_DIR:-/logs}"
SESSION="logs"

mkdir -p "$LOG_DIR"
# tail -F tolerates missing files, but touching them keeps the panes tidy until
# the services start writing.
touch "$LOG_DIR/vllm.log" "$LOG_DIR/agent.log"

# (Re)create the tmux session: one pane per logfile, stacked vertically.
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session  -d -s "$SESSION" -n services "tail -F $LOG_DIR/vllm.log"
tmux split-window -t "$SESSION":0 -v             "tail -F $LOG_DIR/agent.log"
tmux select-layout -t "$SESSION":0 even-vertical

echo "──────────────────────────────────────────────────────────────"
echo "  tmux log-tailer ready. Attach with:"
echo "    docker exec -it llm-log-tailer tmux attach -t $SESSION"
echo "  (detach with Ctrl-B then D)"
echo "──────────────────────────────────────────────────────────────"

# Keep PID 1 (and the tmux server) alive, and mirror logs to docker logs.
exec tail -F "$LOG_DIR/vllm.log" "$LOG_DIR/agent.log"
