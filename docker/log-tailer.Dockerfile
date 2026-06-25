# log-tailer: a tiny tmux "service" that follows every per-service logfile from
# the shared logs volume in split panes. Attach to debug all services at once:
#   docker exec -it llm-log-tailer tmux attach -t logs
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux procps \
    && rm -rf /var/lib/apt/lists/*

COPY docker/log-tailer-entrypoint.sh /usr/local/bin/log-tailer-entrypoint.sh
RUN chmod +x /usr/local/bin/log-tailer-entrypoint.sh

WORKDIR /logs
CMD ["/usr/local/bin/log-tailer-entrypoint.sh"]
