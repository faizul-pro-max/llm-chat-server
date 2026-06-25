# inference-server image: vLLM + the observer/metrics agent in one container,
# both supervised by supervisord. Built on the official vLLM image so CUDA,
# torch, and vLLM are already present — we only add the project's light deps.
FROM vllm/vllm-openai:latest

WORKDIR /app

# Process manager (runs vLLM and the agent side by side) + accelerate, which the
# naive HuggingFace `baseline` backend needs for device placement. vLLM already
# pulls in transformers/torch, so they are not reinstalled here.
RUN pip install --no-cache-dir supervisor accelerate

# Light Python deps (click, fastapi, uvicorn, pynvml, httpx, ...). Copied first
# so this layer is cached when only src/ changes.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY docker/supervisord.conf /etc/supervisor/supervisord.conf

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    LOG_DIR=/app/logs \
    AGENT_PORT=9100 \
    SCENARIO=baseline

RUN mkdir -p /app/logs

EXPOSE 8000 9100

# The base image sets an ENTRYPOINT that runs vLLM directly — clear it so our
# supervisord CMD (which runs vLLM *and* the agent) takes over.
ENTRYPOINT []
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
