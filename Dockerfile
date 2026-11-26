FROM python:3.11-slim-trixie@sha256:61e01f72680f86a5279ad510818ab7b2ee9f049822589d4e34e33e044360356c

# uv 0.12.10, pinned to the verified multi-platform index
COPY --from=ghcr.io/astral-sh/uv@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --group service --no-dev --no-install-project

ARG RUNTIME_WHEEL=sleep_classifier-0.1.0-py3-none-any.whl
COPY dist/ /tmp/wheels/
RUN uv pip install --python /app/.venv/bin/python --no-deps "/tmp/wheels/${RUNTIME_WHEEL}" \
    && rm -rf /tmp/wheels /root/.cache

USER 65532:65532
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "sleep_classifier.http_service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5"]
