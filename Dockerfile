# Railway production image for the optional hosted Agent VCR control plane.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system agentvcr \
    && useradd --system --create-home --gid agentvcr agentvcr \
    && mkdir /data \
    && chown agentvcr:agentvcr /data

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[dashboard,hosted]"

USER agentvcr
EXPOSE 8000

# Railway injects PORT. A mounted Railway Volume should be mounted at /data.
CMD ["sh", "-c", "vcr-server --host 0.0.0.0 --port ${PORT:-8000} --vcr-dir /data --data-dir /data --allowed-origin ${AGENT_VCR_APP_URL} --hosted"]
