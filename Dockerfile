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
#
# Authentication is read from the environment, never baked into the image:
#   AGENT_VCR_ADMIN_TOKEN  admin token (create projects, mint project tokens)
#   AGENT_VCR_AUTH_TOKEN   optional dashboard/recorder token for /api and /ws/live
# vcr-server refuses to bind 0.0.0.0 unless at least one of them is set.
# Tokens are not passed on the command line so they never show up in `ps`.
CMD ["sh", "-c", "exec vcr-server --host 0.0.0.0 --port ${PORT:-8000} --vcr-dir /data --data-dir /data --allowed-origin ${AGENT_VCR_APP_URL} --hosted"]
