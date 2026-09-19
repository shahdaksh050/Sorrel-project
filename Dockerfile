# =============================================================================
# Hardened, Network-Isolated Sandbox Image for DSA Agent
# =============================================================================
# Build (from the repo root; rebuild after changing src/):
#   docker build -t dsa-sandbox:latest .
#
# Containment flags used by DockerSandbox (src/core/sandbox.py), plus the
# seccomp profile documented in docker/README.md:
#   docker run --rm \
#     --network none \
#     --read-only \
#     --cap-drop ALL \
#     --security-opt no-new-privileges \
#     --security-opt seccomp=<abs path>/docker/seccomp-sandbox.json \
#     --tmpfs /tmp:rw,size=100m \
#     -v /path/to/scratch:/scratch:rw \
#     --memory 512m \
#     --cpus 1.0 \
#     --pids-limit 50 \
#     dsa-sandbox:latest /scratch/input.json /scratch/result.json
# =============================================================================

# Same Python minor as the host venv, so sandboxed pandas/numpy match the
# subprocess backend. To pin reproducibly, resolve the digest and append it:
#   docker buildx imagetools inspect python:3.13-slim
#   FROM python:3.13-slim@sha256:<digest>
FROM python:3.13-slim

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root user with no login shell. No apt packages are installed.
RUN groupadd -g 1000 sandbox && \
    useradd -u 1000 -g sandbox -s /usr/sbin/nologin -M sandbox

WORKDIR /app

# Worker dependencies only (not the UI/LLM/reporting stack), then remove pip
# so nothing in the image can install packages at runtime.
COPY docker/requirements-sandbox.txt /tmp/requirements-sandbox.txt
RUN pip install -r /tmp/requirements-sandbox.txt && \
    rm /tmp/requirements-sandbox.txt && \
    pip uninstall -y pip

COPY --chown=root:root src/ /app/src/

# Next step, if the shell must go too: move to a distroless Python base on
# the same minor version; the exec-form ENTRYPOINT below never uses a shell.
USER sandbox

ENTRYPOINT ["python", "-m", "src.core._sandbox_worker"]
