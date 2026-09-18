# =============================================================================
# Hardened, Network-Isolated Sandbox Image for DSA Agent
# =============================================================================
# Build:
#   docker build -t dsa-sandbox:latest .
#
# Containment Flags used by DockerSandbox:
#   docker run --rm \
#     --network none \
#     --read-only \
#     --tmpfs /tmp:rw,size=100m \
#     -v /path/to/scratch:/scratch:rw \
#     --memory 512m \
#     --cpus 1.0 \
#     --pids-limit 50 \
#     dsa-sandbox:latest /scratch/input.json /scratch/result.json
# =============================================================================

FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -g 1000 sandbox && \
    useradd -u 1000 -g sandbox -s /bin/sh -m sandbox

WORKDIR /app

# Install standard analytical packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code into the container image
COPY src/ /app/src/

# Environment configuration
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Run as non-privileged sandbox user
USER sandbox

# Worker entrypoint
ENTRYPOINT ["python", "-m", "src.core._sandbox_worker"]
