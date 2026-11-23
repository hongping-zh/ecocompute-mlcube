# EcoCompute energy-methodology MLCube — production image (multi-stage).
# Stage 1 (builder) installs pinned deps into an isolated venv so the runtime
# image is reproducible and lean; Stage 2 copies only that venv.
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# Build-time identity carried into the runtime stage and read by
# entrypoint.py for the report provenance block (the registry tag is
# a moving label; these fields are what a report can be traced by).
ARG ECOMPUTE_CODE_COMMIT=unavailable
ARG ECOMPUTE_IMAGE_REF=unavailable

# ---------------------------------------------------------------- builder --
FROM ${CUDA_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Install the fully-pinned lock for reproducible, deterministic builds.
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.lock.txt

# ---------------------------------------------------------------- runtime --
FROM ${CUDA_IMAGE} AS runtime
# Re-declare to receive the global ARG values into this stage.
ARG ECOMPUTE_CODE_COMMIT=unavailable
ARG ECOMPUTE_IMAGE_REF=unavailable
ENV ECOMPUTE_CODE_COMMIT=${ECOMPUTE_CODE_COMMIT} \
    ECOMPUTE_IMAGE_REF=${ECOMPUTE_IMAGE_REF}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/workspace/models/.hf
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
COPY entrypoint.py /workspace/entrypoint.py
COPY workspace/parameters /workspace/parameters
# Shipped so every run can self-check its report before claiming schema validity.
COPY schema /workspace/schema
# The perplexity probe's fixed evaluation text: vendored so a run needs no network
# and every contributor scores the same bytes.
COPY quality /workspace/quality

# ------------------------------------------------------------------ non-root --
# The image runs as an unprivileged user. Runtime writable locations (outputs,
# HF cache) are group-writable so an arbitrary `docker run --user UID:GID`
# (what quickstart.sh always passes) keeps working regardless of the host uid.
RUN useradd --create-home --uid 1000 --user-group ecocompute \
 && mkdir -p /workspace/outputs /workspace/models/.hf \
 && chmod 0777 /workspace/outputs /workspace/models /workspace/models/.hf
USER ecocompute

ENTRYPOINT ["python3", "/workspace/entrypoint.py"]
CMD ["energy_estimate"]
