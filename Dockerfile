# EcoCompute energy-methodology MLCube — production image (multi-stage).
# Stage 1 (builder) installs pinned deps into an isolated venv so the runtime
# image is reproducible and lean; Stage 2 copies only that venv.
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------------------------------------------------------------- builder --
FROM ${CUDA_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------- runtime --
FROM ${CUDA_IMAGE} AS runtime
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

ENTRYPOINT ["python3", "/workspace/entrypoint.py"]
CMD ["energy_estimate"]
