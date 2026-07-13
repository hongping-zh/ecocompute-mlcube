# EcoCompute energy-methodology MLCube
# CUDA runtime + PyTorch + transformers + bitsandbytes + NVML bindings.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/models/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /workspace/requirements.txt

COPY entrypoint.py /workspace/entrypoint.py
COPY workspace/parameters /workspace/parameters

# MLCube mounts task inputs/outputs at run time; default task:
ENTRYPOINT ["python3", "/workspace/entrypoint.py"]
CMD ["run"]
