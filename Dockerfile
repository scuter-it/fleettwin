# Default: PyTorch with CUDA 13, for NVIDIA GPUs on x86 and on Jetson with JetPack 7.
# CPU only: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.14.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS="ignore:Failed to initialize NumPy:UserWarning" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch==${TORCH_VERSION}
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY fleettwin ./fleettwin
RUN pip install --no-cache-dir ".[dev]"
COPY tests ./tests
COPY benchmarks ./benchmarks

CMD ["python", "-m", "fleettwin.cli"]
