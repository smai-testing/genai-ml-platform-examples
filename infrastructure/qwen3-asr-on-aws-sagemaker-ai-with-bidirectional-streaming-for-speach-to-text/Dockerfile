# SageMaker Bidirectional Streaming container for Qwen3-ASR
#
# Build:
#   docker build --platform linux/amd64 -t qwen3-asr-sagemaker .
#
# Run locally:
#   docker run --gpus all -p 8080:8080 \
#     -v /path/to/model:/opt/ml/model \
#     qwen3-asr-sagemaker

ARG CUDA_VERSION=12.8.0
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    libsndfile1 \
    ccache \
    software-properties-common \
    ffmpeg \
    ca-certificates \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install

WORKDIR /app

ENV MAX_JOBS=32
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache

ARG BUNDLE_FLASH_ATTENTION=true

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U pip setuptools wheel

RUN apt remove python3-blinker -y

# Install qwen-asr with vLLM backend + FastAPI/uvicorn for WebSocket serving
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U "qwen-asr[vllm]" fastapi "uvicorn[standard]" websockets

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        pip3 install -U flash-attn --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git; \
    fi

RUN rm -rf /root/.cache/pip

# Ensure the CUDA forward compatibility package is installed.
# This provides /usr/local/cuda/compat/libcuda.so.1 which bridges newer
# CUDA toolkit APIs to older host drivers (e.g. driver 535 on g6 instances).
RUN apt-get update && apt-get install -y --no-install-recommends cuda-compat-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && (ls -la /usr/local/cuda/compat/ 2>/dev/null || dpkg -L cuda-compat-12-8 | head -20)

COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# --------------------------------------------------------------------------
# REQUIRED: Docker label that tells SageMaker this container supports
# bidirectional streaming.
# --------------------------------------------------------------------------
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true

# SageMaker expects the container to listen on port 8080
EXPOSE 8080

# Default environment variables
ENV MODEL_PATH=/opt/ml/model
ENV GPU_MEMORY_UTILIZATION=0.9
ENV MAX_NEW_TOKENS=32
ENV HOST=0.0.0.0
ENV PORT=8080
ENV TRANSFORMERS_VERBOSITY=error

ENTRYPOINT ["/app/entrypoint.sh"]
