#!/bin/bash
# Entrypoint script for Qwen3-ASR SageMaker container.
#
# Enables NVIDIA CUDA forward compatibility when the host driver is older
# than the CUDA toolkit bundled in the container.  This allows the container
# (built with CUDA 12.8) to run on SageMaker instances with older drivers
# (e.g. g6 with driver 535 / CUDA 12.2).
#
# See: https://docs.aws.amazon.com/sagemaker/latest/dg/inference-gpu-drivers.html
#      https://docs.nvidia.com/deploy/cuda-compatibility/index.html

set -euo pipefail

log() { echo "[entrypoint] $*" >&2; }

# ---------------------------------------------------------------------------
# Debug: show CUDA and driver environment
# ---------------------------------------------------------------------------
log "=== CUDA Compatibility Check ==="
log "Contents of /usr/local/cuda/compat/:"
ls -la /usr/local/cuda/compat/ >&2 2>/dev/null || log "  (directory not found)"

log "NVIDIA driver version from /proc:"
cat /proc/driver/nvidia/version >&2 2>/dev/null || log "  (not available)"

log "nvidia-smi output:"
nvidia-smi >&2 2>/dev/null || log "  (nvidia-smi not available)"

log "LD_LIBRARY_PATH before compat: ${LD_LIBRARY_PATH:-<unset>}"

# ---------------------------------------------------------------------------
# Dynamic CUDA Compatibility Package activation
# (from AWS docs: https://docs.aws.amazon.com/sagemaker/latest/dg/inference-gpu-drivers.html)
# ---------------------------------------------------------------------------
verlt() {
    [ "$1" = "$2" ] && return 1 || [ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink /usr/local/cuda/compat/libcuda.so.1 | cut -d'.' -f 3-)
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)

    log "Host driver version: ${NVIDIA_DRIVER_VERSION:-unknown}"
    log "Compat max driver version: ${CUDA_COMPAT_MAX_DRIVER_VERSION:-unknown}"

    if [ -n "$NVIDIA_DRIVER_VERSION" ] && [ -n "$CUDA_COMPAT_MAX_DRIVER_VERSION" ]; then
        if verlt "$NVIDIA_DRIVER_VERSION" "$CUDA_COMPAT_MAX_DRIVER_VERSION"; then
            log "Enabling CUDA forward compatibility (host $NVIDIA_DRIVER_VERSION < compat $CUDA_COMPAT_MAX_DRIVER_VERSION)"
            export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
        else
            log "CUDA compat not needed (host $NVIDIA_DRIVER_VERSION >= compat $CUDA_COMPAT_MAX_DRIVER_VERSION)"
        fi
    else
        log "WARNING: Could not determine driver versions, enabling compat as fallback"
        export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
    fi
else
    log "WARNING: CUDA compat package not found at /usr/local/cuda/compat/libcuda.so.1"
fi

log "LD_LIBRARY_PATH after compat: ${LD_LIBRARY_PATH:-<unset>}"
log "=== Starting application ==="

# ---------------------------------------------------------------------------
# Launch the application
# ---------------------------------------------------------------------------
exec python app.py serve
