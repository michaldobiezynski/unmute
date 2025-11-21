#!/bin/bash
# This is the public-facing version.
set -ex

export LD_LIBRARY_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# Fix for "Failed to initialize NVML: Unknown Error" during build
# Default to 80 (Ampere) if not specified
export CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-80}

uvx --from 'huggingface_hub[cli]' huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

CARGO_TARGET_DIR=/app/target CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} cargo install --features cuda moshi-server@0.6.3

# Subtle detail here: We use the full path to `moshi-server` because there is a `moshi-server` binary
# from the `moshi` Python package. We'll fix this conflict soon.
/root/.cargo/bin/moshi-server $@