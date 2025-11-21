#!/bin/bash
# This is the Kyutai-internal version.
set -ex

export LD_LIBRARY_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# Fix for "Failed to initialize NVML: Unknown Error" during build
# Default to 80 (Ampere) if not specified
export CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-80}

uvx --from 'huggingface_hub[cli]' huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} cargo run --features=cuda --bin=moshi-server -r -- $@
