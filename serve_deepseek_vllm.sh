#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate deepseek

# Set offline environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Serve the model with vLLM
vllm serve /opt/llm/deepseek-r1-14b \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 