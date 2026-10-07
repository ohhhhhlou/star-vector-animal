#!/usr/bin/env bash
set -euo pipefail
cd /home/yanyiling/projects/star-vector
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec .venv/bin/python lora_training/train_lora.py   --output-dir outputs/animal-lora-smoke   --epochs 1   --batch-size 1   --gradient-accumulation 1   --learning-rate 1e-4   --max-steps 2   --validate-every 1   --validation-batches 1   --save-every 1   --num-workers 0
