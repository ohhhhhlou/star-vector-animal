#!/usr/bin/env bash
set -euo pipefail
cd /home/yanyiling/projects/star-vector
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec .venv/bin/python lora_training/train_lora.py --output-dir outputs/animal-lora --epochs 3 --batch-size 1 --gradient-accumulation 8 --learning-rate 1e-4 --validate-every 100 --validation-batches 8 --save-every 100 --num-workers 2
