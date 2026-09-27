#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
DATA_ROOT="${1:?Usage: bash run_main.sh DATA_ROOT OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: bash run_main.sh DATA_ROOT OUTPUT_DIR}"
COMMON=(--data-root "$DATA_ROOT" --save-dir "$OUTPUT_DIR" --target_dataset RUNMC --sequence-len 8 --batch-size-single-frame 5 --batch-size-multi-frames 5 --Mixup 1 --SegVREx 1)

# Single-frame initialization, followed by two multi-frame training stages.
python train.py "${COMMON[@]}" --train-single single --epoch-start 0 --epoch-end 40
python train.py "${COMMON[@]}" --epoch-start 40 --epoch-end 70 --checkpoint single_epoch40_last.pth
python train.py "${COMMON[@]}" --epoch-start 70 --epoch-end 100 --checkpoint multi_epoch70_last.pth --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002
