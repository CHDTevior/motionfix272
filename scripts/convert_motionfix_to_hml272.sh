#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/afs/mogo_base/datasets/MotionFix}"

python -m motionfix272.convert \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --force
