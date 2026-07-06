#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/afs/mogo_base/datasets/MotionFix}"

python -m motionfix272.audit \
  --output-root "${OUTPUT_ROOT}"
