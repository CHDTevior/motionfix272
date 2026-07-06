# motionfix272

Standalone conversion code for turning MotionFix raw pose pairs into the
HumanML3D/MotionStreamer 272D representation used by our MotionFix RVQ runs.

This repo is intentionally small. It contains only conversion and verification
code, not generated `.npy` data, checkpoints, or training code.

## What This Reproduces

The conversion target is the dataset currently used in our project:

```text
/mnt/afs/mogo_base/datasets/MotionFix/motionstreamer272_hml_joint_vecs
/mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionstreamer272_hml_{train,val,test}.jsonl
/mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionstreamer272_hml_summary.json
/mnt/afs/mogo_base/datasets/MotionFix/stats/motionstreamer272_hml_source_target_train/Mean.npy
/mnt/afs/mogo_base/datasets/MotionFix/stats/motionstreamer272_hml_source_target_train/Std.npy
```

Source MotionFix raw data on our machine:

```text
/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset/motionfix.pth.tar
/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset/motionfix_val.pth.tar
/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset/motionfix_test.pth.tar
/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset/splits.json
```

Expected full conversion counts from our validated run:

```text
train records: 5387
val records:   330
test records:  1013
motion files:  13460 source/target .npy files
train stats:   1193438 source+target frames
```

## 272D Schema

Each output `.npy` is `float32` with shape `[T, 272]`:

```text
[0:2]     root x/z velocity
[2:8]     heading delta as 6D rotation
[8:74]    22 local joint positions, flattened as 22 * xyz
[74:140]  22 local joint position velocities, flattened as 22 * xyz
[140:272] 22 local joint rotations, flattened as 22 * rot6d
```

Compatibility contract:

```text
conversion_version: motionfix_motionstreamer272_hml
coordinate system:  y-up, x/z ground plane
initial facing:     hips/shoulders skeleton forward aligned to +Z
canonical frame:    per-sequence HumanML3D style
raw MotionFix meta: not needed for recovery
joint correction:   no per-joint correction/inverse is applied
recover function:   recover_motionstreamer272_positions
```

Important: this is not the original MotionFix 207D representation. MotionFix raw
z-up coordinates are converted to MotionStreamer/HumanML-style y-up coordinates
with the basis `(x, y, z)_raw -> (x, z, -y)_hml`.

## Install

Use a Python environment with PyTorch installed. On our machines, this works:

```bash
cd /mnt/afs/UMO_debug/motionfix272
/root/miniconda3/envs/mogo/bin/python -m pip install -e .
```

Generic install:

```bash
python -m pip install -e .
```

## Full Conversion

Reproduce the full dataset at the same absolute output root:

```bash
cd /mnt/afs/UMO_debug/motionfix272
/root/miniconda3/envs/mogo/bin/python -m motionfix272.convert \
  --data-dir /mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset \
  --output-root /mnt/afs/mogo_base/datasets/MotionFix \
  --force
```

Equivalent wrapper:

```bash
cd /mnt/afs/UMO_debug/motionfix272
DATA_DIR=/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset \
OUTPUT_ROOT=/mnt/afs/mogo_base/datasets/MotionFix \
PATH=/root/miniconda3/envs/mogo/bin:$PATH \
bash scripts/convert_motionfix_to_hml272.sh
```

`--force` removes and rewrites only the selected feature directory:

```text
<output-root>/motionstreamer272_hml_joint_vecs
```

It also overwrites the matching manifest summary/stat files under
`<output-root>/manifests` and `<output-root>/stats`.

## Smoke Test

Run a tiny conversion first:

```bash
cd /mnt/afs/UMO_debug/motionfix272
rm -rf /tmp/motionfix272_smoke
/root/miniconda3/envs/mogo/bin/python -m motionfix272.convert \
  --data-dir /mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset \
  --output-root /tmp/motionfix272_smoke \
  --split val \
  --max-items-per-split 3 \
  --force
```

Expected smoke output:

```text
/tmp/motionfix272_smoke/motionstreamer272_hml_joint_vecs/val/*_{source,target}.npy
/tmp/motionfix272_smoke/manifests/motionfix_motionstreamer272_hml_val.jsonl
/tmp/motionfix272_smoke/manifests/motionfix_motionstreamer272_hml_summary.json
```

## Audit Converted Data

Basic shape, finite-value, and root-local-xz checks:

```bash
cd /mnt/afs/UMO_debug/motionfix272
/root/miniconda3/envs/mogo/bin/python -m motionfix272.audit \
  --output-root /mnt/afs/mogo_base/datasets/MotionFix
```

For a quick audit:

```bash
/root/miniconda3/envs/mogo/bin/python -m motionfix272.audit \
  --output-root /tmp/motionfix272_smoke \
  --split val \
  --max-records-per-split 3
```

## Output Manifest

Each manifest row contains:

```json
{
  "id": "002329",
  "instruction": "...",
  "source": "motionstreamer272_hml_joint_vecs/val/002329_source.npy",
  "target": "motionstreamer272_hml_joint_vecs/val/002329_target.npy",
  "source_len": 120,
  "target_len": 120,
  "feature_dim": 272,
  "fps": 30.0,
  "conversion_version": "motionfix_motionstreamer272_hml",
  "canonical_frame": "per_sequence_humanml3d",
  "initial_forward": "hips_shoulders_to_zplus",
  "direct_motionstreamer272_recover": true
}
```

## Notes For Downstream Training

Use the generated `Mean.npy` and `Std.npy` from:

```text
/mnt/afs/mogo_base/datasets/MotionFix/stats/motionstreamer272_hml_source_target_train
```

Those stats are computed over both source and target motions in the train split.
For VQ/RVQ training, the loss is expected to be computed on normalized 272D
features unless the downstream code explicitly changes that behavior.
