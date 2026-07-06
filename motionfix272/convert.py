"""Convert MotionFix raw pose pairs to HumanML3D/MotionStreamer 272D features.

The output layout is the HML272 variant used by MotionStreamer:

  [0:2]     root x/z velocity in local heading-free coordinates
  [2:8]     heading delta as 6D rotation
  [8:74]    22 local joint positions
  [74:140]  22 local joint position velocities
  [140:272] 22 local joint rotations as 6D rotations

This is the standalone version of the conversion used in UMO's MotionFix RVQ
experiments. It intentionally does not depend on the full training repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_DATA_DIR = Path("/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset")
DEFAULT_OUTPUT_ROOT = Path("/mnt/afs/mogo_base/datasets/MotionFix")

FEATURE_DIR_NAME = "motionstreamer272_hml_joint_vecs"
MANIFEST_PREFIX = "motionfix_motionstreamer272_hml"
STATS_DIR_NAME = "motionstreamer272_hml_source_target_train"
SPLITS_FILE_NAME = "splits.json"
CONVERSION_VERSION = "motionfix_motionstreamer272_hml"

NUM_JOINTS = 22
FEATURE_DIM = 272
FPS = 30.0

# MotionFix/SMPL joint positions are z-up. HumanML3D/MotionStreamer 272 uses
# y-up with x/z as the ground plane. This basis maps raw (x, y, z) to
# MotionStreamer (x, z, -y). Rotations are conjugated by the same basis.
MOTIONFIX_TO_MOTIONSTREAMER = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

# HumanML3D-style facing estimate: right hip, left hip, right shoulder,
# left shoulder. The initial frame is canonicalized to face +Z.
FACE_JOINT_INDEX = (2, 1, 17, 16)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices."""
    axis_angle = axis_angle.to(dtype=torch.float32)
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / torch.clamp(angle, min=1e-8)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(axis.shape[:-1] + (3, 3))
    angle = angle[..., None]
    return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * (k @ k)


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Return the same row-major 6D rotation layout used by our training data."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6).astype(np.float32)


def skeleton_forward(positions: np.ndarray) -> np.ndarray:
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDEX
    across = (positions[:, r_hip] - positions[:, l_hip]) + (positions[:, sdr_r] - positions[:, sdr_l])
    across = normalize(across)
    forward = np.cross(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), across, axis=-1)
    forward[..., 1] = 0.0
    return normalize(forward).astype(np.float32)


def heading_remove_from_forward(forward: np.ndarray) -> np.ndarray:
    """Build y-axis yaw-removal matrices from per-frame forward vectors."""
    forward = np.asarray(forward, dtype=np.float32)
    yaw = np.arctan2(forward[..., 0], forward[..., 2])
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    matrix = np.zeros((*yaw.shape, 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = cos
    matrix[..., 0, 2] = -sin
    matrix[..., 1, 1] = 1.0
    matrix[..., 2, 0] = sin
    matrix[..., 2, 2] = cos
    return matrix


def convert_motionfix_motion_hml(raw_motion: dict[str, Any]) -> np.ndarray:
    """Convert one MotionFix raw motion dict to a [T, 272] float32 array."""
    rots = as_numpy(raw_motion["rots"]).astype(np.float32)
    joints_raw = as_numpy(raw_motion["joint_positions"]).astype(np.float32)[:, :NUM_JOINTS]
    if rots.ndim != 2 or rots.shape[-1] != NUM_JOINTS * 3:
        raise ValueError(f"Expected MotionFix rots [T,{NUM_JOINTS * 3}], got {rots.shape}")
    if joints_raw.ndim != 3 or joints_raw.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(f"Expected MotionFix joints [T,{NUM_JOINTS},3], got {joints_raw.shape}")
    if rots.shape[0] != joints_raw.shape[0]:
        raise ValueError(f"Motion length mismatch: rots={rots.shape[0]} joints={joints_raw.shape[0]}")

    nframes = int(rots.shape[0])
    basis = MOTIONFIX_TO_MOTIONSTREAMER

    joints = np.einsum("ij,tkj->tki", basis, joints_raw).astype(np.float32)
    rot_aa = rots.reshape(nframes, NUM_JOINTS, 3)
    rot_mats_raw = (
        axis_angle_to_matrix(torch.from_numpy(rot_aa.reshape(-1, 3)))
        .detach()
        .cpu()
        .numpy()
        .reshape(nframes, NUM_JOINTS, 3, 3)
        .astype(np.float32)
    )
    rot_mats = np.einsum("ij,tkjl,ml->tkim", basis, rot_mats_raw, basis).astype(np.float32)

    origin = joints[0, 0].copy()
    origin[1] = float(joints[:, :, 1].min())
    positions = joints - origin.reshape(1, 1, 3)

    initial_heading_remove = heading_remove_from_forward(skeleton_forward(positions[:1]))[0]
    positions = np.einsum("ij,tkj->tki", initial_heading_remove, positions).astype(np.float32)

    canonical_rot_mats = rot_mats.copy()
    canonical_rot_mats[:, 0] = np.einsum("ij,tjk->tik", initial_heading_remove, rot_mats[:, 0])

    heading_remove = heading_remove_from_forward(skeleton_forward(positions))
    heading_remove[0] = np.eye(3, dtype=np.float32)
    heading_delta = np.tile(np.eye(3, dtype=np.float32), (nframes, 1, 1))
    for frame in range(1, nframes):
        heading_delta[frame] = heading_remove[frame] @ heading_remove[frame - 1].T

    root = positions[:, 0].copy()
    centered = positions.copy()
    centered[:, :, 0] -= root[:, 0, None]
    centered[:, :, 2] -= root[:, 2, None]
    local_positions = np.einsum("tij,tkj->tki", heading_remove, centered).astype(np.float32)

    local_position_velocity = np.zeros_like(local_positions)
    local_position_velocity[1:] = local_positions[1:] - local_positions[:-1]

    canonical_root_velocity = np.zeros_like(root)
    canonical_root_velocity[1:] = root[1:] - root[:-1]
    local_root_velocity = np.zeros_like(canonical_root_velocity)
    local_root_velocity[1:] = np.einsum(
        "tij,tj->ti",
        heading_remove[:-1],
        canonical_root_velocity[1:],
    )

    local_rotations = canonical_rot_mats.copy()
    local_rotations[:, 0] = np.einsum("tij,tjk->tik", heading_remove, canonical_rot_mats[:, 0])

    motion = np.zeros((nframes, FEATURE_DIM), dtype=np.float32)
    motion[:, 0] = local_root_velocity[:, 0]
    motion[:, 1] = local_root_velocity[:, 2]
    motion[:, 2:8] = matrix_to_rotation_6d(heading_delta)
    motion[:, 8 : 8 + 3 * NUM_JOINTS] = local_positions.reshape(nframes, 3 * NUM_JOINTS)
    motion[:, 8 + 3 * NUM_JOINTS : 8 + 6 * NUM_JOINTS] = local_position_velocity.reshape(
        nframes, 3 * NUM_JOINTS
    )
    motion[:, 8 + 6 * NUM_JOINTS : 8 + 12 * NUM_JOINTS] = matrix_to_rotation_6d(local_rotations).reshape(
        nframes, 6 * NUM_JOINTS
    )
    if not np.isfinite(motion).all():
        raise RuntimeError("Converted MotionFix HML272 motion contains non-finite values")
    return motion


def split_file(data_dir: Path, split: str) -> Path:
    if split == "train":
        return data_dir / "motionfix.pth.tar"
    return data_dir / f"motionfix_{split}.pth.tar"


def load_official_splits(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Official MotionFix split file is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a split dictionary")
    splits: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        ids = payload.get(split)
        if not isinstance(ids, list):
            raise ValueError(f"Expected {path} key '{split}' to be a list")
        splits[split] = [str(item) for item in ids]
    return splits


def filter_dataset_to_official_split(
    dataset: dict[str, Any],
    split: str,
    split_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_by_id = {str(key): (str(key), value) for key, value in dataset.items()}
    requested = [str(item) for item in split_ids]
    requested_set = set(requested)
    filtered: dict[str, Any] = {}
    missing: list[str] = []
    for keyid in requested:
        item = dataset_by_id.get(keyid)
        if item is None:
            missing.append(keyid)
            continue
        filtered[keyid] = item[1]
    dropped = sorted(str(key) for key in dataset.keys() if str(key) not in requested_set)
    summary = {
        "split": split,
        "requested": len(requested),
        "raw_records": len(dataset),
        "kept": len(filtered),
        "missing": len(missing),
        "dropped_non_split": len(dropped),
        "missing_examples": missing[:20],
        "dropped_non_split_examples": dropped[:20],
    }
    return filtered, summary


def prepare_output(output_root: Path, feature_dir_name: str, force: bool) -> tuple[Path, Path]:
    feature_root = output_root / feature_dir_name
    manifest_root = output_root / "manifests"
    if feature_root.exists():
        if not force:
            raise FileExistsError(f"{feature_root} exists; pass --force to replace it")
        shutil.rmtree(feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    return feature_root, manifest_root


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def convert_split(
    *,
    split: str,
    dataset: dict[str, Any],
    feature_root: Path,
    output_root: Path,
    feature_dir_name: str,
    manifest_root: Path,
    manifest_prefix: str,
    max_items: int,
) -> dict[str, Any]:
    split_dir = feature_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    items = list(dataset.items())
    if int(max_items) > 0:
        items = items[: int(max_items)]

    records: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for keyid, item in tqdm(items, desc=f"Converting MotionFix {split} hml272"):
        try:
            source = convert_motionfix_motion_hml(item["motion_source"])
            target = convert_motionfix_motion_hml(item["motion_target"])
            source_name = f"{keyid}_source.npy"
            target_name = f"{keyid}_target.npy"
            np.save(split_dir / source_name, source.astype(np.float32))
            np.save(split_dir / target_name, target.astype(np.float32))
            records.append(
                {
                    "id": str(keyid),
                    "instruction": str(item["text"]),
                    "source": f"{feature_dir_name}/{split}/{source_name}",
                    "target": f"{feature_dir_name}/{split}/{target_name}",
                    "source_len": int(source.shape[0]),
                    "target_len": int(target.shape[0]),
                    "feature_dim": FEATURE_DIM,
                    "fps": FPS,
                    "conversion_version": CONVERSION_VERSION,
                    "canonical_frame": "per_sequence_humanml3d",
                    "initial_forward": "hips_shoulders_to_zplus",
                    "direct_motionstreamer272_recover": True,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep converting and report bad samples.
            failed.append({"id": str(keyid), "error": repr(exc)})

    manifest_path = manifest_root / f"{manifest_prefix}_{split}.jsonl"
    write_jsonl(manifest_path, records)
    return {
        "input": len(items),
        "written": len(records),
        "failed": len(failed),
        "manifest": str(manifest_path),
        "failed_examples": failed[:20],
    }


def compute_stats(records: list[dict[str, Any]], output_root: Path, stats_dir: Path) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Cannot compute stats from an empty train manifest")
    sums = np.zeros((FEATURE_DIM,), dtype=np.float64)
    sums_sq = np.zeros((FEATURE_DIM,), dtype=np.float64)
    frames = 0
    for record in tqdm(records, desc="Computing MotionFix hml272 stats"):
        for key in ("source", "target"):
            motion = np.load(output_root / record[key]).astype(np.float64)
            if motion.ndim != 2 or motion.shape[1] != FEATURE_DIM:
                raise ValueError(f"Expected [T,{FEATURE_DIM}] motion at {record[key]}, got {motion.shape}")
            sums += motion.sum(axis=0)
            sums_sq += np.square(motion).sum(axis=0)
            frames += int(motion.shape[0])
    mean = (sums / float(frames)).astype(np.float32)
    var = np.maximum(sums_sq / float(frames) - np.square(sums / float(frames)), 1e-12)
    std = np.maximum(np.sqrt(var), 1e-6).astype(np.float32)
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "Mean.npy", mean)
    np.save(stats_dir / "Std.npy", std)
    meta = {
        "dataset": "motionfix_hml272",
        "conversion_version": CONVERSION_VERSION,
        "num_manifest_records": len(records),
        "num_motion_files": len(records) * 2,
        "num_frames": int(frames),
        "feature_dim": FEATURE_DIM,
        "mean_path": str(stats_dir / "Mean.npy"),
        "std_path": str(stats_dir / "Std.npy"),
    }
    (stats_dir / "stats_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Official MotionFix splits.json. Defaults to --data-dir/splits.json.",
    )
    parser.add_argument("--feature-dir-name", type=str, default=FEATURE_DIR_NAME)
    parser.add_argument("--manifest-prefix", type=str, default=MANIFEST_PREFIX)
    parser.add_argument("--stats-dir-name", type=str, default=STATS_DIR_NAME)
    parser.add_argument("--split", action="append", choices=["train", "val", "test"], default=None)
    parser.add_argument(
        "--max-items-per-split",
        type=int,
        default=0,
        help="Debug limit. 0 converts every item in each requested split.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    splits_path = (args.splits_file if args.splits_file is not None else data_dir / SPLITS_FILE_NAME).expanduser().resolve()

    splits = args.split or ["train", "val", "test"]
    official_splits = load_official_splits(splits_path)
    feature_root, manifest_root = prepare_output(output_root, args.feature_dir_name, args.force)

    summary: dict[str, Any] = {
        "conversion_version": CONVERSION_VERSION,
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "feature_dir": str(feature_root),
        "manifest_prefix": args.manifest_prefix,
        "splits_file": str(splits_path),
        "feature_dim": FEATURE_DIM,
        "fps": FPS,
        "num_joints": NUM_JOINTS,
        "canonical_frame": "per_sequence_humanml3d",
        "compatibility": {
            "channel_semantics": "HumanML3D/MotionStreamer 272",
            "initial_forward": "hips_shoulders_to_zplus",
            "uses_motionfix_meta_for_recovery": False,
            "uses_per_joint_rotation_correction": False,
            "direct_recover_function": "recover_motionstreamer272_positions",
        },
        "layout": {
            "root_xz_velocity": [0, 2],
            "heading_6d": [2, 8],
            "local_positions": [8, 74],
            "local_velocities": [74, 140],
            "local_rotations": [140, 272],
        },
        "splits": {},
    }

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        src_file = split_file(data_dir, split)
        if not src_file.is_file():
            raise FileNotFoundError(src_file)
        raw_dataset = joblib.load(src_file)
        dataset, split_filter_summary = filter_dataset_to_official_split(raw_dataset, split, official_splits[split])
        split_summary = convert_split(
            split=split,
            dataset=dataset,
            feature_root=feature_root,
            output_root=output_root,
            feature_dir_name=args.feature_dir_name,
            manifest_root=manifest_root,
            manifest_prefix=args.manifest_prefix,
            max_items=int(args.max_items_per_split),
        )
        split_summary["split_filter"] = split_filter_summary
        summary["splits"][split] = split_summary
        manifest_path = Path(split_summary["manifest"])
        records_by_split[split] = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]

    if "train" in records_by_split:
        stats_dir = output_root / "stats" / args.stats_dir_name
        summary["stats"] = compute_stats(records_by_split["train"], output_root, stats_dir)

    summary_path = manifest_root / f"{args.manifest_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
