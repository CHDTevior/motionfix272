"""Sanity checks for converted MotionFix HML272 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


FEATURE_DIM = 272
DEFAULT_OUTPUT_ROOT = Path("/mnt/afs/mogo_base/datasets/MotionFix")
DEFAULT_MANIFEST_PREFIX = "motionfix_motionstreamer272_hml"


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = normalize(a1)
    b2 = normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def skeleton_forward(local_positions: np.ndarray) -> np.ndarray:
    across = (local_positions[:, 2] - local_positions[:, 1]) + (local_positions[:, 17] - local_positions[:, 16])
    across = normalize(across)
    up = np.zeros_like(across)
    up[:, 1] = 1.0
    forward = np.cross(up, across, axis=-1)
    forward[:, 1] = 0.0
    return normalize(forward)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-prefix", type=str, default=DEFAULT_MANIFEST_PREFIX)
    parser.add_argument("--split", action="append", choices=["train", "val", "test"], default=None)
    parser.add_argument("--max-records-per-split", type=int, default=0)
    parser.add_argument("--write-json", type=Path, default=None)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def update_extrema(summary: dict[str, Any], arr: np.ndarray) -> None:
    summary["global_min"] = min(summary["global_min"], float(arr.min()))
    summary["global_max"] = max(summary["global_max"], float(arr.max()))
    summary["abs_max"] = max(summary["abs_max"], float(np.abs(arr).max()))


def audit_split(output_root: Path, manifest_path: Path, max_records: int) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    if max_records > 0:
        records = records[:max_records]

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "records_checked": len(records),
        "motion_files_checked": 0,
        "frames_checked": 0,
        "bad_shape": [],
        "non_finite": [],
        "global_min": float("inf"),
        "global_max": float("-inf"),
        "abs_max": 0.0,
        "root_local_xz_abs_max": 0.0,
        "root_plus_z_face_dots": [],
        "root_plus_y_face_dots": [],
        "root_plus_y_abs_up_values": [],
    }
    for record in tqdm(records, desc=f"Auditing {manifest_path.name}"):
        for key in ("source", "target"):
            rel_path = record[key]
            path = output_root / rel_path
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
                summary["bad_shape"].append({"path": str(path), "shape": list(arr.shape)})
                continue
            if not np.isfinite(arr).all():
                summary["non_finite"].append(str(path))
                continue
            update_extrema(summary, np.asarray(arr))
            positions = np.asarray(arr[:, 8:74]).reshape(arr.shape[0], 22, 3)
            root_xz = positions[:, 0, [0, 2]]
            summary["root_local_xz_abs_max"] = max(summary["root_local_xz_abs_max"], float(np.abs(root_xz).max()))
            rotations = rotation_6d_to_matrix(np.asarray(arr[:, 140:272]).reshape(arr.shape[0], 22, 6))
            face = skeleton_forward(positions)
            root_plus_z = rotations[:, 0] @ np.array([0.0, 0.0, 1.0])
            root_plus_y = rotations[:, 0] @ np.array([0.0, 1.0, 0.0])
            root_plus_z_h = normalize(root_plus_z * np.array([1.0, 0.0, 1.0]))
            root_plus_y_h = normalize(root_plus_y * np.array([1.0, 0.0, 1.0]))
            summary["root_plus_z_face_dots"].append(np.sum(root_plus_z_h * face, axis=-1))
            summary["root_plus_y_face_dots"].append(np.sum(root_plus_y_h * face, axis=-1))
            summary["root_plus_y_abs_up_values"].append(np.abs(root_plus_y[:, 1]))
            summary["motion_files_checked"] += 1
            summary["frames_checked"] += int(arr.shape[0])
    if summary["root_plus_z_face_dots"]:
        z_dots = np.concatenate(summary.pop("root_plus_z_face_dots"))
        y_dots = np.concatenate(summary.pop("root_plus_y_face_dots"))
        y_up = np.concatenate(summary.pop("root_plus_y_abs_up_values"))
        summary["root_plus_z_face_dot_median"] = float(np.median(z_dots))
        summary["root_plus_z_face_dot_p05"] = float(np.percentile(z_dots, 5))
        summary["root_plus_y_face_dot_median"] = float(np.median(y_dots))
        summary["root_plus_y_abs_up_median"] = float(np.median(y_up))
    return summary


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    splits = args.split or ["train", "val", "test"]
    result = {
        "output_root": str(output_root),
        "manifest_prefix": args.manifest_prefix,
        "feature_dim": FEATURE_DIM,
        "splits": {},
    }
    for split in splits:
        manifest_path = output_root / "manifests" / f"{args.manifest_prefix}_{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        result["splits"][split] = audit_split(output_root, manifest_path, int(args.max_records_per_split))
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
