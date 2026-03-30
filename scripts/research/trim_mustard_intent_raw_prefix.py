#!/usr/bin/env python3
"""Trim a fixed number of leading steps from mustard-intent raw episodes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim a fixed number of leading timesteps from raw episode_*.npz files."
    )
    parser.add_argument("--raw-dir", required=True, help="Input directory containing episode_*.npz files.")
    parser.add_argument("--out-dir", required=True, help="Output root directory containing raw/.")
    parser.add_argument("--trim-steps", type=int, required=True, help="How many leading steps to drop.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("episode_*.npz") if p.is_file())


def _trim_episode(data: np.lib.npyio.NpzFile, trim_steps: int) -> dict[str, np.ndarray]:
    arrays = {k: data[k] for k in data.files}
    if "action" not in arrays:
        raise KeyError("raw episode missing 'action'")
    total_steps = int(arrays["action"].shape[0])
    if trim_steps >= total_steps:
        raise ValueError(f"trim_steps={trim_steps} >= episode length {total_steps}")

    out: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == total_steps:
            out[key] = arr[trim_steps:].copy()
        else:
            out[key] = arr.copy() if isinstance(arr, np.ndarray) else arr

    out["trim_prefix_steps"] = np.asarray(trim_steps, dtype=np.int32)
    out["trimmed_at"] = np.asarray(datetime.now().isoformat(timespec="seconds"))
    return out


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_raw = out_root / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    files = _episode_files(raw_dir)
    if not files:
        raise SystemExit(f"No episode files found in {raw_dir}")

    if args.overwrite:
        for p in _episode_files(out_raw):
            p.unlink()

    lengths = []
    first_actions = []
    for i, src in enumerate(files):
        data = np.load(src, allow_pickle=True)
        trimmed = _trim_episode(data, args.trim_steps)
        np.savez_compressed(out_raw / f"episode_{i:05d}.npz", **trimmed)
        lengths.append(int(trimmed["action"].shape[0]))
        first_actions.append(np.asarray(trimmed["action"][0], dtype=np.float32))

    summary = {
        "source_raw_dir": str(raw_dir),
        "out_dir": str(out_root),
        "raw_dir": str(out_raw),
        "episodes": len(files),
        "trim_prefix_steps": int(args.trim_steps),
        "min_steps": int(min(lengths)),
        "max_steps": int(max(lengths)),
        "mean_steps": float(np.mean(lengths)),
        "first_action_mean": np.mean(np.stack(first_actions, axis=0), axis=0).astype(float).tolist(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_root / "collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] trimmed raw -> {out_raw}")
    print(f"[saved] summary -> {out_root / 'collection_summary.json'}")


if __name__ == "__main__":
    main()
