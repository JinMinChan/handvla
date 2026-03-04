#!/usr/bin/env python3
"""Convert full_joint mustard raw episodes into joint-synergy latent raw episodes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode full_joint actions into synergy-kD latent actions and export raw episodes."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset/mustard_grasp_full_joint/raw",
        help="Source raw full_joint directory.",
    )
    parser.add_argument(
        "--basis-path",
        type=str,
        required=True,
        help="Path to full_joint PCA basis npz (from scripts/research/build_joint_synergy_basis.py).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output root directory. Default: dataset/mustard_grasp_synergy_k{K}, "
            "where K is loaded from basis."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing episode_*.npz files in output raw directory before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect conversion statistics without writing files.",
    )
    return parser.parse_args()


def _list_episode_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("episode_*.npz") if p.is_file())


def _load_basis(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    if "mu" not in data or "B" not in data:
        raise KeyError(f"Basis file missing mu/B: {path}")
    mu = np.asarray(data["mu"], dtype=np.float64).reshape(-1)
    B = np.asarray(data["B"], dtype=np.float64)
    if mu.shape[0] != 16:
        raise ValueError(f"mu must be length 16, got {mu.shape}")
    if B.ndim != 2 or B.shape[0] != 16:
        raise ValueError(f"B must be shape (16,k), got {B.shape}")
    k = int(B.shape[1])
    return mu, B, k


def _encode_actions(actions_16: np.ndarray, mu: np.ndarray, B: np.ndarray) -> np.ndarray:
    return ((actions_16 - mu[None, :]) @ B).astype(np.float32)


def _source_summary_path(raw_dir: Path) -> Path:
    return raw_dir.parent / "collection_summary.json"


def _load_source_summary(raw_dir: Path) -> dict:
    summary_path = _source_summary_path(raw_dir)
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    basis_path = Path(args.basis_path).expanduser().resolve()
    mu, B, k = _load_basis(basis_path)

    if args.out_dir:
        out_root = Path(args.out_dir).expanduser().resolve()
    else:
        out_root = Path(f"dataset/mustard_grasp_synergy_k{k}").resolve()
    out_raw = out_root / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    files = _list_episode_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No episode files found in {raw_dir}")

    if args.overwrite and not args.dry_run:
        for p in _list_episode_files(out_raw):
            p.unlink()

    src_summary = _load_source_summary(raw_dir)

    total_steps = 0
    recon_err_sum = 0.0
    recon_err_sq_sum = 0.0
    recon_err_max = 0.0
    min_steps = 10**9
    max_steps = 0
    image_shape = None
    state_dim = None

    for ep_idx, src_path in enumerate(files):
        data = np.load(src_path, allow_pickle=True)
        if "action" not in data:
            raise KeyError(f"'action' missing in {src_path}")
        action16 = np.asarray(data["action"], dtype=np.float64)
        if action16.ndim != 2 or action16.shape[1] != 16:
            raise ValueError(f"Expected action shape (T,16), got {action16.shape} in {src_path}")

        z = _encode_actions(action16, mu, B)
        recon = z.astype(np.float64) @ B.T + mu[None, :]
        err = recon - action16
        recon_err_sum += float(np.sum(np.abs(err)))
        recon_err_sq_sum += float(np.sum(np.square(err)))
        recon_err_max = max(recon_err_max, float(np.max(np.abs(err))))

        steps = int(action16.shape[0])
        total_steps += steps
        min_steps = min(min_steps, steps)
        max_steps = max(max_steps, steps)

        if "images" in data and image_shape is None:
            image_shape = list(np.asarray(data["images"]).shape[1:])
        if "state" in data and state_dim is None:
            state_arr = np.asarray(data["state"])
            if state_arr.ndim == 2:
                state_dim = int(state_arr.shape[1])

        if args.dry_run:
            continue

        payload = {key: data[key] for key in data.files}
        payload["action"] = z
        payload["action_interface"] = np.asarray("synergy_kd")
        payload["action_semantics"] = np.asarray("joint_synergy_latent")
        payload["synergy_basis_path"] = np.asarray(str(basis_path))
        payload["synergy_k"] = np.asarray(k, dtype=np.int32)
        payload["source_action_interface"] = np.asarray("joint16")
        payload["converted_at"] = np.asarray(datetime.now().isoformat())

        dst_path = out_raw / f"episode_{ep_idx:05d}.npz"
        np.savez_compressed(dst_path, **payload)

    mean_abs_recon_err = recon_err_sum / max(total_steps * 16, 1)
    rmse_recon = (recon_err_sq_sum / max(total_steps * 16, 1)) ** 0.5

    summary = {
        "finished": True,
        "source_raw_dir": str(raw_dir),
        "out_dir": str(out_root),
        "raw_dir": str(out_raw),
        "episodes": len(files),
        "steps_total": int(total_steps),
        "min_steps": int(min_steps),
        "max_steps": int(max_steps),
        "mean_steps": float(total_steps / len(files)),
        "action_interface": "synergy_kd",
        "action_semantics": "joint_synergy_latent",
        "action_dim": int(k),
        "source_action_dim": 16,
        "basis_path": str(basis_path),
        "reconstruction": {
            "mean_abs_error": float(mean_abs_recon_err),
            "rmse": float(rmse_recon),
            "max_abs_error": float(recon_err_max),
        },
        "image_shape": image_shape,
        "state_dim": state_dim,
        "source_summary": src_summary,
        "dry_run": bool(args.dry_run),
        "created_at": datetime.now().isoformat(),
    }

    summary_path = out_root / "collection_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    status = "[dry-run]" if args.dry_run else "[done]"
    print(
        f"{status} episodes={summary['episodes']} action_dim={summary['action_dim']} "
        f"rmse={summary['reconstruction']['rmse']:.6f} "
        f"out_dir={out_root}"
    )
    if not args.dry_run:
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
