#!/usr/bin/env python3
"""Convert Franka pick-and-lift raw episodes to arm7 + hand-synergy-k latent actions."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
import json

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode pick-and-lift raw episodes from arm7+hand16 absolute targets into "
            "arm7+hand_synergy_k latent action for end-to-end VLA training."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset/franka_pickandlift_object6d_low2e_owrap_20_fast5hz_20260306/raw",
        help="Source raw directory containing episode_*.npz.",
    )
    parser.add_argument(
        "--basis-path",
        type=str,
        required=True,
        help="Path to pick-and-lift hand PCA basis npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output root directory. Default: "
            "dataset/franka_pickandlift_arm_hand_synergy_k{K}_{source_suffix}"
        ),
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="grasp the mustard bottle",
        help="Instruction to write into converted episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing episode_*.npz in output raw directory before writing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _list_episodes(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("episode_*.npz") if p.is_file())


def _load_basis(path: Path) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    data = np.load(path, allow_pickle=True)
    if "mu" not in data or "B" not in data:
        raise KeyError(f"Basis file missing mu/B: {path}")
    mu = np.asarray(data["mu"], dtype=np.float64).reshape(-1)
    B = np.asarray(data["B"], dtype=np.float64)
    if mu.shape[0] != 16:
        raise ValueError(f"mu must be 16-dim, got {mu.shape}")
    if B.ndim != 2 or B.shape[0] != 16:
        raise ValueError(f"B must be (16,k), got {B.shape}")
    k = int(B.shape[1])
    hs = int(data["action_slice_start"]) if "action_slice_start" in data else 7
    hd = int(data["action_slice_dim"]) if "action_slice_dim" in data else 16
    return mu, B, k, hs, hd


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    basis_path = Path(args.basis_path).expanduser().resolve()
    mu, B, k, hand_start, hand_dim = _load_basis(basis_path)
    hand_end = hand_start + hand_dim

    files = _list_episodes(raw_dir)
    if not files:
        raise SystemExit(f"No episode files found in: {raw_dir}")

    if args.out_dir:
        out_root = Path(args.out_dir).expanduser().resolve()
    else:
        suffix = raw_dir.parent.name.replace("franka_pickandlift_", "")
        out_root = Path(
            f"dataset/franka_pickandlift_arm_hand_synergy_k{k}_{suffix}"
        ).resolve()
    out_raw = out_root / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    if args.overwrite and not args.dry_run:
        for p in _list_episodes(out_raw):
            p.unlink()

    src_summary = _load_json(raw_dir.parent / "collection_summary.json")

    total_steps = 0
    recon_abs_sum = 0.0
    recon_sq_sum = 0.0
    recon_max_abs = 0.0
    min_steps = 10**9
    max_steps = 0
    image_shape = None
    state_dim = None

    for ep_idx, src_path in enumerate(files):
        data = np.load(src_path, allow_pickle=True)
        if "action" not in data:
            raise KeyError(f"'action' missing in {src_path}")
        action = np.asarray(data["action"], dtype=np.float64)
        if action.ndim != 2 or action.shape[1] < hand_end:
            raise ValueError(
                f"Expected action shape (T,{hand_end}+), got {action.shape} in {src_path}"
            )

        arm7 = action[:, :hand_start]
        hand16 = action[:, hand_start:hand_end]
        z = ((hand16 - mu[None, :]) @ B).astype(np.float32)
        hand_recon = z.astype(np.float64) @ B.T + mu[None, :]
        err = hand_recon - hand16

        recon_abs_sum += float(np.sum(np.abs(err)))
        recon_sq_sum += float(np.sum(np.square(err)))
        recon_max_abs = max(recon_max_abs, float(np.max(np.abs(err))))

        arm_hand11 = np.concatenate([arm7.astype(np.float32), z], axis=-1)
        steps = int(arm_hand11.shape[0])
        total_steps += steps
        min_steps = min(min_steps, steps)
        max_steps = max(max_steps, steps)

        if image_shape is None and "images" in data:
            image_shape = list(np.asarray(data["images"]).shape[1:])
        if state_dim is None and "state" in data:
            st = np.asarray(data["state"])
            if st.ndim == 2:
                state_dim = int(st.shape[1])

        if args.dry_run:
            continue

        payload = {key: data[key] for key in data.files}
        payload["action_full"] = action.astype(np.float32)
        payload["action_arm"] = arm7.astype(np.float32)
        payload["action_hand16"] = hand16.astype(np.float32)
        payload["action_hand_latent"] = z.astype(np.float32)
        payload["action"] = arm_hand11.astype(np.float32)
        payload["language_instruction"] = np.asarray(args.instruction)
        payload["action_interface"] = np.asarray("arm_joint7_hand_synergy_kd", dtype=object)
        payload["action_semantics"] = np.asarray(
            "joint11_absolute_arm7_hand_synergy_kd",
            dtype=object,
        )
        payload["synergy_basis_path"] = np.asarray(str(basis_path), dtype=object)
        payload["synergy_k"] = np.asarray(k, dtype=np.int32)
        payload["source_action_dim"] = np.asarray(action.shape[1], dtype=np.int32)
        payload["source_action_slice_start"] = np.asarray(hand_start, dtype=np.int32)
        payload["source_action_slice_dim"] = np.asarray(hand_dim, dtype=np.int32)
        payload["converted_at"] = np.asarray(datetime.now().isoformat(), dtype=object)

        dst = out_raw / f"episode_{ep_idx:05d}.npz"
        np.savez_compressed(dst, **payload)

    denom = max(total_steps * hand_dim, 1)
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
        "instruction": args.instruction,
        "action_interface": "arm_joint7_hand_synergy_kd",
        "action_semantics": "joint11_absolute_arm7_hand_synergy_kd",
        "action_dim": int(7 + k),
        "source_action_dim": 23,
        "arm_action_dim": 7,
        "hand_action_dim": int(k),
        "hand_slice": {"start": int(hand_start), "dim": int(hand_dim)},
        "basis_path": str(basis_path),
        "reconstruction": {
            "mean_abs_error": float(recon_abs_sum / denom),
            "rmse": float((recon_sq_sum / denom) ** 0.5),
            "max_abs_error": float(recon_max_abs),
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
        f"rmse={summary['reconstruction']['rmse']:.6f} out_dir={out_root}"
    )
    if not args.dry_run:
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
