#!/usr/bin/env python3
"""Fit a hand-only PCA synergy basis (kD) from Franka pick-and-lift raw episodes."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hand-only PCA basis from pick-and-lift raw actions (23D -> hand16 -> kD)."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset/franka_pickandlift_object6d_low2e_thumb_200_fast5hz_20260305/raw",
        help="Raw episode directory containing episode_*.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/pickandlift_synergy_basis",
        help="Output directory for basis npz and summary json.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Target latent dimension k (default: 4).",
    )
    parser.add_argument(
        "--hand-action-start",
        type=int,
        default=7,
        help="Action slice start index for hand joints in raw action (default: 7).",
    )
    parser.add_argument(
        "--hand-action-dim",
        type=int,
        default=16,
        help="Number of hand action dimensions in raw action (default: 16).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _list_episodes(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("episode_*.npz") if p.is_file())


def _fit_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    dof = max(int(X.shape[0]) - 1, 1)
    explained_var = (s**2) / dof
    explained_ratio = explained_var / max(float(np.sum(explained_var)), 1e-12)
    cum_ratio = np.cumsum(explained_ratio)
    return mu, vt, explained_ratio, cum_ratio


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be >= 1")

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    files = _list_episodes(raw_dir)
    if not files:
        raise SystemExit(f"No episode files found in: {raw_dir}")

    hs = int(args.hand_action_start)
    hd = int(args.hand_action_dim)
    he = hs + hd

    hand_actions = []
    steps_total = 0
    for path in files:
        data = np.load(path, allow_pickle=True)
        if "action" not in data:
            raise KeyError(f"'action' missing in {path}")
        action = np.asarray(data["action"], dtype=np.float64)
        if action.ndim != 2 or action.shape[1] < he:
            raise ValueError(
                f"Expected action shape (T,{he}+), got {action.shape} in {path}"
            )
        hand = action[:, hs:he]
        hand_actions.append(hand)
        steps_total += int(hand.shape[0])

    X = np.concatenate(hand_actions, axis=0)
    mu, vt, explained_ratio, cum_ratio = _fit_pca(X)
    if args.k > vt.shape[0]:
        raise SystemExit(f"--k={args.k} exceeds max rank {vt.shape[0]}")
    B = vt[: args.k].T.astype(np.float32)  # (16, k)
    mu = mu.astype(np.float32)

    Z = (X - mu[None, :]) @ B
    Xhat = Z @ B.T + mu[None, :]
    err = Xhat - X
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mean_abs = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))

    basis_name = f"pickandlift_hand_pca_k{args.k}.npz"
    basis_path = out_dir / basis_name
    alias_path = out_dir / "pickandlift_hand_pca_best.npz"
    summary_path = out_dir / "pickandlift_hand_pca_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at": datetime.now().isoformat(),
        "raw_dir": str(raw_dir),
        "episodes": len(files),
        "steps_total": int(steps_total),
        "action_slice": {"start": hs, "dim": hd},
        "k": int(args.k),
        "basis_path": str(basis_path),
        "explained_variance_ratio_k": float(np.sum(explained_ratio[: args.k])),
        "cumulative_explained_variance_ratio_k": float(cum_ratio[args.k - 1]),
        "reconstruction": {
            "rmse": rmse,
            "mean_abs_error": mean_abs,
            "max_abs_error": max_abs,
        },
        "explained_ratio_full": explained_ratio.astype(np.float32).tolist(),
        "cum_ratio_full": cum_ratio.astype(np.float32).tolist(),
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        np.savez_compressed(
            basis_path,
            mu=mu,
            B=B,
            k=np.int32(args.k),
            action_slice_start=np.int32(hs),
            action_slice_dim=np.int32(hd),
            explained_ratio=explained_ratio.astype(np.float32),
            cumulative_explained_ratio=cum_ratio.astype(np.float32),
        )
        np.savez_compressed(
            alias_path,
            mu=mu,
            B=B,
            k=np.int32(args.k),
            action_slice_start=np.int32(hs),
            action_slice_dim=np.int32(hd),
            explained_ratio=explained_ratio.astype(np.float32),
            cumulative_explained_ratio=cum_ratio.astype(np.float32),
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    status = "[dry-run]" if args.dry_run else "[done]"
    print(
        f"{status} episodes={len(files)} steps={steps_total} k={args.k} "
        f"explained={summary['cumulative_explained_variance_ratio_k']:.6f} "
        f"rmse={rmse:.6f}"
    )
    if not args.dry_run:
        print(f"basis: {basis_path}")
        print(f"alias: {alias_path}")
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

