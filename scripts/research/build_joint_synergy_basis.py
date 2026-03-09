#!/usr/bin/env python3
"""Build joint-synergy PCA basis from full-joint mustard raw dataset."""

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
import shutil

import numpy as np


JOINT_NAMES = tuple(f"{finger}j{j}" for finger in ("ff", "mf", "rf", "th") for j in range(4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PCA basis on full_joint actions and export joint-synergy basis artifacts."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset/mustard_grasp_full_joint/raw",
        help="Directory containing full_joint raw episode_*.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/synergy_basis",
        help="Output directory for basis files and summary json.",
    )
    parser.add_argument(
        "--side",
        choices=("right", "left"),
        default="right",
        help="Hand side used for joint limits and oracle replay.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="4,6,8,10",
        help="Comma-separated synergy dimensions to evaluate.",
    )
    parser.add_argument(
        "--var-threshold",
        type=float,
        default=0.99,
        help="Cumulative explained-variance threshold used to choose best k when oracle is off.",
    )
    parser.add_argument(
        "--save-best-alias",
        action="store_true",
        default=True,
        help="Also save full_joint_pca_best.npz alias for chosen k (default: on).",
    )
    parser.add_argument(
        "--no-save-best-alias",
        dest="save_best_alias",
        action="store_false",
        help="Disable full_joint_pca_best.npz alias export.",
    )
    parser.add_argument(
        "--oracle-eval",
        action="store_true",
        help="Run MuJoCo oracle replay for each k (slower, but recommended).",
    )
    parser.add_argument(
        "--oracle-episodes",
        type=int,
        default=20,
        help="How many episodes to use for oracle replay when --oracle-eval is enabled.",
    )
    parser.add_argument("--control-repeat", type=int, default=5)
    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=2)
    parser.add_argument(
        "--require-thumb-contact",
        action="store_true",
        default=True,
        help="Require thumb contact in oracle success (default: on).",
    )
    parser.add_argument(
        "--no-require-thumb-contact",
        dest="require_thumb_contact",
        action="store_false",
        help="Disable mandatory thumb contact for oracle success.",
    )
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)
    return parser.parse_args()


def _parse_k_values(text: str) -> list[int]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k = int(tok)
        if k <= 0:
            raise ValueError(f"Invalid k value: {k}")
        out.append(k)
    if not out:
        raise ValueError("No valid k values parsed.")
    return sorted(set(out))


def _list_episode_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("episode_*.npz") if p.is_file())


def _load_episode_actions(raw_dir: Path) -> tuple[list[Path], list[np.ndarray], np.ndarray]:
    files = _list_episode_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No episode files found under: {raw_dir}")

    episode_actions: list[np.ndarray] = []
    flat: list[np.ndarray] = []
    action_dim = None
    for path in files:
        data = np.load(path, allow_pickle=True)
        if "action" not in data:
            raise KeyError(f"'action' missing in {path}")
        action = np.asarray(data["action"], dtype=np.float64)
        if action.ndim != 2:
            raise ValueError(f"action must be 2D, got {action.shape} in {path}")
        if action_dim is None:
            action_dim = int(action.shape[1])
        if action.shape[1] != action_dim:
            raise ValueError(
                f"Inconsistent action dim in {path}: {action.shape[1]} vs expected {action_dim}"
            )
        episode_actions.append(action)
        flat.append(action)

    X = np.concatenate(flat, axis=0)
    if action_dim != 16:
        raise ValueError(f"Expected full_joint action dim 16, got {action_dim}")
    return files, episode_actions, X


def _fit_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    dof = max(int(X.shape[0]) - 1, 1)
    explained_var = (s ** 2) / dof
    explained_ratio = explained_var / max(float(np.sum(explained_var)), 1e-12)
    cum_ratio = np.cumsum(explained_ratio)
    return mu, vt, explained_ratio, cum_ratio


def _get_joint_limits(side: str) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    from scripts.data.collect_mustard_grasp import build_hand_config
    from env import allegro_hand_mjcf

    mjcf = allegro_hand_mjcf.load(side=side, add_mustard=False)
    model = mjcf.compile()
    data = mujoco.MjData(model)
    del data
    cfg = build_hand_config(model, side)
    return cfg.q_min.astype(np.float64), cfg.q_max.astype(np.float64)


def _metrics_for_k(
    X: np.ndarray,
    mu: np.ndarray,
    vt: np.ndarray,
    explained_ratio: np.ndarray,
    cum_ratio: np.ndarray,
    k: int,
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> tuple[dict, np.ndarray]:
    B = vt[:k].T  # (16,k), orthonormal columns
    Xc = X - mu
    Z = Xc @ B
    Xhat = Z @ B.T + mu
    err = Xhat - X

    per_joint_rmse = np.sqrt(np.mean(np.square(err), axis=0))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    max_abs_err = float(np.max(np.abs(err)))

    low_viol = Xhat < q_min[None, :]
    high_viol = Xhat > q_max[None, :]
    clip_rate = float(np.mean(low_viol | high_viol))

    metrics = {
        "k": int(k),
        "explained_variance_ratio_k": float(np.sum(explained_ratio[:k])),
        "cumulative_explained_variance_ratio_k": float(cum_ratio[k - 1]),
        "rmse_all_joints": rmse,
        "max_abs_error": max_abs_err,
        "clip_rate": clip_rate,
        "per_joint_rmse": per_joint_rmse.astype(np.float32).tolist(),
        "thumb_rmse_mean": float(np.mean(per_joint_rmse[12:16])),
    }
    return metrics, B


def _oracle_eval(
    side: str,
    mu: np.ndarray,
    B: np.ndarray,
    episodes: list[np.ndarray],
    max_episodes: int,
    control_repeat: int,
    min_contacts: int,
    min_contact_fingers: int,
    require_thumb_contact: bool,
    min_force: float,
    max_force: float,
    stable_steps: int,
) -> dict:
    import mujoco

    from scripts.data.collect_mustard_grasp import (
        build_contact_config,
        build_hand_config,
        build_mustard_config,
        compute_mustard_spawn_pose,
        detect_contact_with_target,
        reset_to_initial,
        set_mustard_pose,
    )
    from env import allegro_hand_mjcf

    use_eps = episodes[: max(0, int(max_episodes))]
    if not use_eps:
        return {"episodes": 0, "success_rate": 0.0}

    mjcf = allegro_hand_mjcf.load(side=side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)
    hand_cfg = build_hand_config(model, side)
    mustard_cfg = build_mustard_config(model)
    contact_cfg = build_contact_config(model, side, mustard_cfg.body_id)
    force_buf = np.zeros(6, dtype=float)

    successes = 0
    thumb_touch_episodes = 0
    mean_best_contacts: list[float] = []
    mean_best_fingers: list[float] = []

    for ep_actions in use_eps:
        reset_to_initial(model, data)
        spawn_pos, spawn_quat = compute_mustard_spawn_pose(model, data, side)
        set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
        mujoco.mj_forward(model, data)

        Z = (ep_actions - mu[None, :]) @ B
        q_seq = Z @ B.T + mu[None, :]
        q_seq = np.clip(q_seq, hand_cfg.q_min[None, :], hand_cfg.q_max[None, :])

        stable_hits = 0
        success = False
        best_contacts = 0
        best_fingers = 0
        thumb_touched = False

        for q_cmd in q_seq:
            for _ in range(control_repeat):
                data.ctrl[:16] = q_cmd.astype(np.float32)
                set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_step(model, data)
                set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_forward(model, data)

                _, n_contacts, total_force, touched = detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                best_contacts = max(best_contacts, int(n_contacts))
                best_fingers = max(best_fingers, len(touched))
                if "th" in touched:
                    thumb_touched = True

                thumb_ok = (not require_thumb_contact) or ("th" in touched)
                meets = (
                    n_contacts >= min_contacts
                    and len(touched) >= min_contact_fingers
                    and thumb_ok
                    and total_force >= min_force
                    and total_force <= max_force
                )
                stable_hits = stable_hits + 1 if meets else 0
                if stable_hits >= stable_steps:
                    success = True
                    break
            if success:
                break

        successes += int(success)
        thumb_touch_episodes += int(thumb_touched)
        mean_best_contacts.append(float(best_contacts))
        mean_best_fingers.append(float(best_fingers))

    return {
        "episodes": len(use_eps),
        "success_rate": float(successes / len(use_eps)),
        "thumb_touch_episode_ratio": float(thumb_touch_episodes / len(use_eps)),
        "best_contacts_mean": float(np.mean(mean_best_contacts)),
        "best_fingers_mean": float(np.mean(mean_best_fingers)),
    }


def _choose_best_k(
    metrics_by_k: dict[int, dict],
    k_values: list[int],
    var_threshold: float,
    oracle_eval: bool,
) -> int:
    if oracle_eval:
        # Max oracle success first, then smaller k.
        ranked = sorted(
            k_values,
            key=lambda k: (
                float(metrics_by_k[k].get("oracle_success_rate", 0.0)),
                -int(k),
            ),
            reverse=True,
        )
        return int(ranked[0])

    candidates = [
        k
        for k in k_values
        if float(metrics_by_k[k]["cumulative_explained_variance_ratio_k"]) >= float(var_threshold)
    ]
    if candidates:
        return int(min(candidates))
    return int(max(k_values))


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = _parse_k_values(args.k_values)
    files, episode_actions, X = _load_episode_actions(raw_dir)
    mu, vt, explained_ratio, cum_ratio = _fit_pca(X)
    q_min, q_max = _get_joint_limits(args.side)

    print(
        f"[basis] episodes={len(files)} transitions={X.shape[0]} action_dim={X.shape[1]} "
        f"k_values={k_values}"
    )

    metrics_by_k: dict[int, dict] = {}
    basis_path_by_k: dict[int, Path] = {}

    for k in k_values:
        metrics, B = _metrics_for_k(X, mu, vt, explained_ratio, cum_ratio, k, q_min, q_max)
        basis_path = out_dir / f"full_joint_pca_k{k}.npz"
        np.savez_compressed(
            basis_path,
            mu=mu.astype(np.float32),
            B=B.astype(np.float32),
            explained_variance_ratio=explained_ratio.astype(np.float32),
            cumulative_explained_variance_ratio=cum_ratio.astype(np.float32),
            joint_names=np.asarray(JOINT_NAMES, dtype=object),
            side=np.asarray(args.side),
            k=np.asarray(k, dtype=np.int32),
            fit_dataset=np.asarray(str(raw_dir)),
            created_at=np.asarray(datetime.now().isoformat()),
        )

        metrics["basis_path"] = str(basis_path)

        if args.oracle_eval:
            oracle = _oracle_eval(
                side=args.side,
                mu=mu,
                B=B,
                episodes=episode_actions,
                max_episodes=args.oracle_episodes,
                control_repeat=args.control_repeat,
                min_contacts=args.min_contacts,
                min_contact_fingers=args.min_contact_fingers,
                require_thumb_contact=args.require_thumb_contact,
                min_force=args.min_force,
                max_force=args.max_force,
                stable_steps=args.stable_steps,
            )
            metrics["oracle"] = oracle
            metrics["oracle_success_rate"] = float(oracle["success_rate"])

        metrics_by_k[k] = metrics
        basis_path_by_k[k] = basis_path

        msg = (
            f"[k={k}] cum_var={metrics['cumulative_explained_variance_ratio_k']:.6f} "
            f"rmse={metrics['rmse_all_joints']:.6f} clip_rate={metrics['clip_rate']:.6f}"
        )
        if args.oracle_eval:
            msg += f" oracle_success={metrics['oracle_success_rate']:.3f}"
        print(msg)

    best_k = _choose_best_k(metrics_by_k, k_values, args.var_threshold, args.oracle_eval)
    best_basis_path = basis_path_by_k[best_k]

    if args.save_best_alias:
        best_alias = out_dir / "full_joint_pca_best.npz"
        shutil.copyfile(best_basis_path, best_alias)
        print(f"[basis] best alias saved: {best_alias}")

    summary = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "episodes": len(files),
        "transitions": int(X.shape[0]),
        "action_dim": int(X.shape[1]),
        "k_values": k_values,
        "var_threshold": float(args.var_threshold),
        "oracle_eval": bool(args.oracle_eval),
        "oracle_episodes": int(args.oracle_episodes),
        "best_k": int(best_k),
        "best_basis_path": str(best_basis_path),
        "metrics_by_k": {str(k): metrics_by_k[k] for k in k_values},
        "created_at": datetime.now().isoformat(),
    }
    summary_path = out_dir / "full_joint_pca_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] summary saved: {summary_path}")


if __name__ == "__main__":
    main()

