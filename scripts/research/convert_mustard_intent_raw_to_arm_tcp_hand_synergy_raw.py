#!/usr/bin/env python3
"""Convert mustard intent benchmark raw episodes to arm-TCP6 + hand-synergy-k actions."""

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

from scripts.data.collect_pickandlift_rlds import _quat_to_rot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode mustard intent raw episodes into arm TCP pose "
            "[x,y,z,roll,pitch,yaw] plus hand synergy-k latent action."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset/mustard_intent_v5_capture5_y0p12_pushclean_20260311",
        help="Root directory containing per-task subdirectories with raw/episode_*.npz files.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="wrap_and_lift,push_over,hook_and_pull",
        help="Comma-separated task subdirectories to include.",
    )
    parser.add_argument(
        "--basis-path",
        type=str,
        required=True,
        help="Path to mustard intent hand PCA basis npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output root directory. Default: "
            "dataset/mustard_intent_arm_tcp_hand_synergy_k{K}_{source_suffix}"
        ),
    )
    parser.add_argument(
        "--arm-action-type",
        choices=("absolute_world", "delta_local"),
        default="absolute_world",
        help=(
            "Arm TCP semantics for the first 6 action dims. "
            "absolute_world stores world-frame [xyz,rpy]. "
            "delta_local stores command deltas in the current palm local frame."
        ),
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


def _rot_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    sy = float(np.hypot(r[0, 0], r[1, 0]))
    singular = sy < 1e-8
    if not singular:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    else:
        roll = float(np.arctan2(-r[1, 2], r[1, 1]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    return _rot_to_euler_xyz(_quat_to_rot(np.asarray(quat_wxyz, dtype=np.float64)))


def _pose_to_tcp6_absolute(pose_wxyz: np.ndarray) -> np.ndarray:
    out = np.zeros((6,), dtype=np.float32)
    out[:3] = np.asarray(pose_wxyz[:3], dtype=np.float32)
    out[3:6] = _quat_wxyz_to_euler_xyz(np.asarray(pose_wxyz[3:7], dtype=np.float64))
    return out


def _pose_to_tcp6_delta_local(cmd_pose_wxyz: np.ndarray, obs_pose_wxyz: np.ndarray) -> np.ndarray:
    obs_rot = _quat_to_rot(np.asarray(obs_pose_wxyz[3:7], dtype=np.float64))
    cmd_rot = _quat_to_rot(np.asarray(cmd_pose_wxyz[3:7], dtype=np.float64))
    pos_delta_world = np.asarray(cmd_pose_wxyz[:3], dtype=np.float64) - np.asarray(
        obs_pose_wxyz[:3], dtype=np.float64
    )
    pos_delta_local = obs_rot.T @ pos_delta_world
    rot_delta_local = obs_rot.T @ cmd_rot

    out = np.zeros((6,), dtype=np.float32)
    out[:3] = pos_delta_local.astype(np.float32)
    out[3:6] = _rot_to_euler_xyz(rot_delta_local)
    return out


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    basis_path = Path(args.basis_path).expanduser().resolve()
    mu, B, k, hand_start, hand_dim = _load_basis(basis_path)
    hand_end = hand_start + hand_dim

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    by_task_files: dict[str, list[Path]] = {}
    for task in tasks:
        raw_dir = dataset_root / task / "raw"
        files = _list_episodes(raw_dir)
        if not files:
            raise SystemExit(f"No episode files found for task '{task}' in: {raw_dir}")
        by_task_files[task] = files

    if args.out_dir:
        out_root = Path(args.out_dir).expanduser().resolve()
    else:
        suffix = dataset_root.name.replace("mustard_intent_", "")
        out_root = Path(
            f"dataset/mustard_intent_arm_tcp_hand_synergy_k{k}_{suffix}"
        ).resolve()
    out_raw = out_root / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    if args.overwrite and not args.dry_run:
        for p in _list_episodes(out_raw):
            p.unlink()

    src_summary = _load_json(dataset_root / "collection_summary.json")

    total_steps = 0
    total_eps = 0
    recon_abs_sum = 0.0
    recon_sq_sum = 0.0
    recon_max_abs = 0.0
    arm_pose_abs_sum = 0.0
    arm_pose_sq_sum = 0.0
    arm_pose_max_abs = 0.0
    min_steps = 10**9
    max_steps = 0
    image_shape = None
    state_dim = None
    task_counts: dict[str, int] = {task: 0 for task in tasks}

    ep_idx = 0
    for task in tasks:
        for src_path in by_task_files[task]:
            data = np.load(src_path, allow_pickle=True)
            if "action" not in data:
                raise KeyError(f"'action' missing in {src_path}")
            if "arm_cmd_pose_wxyz" not in data or "arm_obs_pose_wxyz" not in data:
                raise KeyError(f"arm_cmd/obs_pose_wxyz missing in {src_path}")

            action_full = np.asarray(data["action"], dtype=np.float64)
            if action_full.ndim != 2 or action_full.shape[1] < hand_end:
                raise ValueError(
                    f"Expected action shape (T,{hand_end}+), got {action_full.shape} in {src_path}"
                )

            arm_cmd_pose = np.asarray(data["arm_cmd_pose_wxyz"], dtype=np.float64)
            arm_obs_pose = np.asarray(data["arm_obs_pose_wxyz"], dtype=np.float64)
            if arm_cmd_pose.ndim != 2 or arm_cmd_pose.shape[1] != 7:
                raise ValueError(f"Expected arm_cmd_pose_wxyz (T,7), got {arm_cmd_pose.shape}")
            if arm_obs_pose.ndim != 2 or arm_obs_pose.shape[1] != 7:
                raise ValueError(f"Expected arm_obs_pose_wxyz (T,7), got {arm_obs_pose.shape}")

            steps = int(min(action_full.shape[0], arm_cmd_pose.shape[0], arm_obs_pose.shape[0]))
            action_full = action_full[:steps]
            arm_cmd_pose = arm_cmd_pose[:steps]
            arm_obs_pose = arm_obs_pose[:steps]

            hand16 = action_full[:, hand_start:hand_end]
            z = ((hand16 - mu[None, :]) @ B).astype(np.float32)
            hand_recon = z.astype(np.float64) @ B.T + mu[None, :]
            hand_err = hand_recon - hand16

            arm_cmd_tcp6 = np.zeros((steps, 6), dtype=np.float32)
            arm_obs_tcp6 = np.zeros((steps, 6), dtype=np.float32)
            arm_delta_local_tcp6 = np.zeros((steps, 6), dtype=np.float32)
            for t in range(steps):
                arm_cmd_tcp6[t] = _pose_to_tcp6_absolute(arm_cmd_pose[t])
                arm_obs_tcp6[t] = _pose_to_tcp6_absolute(arm_obs_pose[t])
                arm_delta_local_tcp6[t] = _pose_to_tcp6_delta_local(
                    arm_cmd_pose[t], arm_obs_pose[t]
                )

            if args.arm_action_type == "absolute_world":
                arm_action_tcp6 = arm_cmd_tcp6
                action_semantics = "tcp8_absolute_world_xyz_rpy_hand_synergy_kd"
                arm_pose_representation = "world_xyz_rpy_absolute"
                arm_pose_err = arm_cmd_tcp6.astype(np.float64) - arm_obs_tcp6.astype(np.float64)
            else:
                arm_action_tcp6 = arm_delta_local_tcp6
                action_semantics = "tcp8_delta_local_xyz_rpy_hand_synergy_kd"
                arm_pose_representation = "palm_local_xyz_rpy_delta"
                arm_pose_err = arm_delta_local_tcp6.astype(np.float64)

            arm_hand = np.concatenate([arm_action_tcp6, z], axis=-1).astype(np.float32)

            recon_abs_sum += float(np.sum(np.abs(hand_err)))
            recon_sq_sum += float(np.sum(np.square(hand_err)))
            recon_max_abs = max(recon_max_abs, float(np.max(np.abs(hand_err))))
            arm_pose_abs_sum += float(np.sum(np.abs(arm_pose_err)))
            arm_pose_sq_sum += float(np.sum(np.square(arm_pose_err)))
            arm_pose_max_abs = max(arm_pose_max_abs, float(np.max(np.abs(arm_pose_err))))

            total_steps += steps
            total_eps += 1
            task_counts[task] += 1
            min_steps = min(min_steps, steps)
            max_steps = max(max_steps, steps)

            if image_shape is None and "images" in data:
                image_shape = list(np.asarray(data["images"]).shape[1:])
            if state_dim is None and "state" in data:
                st = np.asarray(data["state"])
                if st.ndim == 2:
                    state_dim = int(st.shape[1])

            if args.dry_run:
                ep_idx += 1
                continue

            payload = {key: data[key] for key in data.files}
            payload["action_full"] = action_full.astype(np.float32)
            payload["action_arm_tcp6_cmd"] = arm_cmd_tcp6.astype(np.float32)
            payload["action_arm_tcp6_obs"] = arm_obs_tcp6.astype(np.float32)
            payload["action_arm_tcp6_delta_local"] = arm_delta_local_tcp6.astype(np.float32)
            payload["action_hand16"] = hand16.astype(np.float32)
            payload["action_hand_latent"] = z.astype(np.float32)
            payload["action"] = arm_hand.astype(np.float32)
            payload["action_interface"] = np.asarray("arm_tcp6_hand_synergy_kd", dtype=object)
            payload["action_semantics"] = np.asarray(action_semantics, dtype=object)
            payload["arm_pose_representation"] = np.asarray(arm_pose_representation, dtype=object)
            payload["arm_action_type"] = np.asarray(args.arm_action_type, dtype=object)
            payload["synergy_basis_path"] = np.asarray(str(basis_path), dtype=object)
            payload["synergy_k"] = np.asarray(k, dtype=np.int32)
            payload["source_action_dim"] = np.asarray(action_full.shape[1], dtype=np.int32)
            payload["source_action_slice_start"] = np.asarray(hand_start, dtype=np.int32)
            payload["source_action_slice_dim"] = np.asarray(hand_dim, dtype=np.int32)
            payload["converted_at"] = np.asarray(datetime.now().isoformat(), dtype=object)

            dst = out_raw / f"episode_{ep_idx:05d}.npz"
            np.savez_compressed(dst, **payload)
            ep_idx += 1

    hand_denom = max(total_steps * hand_dim, 1)
    arm_denom = max(total_steps * 6, 1)
    summary = {
        "finished": True,
        "source_dataset_root": str(dataset_root),
        "tasks": tasks,
        "task_episode_counts": task_counts,
        "out_dir": str(out_root),
        "raw_dir": str(out_raw),
        "episodes": int(total_eps),
        "steps_total": int(total_steps),
        "min_steps": int(min_steps),
        "max_steps": int(max_steps),
        "mean_steps": float(total_steps / max(total_eps, 1)),
        "action_interface": "arm_tcp6_hand_synergy_kd",
        "action_semantics": action_semantics,
        "arm_action_type": args.arm_action_type,
        "action_dim": int(6 + k),
        "source_action_dim": 23,
        "arm_action_dim": 6,
        "hand_action_dim": int(k),
        "hand_slice": {"start": int(hand_start), "dim": int(hand_dim)},
        "basis_path": str(basis_path),
        "hand_reconstruction": {
            "mean_abs_error": float(recon_abs_sum / hand_denom),
            "rmse": float((recon_sq_sum / hand_denom) ** 0.5),
            "max_abs_error": float(recon_max_abs),
        },
        "arm_cmd_minus_obs_tcp6": {
            "mean_abs_error": float(arm_pose_abs_sum / arm_denom),
            "rmse": float((arm_pose_sq_sum / arm_denom) ** 0.5),
            "max_abs_error": float(arm_pose_max_abs),
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
        f"hand_rmse={summary['hand_reconstruction']['rmse']:.6f} out_dir={out_root}"
    )
    if not args.dry_run:
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
