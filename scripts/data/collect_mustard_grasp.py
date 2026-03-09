#!/usr/bin/env python3
"""Mustard fixed-air grasp rollout for data collection with geom-contact labels."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from mujoco import viewer

from env import allegro_hand_mjcf
from env.allegro_hand_trajectories import (
    POWER_GRASP_CLOSE_QPOS,
    POWER_GRASP_PRESHAPE_QPOS,
)
from env.viewer_utils import set_default_hand_camera

MUSTARD_JOINT_NAME = f"{allegro_hand_mjcf.MUSTARD_PREFIX}006_mustard_bottle"
MUSTARD_BODY_NAME = f"{allegro_hand_mjcf.MUSTARD_PREFIX}006_mustard_bottle"
MUSTARD_LOCAL_OFFSET = np.array([0.10, 0.00, 0.02], dtype=float)
# Horizontal bottle pose + 90 deg clockwise about world Z.
MUSTARD_HORIZONTAL_QUAT = np.array([0.5, 0.5, 0.5, -0.5], dtype=float)
FINGER_KEYS = ("ff", "mf", "rf", "th")
FINGER_CONTACT_SEGMENTS = ("distal", "tip")

@dataclass(frozen=True)
class MustardConfig:
    qpos_adr: int
    qvel_adr: int
    body_id: int


@dataclass(frozen=True)
class HandConfig:
    qpos_ids: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray


@dataclass(frozen=True)
class ContactConfig:
    geom_to_finger: dict[int, str]
    target_geom_ids: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fixed-air mustard grasp rollouts with geom-based grasp labels."
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--episodes", type=int, default=10, help="Number of rollouts.")
    parser.add_argument("--open-steps", type=int, default=30, help="Warmup open-hand steps.")
    parser.add_argument(
        "--preshape-steps",
        type=int,
        default=25,
        help="Steps for open->thumb-opposition pre-shape interpolation.",
    )
    parser.add_argument(
        "--close-steps", type=int, default=100, help="Steps for pre-shape->close interpolation."
    )
    parser.add_argument("--hold-steps", type=int, default=40, help="Hold closed-hand steps.")
    parser.add_argument(
        "--min-contacts",
        type=int,
        default=2,
        help="Minimum hand-object contact pairs for grasp success.",
    )
    parser.add_argument(
        "--min-contact-fingers",
        type=int,
        default=2,
        help="Minimum unique fingers touching object for grasp success.",
    )
    parser.add_argument(
        "--require-thumb-contact",
        action="store_true",
        default=True,
        help="Require thumb contact for grasp success (default: on).",
    )
    parser.add_argument(
        "--no-require-thumb-contact",
        dest="require_thumb_contact",
        action="store_false",
        help="Disable mandatory thumb contact for success.",
    )
    parser.add_argument(
        "--min-force",
        type=float,
        default=0.5,
        help="Minimum total contact force (N) for grasp success.",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=1000.0,
        help="Maximum allowed total contact force (N) to reject penetration artifacts.",
    )
    parser.add_argument(
        "--stable-steps",
        type=int,
        default=3,
        help="Required consecutive steps satisfying success criteria.",
    )
    parser.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Visualize rollouts (default on).",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Run headless without opening viewer.",
    )
    parser.set_defaults(viewer=True)
    parser.add_argument(
        "--viewer-step",
        type=float,
        default=60.0,
        help="Viewer update frequency in Hz (default: 60).",
    )
    parser.add_argument(
        "--viewer-step-delay",
        type=float,
        default=None,
        help="Manual viewer delay in seconds. If set, overrides --viewer-step.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep viewer open after collection until window is closed manually.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output path for JSON summary.",
    )
    parser.add_argument("--record", action="store_true", help="Record MP4 during collection.")
    parser.add_argument(
        "--record-path",
        type=str,
        default="",
        help="Output mp4 path. Default: codex/logs/mustard_grasp_<timestamp>.mp4",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=1920,
        help="Recording width (default: 1920).",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=1080,
        help="Recording height (default: 1080).",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=60,
        help="Recording frame rate (default: 60).",
    )
    return parser.parse_args()


def _default_record_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"mustard_grasp_{ts}.mp4"


def build_hand_config(model: mujoco.MjModel, side: str) -> HandConfig:
    prefix = f"allegro_{side}"
    qpos_ids: list[int] = []
    q_min: list[float] = []
    q_max: list[float] = []

    for finger in ("ff", "mf", "rf", "th"):
        for j in range(4):
            joint_name = f"{prefix}/{finger}j{j}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            qpos_ids.append(int(model.jnt_qposadr[joint_id]))
            q_min.append(float(model.jnt_range[joint_id][0]))
            q_max.append(float(model.jnt_range[joint_id][1]))

    return HandConfig(
        qpos_ids=np.asarray(qpos_ids, dtype=int),
        q_min=np.asarray(q_min, dtype=float),
        q_max=np.asarray(q_max, dtype=float),
    )


def build_mustard_config(model: mujoco.MjModel) -> MustardConfig:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MUSTARD_JOINT_NAME)
    if joint_id < 0:
        raise RuntimeError(f"Mustard joint not found: {MUSTARD_JOINT_NAME}")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MUSTARD_BODY_NAME)
    if body_id < 0:
        raise RuntimeError(f"Mustard body not found: {MUSTARD_BODY_NAME}")
    return MustardConfig(
        qpos_adr=int(model.jnt_qposadr[joint_id]),
        qvel_adr=int(model.jnt_dofadr[joint_id]),
        body_id=body_id,
    )


def build_contact_config(
    model: mujoco.MjModel, side: str, mustard_body_id: int
) -> ContactConfig:
    hand_body_prefix = f"allegro_{side}/"
    geom_to_finger: dict[int, str] = {}
    target_geom_ids: set[int] = set()

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) if body_id >= 0 else ""
        )
        if body_id == mustard_body_id:
            target_geom_ids.add(geom_id)
            continue
        if not body_name:
            continue
        if not body_name.startswith(hand_body_prefix):
            continue
        if "target_body" in body_name:
            continue
        for finger in FINGER_KEYS:
            for seg in FINGER_CONTACT_SEGMENTS:
                if body_name.endswith(f"{finger}_{seg}"):
                    geom_to_finger[geom_id] = finger
                    break
            if geom_id in geom_to_finger:
                break

    if not geom_to_finger:
        raise RuntimeError("No fingertip/distal hand geoms found for contact detection.")
    if not target_geom_ids:
        raise RuntimeError("No target geoms found for contact detection.")

    return ContactConfig(geom_to_finger=geom_to_finger, target_geom_ids=target_geom_ids)


def detect_contact_with_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: ContactConfig,
    force_buf: np.ndarray,
) -> tuple[bool, int, float, set[str]]:
    num_contacts = 0
    total_force = 0.0
    touched_fingers: set[str] = set()

    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        hand_contact = g1 in cfg.geom_to_finger and g2 in cfg.target_geom_ids
        target_contact = g2 in cfg.geom_to_finger and g1 in cfg.target_geom_ids
        if not (hand_contact or target_contact):
            continue

        num_contacts += 1
        force_buf[:] = 0.0
        mujoco.mj_contactForce(model, data, i, force_buf)
        total_force += float(np.linalg.norm(force_buf[:3]))

        hand_geom_id = g1 if g1 in cfg.geom_to_finger else g2
        touched_fingers.add(cfg.geom_to_finger[hand_geom_id])

    return num_contacts > 0, num_contacts, total_force, touched_fingers


def reset_to_initial(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    initial_id = model.key("initial_state").id
    mujoco.mj_resetDataKeyframe(model, data, initial_id)
    mujoco.mj_forward(model, data)


def compute_mustard_spawn_pose(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> tuple[np.ndarray, np.ndarray]:
    palm_body_name = f"allegro_{side}/palm"
    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name)
    if palm_body_id < 0:
        raise RuntimeError(f"Palm body not found: {palm_body_name}")
    palm_pos = data.xpos[palm_body_id].copy()
    palm_rot = data.xmat[palm_body_id].reshape(3, 3)
    spawn_pos = palm_pos + palm_rot @ MUSTARD_LOCAL_OFFSET
    return spawn_pos, MUSTARD_HORIZONTAL_QUAT.copy()


def set_mustard_pose(
    data: mujoco.MjData, mustard_cfg: MustardConfig, pos: np.ndarray, quat: np.ndarray
) -> None:
    data.qpos[mustard_cfg.qpos_adr : mustard_cfg.qpos_adr + 3] = pos
    data.qpos[mustard_cfg.qpos_adr + 3 : mustard_cfg.qpos_adr + 7] = quat
    data.qvel[mustard_cfg.qvel_adr : mustard_cfg.qvel_adr + 6] = 0.0


def run_collection(args: argparse.Namespace) -> dict:
    mjcf = allegro_hand_mjcf.load(side=args.side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)

    hand_cfg = build_hand_config(model, args.side)
    mustard_cfg = build_mustard_config(model)
    contact_cfg = build_contact_config(model, args.side, mustard_cfg.body_id)

    open_pose = np.clip(np.zeros(16, dtype=float), hand_cfg.q_min, hand_cfg.q_max)
    preshape_pose = np.clip(POWER_GRASP_PRESHAPE_QPOS, hand_cfg.q_min, hand_cfg.q_max)
    close_pose = np.clip(POWER_GRASP_CLOSE_QPOS, hand_cfg.q_min, hand_cfg.q_max)

    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )
    force_buf = np.zeros(6, dtype=float)

    result = {
        "side": args.side,
        "episodes": args.episodes,
        "open_steps": args.open_steps,
        "preshape_steps": args.preshape_steps,
        "close_steps": args.close_steps,
        "hold_steps": args.hold_steps,
        "criteria": {
            "min_contacts": args.min_contacts,
            "min_contact_fingers": args.min_contact_fingers,
            "require_thumb_contact": bool(args.require_thumb_contact),
            "min_force": args.min_force,
            "max_force": args.max_force,
            "stable_steps": args.stable_steps,
        },
        "spawn": {
            "offset": MUSTARD_LOCAL_OFFSET.tolist(),
            "quat_wxyz": MUSTARD_HORIZONTAL_QUAT.tolist(),
        },
        "grasp_pose": {
            "preshape_qpos": preshape_pose.tolist(),
            "close_qpos": close_pose.tolist(),
        },
        "episodes_data": [],
    }

    viewer_ctx = None
    renderer = None
    writer = None
    if args.viewer:
        viewer_ctx = viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=True
        )
        viewer_ctx.__enter__()
        set_default_hand_camera(viewer_ctx.cam)

    if args.record:
        record_path = Path(args.record_path) if args.record_path else _default_record_path()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(
            model, width=args.record_width, height=args.record_height
        )
        writer = imageio.get_writer(
            record_path,
            fps=args.record_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        print(
            f"[record] writing {record_path} "
            f"({args.record_width}x{args.record_height}@{args.record_fps}fps)",
            flush=True,
        )
    sync_enabled = viewer_ctx is not None or writer is not None

    def sync_and_record() -> None:
        if viewer_ctx is not None:
            viewer_ctx.sync()
        if renderer is not None and writer is not None:
            cam = viewer_ctx.cam if viewer_ctx is not None else None
            if cam is not None:
                renderer.update_scene(data, camera=cam)
            else:
                renderer.update_scene(data)
            writer.append_data(renderer.render())
        if viewer_delay > 0.0:
            time.sleep(viewer_delay)

    try:
        print(
            "[contact] fingertip_geoms="
            f"{len(contact_cfg.geom_to_finger)} target_geoms={len(contact_cfg.target_geom_ids)}",
            flush=True,
        )
        for ep in range(args.episodes):
            reset_to_initial(model, data)
            spawn_pos, spawn_quat = compute_mustard_spawn_pose(model, data, args.side)
            set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)

            stable_hits = 0
            first_contact_step = -1
            first_success_step = -1
            best_contacts = 0
            best_contact_fingers = 0
            best_force = 0.0
            best_finger_set: set[str] = set()
            total_steps = 0

            def _step(q_cmd: np.ndarray) -> None:
                nonlocal stable_hits
                nonlocal first_contact_step
                nonlocal first_success_step
                nonlocal best_contacts
                nonlocal best_contact_fingers
                nonlocal best_force
                nonlocal best_finger_set
                nonlocal total_steps

                data.ctrl[:16] = q_cmd
                set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_step(model, data)
                set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_forward(model, data)

                is_contact, n_contacts, total_force, touched_fingers = detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                if is_contact and first_contact_step < 0:
                    first_contact_step = total_steps
                best_contacts = max(best_contacts, n_contacts)
                best_contact_fingers = max(best_contact_fingers, len(touched_fingers))
                if len(touched_fingers) >= len(best_finger_set):
                    best_finger_set = set(touched_fingers)
                best_force = max(best_force, total_force)

                thumb_ok = (not args.require_thumb_contact) or ("th" in touched_fingers)
                meets = (
                    n_contacts >= args.min_contacts
                    and len(touched_fingers) >= args.min_contact_fingers
                    and thumb_ok
                    and total_force >= args.min_force
                    and total_force <= args.max_force
                )
                stable_hits = stable_hits + 1 if meets else 0
                if stable_hits >= args.stable_steps and first_success_step < 0:
                    first_success_step = total_steps

                if sync_enabled:
                    sync_and_record()
                total_steps += 1

            for _ in range(args.open_steps):
                _step(open_pose)
            for i in range(args.preshape_steps):
                alpha = (i + 1) / max(args.preshape_steps, 1)
                q_cmd = (1.0 - alpha) * open_pose + alpha * preshape_pose
                _step(q_cmd)
            for i in range(args.close_steps):
                alpha = (i + 1) / max(args.close_steps, 1)
                q_cmd = (1.0 - alpha) * preshape_pose + alpha * close_pose
                _step(q_cmd)
            for _ in range(args.hold_steps):
                _step(close_pose)

            success = first_success_step >= 0
            episode_result = {
                "episode": ep,
                "success": success,
                "first_contact_step": first_contact_step,
                "first_success_step": first_success_step,
                "best_contacts": best_contacts,
                "best_contact_fingers": best_contact_fingers,
                "best_fingers": sorted(best_finger_set),
                "best_force": best_force,
                "total_steps": total_steps,
            }
            result["episodes_data"].append(episode_result)
            print(
                f"[episode {ep:03d}] success={int(success)} contacts={best_contacts} "
                f"fingers={best_contact_fingers}({','.join(sorted(best_finger_set))}) "
                f"force={best_force:.3f}N",
                flush=True,
            )

        if viewer_ctx is not None and args.keep_open:
            print("Viewer is kept open. Close the MuJoCo window to finish.", flush=True)
            while viewer_ctx.is_running():
                sync_and_record()

    finally:
        if writer is not None:
            writer.close()
            print("[record] finished", flush=True)
        if renderer is not None:
            renderer.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}")

    successes = [ep["success"] for ep in result["episodes_data"]]
    best_contacts = [ep["best_contacts"] for ep in result["episodes_data"]]
    best_forces = [ep["best_force"] for ep in result["episodes_data"]]
    best_fingers = [ep["best_contact_fingers"] for ep in result["episodes_data"]]
    if successes:
        result["summary"] = {
            "success_rate": float(np.mean(np.asarray(successes, dtype=float))),
            "mean_best_contacts": float(np.mean(np.asarray(best_contacts, dtype=float))),
            "mean_best_contact_fingers": float(np.mean(np.asarray(best_fingers, dtype=float))),
            "mean_best_force": float(np.mean(np.asarray(best_forces, dtype=float))),
        }
    else:
        result["summary"] = {
            "success_rate": 0.0,
            "mean_best_contacts": 0.0,
            "mean_best_contact_fingers": 0.0,
            "mean_best_force": 0.0,
        }
    return result


def print_result(result: dict) -> None:
    s = result["summary"]
    print("\n=== Mustard Grasp Collection Summary ===")
    print(
        f"side={result['side']} episodes={result['episodes']} "
        f"success_rate={100.0 * s['success_rate']:.1f}% "
        f"mean_contacts={s['mean_best_contacts']:.2f} "
        f"mean_fingers={s['mean_best_contact_fingers']:.2f} "
        f"mean_force={s['mean_best_force']:.3f}N"
    )


def main() -> None:
    args = parse_args()
    result = run_collection(args)
    print_result(result)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
