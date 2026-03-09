#!/usr/bin/env python3
"""Launch Franka + Allegro + mustard tabletop scene with slider controls."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from mujoco import viewer

from env import franka_allegro_mjcf
from env.viewer_utils import set_default_franka_allegro_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Franka + Allegro + mustard tabletop MuJoCo scene."
    )
    parser.add_argument(
        "--side",
        choices=("right", "left"),
        default="right",
        help="Which Allegro hand to mount (default: right).",
    )
    parser.add_argument(
        "--no-mustard",
        action="store_true",
        help="Disable mustard bottle in scene.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless for fixed simulation steps.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Headless simulation steps when --no-viewer is set.",
    )
    parser.add_argument(
        "--viewer-step",
        type=float,
        default=60.0,
        help=(
            "Target viewer/control update rate in Hz. "
            "Simulation runs multiple mj steps per viewer sync (default: 60)."
        ),
    )
    parser.add_argument(
        "--viewer-step-delay",
        type=float,
        default=None,
        help="Optional extra sleep (seconds) after each viewer sync.",
    )
    parser.add_argument(
        "--show-right-ui",
        action="store_true",
        help="Show MuJoCo right-side control panel (off by default for speed).",
    )
    parser.add_argument(
        "--arm-mode",
        choices=("manual", "ik-approach"),
        default="manual",
        help="Arm control mode: manual sliders or IK approach to mustard (default: manual).",
    )
    parser.add_argument(
        "--ik-target-offset",
        type=float,
        nargs=3,
        default=(-0.08, 0.0, 0.05),
        metavar=("DX", "DY", "DZ"),
        help=(
            "Target offset from mustard body center in world frame for IK approach "
            "(default: -0.08 0.0 0.05)."
        ),
    )
    parser.add_argument(
        "--ik-gain",
        type=float,
        default=0.9,
        help="Cartesian IK proportional gain (default: 0.9).",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=0.06,
        help="Damped least-squares lambda (default: 0.06).",
    )
    parser.add_argument(
        "--ik-stop-error",
        type=float,
        default=0.015,
        help="Position error threshold in meters to mark reach complete (default: 0.015).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record MP4 while simulation is running.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default="",
        help="Output mp4 path. Default: codex/logs/franka_allegro_mustard_<timestamp>.mp4",
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
    return Path("codex/logs") / f"franka_allegro_mustard_{ts}.mp4"


def _resolve_viewer_nstep(model: mujoco.MjModel, viewer_step_hz: float) -> tuple[int, float]:
    if viewer_step_hz <= 0.0:
        raise ValueError("--viewer-step must be > 0")
    target_dt = 1.0 / viewer_step_hz
    nstep = max(1, int(round(target_dt / model.opt.timestep)))
    actual_dt = nstep * model.opt.timestep
    return nstep, actual_dt


def _setup_recorder(args: argparse.Namespace):
    if not args.record:
        return None, None, 0.0

    record_path = Path(args.record_path) if args.record_path else _default_record_path()
    record_path.parent.mkdir(parents=True, exist_ok=True)

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
    frame_dt = 1.0 / max(args.record_fps, 1)
    return writer, record_path, frame_dt


def _resolve_arm_ik_handles(model: mujoco.MjModel, side: str) -> dict[str, np.ndarray | int]:
    arm_joint_names = [f"franka/joint{i}" for i in range(1, 8)]
    arm_joint_ids = np.array([model.joint(name).id for name in arm_joint_names], dtype=np.int32)
    arm_qpos_ids = np.array([model.jnt_qposadr[jid] for jid in arm_joint_ids], dtype=np.int32)
    arm_dof_ids = np.array([model.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=np.int32)
    arm_act_ids = np.array(
        [model.actuator(f"franka/actuator{i}").id for i in range(1, 8)],
        dtype=np.int32,
    )

    ee_body_id = model.body(f"franka/allegro_{side}/palm").id
    mustard_body_id = model.body("mustard/006_mustard_bottle").id
    joint_ranges = model.jnt_range[arm_joint_ids]

    return {
        "qpos_ids": arm_qpos_ids,
        "dof_ids": arm_dof_ids,
        "act_ids": arm_act_ids,
        "ee_body_id": ee_body_id,
        "mustard_body_id": mustard_body_id,
        "qmin": joint_ranges[:, 0].copy(),
        "qmax": joint_ranges[:, 1].copy(),
    }


def _apply_arm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: dict[str, np.ndarray | int],
    target_offset_world: np.ndarray,
    gain: float,
    damping: float,
    hand_ctrl_ref: np.ndarray,
) -> float:
    ee_body_id = int(handles["ee_body_id"])
    mustard_body_id = int(handles["mustard_body_id"])
    qpos_ids = handles["qpos_ids"]
    dof_ids = handles["dof_ids"]
    act_ids = handles["act_ids"]
    qmin = handles["qmin"]
    qmax = handles["qmax"]

    target_pos = data.xpos[mustard_body_id].copy() + target_offset_world
    ee_pos = data.xpos[ee_body_id].copy()
    pos_err = target_pos - ee_pos

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, None, ee_body_id)
    j_arm = jacp[:, dof_ids]

    jjt = j_arm @ j_arm.T
    dq = j_arm.T @ np.linalg.solve(
        jjt + (damping**2) * np.eye(3, dtype=np.float64),
        gain * pos_err,
    )

    q_cur = data.qpos[qpos_ids].copy()
    q_des = np.clip(q_cur + dq, qmin, qmax)
    data.ctrl[act_ids] = q_des

    # Keep Allegro hand fixed during arm-only IK approach.
    data.ctrl[7:23] = hand_ctrl_ref

    return float(np.linalg.norm(pos_err))


def main() -> None:
    args = parse_args()

    if args.arm_mode == "ik-approach" and args.no_mustard:
        raise ValueError("--arm-mode ik-approach requires mustard object (remove --no-mustard)")

    mjcf = franka_allegro_mjcf.load(side=args.side, add_mustard=not args.no_mustard)
    model = mjcf.compile()
    data = mujoco.MjData(model)
    initial_state = model.key("initial_state").id
    mujoco.mj_resetDataKeyframe(model, data, initial_state)
    mujoco.mj_forward(model, data)

    ik_handles = None
    if args.arm_mode == "ik-approach":
        ik_handles = _resolve_arm_ik_handles(model, args.side)
        print(
            "[ik] enabled: end-effector body=franka/allegro_"
            f"{args.side}/palm, target=mustard+offset({args.ik_target_offset[0]:.3f}, "
            f"{args.ik_target_offset[1]:.3f}, {args.ik_target_offset[2]:.3f})",
            flush=True,
        )

    writer, record_path, frame_dt = _setup_recorder(args)

    renderer = None
    if writer is not None:
        renderer = mujoco.Renderer(
            model,
            width=args.record_width,
            height=args.record_height,
        )

    next_frame_t = data.time
    ik_target_offset = np.asarray(args.ik_target_offset, dtype=np.float64)
    hand_ctrl_ref = data.ctrl[7:23].copy()
    last_ik_error = np.nan
    ik_reached = False

    if args.no_viewer:
        try:
            for _ in range(max(args.steps, 0)):
                if ik_handles is not None:
                    last_ik_error = _apply_arm_ik(
                        model=model,
                        data=data,
                        handles=ik_handles,
                        target_offset_world=ik_target_offset,
                        gain=args.ik_gain,
                        damping=args.ik_damping,
                        hand_ctrl_ref=hand_ctrl_ref,
                    )
                    if last_ik_error < args.ik_stop_error:
                        ik_reached = True
                mujoco.mj_step(model, data)
                if renderer is not None and writer is not None and data.time >= next_frame_t:
                    renderer.update_scene(data)
                    writer.append_data(renderer.render())
                    next_frame_t += frame_dt
        finally:
            if ik_handles is not None:
                print(
                    f"[ik] final_error={last_ik_error:.4f} m, reached={ik_reached}",
                    flush=True,
                )
            if writer is not None:
                writer.close()
                print("[record] finished", flush=True)
            if renderer is not None:
                renderer.close()
        return

    viewer_nstep, actual_viewer_dt = _resolve_viewer_nstep(model, args.viewer_step)
    print(
        f"[viewer] target_hz={args.viewer_step:.1f} "
        f"nstep={viewer_nstep} (sim_dt={actual_viewer_dt:.4f}s per sync)",
        flush=True,
    )
    if args.viewer_step_delay is not None:
        print(f"[viewer] extra_delay={args.viewer_step_delay:.4f}s per sync", flush=True)

    with viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=args.show_right_ui,
    ) as passive_viewer:
        set_default_franka_allegro_camera(passive_viewer.cam)
        next_frame_t = data.time

        try:
            while passive_viewer.is_running():
                if ik_handles is not None:
                    last_ik_error = _apply_arm_ik(
                        model=model,
                        data=data,
                        handles=ik_handles,
                        target_offset_world=ik_target_offset,
                        gain=args.ik_gain,
                        damping=args.ik_damping,
                        hand_ctrl_ref=hand_ctrl_ref,
                    )
                    if (not ik_reached) and last_ik_error < args.ik_stop_error:
                        ik_reached = True
                        print(
                            f"[ik] reached stop error {args.ik_stop_error:.3f} m "
                            f"(current {last_ik_error:.4f} m)",
                            flush=True,
                        )
                mujoco.mj_step(model, data, nstep=viewer_nstep)

                if renderer is not None and writer is not None and data.time >= next_frame_t:
                    renderer.update_scene(data, camera=passive_viewer.cam)
                    writer.append_data(renderer.render())
                    next_frame_t += frame_dt

                passive_viewer.sync()
                if args.viewer_step_delay is not None and args.viewer_step_delay > 0:
                    time.sleep(args.viewer_step_delay)
        finally:
            if ik_handles is not None:
                print(
                    f"[ik] final_error={last_ik_error:.4f} m, reached={ik_reached}",
                    flush=True,
                )
            if writer is not None:
                writer.close()
                print("[record] finished", flush=True)
            if renderer is not None:
                renderer.close()


if __name__ == "__main__":
    main()
