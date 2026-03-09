#!/usr/bin/env python3
"""Run 6D pre-grasp IK for Franka+Allegro to reach a mustard pre-grasp pose."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from dataclasses import dataclass
from datetime import datetime
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from mujoco import viewer

from env import franka_allegro_mjcf
from env.allegro_hand_trajectories import interpolate_allegro_hand_pose
from env.viewer_utils import set_default_franka_allegro_camera


MUSTARD_BODY_NAME = "mustard/006_mustard_bottle"
MUSTARD_JOINT_NAME = "mustard/006_mustard_bottle"

@dataclass(frozen=True)
class ArmIkHandles:
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    act_ids: np.ndarray
    qmin: np.ndarray
    qmax: np.ndarray
    palm_body_id: int
    mustard_body_id: int
    mustard_qpos_adr: int
    mustard_qvel_adr: int


@dataclass(frozen=True)
class HandHandles:
    qmin: np.ndarray
    qmax: np.ndarray
    act_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Franka 6D IK pre-grasp demo: move arm to a precise mustard pre-grasp pose "
            "(no grasp close/lift)."
        )
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--show-right-ui", action="store_true")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)

    parser.add_argument(
        "--spawn-pos",
        type=float,
        nargs=3,
        default=(0.62, 0.06, 0.82),
        metavar=("X", "Y", "Z"),
        help="Mustard spawn world position.",
    )
    parser.add_argument(
        "--spawn-quat",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Mustard spawn world quaternion (wxyz).",
    )
    parser.add_argument(
        "--lock-object",
        action="store_true",
        default=True,
        help="Clamp mustard freejoint each step (default: on).",
    )
    parser.add_argument(
        "--no-lock-object",
        dest="lock_object",
        action="store_false",
        help="Disable object clamp.",
    )

    parser.add_argument(
        "--target-offset",
        type=float,
        nargs=3,
        default=(-0.078, -0.015, 0.055),
        metavar=("DX", "DY", "DZ"),
        help=(
            "Pre-grasp target offset from mustard center. "
            "Interpreted in object frame by default."
        ),
    )
    parser.add_argument(
        "--target-offset-frame",
        choices=("object", "world"),
        default="object",
        help="Frame for --target-offset.",
    )
    parser.add_argument(
        "--target-rot-quat",
        type=float,
        nargs=4,
        default=(0.70710678, 0.70710678, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help=(
            "Desired palm orientation quaternion (wxyz). "
            "Interpreted in object frame by default."
        ),
    )
    parser.add_argument(
        "--target-rot-frame",
        choices=("object", "world"),
        default="object",
        help="Frame for --target-rot-quat.",
    )
    parser.add_argument("--ik-pos-gain", type=float, default=0.95)
    parser.add_argument("--ik-rot-gain", type=float, default=0.90)
    parser.add_argument("--ik-damping", type=float, default=0.10)
    parser.add_argument("--ik-pos-weight", type=float, default=1.0)
    parser.add_argument("--ik-rot-weight", type=float, default=0.35)
    parser.add_argument(
        "--ik-max-joint-step",
        type=float,
        default=0.08,
        help="Max arm joint change per IK update (rad).",
    )
    parser.add_argument("--stop-pos-error", type=float, default=0.012)
    parser.add_argument("--stop-rot-error-deg", type=float, default=6.0)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.2,
        help="Required stable duration under stop thresholds.",
    )
    parser.add_argument(
        "--hand-preshape-ratio",
        type=float,
        default=0.0,
        help="Interpolation ratio [0,1] from open hand to chosen pre-grasp trajectory.",
    )
    parser.add_argument(
        "--hand-trajectory",
        choices=("keti_human", "power_linear", "thumb_opposition", "thumb_o_wrap"),
        default="thumb_o_wrap",
        help="Hand trajectory used for pre-grasp visualization.",
    )

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-path", type=str, default="")
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument(
        "--capture-path",
        type=str,
        default="",
        help="Optional final-frame PNG path. Default: codex/logs/franka_pregrasp_*.png",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        default=True,
        help="Render XYZ axes on palm and mustard bodies (default: on).",
    )
    parser.add_argument(
        "--no-show-axes",
        dest="show_axes",
        action="store_false",
        help="Disable palm/object axis visualization.",
    )
    return parser.parse_args()


def _default_record_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"franka_pregrasp_ik_{ts}.mp4"


def _default_capture_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"franka_pregrasp_ik_{ts}.png"


def _resolve_viewer_nstep(model: mujoco.MjModel, viewer_step_hz: float) -> tuple[int, float]:
    if viewer_step_hz <= 0.0:
        raise ValueError("--viewer-step must be > 0")
    target_dt = 1.0 / viewer_step_hz
    nstep = max(1, int(round(target_dt / model.opt.timestep)))
    actual_dt = nstep * model.opt.timestep
    return nstep, actual_dt


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    return q / norm


def _quat_to_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(quat)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    return quat


def _orientation_error_world(cur_rot: np.ndarray, tgt_rot: np.ndarray) -> np.ndarray:
    # Small-angle orientation error in world frame.
    return 0.5 * (
        np.cross(cur_rot[:, 0], tgt_rot[:, 0])
        + np.cross(cur_rot[:, 1], tgt_rot[:, 1])
        + np.cross(cur_rot[:, 2], tgt_rot[:, 2])
    )


def _build_arm_handles(model: mujoco.MjModel, side: str) -> ArmIkHandles:
    arm_joint_names = [f"franka/joint{i}" for i in range(1, 8)]
    arm_joint_ids = np.array([model.joint(name).id for name in arm_joint_names], dtype=np.int32)
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in arm_joint_ids], dtype=np.int32)
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=np.int32)
    act_ids = np.array(
        [model.actuator(f"franka/actuator{i}").id for i in range(1, 8)],
        dtype=np.int32,
    )
    palm_body_id = model.body(f"franka/allegro_{side}/palm").id

    mustard_body_id = model.body(MUSTARD_BODY_NAME).id
    mustard_joint_id = model.joint(MUSTARD_JOINT_NAME).id

    return ArmIkHandles(
        qpos_ids=qpos_ids,
        dof_ids=dof_ids,
        act_ids=act_ids,
        qmin=model.jnt_range[arm_joint_ids, 0].copy(),
        qmax=model.jnt_range[arm_joint_ids, 1].copy(),
        palm_body_id=palm_body_id,
        mustard_body_id=mustard_body_id,
        mustard_qpos_adr=int(model.jnt_qposadr[mustard_joint_id]),
        mustard_qvel_adr=int(model.jnt_dofadr[mustard_joint_id]),
    )


def _build_hand_handles(model: mujoco.MjModel, side: str, arm_act_ids: np.ndarray) -> HandHandles:
    prefix = f"franka/allegro_{side}"
    hand_joint_ids: list[int] = []
    for finger in ("ff", "mf", "rf", "th"):
        for j in range(4):
            hand_joint_ids.append(model.joint(f"{prefix}/{finger}j{j}").id)
    qmin = model.jnt_range[np.asarray(hand_joint_ids, dtype=np.int32), 0].copy()
    qmax = model.jnt_range[np.asarray(hand_joint_ids, dtype=np.int32), 1].copy()

    all_act_ids = np.arange(model.nu, dtype=np.int32)
    hand_act_ids = np.setdiff1d(all_act_ids, arm_act_ids)
    if hand_act_ids.size != 16:
        raise RuntimeError(
            f"Expected 16 hand actuators, got {hand_act_ids.size}. "
            "Check Franka+Allegro actuator ordering."
        )
    return HandHandles(qmin=qmin, qmax=qmax, act_ids=hand_act_ids)


def _set_mustard_pose(data: mujoco.MjData, handles: ArmIkHandles, pos: np.ndarray, quat: np.ndarray) -> None:
    adr_q = handles.mustard_qpos_adr
    adr_v = handles.mustard_qvel_adr
    data.qpos[adr_q : adr_q + 3] = pos
    data.qpos[adr_q + 3 : adr_q + 7] = quat
    data.qvel[adr_v : adr_v + 6] = 0.0


def _compute_world_target(
    data: mujoco.MjData,
    handles: ArmIkHandles,
    target_offset: np.ndarray,
    target_offset_frame: str,
    target_rot_quat: np.ndarray,
    target_rot_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    obj_pos = data.xpos[handles.mustard_body_id].copy()
    obj_rot = data.xmat[handles.mustard_body_id].reshape(3, 3).copy()

    if target_offset_frame == "object":
        tgt_pos = obj_pos + obj_rot @ target_offset
    else:
        tgt_pos = obj_pos + target_offset

    rot_offset = _quat_to_rot_wxyz(target_rot_quat)
    tgt_rot = obj_rot @ rot_offset if target_rot_frame == "object" else rot_offset
    return tgt_pos, tgt_rot


def _step_arm_ik_6d(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ArmIkHandles,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    pos_gain: float,
    rot_gain: float,
    damping: float,
    pos_weight: float,
    rot_weight: float,
    max_joint_step: float,
) -> tuple[float, float]:
    ee_pos = data.xpos[handles.palm_body_id].copy()
    ee_rot = data.xmat[handles.palm_body_id].reshape(3, 3).copy()
    pos_err = target_pos - ee_pos
    rot_err_vec = _orientation_error_world(ee_rot, target_rot)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, handles.palm_body_id)
    j_pos = jacp[:, handles.dof_ids]
    j_rot = jacr[:, handles.dof_ids]

    j = np.vstack([pos_weight * j_pos, rot_weight * j_rot])
    err6 = np.concatenate([pos_weight * pos_gain * pos_err, rot_weight * rot_gain * rot_err_vec])

    jjt = j @ j.T
    dq = j.T @ np.linalg.solve(
        jjt + (damping**2) * np.eye(6, dtype=np.float64),
        err6,
    )
    if max_joint_step > 0.0:
        dq = np.clip(dq, -max_joint_step, max_joint_step)

    q_cur = data.qpos[handles.qpos_ids].copy()
    q_des = np.clip(q_cur + dq, handles.qmin, handles.qmax)
    data.ctrl[handles.act_ids] = q_des

    pos_err_norm = float(np.linalg.norm(pos_err))
    rot_err_rad = float(np.linalg.norm(rot_err_vec))
    return pos_err_norm, rot_err_rad


def _setup_recording(args: argparse.Namespace):
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
    frame_dt = 1.0 / max(int(args.record_fps), 1)
    print(
        f"[record] writing {record_path} "
        f"({args.record_width}x{args.record_height}@{args.record_fps}fps)",
        flush=True,
    )
    return writer, record_path, frame_dt


def _make_capture_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    set_default_franka_allegro_camera(cam)
    return cam


def main() -> None:
    args = parse_args()

    mjcf = franka_allegro_mjcf.load(
        side=args.side,
        add_mustard=True,
        add_frame_axes=args.show_axes,
    )
    model = mjcf.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("initial_state").id)
    mujoco.mj_forward(model, data)

    arm = _build_arm_handles(model, args.side)
    hand = _build_hand_handles(model, args.side, arm.act_ids)

    spawn_pos = np.asarray(args.spawn_pos, dtype=np.float64)
    spawn_quat = _normalize_quat_wxyz(np.asarray(args.spawn_quat, dtype=np.float64))
    _set_mustard_pose(data, arm, spawn_pos, spawn_quat)
    mujoco.mj_forward(model, data)

    pre_ratio = float(np.clip(args.hand_preshape_ratio, 0.0, 1.0))
    hand_cmd = interpolate_allegro_hand_pose(
        grasp_value=pre_ratio,
        q_min=hand.qmin,
        q_max=hand.qmax,
        trajectory=args.hand_trajectory,
        preshape_pivot=0.45,
    ).astype(np.float64)
    data.ctrl[hand.act_ids] = hand_cmd

    target_offset = np.asarray(args.target_offset, dtype=np.float64)
    target_rot_quat = _normalize_quat_wxyz(np.asarray(args.target_rot_quat, dtype=np.float64))

    target_pos, target_rot = _compute_world_target(
        data=data,
        handles=arm,
        target_offset=target_offset,
        target_offset_frame=args.target_offset_frame,
        target_rot_quat=target_rot_quat,
        target_rot_frame=args.target_rot_frame,
    )
    target_quat_world = _rot_to_quat_wxyz(target_rot)
    print(
        "[target] pos="
        f"[{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}] "
        "quat_wxyz="
        f"[{target_quat_world[0]:.6f}, {target_quat_world[1]:.6f}, "
        f"{target_quat_world[2]:.6f}, {target_quat_world[3]:.6f}]",
        flush=True,
    )

    writer, record_path, frame_dt = _setup_recording(args)
    renderer = None
    if writer is not None:
        renderer = mujoco.Renderer(model, width=args.record_width, height=args.record_height)
    capture_cam = _make_capture_camera()
    next_frame_t = data.time

    viewer_nstep, actual_viewer_dt = _resolve_viewer_nstep(model, args.viewer_step)
    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else max(0.0, (1.0 / max(args.viewer_step, 1e-6)) - actual_viewer_dt)
    )

    pos_err = np.inf
    rot_err_rad = np.inf
    reached_time = 0.0
    reached = False

    def _loop_step(render_cam=None):
        nonlocal pos_err, rot_err_rad, reached, reached_time, next_frame_t

        if args.lock_object:
            _set_mustard_pose(data, arm, spawn_pos, spawn_quat)

        target_pos_now, target_rot_now = _compute_world_target(
            data=data,
            handles=arm,
            target_offset=target_offset,
            target_offset_frame=args.target_offset_frame,
            target_rot_quat=target_rot_quat,
            target_rot_frame=args.target_rot_frame,
        )
        pos_err, rot_err_rad = _step_arm_ik_6d(
            model=model,
            data=data,
            handles=arm,
            target_pos=target_pos_now,
            target_rot=target_rot_now,
            pos_gain=args.ik_pos_gain,
            rot_gain=args.ik_rot_gain,
            damping=args.ik_damping,
            pos_weight=args.ik_pos_weight,
            rot_weight=args.ik_rot_weight,
            max_joint_step=args.ik_max_joint_step,
        )
        data.ctrl[hand.act_ids] = hand_cmd

        mujoco.mj_step(model, data, nstep=viewer_nstep)

        rot_err_deg = float(np.rad2deg(rot_err_rad))
        in_band = (pos_err <= args.stop_pos_error) and (rot_err_deg <= args.stop_rot_error_deg)
        reached_time = reached_time + actual_viewer_dt if in_band else 0.0
        reached = reached_time >= args.hold_seconds

        if renderer is not None and writer is not None and data.time >= next_frame_t:
            if render_cam is None:
                renderer.update_scene(data, camera=capture_cam)
            else:
                renderer.update_scene(data, camera=render_cam)
            writer.append_data(renderer.render())
            next_frame_t += frame_dt

    if args.no_viewer:
        try:
            for _ in range(max(int(args.steps), 0)):
                _loop_step(render_cam=None)
                if reached:
                    break
        finally:
            pass
    else:
        print(
            f"[viewer] target_hz={args.viewer_step:.1f}, nstep={viewer_nstep}, "
            f"sim_dt={actual_viewer_dt:.4f}s",
            flush=True,
        )
        if args.viewer_step_delay is not None:
            print(f"[viewer] extra_delay={args.viewer_step_delay:.4f}s", flush=True)
        with viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=args.show_right_ui,
        ) as passive_viewer:
            set_default_franka_allegro_camera(passive_viewer.cam)
            while passive_viewer.is_running():
                _loop_step(render_cam=passive_viewer.cam)
                passive_viewer.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)
                if reached:
                    break

    if args.lock_object:
        _set_mustard_pose(data, arm, spawn_pos, spawn_quat)
        mujoco.mj_forward(model, data)

    final_pos = data.xpos[arm.palm_body_id].copy()
    final_rot = data.xmat[arm.palm_body_id].reshape(3, 3).copy()
    final_quat = _rot_to_quat_wxyz(final_rot)
    final_rot_deg = float(np.rad2deg(rot_err_rad))
    print(
        "[result] reached="
        f"{reached} pos_err={pos_err:.5f}m rot_err={final_rot_deg:.3f}deg",
        flush=True,
    )
    print(
        "[result] palm_pos="
        f"[{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}] "
        "palm_quat_wxyz="
        f"[{final_quat[0]:.6f}, {final_quat[1]:.6f}, {final_quat[2]:.6f}, {final_quat[3]:.6f}]",
        flush=True,
    )

    capture_path = Path(args.capture_path) if args.capture_path else _default_capture_path()
    capture_path.parent.mkdir(parents=True, exist_ok=True)

    if renderer is None:
        renderer = mujoco.Renderer(model, width=args.record_width, height=args.record_height)
    renderer.update_scene(data, camera=capture_cam)
    imageio.imwrite(capture_path, renderer.render())
    print(f"[capture] {capture_path}", flush=True)

    if writer is not None:
        writer.close()
        print("[record] finished", flush=True)
        print(f"[record] path={record_path}", flush=True)
    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
