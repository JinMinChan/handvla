#!/usr/bin/env python3
"""Collect successful Franka+Allegro mustard pick-and-lift episodes (raw NPZ)."""

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

import mujoco
import numpy as np
from mujoco import viewer

from env import franka_allegro_mjcf
from env.allegro_hand_trajectories import (
    KETI_HUMAN_CLOSE_QPOS,
    POWER_GRASP_CLOSE_QPOS,
    POWER_GRASP_PRESHAPE_QPOS,
    interpolate_allegro_hand_pose,
)
from env.viewer_utils import set_default_franka_allegro_camera

FINGER_KEYS = ("ff", "mf", "rf", "th")
FINGER_CONTACT_SEGMENTS = ("distal", "tip")

PHASE_SETTLE = 0
PHASE_APPROACH = 1
PHASE_PRESHAPE = 2
PHASE_CLOSE = 3
PHASE_LIFT = 4
PHASE_LIFT_HOLD = 5

@dataclass(frozen=True)
class ArmConfig:
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    act_ids: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    palm_body_id: int


@dataclass(frozen=True)
class HandConfig:
    qpos_ids: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray


@dataclass(frozen=True)
class MustardConfig:
    qpos_adr: int
    qvel_adr: int
    body_id: int


@dataclass(frozen=True)
class ContactConfig:
    geom_to_finger: dict[int, str]
    target_geom_ids: set[int]


@dataclass(frozen=True)
class ArmTargetPose:
    pos: np.ndarray
    quat_wxyz: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect pick-and-lift episodes in Franka+Allegro mustard scene. "
            "Save only successful episodes."
        )
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--target-episodes", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/franka_pickandlift",
        help="Output root directory. Raw episodes are saved in <out-dir>/raw.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="reach, grasp, and lift the mustard bottle",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--capture-hz", type=float, default=20.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)

    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--approach-steps", type=int, default=700)
    parser.add_argument("--preshape-steps", type=int, default=150)
    parser.add_argument("--close-steps", type=int, default=280)
    parser.add_argument("--lift-steps", type=int, default=260)
    parser.add_argument(
        "--hand-trajectory",
        choices=("keti_human", "power_linear", "thumb_opposition", "thumb_o_wrap"),
        default="thumb_o_wrap",
        help="Hand closing profile (default thumb_o_wrap for wider thumb opposition).",
    )
    parser.add_argument(
        "--preshape-grasp-ratio",
        type=float,
        default=0.45,
        help="Grasp interpolation ratio at end of preshape phase [0,1].",
    )

    parser.add_argument(
        "--spawn-pos",
        type=float,
        nargs=3,
        default=(0.62, 0.06, 0.82),
        metavar=("X", "Y", "Z"),
        help="Mustard initial world position (default: 0.62 0.06 0.82).",
    )
    parser.add_argument(
        "--spawn-quat",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Mustard initial world quaternion wxyz (default: upright).",
    )
    parser.add_argument(
        "--spawn-jitter-xy",
        type=float,
        default=0.0,
        help="Uniform XY jitter (meters) applied to spawn position per attempt.",
    )
    parser.add_argument(
        "--spawn-yaw-jitter-deg",
        type=float,
        default=0.0,
        help="Uniform yaw jitter (deg, world Z) applied per attempt.",
    )

    parser.add_argument(
        "--arm-approach-offset",
        type=float,
        nargs=3,
        default=(-0.09, -0.015, 0.04),
        metavar=("DX", "DY", "DZ"),
        help="World offset from mustard center for approach target.",
    )
    parser.add_argument(
        "--arm-push-offset",
        type=float,
        nargs=3,
        default=(-0.078, -0.015, 0.01),
        metavar=("DX", "DY", "DZ"),
        help="World offset from mustard center for close/lift target.",
    )
    parser.add_argument(
        "--arm-offset-frame",
        choices=("object", "world"),
        default="object",
        help="Interpret arm offsets in object frame (recommended) or world frame.",
    )
    parser.add_argument(
        "--arm-rot-frame",
        choices=("object", "world"),
        default="object",
        help="Interpret arm target quaternion offsets in object/world frame.",
    )
    parser.add_argument(
        "--arm-approach-rot-quat",
        type=float,
        nargs=4,
        default=(0.70710678, 0.70710678, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Approach palm orientation quaternion offset (wxyz).",
    )
    parser.add_argument(
        "--arm-push-rot-quat",
        type=float,
        nargs=4,
        default=(0.70710678, 0.70710678, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Push/close palm orientation quaternion offset (wxyz).",
    )
    parser.add_argument(
        "--lock-object-until-close",
        action="store_true",
        default=True,
        help="Keep mustard fixed at spawn pose until close phase starts (default: on).",
    )
    parser.add_argument(
        "--no-lock-object-until-close",
        dest="lock_object_until_close",
        action="store_false",
        help="Disable pre-grasp object lock.",
    )
    parser.add_argument("--ik-gain", type=float, default=0.95)
    parser.add_argument("--ik-rot-gain", type=float, default=0.9)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument(
        "--ik-rot-weight",
        type=float,
        default=0.20,
        help="Rotation Jacobian weight for 6D IK.",
    )
    parser.add_argument(
        "--ik-max-joint-step",
        type=float,
        default=0.08,
        help="Max per-joint update per control step (rad).",
    )
    parser.add_argument(
        "--close-z-clearance",
        type=float,
        default=0.015,
        help="Extra +Z clearance (m) applied to push target during preshape/close/lift to avoid table snag.",
    )
    parser.add_argument(
        "--arm-reach-threshold",
        type=float,
        default=0.05,
        help="Reach success threshold in meters for approach phase.",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.20,
        help="Planned lift height in meters.",
    )
    parser.add_argument(
        "--lift-success-delta",
        type=float,
        default=0.08,
        help="Object z increase threshold in meters to count as lifted (stricter).",
    )
    parser.add_argument(
        "--lift-hold-seconds",
        type=float,
        default=1.5,
        help="Required hold time (seconds) after lift before success.",
    )

    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=3)
    parser.add_argument(
        "--require-thumb-contact",
        action="store_true",
        default=True,
        help="Require thumb contact for grasp/lift success (default: on).",
    )
    parser.add_argument(
        "--no-require-thumb-contact",
        dest="require_thumb_contact",
        action="store_false",
    )
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument(
        "--stable-steps",
        type=int,
        default=3,
        help="Consecutive steps required for grasp/lift event.",
    )

    parser.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Show interactive viewer.",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Run headless without viewer.",
    )
    parser.set_defaults(viewer=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")
    return parser.parse_args()


def _capture_every(control_hz: float, capture_hz: float) -> int:
    if capture_hz <= 0:
        return 1
    ratio = control_hz / capture_hz
    if ratio <= 1.0:
        return 1
    return max(1, int(round(ratio)))


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).copy()
    nrm = float(np.linalg.norm(q))
    if nrm < 1e-12:
        raise ValueError("Quaternion norm too small.")
    return q / nrm


def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(quat_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).reshape(-1))
    return _normalize_quat(quat)


def _quat_lerp_normalize(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    q = (1.0 - a) * _normalize_quat(q0) + a * _normalize_quat(q1)
    return _normalize_quat(q)


def _sample_spawn_pose(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    pos = base_pos.astype(np.float64).copy()
    quat = base_quat.astype(np.float64).copy()

    if args.spawn_jitter_xy > 0.0:
        pos[0] += float(rng.uniform(-args.spawn_jitter_xy, args.spawn_jitter_xy))
        pos[1] += float(rng.uniform(-args.spawn_jitter_xy, args.spawn_jitter_xy))
    if args.spawn_yaw_jitter_deg > 0.0:
        yaw = np.deg2rad(
            float(rng.uniform(-args.spawn_yaw_jitter_deg, args.spawn_yaw_jitter_deg))
        )
        qz = np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)
        quat = _quat_mul(qz, quat)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return pos.astype(np.float32), quat.astype(np.float32)


def _resolve_arm_targets(
    data: mujoco.MjData,
    mustard_cfg: MustardConfig,
    approach_offset: np.ndarray,
    push_offset: np.ndarray,
    approach_rot_quat: np.ndarray,
    push_rot_quat: np.ndarray,
    offset_frame: str,
    rot_frame: str,
) -> tuple[ArmTargetPose, ArmTargetPose]:
    mustard_pos = data.xpos[mustard_cfg.body_id].copy()
    obj_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3)

    if offset_frame == "world":
        approach_pos = mustard_pos + approach_offset
        push_pos = mustard_pos + push_offset
    else:
        approach_pos = mustard_pos + obj_rot @ approach_offset
        push_pos = mustard_pos + obj_rot @ push_offset

    approach_rot_offset = _quat_to_rot(approach_rot_quat)
    push_rot_offset = _quat_to_rot(push_rot_quat)
    if rot_frame == "world":
        approach_rot = approach_rot_offset
        push_rot = push_rot_offset
    else:
        approach_rot = obj_rot @ approach_rot_offset
        push_rot = obj_rot @ push_rot_offset

    return (
        ArmTargetPose(pos=approach_pos.astype(np.float64), quat_wxyz=_rot_to_quat(approach_rot)),
        ArmTargetPose(pos=push_pos.astype(np.float64), quat_wxyz=_rot_to_quat(push_rot)),
    )


def _interpolate_hand_pose(
    grasp_value: float,
    hand_cfg: HandConfig,
    hand_trajectory: str,
    preshape_pivot: float = 0.45,
) -> np.ndarray:
    return interpolate_allegro_hand_pose(
        grasp_value=grasp_value,
        q_min=hand_cfg.q_min,
        q_max=hand_cfg.q_max,
        trajectory=hand_trajectory,
        preshape_pivot=preshape_pivot,
    )


def _build_arm_config(model: mujoco.MjModel, side: str) -> ArmConfig:
    arm_joint_names = [f"franka/joint{i}" for i in range(1, 8)]
    arm_joint_ids = np.array([model.joint(name).id for name in arm_joint_names], dtype=np.int32)
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in arm_joint_ids], dtype=np.int32)
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=np.int32)
    act_ids = np.array(
        [model.actuator(f"franka/actuator{i}").id for i in range(1, 8)],
        dtype=np.int32,
    )
    qmin = model.jnt_range[arm_joint_ids, 0].copy()
    qmax = model.jnt_range[arm_joint_ids, 1].copy()
    palm_body_id = model.body(f"franka/allegro_{side}/palm").id

    return ArmConfig(
        qpos_ids=qpos_ids,
        dof_ids=dof_ids,
        act_ids=act_ids,
        q_min=qmin,
        q_max=qmax,
        palm_body_id=palm_body_id,
    )


def _build_hand_config(model: mujoco.MjModel, side: str) -> HandConfig:
    prefix = f"franka/allegro_{side}"
    qpos_ids: list[int] = []
    q_min: list[float] = []
    q_max: list[float] = []

    for finger in FINGER_KEYS:
        for j in range(4):
            joint_name = f"{prefix}/{finger}j{j}"
            joint_id = model.joint(joint_name).id
            qpos_ids.append(int(model.jnt_qposadr[joint_id]))
            q_min.append(float(model.jnt_range[joint_id][0]))
            q_max.append(float(model.jnt_range[joint_id][1]))

    return HandConfig(
        qpos_ids=np.asarray(qpos_ids, dtype=int),
        q_min=np.asarray(q_min, dtype=float),
        q_max=np.asarray(q_max, dtype=float),
    )


def _build_mustard_config(model: mujoco.MjModel) -> MustardConfig:
    joint_name = f"{franka_allegro_mjcf.MUSTARD_PREFIX}006_mustard_bottle"
    body_name = f"{franka_allegro_mjcf.MUSTARD_PREFIX}006_mustard_bottle"
    joint_id = model.joint(joint_name).id
    body_id = model.body(body_name).id
    return MustardConfig(
        qpos_adr=int(model.jnt_qposadr[joint_id]),
        qvel_adr=int(model.jnt_dofadr[joint_id]),
        body_id=body_id,
    )


def _build_contact_config(
    model: mujoco.MjModel, side: str, mustard_body_id: int
) -> ContactConfig:
    hand_body_prefix = f"franka/allegro_{side}/"
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
        if not body_name or not body_name.startswith(hand_body_prefix):
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
        raise RuntimeError("No fingertip/distal geoms found for contact detection.")
    if not target_geom_ids:
        raise RuntimeError("No mustard target geoms found for contact detection.")

    return ContactConfig(geom_to_finger=geom_to_finger, target_geom_ids=target_geom_ids)


def _detect_contact_with_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: ContactConfig,
    force_buf: np.ndarray,
) -> tuple[int, float, set[str]]:
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

    return num_contacts, total_force, touched_fingers


def _set_mustard_pose(
    data: mujoco.MjData, mustard_cfg: MustardConfig, pos: np.ndarray, quat: np.ndarray
) -> None:
    data.qpos[mustard_cfg.qpos_adr : mustard_cfg.qpos_adr + 3] = pos
    data.qpos[mustard_cfg.qpos_adr + 3 : mustard_cfg.qpos_adr + 7] = quat
    data.qvel[mustard_cfg.qvel_adr : mustard_cfg.qvel_adr + 6] = 0.0


def _capture_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    cam: mujoco.MjvCamera | None,
) -> np.ndarray:
    if cam is None:
        renderer.update_scene(data)
    else:
        renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def _make_state_vector(
    data: mujoco.MjData,
    arm_cfg: ArmConfig,
    hand_cfg: HandConfig,
    mustard_cfg: MustardConfig,
    contact_stats: np.ndarray,
) -> np.ndarray:
    arm_q = data.qpos[arm_cfg.qpos_ids]
    hand_q = data.qpos[hand_cfg.qpos_ids]
    arm_dq = data.qvel[arm_cfg.dof_ids]
    hand_dq = data.qvel[hand_cfg.qpos_ids]
    palm_pos = data.xpos[arm_cfg.palm_body_id]
    mustard_pos = data.qpos[mustard_cfg.qpos_adr : mustard_cfg.qpos_adr + 3]
    mustard_quat = data.qpos[mustard_cfg.qpos_adr + 3 : mustard_cfg.qpos_adr + 7]
    return np.concatenate(
        [arm_q, hand_q, arm_dq, hand_dq, palm_pos, mustard_pos, mustard_quat, contact_stats],
        axis=0,
    ).astype(np.float32)


def _orientation_error_world(cur_rot: np.ndarray, tgt_rot: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(cur_rot[:, 0], tgt_rot[:, 0])
        + np.cross(cur_rot[:, 1], tgt_rot[:, 1])
        + np.cross(cur_rot[:, 2], tgt_rot[:, 2])
    )


def _step_arm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_cfg: ArmConfig,
    target: ArmTargetPose,
    gain: float,
    rot_gain: float,
    damping: float,
    rot_weight: float,
    max_joint_step: float,
) -> tuple[np.ndarray, float, float]:
    ee_pos = data.xpos[arm_cfg.palm_body_id].copy()
    ee_rot = data.xmat[arm_cfg.palm_body_id].reshape(3, 3).copy()
    pos_err = target.pos - ee_pos
    tgt_rot = _quat_to_rot(target.quat_wxyz)
    rot_err_vec = _orientation_error_world(ee_rot, tgt_rot)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, arm_cfg.palm_body_id)
    j_pos = jacp[:, arm_cfg.dof_ids]
    j_rot = jacr[:, arm_cfg.dof_ids]

    j = np.vstack([j_pos, rot_weight * j_rot])
    err6 = np.concatenate(
        [gain * pos_err, rot_weight * rot_gain * rot_err_vec],
        axis=0,
    )
    jjt = j @ j.T
    dq = j.T @ np.linalg.solve(jjt + (damping**2) * np.eye(6, dtype=np.float64), err6)

    if max_joint_step > 0.0:
        dq = np.clip(dq, -max_joint_step, max_joint_step)

    q_cur = data.qpos[arm_cfg.qpos_ids].copy()
    q_des = np.clip(q_cur + dq, arm_cfg.q_min, arm_cfg.q_max)
    return (
        q_des.astype(np.float32),
        float(np.linalg.norm(pos_err)),
        float(np.rad2deg(np.linalg.norm(rot_err_vec))),
    )


def _contact_meets(
    n_contacts: int,
    total_force: float,
    touched_fingers: set[str],
    args: argparse.Namespace,
) -> bool:
    thumb_ok = (not args.require_thumb_contact) or ("th" in touched_fingers)
    return (
        n_contacts >= args.min_contacts
        and len(touched_fingers) >= args.min_contact_fingers
        and thumb_ok
        and total_force >= args.min_force
        and total_force <= args.max_force
    )


def _save_episode_npz(
    episode_path: Path,
    images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    phases: list[int],
    contacts: list[np.ndarray],
    arm_cmd_pose_wxyz: list[np.ndarray],
    arm_obs_pose_wxyz: list[np.ndarray],
    arm_pose_error: list[np.ndarray],
    success: bool,
    instruction: str,
    side: str,
    control_hz: float,
    capture_hz: float,
    object_qpos: np.ndarray,
    criteria: dict,
    metrics: dict,
) -> None:
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": np.asarray(images, dtype=np.uint8),
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "phase": np.asarray(phases, dtype=np.int32),
        "contact": np.asarray(contacts, dtype=np.float32),
        "arm_cmd_pose_wxyz": np.asarray(arm_cmd_pose_wxyz, dtype=np.float32),
        "arm_obs_pose_wxyz": np.asarray(arm_obs_pose_wxyz, dtype=np.float32),
        "arm_pose_error": np.asarray(arm_pose_error, dtype=np.float32),
        "success": np.asarray(success, dtype=np.bool_),
        "language_instruction": np.asarray(instruction),
        "side": np.asarray(side),
        "control_hz": np.asarray(control_hz, dtype=np.float32),
        "capture_hz": np.asarray(capture_hz, dtype=np.float32),
        "object_qpos": np.asarray(object_qpos, dtype=np.float32),
        "criteria_json": np.asarray(json.dumps(criteria), dtype=object),
        "metrics_json": np.asarray(json.dumps(metrics), dtype=object),
        "action_semantics": np.asarray("joint23_absolute_target_object6d_ik", dtype=object),
        "saved_at": np.asarray(datetime.now().isoformat()),
    }
    np.savez_compressed(episode_path, **payload)


def run_collection(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    output_root = Path(args.out_dir)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    mjcf = franka_allegro_mjcf.load(side=args.side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)

    control_target_dt = 1.0 / max(float(args.control_hz), 1e-6)
    control_nstep = max(1, int(round(control_target_dt / model.opt.timestep)))
    effective_control_dt = control_nstep * model.opt.timestep
    effective_control_hz = 1.0 / max(effective_control_dt, 1e-9)

    capture_every = _capture_every(effective_control_hz, args.capture_hz)
    effective_capture_hz = effective_control_hz / capture_every

    arm_cfg = _build_arm_config(model, args.side)
    hand_cfg = _build_hand_config(model, args.side)
    mustard_cfg = _build_mustard_config(model)
    contact_cfg = _build_contact_config(model, args.side, mustard_cfg.body_id)

    open_pose = np.clip(np.zeros(16, dtype=np.float32), hand_cfg.q_min, hand_cfg.q_max)
    preshape_ratio = float(np.clip(args.preshape_grasp_ratio, 0.05, 0.95))

    renderer = mujoco.Renderer(model, width=args.image_width, height=args.image_height)

    viewer_ctx = None
    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )
    capture_cam = None
    if args.viewer:
        viewer_ctx = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
        viewer_ctx.__enter__()
        set_default_franka_allegro_camera(viewer_ctx.cam)
    else:
        capture_cam = mujoco.MjvCamera()
        set_default_franka_allegro_camera(capture_cam)

    force_buf = np.zeros(6, dtype=float)
    base_spawn_pos = np.asarray(args.spawn_pos, dtype=np.float32)
    base_spawn_quat = np.asarray(args.spawn_quat, dtype=np.float32)
    approach_offset = np.asarray(args.arm_approach_offset, dtype=np.float32)
    push_offset = np.asarray(args.arm_push_offset, dtype=np.float32)
    approach_rot_quat = _normalize_quat(np.asarray(args.arm_approach_rot_quat, dtype=np.float64))
    push_rot_quat = _normalize_quat(np.asarray(args.arm_push_rot_quat, dtype=np.float64))

    criteria = {
        "arm_reach_threshold": float(args.arm_reach_threshold),
        "ik_rot_gain": float(args.ik_rot_gain),
        "ik_rot_weight": float(args.ik_rot_weight),
        "ik_max_joint_step": float(args.ik_max_joint_step),
        "min_contacts": int(args.min_contacts),
        "min_contact_fingers": int(args.min_contact_fingers),
        "require_thumb_contact": bool(args.require_thumb_contact),
        "min_force": float(args.min_force),
        "max_force": float(args.max_force),
        "stable_steps": int(args.stable_steps),
        "lift_success_delta": float(args.lift_success_delta),
        "lift_hold_seconds": float(args.lift_hold_seconds),
    }

    saved_success = 0
    attempts = 0

    try:
        print(
            f"[collect] target={args.target_episodes} max_attempts={args.max_attempts} "
            f"control_hz={effective_control_hz:.1f}(req:{args.control_hz:.1f}) "
            f"capture_hz={effective_capture_hz:.1f} "
            f"(capture_every={capture_every}) out_dir={output_root}"
        )
        print(
            "[contact] fingertip_geoms="
            f"{len(contact_cfg.geom_to_finger)} target_geoms={len(contact_cfg.target_geom_ids)}"
        )

        while saved_success < args.target_episodes and attempts < args.max_attempts:
            attempts += 1
            initial_state = model.key("initial_state").id
            mujoco.mj_resetDataKeyframe(model, data, initial_state)
            mujoco.mj_forward(model, data)

            spawn_pos, spawn_quat = _sample_spawn_pose(base_spawn_pos, base_spawn_quat, rng, args)
            _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)

            images: list[np.ndarray] = []
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            phases: list[int] = []
            contacts: list[np.ndarray] = []
            arm_cmd_pose_wxyz: list[np.ndarray] = []
            arm_obs_pose_wxyz: list[np.ndarray] = []
            arm_pose_error: list[np.ndarray] = []

            step_counter = 0
            best_contacts = 0
            best_force = 0.0
            best_fingers: set[str] = set()
            approach_min_err = np.inf
            approach_min_rot_err = np.inf
            grasp_stable_hits = 0
            lift_stable_hits = 0
            grasp_acquired = False
            lift_acquired = False
            first_grasp_step = -1
            first_lift_step = -1

            object_qpos = np.concatenate([spawn_pos, spawn_quat], axis=0).astype(np.float32)
            lift_hold_steps = max(1, int(np.ceil(args.lift_hold_seconds * effective_control_hz)))

            def _record_and_step(hand_cmd: np.ndarray, arm_target: ArmTargetPose | None, phase_id: int) -> dict:
                nonlocal step_counter
                nonlocal best_contacts
                nonlocal best_force
                nonlocal best_fingers

                if args.lock_object_until_close and phase_id in (
                    PHASE_SETTLE,
                    PHASE_APPROACH,
                    PHASE_PRESHAPE,
                ):
                    # Clamp before sensing/IK so target/contact are evaluated on the fixed spawn pose.
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)

                n_pre, f_pre, touched_pre = _detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )

                ee_pos_pre = data.xpos[arm_cfg.palm_body_id].copy()
                ee_quat_pre = _rot_to_quat(data.xmat[arm_cfg.palm_body_id].reshape(3, 3))

                if arm_target is None:
                    arm_cmd = data.qpos[arm_cfg.qpos_ids].copy().astype(np.float32)
                    arm_err = 0.0
                    arm_rot_err_deg = 0.0
                    cmd_pose = np.concatenate([ee_pos_pre, ee_quat_pre], axis=0).astype(np.float32)
                else:
                    arm_cmd, arm_err, arm_rot_err_deg = _step_arm_ik(
                        model=model,
                        data=data,
                        arm_cfg=arm_cfg,
                        target=arm_target,
                        gain=args.ik_gain,
                        rot_gain=args.ik_rot_gain,
                        damping=args.ik_damping,
                        rot_weight=args.ik_rot_weight,
                        max_joint_step=args.ik_max_joint_step,
                    )
                    cmd_pose = np.concatenate([arm_target.pos, arm_target.quat_wxyz], axis=0).astype(
                        np.float32
                    )

                if step_counter % capture_every == 0:
                    thumb_pre = 1.0 if "th" in touched_pre else 0.0
                    contact_stats = np.asarray(
                        [float(n_pre), float(f_pre), float(len(touched_pre)), thumb_pre],
                        dtype=np.float32,
                    )
                    cam = viewer_ctx.cam if viewer_ctx is not None else capture_cam
                    images.append(_capture_frame(renderer, data, cam))
                    states.append(_make_state_vector(data, arm_cfg, hand_cfg, mustard_cfg, contact_stats))
                    actions.append(np.concatenate([arm_cmd, hand_cmd], axis=0).astype(np.float32))
                    phases.append(int(phase_id))
                    contacts.append(contact_stats.copy())
                    arm_cmd_pose_wxyz.append(cmd_pose.copy())
                    arm_obs_pose_wxyz.append(
                        np.concatenate([ee_pos_pre, ee_quat_pre], axis=0).astype(np.float32)
                    )
                    arm_pose_error.append(
                        np.asarray([float(arm_err), float(arm_rot_err_deg)], dtype=np.float32)
                    )

                data.ctrl[arm_cfg.act_ids] = arm_cmd
                data.ctrl[7:23] = hand_cmd
                if args.lock_object_until_close and phase_id in (
                    PHASE_SETTLE,
                    PHASE_APPROACH,
                    PHASE_PRESHAPE,
                ):
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_step(model, data, nstep=control_nstep)

                n_post, f_post, touched_post = _detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                best_contacts = max(best_contacts, int(n_post))
                best_force = max(best_force, float(f_post))
                if len(touched_post) >= len(best_fingers):
                    best_fingers = set(touched_post)

                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    if viewer_delay > 0.0:
                        time.sleep(viewer_delay)

                step_counter += 1
                return {
                    "arm_err": float(arm_err),
                    "arm_rot_err_deg": float(arm_rot_err_deg),
                    "n_contacts": int(n_post),
                    "force": float(f_post),
                    "touched": touched_post,
                }

            for _ in range(args.settle_steps):
                _record_and_step(open_pose, None, PHASE_SETTLE)

            approach_target, push_target = _resolve_arm_targets(
                data=data,
                mustard_cfg=mustard_cfg,
                approach_offset=approach_offset,
                push_offset=push_offset,
                approach_rot_quat=approach_rot_quat,
                push_rot_quat=push_rot_quat,
                offset_frame=args.arm_offset_frame,
                rot_frame=args.arm_rot_frame,
            )
            mustard_pos_after_settle = data.xpos[mustard_cfg.body_id].copy()
            object_z_ref = float(mustard_pos_after_settle[2])
            lift_threshold_z = float(object_z_ref + args.lift_success_delta)
            object_z_max = object_z_ref

            for _ in range(args.approach_steps):
                if args.lock_object_until_close:
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)
                approach_target, _ = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=push_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=push_rot_quat,
                    offset_frame=args.arm_offset_frame,
                    rot_frame=args.arm_rot_frame,
                )
                info = _record_and_step(open_pose, approach_target, PHASE_APPROACH)
                approach_min_err = min(approach_min_err, info["arm_err"])
                approach_min_rot_err = min(approach_min_rot_err, info["arm_rot_err_deg"])

            for i in range(args.preshape_steps):
                alpha = (i + 1) / max(args.preshape_steps, 1)
                grasp_value = preshape_ratio * alpha
                hand_cmd = _interpolate_hand_pose(
                    grasp_value=grasp_value,
                    hand_cfg=hand_cfg,
                    hand_trajectory=args.hand_trajectory,
                    preshape_pivot=preshape_ratio,
                )
                if args.lock_object_until_close:
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)
                approach_target, push_target = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=push_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=push_rot_quat,
                    offset_frame=args.arm_offset_frame,
                    rot_frame=args.arm_rot_frame,
                )
                push_target_lifted = ArmTargetPose(
                    pos=(
                        push_target.pos + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                    ).astype(np.float64),
                    quat_wxyz=push_target.quat_wxyz.copy(),
                )
                arm_target = ArmTargetPose(
                    pos=((1.0 - alpha) * approach_target.pos + alpha * push_target_lifted.pos).astype(
                        np.float64
                    ),
                    quat_wxyz=_quat_lerp_normalize(
                        approach_target.quat_wxyz, push_target_lifted.quat_wxyz, alpha
                    ),
                )
                _record_and_step(hand_cmd, arm_target, PHASE_PRESHAPE)

            for i in range(args.close_steps):
                alpha = (i + 1) / max(args.close_steps, 1)
                grasp_value = preshape_ratio + (1.0 - preshape_ratio) * alpha
                hand_cmd = _interpolate_hand_pose(
                    grasp_value=grasp_value,
                    hand_cfg=hand_cfg,
                    hand_trajectory=args.hand_trajectory,
                    preshape_pivot=preshape_ratio,
                )
                _, push_target = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=push_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=push_rot_quat,
                    offset_frame=args.arm_offset_frame,
                    rot_frame=args.arm_rot_frame,
                )
                push_target_lifted = ArmTargetPose(
                    pos=(
                        push_target.pos + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                    ).astype(np.float64),
                    quat_wxyz=push_target.quat_wxyz.copy(),
                )
                info = _record_and_step(hand_cmd, push_target_lifted, PHASE_CLOSE)
                meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)
                grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                if grasp_stable_hits >= args.stable_steps and not grasp_acquired:
                    grasp_acquired = True
                    first_grasp_step = step_counter

            close_pose = _interpolate_hand_pose(
                grasp_value=1.0,
                hand_cfg=hand_cfg,
                hand_trajectory=args.hand_trajectory,
                preshape_pivot=preshape_ratio,
            )
            _, push_target = _resolve_arm_targets(
                data=data,
                mustard_cfg=mustard_cfg,
                approach_offset=approach_offset,
                push_offset=push_offset,
                approach_rot_quat=approach_rot_quat,
                push_rot_quat=push_rot_quat,
                offset_frame=args.arm_offset_frame,
                rot_frame=args.arm_rot_frame,
            )
            for i in range(args.lift_steps):
                alpha = (i + 1) / max(args.lift_steps, 1)
                push_pos_lifted = (
                    push_target.pos + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                )
                lift_target = ArmTargetPose(
                    pos=push_pos_lifted + np.array([0.0, 0.0, args.lift_height * alpha], dtype=float),
                    quat_wxyz=push_target.quat_wxyz.copy(),
                )
                info = _record_and_step(close_pose, lift_target, PHASE_LIFT)
                object_z = float(data.xpos[mustard_cfg.body_id][2])
                object_z_max = max(object_z_max, object_z)
                lifted = (object_z - object_z_ref) >= args.lift_success_delta
                meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)

                # If stable grasp emerges during lift, still count it as grasp-acquired.
                grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                if grasp_stable_hits >= args.stable_steps and not grasp_acquired:
                    grasp_acquired = True
                    first_grasp_step = step_counter

                lift_ok = lifted and meets
                lift_stable_hits = lift_stable_hits + 1 if lift_ok else 0

            hold_target = ArmTargetPose(
                pos=push_target.pos
                + np.array([0.0, 0.0, args.close_z_clearance + args.lift_height], dtype=float),
                quat_wxyz=push_target.quat_wxyz.copy(),
            )
            # Success requires no drop during the full hold window after lift.
            no_drop_during_hold = True
            hold_contact_ok = True
            for _ in range(lift_hold_steps):
                info = _record_and_step(close_pose, hold_target, PHASE_LIFT_HOLD)
                object_z = float(data.xpos[mustard_cfg.body_id][2])
                object_z_max = max(object_z_max, object_z)
                lifted = object_z >= lift_threshold_z
                meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)
                if not lifted:
                    no_drop_during_hold = False
                if not meets:
                    hold_contact_ok = False

            if no_drop_during_hold and not lift_acquired:
                lift_acquired = True
                first_lift_step = step_counter

            reached = bool(approach_min_err <= args.arm_reach_threshold)
            success = bool(reached and grasp_acquired and lift_acquired)

            metrics = {
                "approach_min_err": float(approach_min_err),
                "approach_min_rot_err_deg": float(approach_min_rot_err),
                "reached": reached,
                "grasp_acquired": grasp_acquired,
                "lift_acquired": lift_acquired,
                "first_grasp_step": int(first_grasp_step),
                "first_lift_step": int(first_lift_step),
                "best_contacts": int(best_contacts),
                "best_force": float(best_force),
                "best_fingers": sorted(best_fingers),
                "object_z_ref": float(object_z_ref),
                "object_z_max": float(object_z_max),
                "object_dz_max": float(object_z_max - object_z_ref),
                "lift_threshold_z": float(lift_threshold_z),
                "no_drop_during_hold": bool(no_drop_during_hold),
                "hold_contact_ok": bool(hold_contact_ok),
                "lift_hold_steps": int(lift_hold_steps),
                "steps": int(step_counter),
            }

            if success:
                ep_name = f"episode_{saved_success:05d}.npz"
                ep_path = raw_dir / ep_name
                _save_episode_npz(
                    episode_path=ep_path,
                    images=images,
                    states=states,
                    actions=actions,
                    phases=phases,
                    contacts=contacts,
                    arm_cmd_pose_wxyz=arm_cmd_pose_wxyz,
                    arm_obs_pose_wxyz=arm_obs_pose_wxyz,
                    arm_pose_error=arm_pose_error,
                    success=success,
                    instruction=args.instruction,
                    side=args.side,
                    control_hz=float(effective_control_hz),
                    capture_hz=float(effective_capture_hz),
                    object_qpos=object_qpos,
                    criteria=criteria,
                    metrics=metrics,
                )
                saved_success += 1
                status = "✅ SAVE"
            else:
                status = "❌ SKIP"

            print(
                f"[attempt {attempts:04d}] {status} success={int(success)} "
                f"saved={saved_success}/{args.target_episodes} "
                f"reach_err={approach_min_err:.4f} rot_err={approach_min_rot_err:.2f}deg "
                f"grasp={int(grasp_acquired)} lift={int(lift_acquired)} "
                f"no_drop={int(metrics['no_drop_during_hold'])} hold_contact={int(metrics['hold_contact_ok'])} "
                f"contacts={best_contacts} fingers={','.join(sorted(best_fingers)) or '-'} "
                f"dz_max={metrics['object_dz_max']:.4f}"
            )

        if viewer_ctx is not None and args.keep_open:
            print("Viewer is kept open. Close the MuJoCo window to finish.")
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)

    finally:
        renderer.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}")

    finished = saved_success >= args.target_episodes
    summary = {
        "finished": finished,
        "saved_success_episodes": int(saved_success),
        "target_episodes": int(args.target_episodes),
        "attempts": int(attempts),
        "max_attempts": int(args.max_attempts),
        "out_dir": str(output_root),
        "raw_dir": str(raw_dir),
        "instruction": args.instruction,
        "side": args.side,
        "control_hz": float(effective_control_hz),
        "control_hz_requested": float(args.control_hz),
        "capture_hz": float(effective_capture_hz),
        "capture_every": int(capture_every),
        "action_semantics": "joint23_absolute_target_object6d_ik",
        "image_shape": [args.image_height, args.image_width, 3],
        "state_dim": 7 + 16 + 7 + 16 + 3 + 3 + 4 + 4,
        "action_dim": 23,
        "criteria": criteria,
        "policy": {
            "settle_steps": int(args.settle_steps),
            "approach_steps": int(args.approach_steps),
            "preshape_steps": int(args.preshape_steps),
            "close_steps": int(args.close_steps),
            "lift_steps": int(args.lift_steps),
            "lift_hold_seconds": float(args.lift_hold_seconds),
            "spawn_pos": [float(x) for x in args.spawn_pos],
            "spawn_quat_wxyz": [float(x) for x in args.spawn_quat],
            "spawn_jitter_xy": float(args.spawn_jitter_xy),
            "spawn_yaw_jitter_deg": float(args.spawn_yaw_jitter_deg),
            "lock_object_until_close": bool(args.lock_object_until_close),
            "arm_approach_offset": [float(x) for x in args.arm_approach_offset],
            "arm_push_offset": [float(x) for x in args.arm_push_offset],
            "arm_offset_frame": args.arm_offset_frame,
            "arm_rot_frame": args.arm_rot_frame,
            "arm_approach_rot_quat_wxyz": [float(x) for x in args.arm_approach_rot_quat],
            "arm_push_rot_quat_wxyz": [float(x) for x in args.arm_push_rot_quat],
            "hand_trajectory": args.hand_trajectory,
            "preshape_grasp_ratio": float(preshape_ratio),
            "ik_gain": float(args.ik_gain),
            "ik_rot_gain": float(args.ik_rot_gain),
            "ik_damping": float(args.ik_damping),
            "ik_rot_weight": float(args.ik_rot_weight),
            "ik_max_joint_step": float(args.ik_max_joint_step),
            "close_z_clearance": float(args.close_z_clearance),
            "lift_height": float(args.lift_height),
            "control_nstep": int(control_nstep),
            "seed": int(args.seed),
        },
    }

    summary_path = output_root / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_collection(args)
    status = "DONE" if summary["finished"] else "STOPPED"
    print(
        f"\n[{status}] saved={summary['saved_success_episodes']}/{summary['target_episodes']} "
        f"attempts={summary['attempts']}"
    )


if __name__ == "__main__":
    main()
