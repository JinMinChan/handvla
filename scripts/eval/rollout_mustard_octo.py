#!/usr/bin/env python3
"""Run closed-loop rollout in mustard grasp scene using a finetuned Octo model."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import jax
import mujoco
from mujoco import viewer
import numpy as np
from PIL import Image
import tensorflow as tf

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
from env.viewer_utils import set_default_hand_camera

_DEFAULT_OCTO_SRC = "/home/minchan/Downloads/dual_arm_VLA/octo"
if importlib.util.find_spec("octo") is None:
    octo_src = Path(
        (Path.cwd() / ".octo_src_path").read_text(encoding="utf-8").strip()
        if (Path.cwd() / ".octo_src_path").exists()
        else _DEFAULT_OCTO_SRC
    )
    if octo_src.exists():
        sys.path.insert(0, str(octo_src))

from octo.model.octo_model import OctoModel

FINGER_KEYS = ("ff", "mf", "rf", "th")
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class JointIndexConfig:
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    tcp_site_ids: np.ndarray
    palm_body_id: int


@dataclass(frozen=True)
class FingerIKConfig:
    name: str
    ctrl_ids: np.ndarray
    dof_ids: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    tcp_site_id: int


@dataclass(frozen=True)
class FKTcpVizConfig:
    mocap_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rollout finetuned Octo policy in fixed-air mustard grasp environment."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mustard_octo_overfit_full_joint/run_260223_171418/final_model",
        help="Path to finetuned Octo checkpoint directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mustard_grasp_oxe",
        help="Key inside model.dataset_statistics for action/proprio normalization.",
    )
    parser.add_argument("--task", type=str, default="grasp the mustard bottle")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=100,
        help="Policy decisions per episode. Total sim steps = max_policy_steps * control_repeat.",
    )
    parser.add_argument(
        "--control-repeat",
        type=int,
        default=5,
        help="How many MuJoCo steps to hold one predicted action.",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=0.35,
        help="EMA blend for predicted action target (1.0 means no smoothing).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--policy-image-size", type=int, default=256)
    parser.add_argument(
        "--action-interface",
        choices=("auto", "tcp12", "joint16"),
        default="tcp12",
        help="Action interpretation for model outputs. Default is tcp12 (4 fingertip TCP xyz).",
    )
    parser.add_argument(
        "--tcp12-action-type",
        choices=("delta", "absolute"),
        default="delta",
        help="How to interpret 12D tcp action. delta: tcp_target=tcp_now+pred, absolute: tcp_target=pred.",
    )
    parser.add_argument(
        "--tcp12-frame",
        choices=("palm_local", "world"),
        default="palm_local",
        help="Frame for tcp12 delta actions. Use world for legacy tcp12 datasets.",
    )
    parser.add_argument("--ik-max-iters", type=int, default=30)
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--ik-step-scale", type=float, default=0.75)
    parser.add_argument("--ik-max-dq", type=float, default=0.08)
    parser.add_argument("--ik-tol-mm", type=float, default=2.5)

    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=2)
    parser.add_argument(
        "--require-thumb-contact",
        action="store_true",
        default=True,
        help="Require thumb contact for success (default: on).",
    )
    parser.add_argument(
        "--no-require-thumb-contact",
        dest="require_thumb_contact",
        action="store_false",
        help="Disable thumb-contact requirement.",
    )
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)

    parser.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Open MuJoCo viewer.",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Run headless.",
    )
    parser.set_defaults(viewer=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")

    parser.add_argument("--record", action="store_true", help="Record MP4.")
    parser.add_argument(
        "--record-path",
        type=str,
        default="",
        help="Output path (default: codex/logs/rollout_mustard_octo_<timestamp>.mp4).",
    )
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument(
        "--show-fk-tcp-markers",
        action="store_true",
        help="Render blue mocap markers at current FK TCP positions for verification.",
    )
    return parser.parse_args()


def _default_record_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"rollout_mustard_octo_{ts}.mp4"


def _build_joint_indices(model: mujoco.MjModel, side: str) -> JointIndexConfig:
    prefix = f"allegro_{side}"
    qpos_ids: list[int] = []
    dof_ids: list[int] = []
    tcp_site_ids: list[int] = []

    for finger in FINGER_KEYS:
        for j in range(4):
            joint_name = f"{prefix}/{finger}j{j}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            qpos_ids.append(int(model.jnt_qposadr[joint_id]))
            dof_ids.append(int(model.jnt_dofadr[joint_id]))

        tcp_name = f"{prefix}/{finger}_tcp"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
        if site_id < 0:
            raise RuntimeError(f"TCP site not found: {tcp_name}")
        tcp_site_ids.append(int(site_id))

    palm_body_name = f"{prefix}/palm"
    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name)
    if palm_body_id < 0:
        raise RuntimeError(f"Palm body not found: {palm_body_name}")

    return JointIndexConfig(
        qpos_ids=np.asarray(qpos_ids, dtype=int),
        dof_ids=np.asarray(dof_ids, dtype=int),
        tcp_site_ids=np.asarray(tcp_site_ids, dtype=int),
        palm_body_id=int(palm_body_id),
    )


def _build_finger_ik_configs(model: mujoco.MjModel, side: str) -> dict[str, FingerIKConfig]:
    prefix = f"allegro_{side}"
    cfgs: dict[str, FingerIKConfig] = {}
    for finger_idx, finger in enumerate(FINGER_KEYS):
        dof_ids = []
        q_min = []
        q_max = []
        for j in range(4):
            joint_name = f"{prefix}/{finger}j{j}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            dof_ids.append(int(model.jnt_dofadr[joint_id]))
            q_min.append(float(model.jnt_range[joint_id][0]))
            q_max.append(float(model.jnt_range[joint_id][1]))

        tcp_name = f"{prefix}/{finger}_tcp"
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
        if tcp_site_id < 0:
            raise RuntimeError(f"TCP site not found: {tcp_name}")

        cfgs[finger] = FingerIKConfig(
            name=finger,
            ctrl_ids=np.arange(4 * finger_idx, 4 * finger_idx + 4, dtype=int),
            dof_ids=np.asarray(dof_ids, dtype=int),
            q_min=np.asarray(q_min, dtype=float),
            q_max=np.asarray(q_max, dtype=float),
            tcp_site_id=int(tcp_site_id),
        )
    return cfgs


def _build_fk_tcp_viz_config(model: mujoco.MjModel, side: str) -> FKTcpVizConfig:
    prefix = f"allegro_{side}"
    mocap_ids: list[int] = []
    for finger in FINGER_KEYS:
        body_name = f"{prefix}/{finger}_fk_tcp_vis"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(
                f"FK TCP marker body not found: {body_name}. "
                "Rebuild model with add_fk_tcp_markers=True."
            )
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"Body is not mocap-enabled: {body_name}")
        mocap_ids.append(mocap_id)
    return FKTcpVizConfig(mocap_ids=np.asarray(mocap_ids, dtype=int))


def _update_fk_tcp_markers(
    data: mujoco.MjData,
    hand_idx: JointIndexConfig,
    fk_viz_cfg: FKTcpVizConfig,
) -> None:
    tcp_world = data.site_xpos[hand_idx.tcp_site_ids]
    for i, mocap_id in enumerate(fk_viz_cfg.mocap_ids):
        data.mocap_pos[mocap_id] = tcp_world[i]
        data.mocap_quat[mocap_id] = IDENTITY_QUAT


def _make_state_vector(
    data: mujoco.MjData,
    hand_idx: JointIndexConfig,
    mustard_cfg,
    contact_stats: np.ndarray,
) -> np.ndarray:
    qpos = data.qpos[hand_idx.qpos_ids]
    qvel = data.qvel[hand_idx.dof_ids]
    tcp_xyz = data.site_xpos[hand_idx.tcp_site_ids].reshape(-1)
    mustard_pos = data.qpos[mustard_cfg.qpos_adr : mustard_cfg.qpos_adr + 3]
    mustard_quat = data.qpos[mustard_cfg.qpos_adr + 3 : mustard_cfg.qpos_adr + 7]
    return np.concatenate(
        [qpos, qvel, tcp_xyz, mustard_pos, mustard_quat, contact_stats], axis=0
    ).astype(np.float32)


def _find_stats(model: OctoModel, dataset_name: str) -> tuple[dict | None, dict | None]:
    stats = model.dataset_statistics
    if not isinstance(stats, dict):
        return None, None
    if "action" in stats:
        return stats.get("action"), stats.get("proprio")
    if dataset_name in stats:
        ds = stats[dataset_name]
        if isinstance(ds, dict):
            return ds.get("action"), ds.get("proprio")
    first_key = next(iter(stats.keys()), None)
    if first_key is None:
        return None, None
    ds = stats[first_key]
    if isinstance(ds, dict):
        return ds.get("action"), ds.get("proprio")
    return None, None


def _solve_tcp12_to_joint_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_qpos_ids: np.ndarray,
    q_seed: np.ndarray,
    finger_cfgs: dict[str, FingerIKConfig],
    tcp_targets: np.ndarray,
    ik_max_iters: int,
    ik_damping: float,
    ik_step_scale: float,
    ik_max_dq: float,
    ik_tol_m: float,
) -> np.ndarray:
    # Damped least-squares IK per finger in fixed order [ff, mf, rf, th].
    q_cmd = q_seed.astype(np.float64).copy()
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    for finger_idx, finger in enumerate(FINGER_KEYS):
        cfg = finger_cfgs[finger]
        target = tcp_targets[finger_idx].astype(np.float64)

        for _ in range(ik_max_iters):
            data.qpos[hand_qpos_ids] = q_cmd
            mujoco.mj_forward(model, data)
            err = target - data.site_xpos[cfg.tcp_site_id]
            if float(np.linalg.norm(err)) <= ik_tol_m:
                break

            mujoco.mj_jacSite(model, data, jacp, jacr, cfg.tcp_site_id)
            J = jacp[:, cfg.dof_ids]  # (3,4)
            lhs = J @ J.T + ik_damping * np.eye(3)
            try:
                dq = J.T @ np.linalg.solve(lhs, err)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.lstsq(lhs, err, rcond=None)[0]
            dq = np.clip(dq, -ik_max_dq, ik_max_dq)

            q_next = q_cmd[cfg.ctrl_ids] + ik_step_scale * dq
            q_cmd[cfg.ctrl_ids] = np.clip(q_next, cfg.q_min, cfg.q_max)

    return q_cmd.astype(np.float32)


def _tcp_delta_local_to_world(
    data: mujoco.MjData,
    hand_idx: JointIndexConfig,
    tcp_delta_local: np.ndarray,
) -> np.ndarray:
    # data.xmat is body->world rotation. Row-vector conversion local->world is v_world = v_local @ R^T.
    rot_body_to_world = data.xmat[hand_idx.palm_body_id].reshape(3, 3).astype(np.float32)
    return (tcp_delta_local @ rot_body_to_world.T).astype(np.float32)


def run_rollout(args: argparse.Namespace) -> dict:
    tf.config.set_visible_devices([], "GPU")
    rng = jax.random.PRNGKey(args.seed)

    mjcf = allegro_hand_mjcf.load(
        side=args.side,
        add_mustard=True,
        add_fk_tcp_markers=args.show_fk_tcp_markers,
    )
    model = mjcf.compile()
    data = mujoco.MjData(model)

    hand_cfg = build_hand_config(model, args.side)
    hand_idx = _build_joint_indices(model, args.side)
    finger_ik_cfgs = _build_finger_ik_configs(model, args.side)
    fk_viz_cfg = _build_fk_tcp_viz_config(model, args.side) if args.show_fk_tcp_markers else None
    mustard_cfg = build_mustard_config(model)
    contact_cfg = build_contact_config(model, args.side, mustard_cfg.body_id)
    force_buf = np.zeros(6, dtype=float)

    policy_model = OctoModel.load_pretrained(args.model_path)
    if policy_model.text_processor is None:
        raise RuntimeError("Loaded model has no text_processor; language-conditioned rollout cannot run.")
    task = {
        "language_instruction": policy_model.text_processor.encode([args.task]),
    }
    action_stats, proprio_stats = _find_stats(policy_model, args.dataset_name)

    viewer_ctx = None
    if args.viewer:
        viewer_ctx = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
        viewer_ctx.__enter__()
        set_default_hand_camera(viewer_ctx.cam)

    renderer_obs = mujoco.Renderer(model, width=args.image_width, height=args.image_height)
    record_renderer = None
    record_fallback_cam = None
    writer = None
    if args.record:
        record_path = Path(args.record_path) if args.record_path else _default_record_path()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_renderer = mujoco.Renderer(
            model, width=args.record_width, height=args.record_height
        )
        # Keep recording view consistent with our default frontal hand shot even in headless mode.
        record_fallback_cam = mujoco.MjvCamera()
        set_default_hand_camera(record_fallback_cam)
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

    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )

    episodes_data = []
    resolved_action_interface: str | None = None
    try:
        for ep in range(args.episodes):
            reset_to_initial(model, data)
            spawn_pos, spawn_quat = compute_mustard_spawn_pose(model, data, args.side)
            set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)
            if fk_viz_cfg is not None:
                _update_fk_tcp_markers(data, hand_idx, fk_viz_cfg)
                mujoco.mj_forward(model, data)

            q_cmd = data.qpos[hand_idx.qpos_ids].astype(np.float32).copy()
            q_cmd = np.clip(q_cmd, hand_cfg.q_min, hand_cfg.q_max)
            stable_hits = 0
            first_success_policy_step = -1
            best_contacts = 0
            best_fingers = 0
            best_force = 0.0
            best_finger_set: set[str] = set()
            nan_action = False

            sim_step_counter = 0
            for p_step in range(args.max_policy_steps):
                _, n_contacts, total_force, touched = detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                thumb_contact = 1.0 if "th" in touched else 0.0
                contact_stats = np.asarray(
                    [float(n_contacts), float(total_force), float(len(touched)), thumb_contact],
                    dtype=np.float32,
                )
                state = _make_state_vector(data, hand_idx, mustard_cfg, contact_stats)
                if proprio_stats is not None:
                    mean = np.asarray(proprio_stats["mean"], dtype=np.float32)
                    std = np.asarray(proprio_stats["std"], dtype=np.float32)
                    state = (state - mean) / np.maximum(std, 1e-6)

                renderer_obs.update_scene(data)
                obs_image = renderer_obs.render()
                obs_image = np.asarray(
                    Image.fromarray(obs_image).resize(
                        (args.policy_image_size, args.policy_image_size), Image.BILINEAR
                    ),
                    dtype=np.uint8,
                )

                obs = {
                    "image_primary": obs_image[None, None, ...],
                    "proprio": state[None, None, :],
                    "timestep_pad_mask": np.array([[True]], dtype=np.bool_),
                }

                rng, key = jax.random.split(rng)
                if action_stats is not None:
                    pred = policy_model.sample_actions(
                        obs,
                        task,
                        unnormalization_statistics=action_stats,
                        rng=key,
                    )[0, 0]
                else:
                    pred = policy_model.sample_actions(obs, task, rng=key)[0, 0]
                pred = np.asarray(pred, dtype=np.float32)

                if not np.all(np.isfinite(pred)):
                    nan_action = True
                    print(f"[episode {ep:03d}] non-finite action detected, stopping episode.")
                    break

                pred_dim = int(pred.shape[-1])
                if resolved_action_interface is None:
                    if args.action_interface == "auto":
                        if pred_dim == 12:
                            resolved_action_interface = "tcp12"
                        elif pred_dim == 16:
                            resolved_action_interface = "joint16"
                        else:
                            raise RuntimeError(
                                f"Unsupported model action dim={pred_dim}. Expected 12 or 16."
                            )
                    else:
                        resolved_action_interface = args.action_interface
                        expected_dim = 12 if resolved_action_interface == "tcp12" else 16
                        if pred_dim != expected_dim:
                            raise RuntimeError(
                                f"Model action dim={pred_dim}, but --action-interface="
                                f"{resolved_action_interface} expects {expected_dim}."
                            )
                    print(
                        f"[action] interface={resolved_action_interface} action_dim={pred_dim}",
                        flush=True,
                    )

                alpha = float(np.clip(args.action_smoothing, 0.0, 1.0))
                if resolved_action_interface == "joint16":
                    q_target = np.clip(pred, hand_cfg.q_min, hand_cfg.q_max)
                else:
                    pred_tcp = pred.reshape(4, 3)
                    if args.tcp12_action_type == "delta":
                        tcp_now = data.site_xpos[hand_idx.tcp_site_ids].reshape(4, 3).astype(np.float32)
                        if args.tcp12_frame == "palm_local":
                            pred_tcp_world = _tcp_delta_local_to_world(
                                data=data,
                                hand_idx=hand_idx,
                                tcp_delta_local=pred_tcp,
                            )
                        else:
                            pred_tcp_world = pred_tcp
                        tcp_targets = tcp_now + pred_tcp_world
                    else:
                        tcp_targets = pred_tcp
                    q_target = _solve_tcp12_to_joint_targets(
                        model=model,
                        data=data,
                        hand_qpos_ids=hand_idx.qpos_ids,
                        q_seed=q_cmd,
                        finger_cfgs=finger_ik_cfgs,
                        tcp_targets=tcp_targets,
                        ik_max_iters=args.ik_max_iters,
                        ik_damping=args.ik_damping,
                        ik_step_scale=args.ik_step_scale,
                        ik_max_dq=args.ik_max_dq,
                        ik_tol_m=args.ik_tol_mm / 1000.0,
                    )
                q_cmd = (1.0 - alpha) * q_cmd + alpha * q_target

                for _ in range(args.control_repeat):
                    data.ctrl[:16] = q_cmd
                    set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_step(model, data)
                    set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)
                    if fk_viz_cfg is not None:
                        _update_fk_tcp_markers(data, hand_idx, fk_viz_cfg)
                        mujoco.mj_forward(model, data)

                    _, n_c, f_c, touched_c = detect_contact_with_target(
                        model, data, contact_cfg, force_buf
                    )
                    best_contacts = max(best_contacts, int(n_c))
                    best_fingers = max(best_fingers, len(touched_c))
                    best_force = max(best_force, float(f_c))
                    if len(touched_c) >= len(best_finger_set):
                        best_finger_set = set(touched_c)

                    thumb_ok = (not args.require_thumb_contact) or ("th" in touched_c)
                    meets = (
                        n_c >= args.min_contacts
                        and len(touched_c) >= args.min_contact_fingers
                        and thumb_ok
                        and f_c >= args.min_force
                        and f_c <= args.max_force
                    )
                    stable_hits = stable_hits + 1 if meets else 0
                    if stable_hits >= args.stable_steps and first_success_policy_step < 0:
                        first_success_policy_step = p_step

                    if viewer_ctx is not None:
                        viewer_ctx.sync()
                    if writer is not None and record_renderer is not None:
                        cam = viewer_ctx.cam if viewer_ctx is not None else record_fallback_cam
                        if cam is not None:
                            record_renderer.update_scene(data, camera=cam)
                        else:
                            record_renderer.update_scene(data)
                        writer.append_data(record_renderer.render())
                    if viewer_delay > 0.0:
                        time.sleep(viewer_delay)

                    sim_step_counter += 1

                if first_success_policy_step >= 0:
                    break

            success = first_success_policy_step >= 0 and not nan_action
            ep_result = {
                "episode": ep,
                "success": success,
                "first_success_policy_step": first_success_policy_step,
                "best_contacts": best_contacts,
                "best_contact_fingers": best_fingers,
                "best_fingers": sorted(best_finger_set),
                "best_force": best_force,
                "sim_steps": sim_step_counter,
                "nan_action": nan_action,
            }
            episodes_data.append(ep_result)
            print(
                f"[episode {ep:03d}] success={int(success)} "
                f"contacts={best_contacts} fingers={best_fingers} "
                f"({','.join(sorted(best_finger_set))}) force={best_force:.3f}N",
                flush=True,
            )

        if viewer_ctx is not None and args.keep_open:
            print("Viewer kept open. Close window to finish.", flush=True)
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)

    finally:
        if writer is not None:
            writer.close()
            print("[record] finished", flush=True)
        if record_renderer is not None:
            record_renderer.close()
        renderer_obs.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}")

    success_rate = (
        float(np.mean([float(ep["success"]) for ep in episodes_data])) if episodes_data else 0.0
    )
    summary = {
        "model_path": args.model_path,
        "task": args.task,
        "episodes": args.episodes,
        "max_policy_steps": args.max_policy_steps,
        "control_repeat": args.control_repeat,
        "action_interface": resolved_action_interface or args.action_interface,
        "tcp12_action_type": args.tcp12_action_type,
        "tcp12_frame": args.tcp12_frame,
        "success_rate": success_rate,
        "episodes_data": episodes_data,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = run_rollout(args)
    print(
        "\n=== Mustard Octo Rollout Summary ===\n"
        f"episodes={summary['episodes']} success_rate={100.0 * summary['success_rate']:.1f}%\n"
        f"model={summary['model_path']}"
    )


if __name__ == "__main__":
    main()
