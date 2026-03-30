#!/usr/bin/env python3
"""Collect wrap-only DAgger-lite corrective episodes from official Octo rollout states."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
import importlib.util
import json
import time

import gym
import imageio.v2 as imageio
import jax
import mujoco
import numpy as np
import tensorflow as tf

from scripts.data.collect_mustard_intent_benchmark import (
    TASK_WRAP_AND_LIFT,
    _contact_meets,
    _save_episode_npz,
    _task_spec,
)
from scripts.data.collect_pickandlift_rlds import (
    ArmTargetPose,
    _capture_every,
    _capture_frame,
    _interpolate_hand_pose,
    _quat_lerp_normalize,
    _resolve_arm_targets,
    _rot_to_quat,
    _step_arm_ik,
)
from scripts.eval.mustard_intent_gym_env import MustardIntentEnvConfig, MustardIntentGymEnv

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
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, ResizeImageWrapper, RHCWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect wrap corrective episodes by rolling out an official Octo policy until "
            "a reached-but-no-contact state, then switching to scripted expert recovery."
        )
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--basis-path", type=str, required=True)
    parser.add_argument("--goal-episode", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/mustard_intent_wrap_daggerlite",
        help="Output root; raw episodes are written to <out-dir>/raw.",
    )
    parser.add_argument("--dataset-name", type=str, default="mustard_grasp_oxe")
    parser.add_argument("--target-episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--spawn-jitter-xy", type=float, default=0.02)
    parser.add_argument("--spawn-yaw-jitter-deg", type=float, default=10.0)

    parser.add_argument("--policy-max-steps", type=int, default=30)
    parser.add_argument("--policy-repeat", type=int, default=20)
    parser.add_argument("--handoff-min-step", type=int, default=18)
    parser.add_argument("--handoff-max-step", type=int, default=30)
    parser.add_argument("--handoff-reach-threshold", type=float, default=0.04)
    parser.add_argument("--handoff-max-contacts", type=int, default=0)

    parser.add_argument("--recenter-steps", type=int, default=70)
    parser.add_argument("--close-steps", type=int, default=280)
    parser.add_argument("--close-hold-steps", type=int, default=40)
    parser.add_argument("--lift-steps", type=int, default=260)
    parser.add_argument("--lift-hold-seconds", type=float, default=1.5)
    parser.add_argument("--stable-steps", type=int, default=3)
    parser.add_argument("--capture-hz", type=float, default=5.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--close-z-clearance", type=float, default=0.015)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--lift-success-delta", type=float, default=0.08)

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-fps", type=int, default=5)
    parser.add_argument("--record-dir", type=str, default="codex/logs/videos")
    return parser.parse_args()


class AddObservationPadMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        spaces = dict(self.observation_space.spaces)
        spaces["pad_mask_dict"] = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(low=0, high=1, shape=(horizon,), dtype=bool),
                "proprio": gym.spaces.Box(low=0, high=1, shape=(horizon,), dtype=bool),
                "timestep": gym.spaces.Box(low=0, high=1, shape=(horizon,), dtype=bool),
            }
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        timestep_pad_mask = np.asarray(observation["timestep_pad_mask"], dtype=bool)
        observation["pad_mask_dict"] = {
            "image_primary": timestep_pad_mask.copy(),
            "proprio": timestep_pad_mask.copy(),
            "timestep": timestep_pad_mask.copy(),
        }
        return observation


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


def _resize_goal_like_octo(goal: np.ndarray, size: int) -> np.ndarray:
    goal = tf.convert_to_tensor(goal, dtype=tf.uint8)
    image = tf.image.resize(goal, size=(size, size), method="lanczos3", antialias=True)
    new_height = tf.clip_by_value(tf.sqrt(0.9 / 1.0), 0, 1)
    new_width = tf.clip_by_value(tf.sqrt(0.9 * 1.0), 0, 1)
    height_offset = (1 - new_height) / 2
    width_offset = (1 - new_width) / 2
    bbox = tf.stack([height_offset, width_offset, height_offset + new_height, width_offset + new_width])
    image = tf.image.crop_and_resize(image[None], bbox[None], [0], (size, size))[0]
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()


def _build_task(model: OctoModel, instruction: str, goal_episode: str, image_size: int):
    if model.text_processor is None:
        raise RuntimeError("Loaded model has no text processor.")
    goal_image = MustardIntentGymEnv.goal_image_from_episode(goal_episode, which="last")
    goal_image = _resize_goal_like_octo(goal_image, image_size)
    return model.create_tasks(
        goals={"image_primary": goal_image[None, ...]},
        texts=[instruction],
    )


def _policy_rollout_to_handoff(
    model: OctoModel,
    env,
    base_env: MustardIntentGymEnv,
    task,
    action_stats: dict,
    args: argparse.Namespace,
    rng_key: jax.Array,
) -> tuple[bool, dict]:
    obs, _ = env.reset(seed=int(base_env.cfg.seed))
    handoff = None
    for step in range(1, int(args.policy_max_steps) + 1):
        obs_batch = jax.tree_map(lambda x: x[None], obs)
        rng_key, sample_rng = jax.random.split(rng_key)
        action = model.sample_actions(
            obs_batch,
            task,
            unnormalization_statistics=action_stats,
            rng=sample_rng,
            argmax=False,
            temperature=1.0,
        )[0]
        obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        summary = base_env.get_episode_summary()
        if terminated:
            return False, {"reason": "policy_success", "summary": summary, "step": step}
        if summary["best_contacts"] > int(args.handoff_max_contacts):
            return False, {"reason": "policy_contact", "summary": summary, "step": step}
        reached_now = (
            step >= int(args.handoff_min_step)
            and step <= int(args.handoff_max_step)
            and float(summary["approach_min_err"]) <= float(args.handoff_reach_threshold)
            and int(summary["best_contacts"]) <= int(args.handoff_max_contacts)
        )
        if reached_now:
            handoff = {
                "policy_step": int(step),
                "policy_summary": summary,
                "qpos": base_env.data.qpos.copy().astype(np.float32),
                "qvel": base_env.data.qvel.copy().astype(np.float32),
                "q_arm_cmd": base_env.q_arm_cmd.copy().astype(np.float32),
                "q_hand_cmd": base_env.q_hand_cmd.copy().astype(np.float32),
            }
            break
        if truncated:
            break
    if handoff is None:
        final_summary = base_env.get_episode_summary()
        if (
            bool(final_summary["reached"])
            and int(final_summary["best_contacts"]) <= int(args.handoff_max_contacts)
        ):
            handoff = {
                "policy_step": int(base_env.step_count),
                "policy_summary": final_summary,
                "qpos": base_env.data.qpos.copy().astype(np.float32),
                "qvel": base_env.data.qvel.copy().astype(np.float32),
                "q_arm_cmd": base_env.q_arm_cmd.copy().astype(np.float32),
                "q_hand_cmd": base_env.q_hand_cmd.copy().astype(np.float32),
                "handoff_reason": "end_of_rollout_fallback",
            }
            return True, handoff
        return False, {"reason": "no_handoff_state", "summary": final_summary}
    return True, handoff


def _run_wrap_recovery(
    base_env: MustardIntentGymEnv,
    args: argparse.Namespace,
    handoff: dict,
    writer=None,
) -> tuple[bool, dict]:
    spec = _task_spec(TASK_WRAP_AND_LIFT)
    data = base_env.data
    model = base_env.model
    arm_cfg = base_env.arm_cfg
    hand_cfg = base_env.hand_cfg
    mustard_cfg = base_env.mustard_cfg
    contact_cfg = base_env.contact_cfg
    force_buf = base_env.force_buf
    control_nstep = int(base_env.control_nstep)
    effective_control_hz = float(base_env.effective_control_hz)
    capture_every = _capture_every(effective_control_hz, float(args.capture_hz))
    lift_hold_steps = max(1, int(np.ceil(float(args.lift_hold_seconds) * effective_control_hz)))

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
    grasp_acquired = False
    first_grasp_step = -1
    task_success = False
    first_success_step = -1
    object_z_ref = float(data.xpos[mustard_cfg.body_id][2])
    object_z_max = object_z_ref
    no_drop_during_hold = False
    hold_contact_ok = False

    current_palm_pose = ArmTargetPose(
        pos=data.xpos[arm_cfg.palm_body_id].astype(np.float64).copy(),
        quat_wxyz=_rot_to_quat(data.xmat[arm_cfg.palm_body_id].reshape(3, 3).copy()).astype(np.float64),
    )
    preshape_ratio = float(np.clip(spec.preshape_ratio, 0.05, 0.95))

    def _record_and_step(hand_cmd: np.ndarray, arm_target: ArmTargetPose, phase_id: int):
        nonlocal step_counter
        nonlocal best_contacts
        nonlocal best_force
        nonlocal best_fingers
        nonlocal approach_min_err
        nonlocal approach_min_rot_err
        n_pre, f_pre, touched_pre, _ = base_env._contact_stats()
        ee_pos_pre = data.xpos[arm_cfg.palm_body_id].copy()
        ee_quat_pre = _rot_to_quat(data.xmat[arm_cfg.palm_body_id].reshape(3, 3))

        arm_cmd, arm_err, arm_rot_err_deg = _step_arm_ik(
            model=model,
            data=data,
            arm_cfg=arm_cfg,
            target=arm_target,
            gain=float(base_env.cfg.ik_gain),
            rot_gain=float(base_env.cfg.ik_rot_gain),
            damping=float(base_env.cfg.ik_damping),
            rot_weight=float(base_env.cfg.ik_rot_weight),
            max_joint_step=float(base_env.cfg.ik_max_joint_step),
        )
        cmd_pose = np.concatenate([arm_target.pos, arm_target.quat_wxyz], axis=0).astype(np.float32)

        if step_counter % capture_every == 0:
            thumb_pre = 1.0 if "th" in touched_pre else 0.0
            contact_stats = np.asarray(
                [float(n_pre), float(f_pre), float(len(touched_pre)), thumb_pre],
                dtype=np.float32,
            )
            images.append(_capture_frame(base_env.obs_renderer, data, base_env.obs_cam))
            states.append(
                base_env._current_observation()["proprio"].astype(np.float32).copy()
            )
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
        mujoco.mj_step(model, data, nstep=control_nstep)
        if writer is not None:
            writer.append_data(base_env.render())

        n_post, f_post, touched_post, _ = base_env._contact_stats()
        best_contacts = max(best_contacts, int(n_post))
        best_force = max(best_force, float(f_post))
        if len(touched_post) >= len(best_fingers):
            best_fingers = set(touched_post)
        approach_min_err = min(approach_min_err, float(arm_err))
        approach_min_rot_err = min(approach_min_rot_err, float(arm_rot_err_deg))
        step_counter += 1
        return {
            "n_contacts": int(n_post),
            "force": float(f_post),
            "touched": set(touched_post),
        }

    def _current_targets() -> tuple[ArmTargetPose, ArmTargetPose]:
        return _resolve_arm_targets(
            data=data,
            mustard_cfg=mustard_cfg,
            approach_offset=np.asarray(spec.approach_offset, dtype=np.float32),
            push_offset=np.asarray(spec.interact_offset, dtype=np.float32),
            approach_rot_quat=np.asarray(spec.approach_rot_quat, dtype=np.float64),
            push_rot_quat=np.asarray(spec.interact_rot_quat, dtype=np.float64),
            offset_frame="object",
            rot_frame="object",
        )

    for i in range(int(args.recenter_steps)):
        alpha = (i + 1) / max(int(args.recenter_steps), 1)
        grasp_value = preshape_ratio * alpha
        hand_cmd = _interpolate_hand_pose(
            grasp_value=grasp_value,
            hand_cfg=hand_cfg,
            hand_trajectory=spec.hand_trajectory,
            preshape_pivot=preshape_ratio,
        )
        _, interact_target = _current_targets()
        arm_target = ArmTargetPose(
            pos=((1.0 - alpha) * current_palm_pose.pos + alpha * interact_target.pos).astype(np.float64),
            quat_wxyz=_quat_lerp_normalize(current_palm_pose.quat_wxyz, interact_target.quat_wxyz, alpha),
        )
        _record_and_step(hand_cmd, arm_target, phase_id=2)

    for i in range(int(args.close_steps)):
        alpha = (i + 1) / max(int(args.close_steps), 1)
        grasp_value = preshape_ratio + (1.0 - preshape_ratio) * alpha
        hand_cmd = _interpolate_hand_pose(
            grasp_value=grasp_value,
            hand_cfg=hand_cfg,
            hand_trajectory=spec.hand_trajectory,
            preshape_pivot=preshape_ratio,
        )
        _, interact_target = _current_targets()
        close_target = ArmTargetPose(
            pos=interact_target.pos.astype(np.float64),
            quat_wxyz=interact_target.quat_wxyz.copy(),
        )
        info = _record_and_step(hand_cmd, close_target, phase_id=3)
        meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], base_env.cfg)
        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
        if grasp_stable_hits >= int(args.stable_steps) and not grasp_acquired:
            grasp_acquired = True
            first_grasp_step = step_counter

    close_pose = _interpolate_hand_pose(
        grasp_value=1.0,
        hand_cfg=hand_cfg,
        hand_trajectory=spec.hand_trajectory,
        preshape_pivot=preshape_ratio,
    )
    _, interact_target = _current_targets()
    close_target = ArmTargetPose(
        pos=interact_target.pos.astype(np.float64),
        quat_wxyz=interact_target.quat_wxyz.copy(),
    )
    for _ in range(int(args.close_hold_steps)):
        info = _record_and_step(close_pose, close_target, phase_id=3)
        meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], base_env.cfg)
        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
        if grasp_stable_hits >= int(args.stable_steps) and not grasp_acquired:
            grasp_acquired = True
            first_grasp_step = step_counter

    if grasp_acquired:
        for i in range(int(args.lift_steps)):
            alpha = (i + 1) / max(int(args.lift_steps), 1)
            lift_target = ArmTargetPose(
                pos=interact_target.pos
                + np.array(
                    [0.0, 0.0, float(args.close_z_clearance) + float(args.lift_height) * alpha],
                    dtype=np.float64,
                ),
                quat_wxyz=interact_target.quat_wxyz.copy(),
            )
            info = _record_and_step(close_pose, lift_target, phase_id=4)
            object_z = float(data.xpos[mustard_cfg.body_id][2])
            object_z_max = max(object_z_max, object_z)
            lifted = (object_z - object_z_ref) >= float(args.lift_success_delta)
            meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], base_env.cfg)
            grasp_stable_hits = grasp_stable_hits + 1 if meets else 0

        hold_target = ArmTargetPose(
            pos=interact_target.pos
            + np.array(
                [0.0, 0.0, float(args.close_z_clearance) + float(args.lift_height)],
                dtype=np.float64,
            ),
            quat_wxyz=interact_target.quat_wxyz.copy(),
        )
        no_drop_during_hold = True
        hold_contact_ok = True
        for _ in range(lift_hold_steps):
            info = _record_and_step(close_pose, hold_target, phase_id=5)
            object_z = float(data.xpos[mustard_cfg.body_id][2])
            object_z_max = max(object_z_max, object_z)
            lifted = (object_z - object_z_ref) >= float(args.lift_success_delta)
            meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], base_env.cfg)
            if not lifted:
                no_drop_during_hold = False
            if not meets:
                hold_contact_ok = False
        if no_drop_during_hold:
            task_success = True
            first_success_step = step_counter

    metrics = {
        "task_success": bool(task_success),
        "reached": bool(approach_min_err <= float(base_env.cfg.arm_reach_threshold)),
        "best_contacts": int(best_contacts),
        "best_force": float(best_force),
        "best_fingers": sorted(best_fingers),
        "approach_min_err": float(approach_min_err),
        "approach_min_rot_err_deg": float(approach_min_rot_err),
        "object_z_ref": float(object_z_ref),
        "object_z_max": float(object_z_max),
        "object_dz_max": float(object_z_max - object_z_ref),
        "grasp_acquired": bool(grasp_acquired),
        "first_grasp_step": int(first_grasp_step),
        "first_success_step": int(first_success_step),
        "no_drop_during_hold": bool(no_drop_during_hold),
        "hold_contact_ok": bool(hold_contact_ok),
        "lift_hold_steps": int(lift_hold_steps),
    }
    return bool(task_success), {
        "images": images,
        "states": states,
        "actions": actions,
        "phases": phases,
        "contacts": contacts,
        "arm_cmd_pose_wxyz": arm_cmd_pose_wxyz,
        "arm_obs_pose_wxyz": arm_obs_pose_wxyz,
        "arm_pose_error": arm_pose_error,
        "metrics": metrics,
    }


def _init_writer(path: Path, fps: int) -> imageio.Writer:
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )


def run_collection(args: argparse.Namespace) -> dict:
    tf.config.set_visible_devices([], "GPU")
    model = OctoModel.load_pretrained(args.model_path)
    action_stats, proprio_stats = _find_stats(model, args.dataset_name)
    if action_stats is None:
        raise RuntimeError(f"Could not find action stats for dataset '{args.dataset_name}'.")
    window_size = int(model.example_batch["observation"]["timestep_pad_mask"].shape[1])
    pred_horizon = int(model.example_batch["action"].shape[2])
    spec = _task_spec(TASK_WRAP_AND_LIFT)

    output_root = Path(args.out_dir)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_success = 0
    attempts = 0
    handoff_success = 0
    policy_rng = jax.random.PRNGKey(int(args.seed))
    attempt_logs: list[dict] = []

    while saved_success < int(args.target_episodes) and attempts < int(args.max_attempts):
        attempts += 1
        env_cfg = MustardIntentEnvConfig(
            task_name=TASK_WRAP_AND_LIFT,
            basis_path=args.basis_path,
            max_episode_steps=int(args.policy_max_steps),
            control_hz=float(args.control_hz),
            policy_repeat=int(args.policy_repeat),
            action_smoothing=1.0,
            action_horizon=int(pred_horizon),
            spawn_pos=(0.78, 0.12, 0.82),
            spawn_quat=(1.0, 0.0, 0.0, 0.0),
            spawn_jitter_xy=float(args.spawn_jitter_xy),
            spawn_yaw_jitter_deg=float(args.spawn_yaw_jitter_deg),
            seed=int(args.seed) + attempts,
            lift_success_delta=float(args.lift_success_delta),
            lift_hold_seconds=float(args.lift_hold_seconds),
        )
        base_env = MustardIntentGymEnv(env_cfg)
        env = NormalizeProprio(base_env, {"proprio": proprio_stats} if proprio_stats is not None else {})
        env = ResizeImageWrapper(env, resize_size={"primary": (256, 256)})
        env = HistoryWrapper(env, window_size)
        env = AddObservationPadMaskWrapper(env, window_size)
        env = RHCWrapper(env, 2)
        task = _build_task(model, spec.instruction, args.goal_episode, 256)

        ok, handoff = _policy_rollout_to_handoff(model, env, base_env, task, action_stats, args, policy_rng)
        policy_rng, _ = jax.random.split(policy_rng)

        if not ok:
            attempt_logs.append({"attempt": attempts, "status": "skip", **handoff})
            base_env.close()
            continue

        handoff_success += 1
        writer = None
        if args.record and saved_success == 0:
            video_path = Path(args.record_dir) / f"mustard_wrap_daggerlite_{datetime.now().strftime('%y%m%d_%H%M%S')}.mp4"
            writer = _init_writer(video_path, int(args.record_fps))
            writer.append_data(base_env.render())

        success, episode = _run_wrap_recovery(base_env, args, handoff, writer=writer)
        if writer is not None:
            writer.close()

        episode_metrics = dict(episode["metrics"])
        episode_metrics["policy_step"] = int(handoff["policy_step"])
        episode_metrics["policy_summary"] = handoff["policy_summary"]
        episode_metrics["handoff_reached"] = bool(
            handoff["policy_summary"]["approach_min_err"] <= float(args.handoff_reach_threshold)
        )

        if success:
            episode_idx = saved_success
            episode_path = raw_dir / f"episode_{episode_idx:05d}.npz"
            criteria = {
                "collection_mode": "dagger_lite_policy_to_expert_wrap_recovery",
                "handoff_reach_threshold": float(args.handoff_reach_threshold),
                "handoff_max_contacts": int(args.handoff_max_contacts),
                "recenter_steps": int(args.recenter_steps),
                "close_steps": int(args.close_steps),
                "close_hold_steps": int(args.close_hold_steps),
                "lift_steps": int(args.lift_steps),
                "lift_hold_seconds": float(args.lift_hold_seconds),
                "policy_repeat": int(args.policy_repeat),
            }
            _save_episode_npz(
                episode_path=episode_path,
                images=episode["images"],
                states=episode["states"],
                actions=episode["actions"],
                phases=episode["phases"],
                contacts=episode["contacts"],
                arm_cmd_pose_wxyz=episode["arm_cmd_pose_wxyz"],
                arm_obs_pose_wxyz=episode["arm_obs_pose_wxyz"],
                arm_pose_error=episode["arm_pose_error"],
                success=True,
                instruction=spec.instruction,
                task_name=TASK_WRAP_AND_LIFT,
                intent=spec.intent,
                side=base_env.cfg.side,
                control_hz=float(args.control_hz),
                capture_hz=float(args.capture_hz),
                object_qpos=np.concatenate(
                    [
                        base_env.data.qpos[base_env.mustard_cfg.qpos_adr : base_env.mustard_cfg.qpos_adr + 3],
                        base_env.data.qpos[base_env.mustard_cfg.qpos_adr + 3 : base_env.mustard_cfg.qpos_adr + 7],
                    ],
                    axis=0,
                ).astype(np.float32),
                criteria=criteria,
                metrics=episode_metrics,
            )
            # Save extra DAgger metadata alongside the standard payload.
            with np.load(episode_path, allow_pickle=True) as existing:
                payload = {k: existing[k] for k in existing.files}
            payload["dagger_model_path"] = np.asarray(str(Path(args.model_path).resolve()), dtype=object)
            payload["dagger_goal_episode"] = np.asarray(str(Path(args.goal_episode).resolve()), dtype=object)
            payload["dagger_handoff_step"] = np.asarray(int(handoff["policy_step"]), dtype=np.int32)
            payload["dagger_handoff_policy_json"] = np.asarray(json.dumps(handoff["policy_summary"]), dtype=object)
            payload["dagger_handoff_qpos"] = handoff["qpos"].astype(np.float32)
            payload["dagger_handoff_qvel"] = handoff["qvel"].astype(np.float32)
            payload["dagger_handoff_q_arm_cmd"] = handoff["q_arm_cmd"].astype(np.float32)
            payload["dagger_handoff_q_hand_cmd"] = handoff["q_hand_cmd"].astype(np.float32)
            payload["saved_at"] = np.asarray(datetime.now().isoformat())
            np.savez_compressed(episode_path, **payload)
            saved_success += 1
            status = "saved"
        else:
            status = "recovery_failed"

        attempt_logs.append(
            {
                "attempt": attempts,
                "status": status,
                "policy_step": int(handoff["policy_step"]),
                "policy_summary": handoff["policy_summary"],
                "episode_metrics": episode_metrics,
            }
        )
        base_env.close()
        print(
            f"[attempt {attempts:04d}] {status} handoff_step={handoff['policy_step']} "
            f"approach={handoff['policy_summary']['approach_min_err']:.4f} "
            f"saved={saved_success}/{args.target_episodes}",
            flush=True,
        )

    summary = {
        "collection_mode": "dagger_lite_policy_to_expert_wrap_recovery",
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "basis_path": str(Path(args.basis_path).expanduser().resolve()),
        "goal_episode": str(Path(args.goal_episode).expanduser().resolve()),
        "target_episodes": int(args.target_episodes),
        "saved_success_episodes": int(saved_success),
        "attempts": int(attempts),
        "handoff_successes": int(handoff_success),
        "out_dir": str(output_root.resolve()),
        "attempt_log_tail": attempt_logs[-20:],
    }
    summary_path = output_root / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_collection(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
