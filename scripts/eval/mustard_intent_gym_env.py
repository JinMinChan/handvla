#!/usr/bin/env python3
"""Gym-style MuJoCo env for mustard intent evaluation with dataset-rate semantics."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from dataclasses import dataclass, field
import time
from typing import Any

import gym
from gym import spaces
import mujoco
import numpy as np

from env import franka_allegro_mjcf
from env.viewer_utils import set_default_franka_allegro_camera
from scripts.data.collect_mustard_intent_benchmark import (
    TASK_HOOK_AND_PULL,
    TASK_PUSH_OVER,
    TASK_WRAP_AND_LIFT,
    _build_contact_config,
    _build_hand_config,
    _build_mustard_config,
    _contact_meets,
    _detect_contact_with_target,
    _make_state_vector,
    _normalize_quat,
    _pull_contact_meets,
    _push_contact_meets,
    _sample_spawn_pose,
    _set_mustard_pose,
    _task_spec,
    _tilt_deg,
    _yaw_rad,
)
from scripts.data.collect_pickandlift_rlds import (
    ArmTargetPose,
    _build_arm_config,
    _capture_frame,
    _quat_to_rot,
    _rot_to_quat,
    _step_arm_ik,
)


TASKS = (TASK_WRAP_AND_LIFT, TASK_PUSH_OVER, TASK_HOOK_AND_PULL)


def _load_basis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
    B = np.asarray(data["B"], dtype=np.float32)
    if mu.shape[0] != 16 or B.ndim != 2 or B.shape[0] != 16:
        raise ValueError(f"Unexpected basis shapes in {path}: mu={mu.shape}, B={B.shape}")
    return mu, B


def _euler_xyz_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in np.asarray(euler_xyz, dtype=np.float64)]
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    quat = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return _normalize_quat(quat).astype(np.float32)


def _resolve_target(
    data: mujoco.MjData,
    mustard_cfg,
    offset: np.ndarray,
    rot_quat: np.ndarray,
) -> ArmTargetPose:
    mustard_pos = data.xpos[mustard_cfg.body_id].copy()
    obj_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3)
    target_pos = mustard_pos + obj_rot @ offset
    target_rot = obj_rot @ _quat_to_rot(rot_quat)
    return ArmTargetPose(pos=target_pos.astype(np.float64), quat_wxyz=_rot_to_quat(target_rot))


@dataclass
class MustardIntentEnvConfig:
    task_name: str
    basis_path: str
    side: str = "right"
    max_episode_steps: int = 60
    control_hz: float = 100.0
    policy_repeat: int = 20
    action_smoothing: float = 1.0
    image_width: int = 640
    image_height: int = 480
    render_width: int = 1280
    render_height: int = 720
    policy_image_size: int = 256
    action_horizon: int = 8
    spawn_pos: tuple[float, float, float] = (0.78, 0.12, 0.82)
    spawn_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    spawn_jitter_xy: float = 0.0
    spawn_yaw_jitter_deg: float = 0.0
    seed: int = 0
    ik_gain: float = 0.95
    ik_rot_gain: float = 0.9
    ik_damping: float = 0.08
    ik_rot_weight: float = 0.20
    ik_max_joint_step: float = 0.08
    arm_reach_threshold: float = 0.05
    lift_success_delta: float = 0.08
    lift_hold_seconds: float = 1.5
    min_contacts: int = 2
    min_contact_fingers: int = 3
    require_thumb_contact: bool = True
    min_force: float = 0.5
    max_force: float = 1000.0
    push_success_tilt_deg: float = 55.0
    push_max_lift_dz: float = 0.04
    push_release_steps: int = 3
    pull_success_dx: float = 0.08
    pull_max_lift_dz: float = 0.04
    post_settle_steps: int = 200


@dataclass
class EpisodeMetrics:
    success: bool = False
    reached: bool = False
    best_contacts: int = 0
    best_force: float = 0.0
    best_fingers: set[str] = field(default_factory=set)
    approach_min_err: float = 1e9
    approach_min_rot_err_deg: float = 1e9
    object_z_ref: float = 0.0
    object_z_max: float = -1e9
    topple_angle_max: float = 0.0
    release_after_topple: bool = False
    push_contact_seen: bool = False
    push_planar_disp_max: float = 0.0
    hook_contact_seen: bool = False
    pull_dx_max: float = 0.0
    wrap_hold_hits: int = 0
    steps: int = 0


class MustardIntentGymEnv(gym.Env):
    """Dataset-rate environment suitable for official Octo wrappers."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, config: MustardIntentEnvConfig):
        super().__init__()
        if config.task_name not in TASKS:
            raise ValueError(f"Unsupported task: {config.task_name}")
        self.cfg = config
        self.spec = _task_spec(config.task_name)
        self.mu, self.B = _load_basis(Path(config.basis_path).expanduser().resolve())
        self.action_dim = 6 + int(self.B.shape[1])

        self.mjcf = franka_allegro_mjcf.load(side=config.side, add_mustard=True)
        self.model = self.mjcf.compile()
        self.data = mujoco.MjData(self.model)

        target_dt = 1.0 / max(float(config.control_hz), 1e-6)
        self.control_nstep = max(1, int(round(target_dt / self.model.opt.timestep)))
        self.effective_control_dt = self.control_nstep * self.model.opt.timestep
        self.effective_control_hz = 1.0 / max(self.effective_control_dt, 1e-9)
        self.lift_hold_steps = max(
            1, int(np.ceil(float(config.lift_hold_seconds) * self.effective_control_hz))
        )

        self.arm_cfg = _build_arm_config(self.model, config.side)
        self.hand_cfg = _build_hand_config(self.model, config.side)
        self.mustard_cfg = _build_mustard_config(self.model)
        self.contact_cfg = _build_contact_config(self.model, config.side, self.mustard_cfg.body_id)
        self.force_buf = np.zeros(6, dtype=float)

        self.obs_renderer = mujoco.Renderer(
            self.model, width=config.image_width, height=config.image_height
        )
        self.render_renderer = mujoco.Renderer(
            self.model, width=config.render_width, height=config.render_height
        )
        self.obs_cam = mujoco.MjvCamera()
        self.render_cam = mujoco.MjvCamera()
        set_default_franka_allegro_camera(self.obs_cam)
        set_default_franka_allegro_camera(self.render_cam)

        self.rng = np.random.default_rng(config.seed)
        self.base_spawn_pos = np.asarray(config.spawn_pos, dtype=np.float32)
        self.base_spawn_quat = np.asarray(config.spawn_quat, dtype=np.float32)
        self.alpha = float(np.clip(config.action_smoothing, 0.0, 1.0))

        self.observation_space = spaces.Dict(
            {
                "image_primary": spaces.Box(
                    low=0,
                    high=255,
                    shape=(config.image_height, config.image_width, 3),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(60,),
                    dtype=np.float32,
                ),
                "timestep": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max,
                    shape=(),
                    dtype=np.int32,
                ),
                "task_completed": spaces.Box(
                    low=0,
                    high=1,
                    shape=(int(config.action_horizon),),
                    dtype=bool,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.q_arm_cmd = np.zeros((7,), dtype=np.float32)
        self.q_hand_cmd = np.zeros((16,), dtype=np.float32)
        self.cached_arm_target: ArmTargetPose | None = None
        self.cached_hand_target = np.zeros((16,), dtype=np.float32)
        self.initial_xy = np.zeros((2,), dtype=np.float32)
        self.initial_x = 0.0
        self.metrics = EpisodeMetrics()
        self.step_count = 0
        self._episode_done = False
        self._viewer_sync_callback = None
        self._viewer_sync_delay = 0.0

    @staticmethod
    def goal_image_from_episode(path: str | Path, which: str = "last") -> np.ndarray:
        data = np.load(Path(path).expanduser().resolve(), allow_pickle=True)
        images = np.asarray(data["images"], dtype=np.uint8)
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError(f"Invalid images in {path}: {images.shape}")
        if which == "first":
            return images[0].copy()
        return images[-1].copy()

    def get_instruction(self) -> str:
        return str(self.spec.instruction)

    def set_step_sync_callback(self, callback, delay: float = 0.0) -> None:
        self._viewer_sync_callback = callback
        self._viewer_sync_delay = max(float(delay), 0.0)

    def _maybe_sync_viewer(self) -> None:
        if self._viewer_sync_callback is None:
            return
        self._viewer_sync_callback()
        if self._viewer_sync_delay > 0.0:
            time.sleep(self._viewer_sync_delay)

    def _contact_stats(self) -> tuple[int, float, set[str], np.ndarray]:
        n_contacts, total_force, touched = _detect_contact_with_target(
            self.model, self.data, self.contact_cfg, self.force_buf
        )
        thumb_pre = 1.0 if "th" in touched else 0.0
        contact_stats = np.asarray(
            [float(n_contacts), float(total_force), float(len(touched)), thumb_pre],
            dtype=np.float32,
        )
        return int(n_contacts), float(total_force), set(touched), contact_stats

    def _current_observation(self) -> dict[str, np.ndarray]:
        _, _, _, contact_stats = self._contact_stats()
        state = _make_state_vector(
            self.data, self.arm_cfg, self.hand_cfg, self.mustard_cfg, contact_stats
        )
        image = _capture_frame(self.obs_renderer, self.data, self.obs_cam)
        return {
            "image_primary": image.astype(np.uint8),
            "proprio": state.astype(np.float32),
            "timestep": np.asarray(self.step_count, dtype=np.int32),
            # In deployment we do not know a future demo goal timestep; keep this
            # false so inference sees the same field structure as training.
            "task_completed": np.zeros((int(self.cfg.action_horizon),), dtype=bool),
        }

    def _update_metrics(self, n_post: int, f_post: float, touched_post: set[str]) -> None:
        self.metrics.best_contacts = max(self.metrics.best_contacts, int(n_post))
        self.metrics.best_force = max(self.metrics.best_force, float(f_post))
        if len(touched_post) >= len(self.metrics.best_fingers):
            self.metrics.best_fingers = set(touched_post)

        object_pos = self.data.xpos[self.mustard_cfg.body_id].copy()
        object_rot = self.data.xmat[self.mustard_cfg.body_id].reshape(3, 3).copy()
        self.metrics.object_z_max = max(self.metrics.object_z_max, float(object_pos[2]))

        current_target = _resolve_target(
            self.data,
            self.mustard_cfg,
            np.asarray(self.spec.interact_offset, dtype=np.float32),
            np.asarray(self.spec.interact_rot_quat, dtype=np.float64),
        )
        ee_pos = self.data.xpos[self.arm_cfg.palm_body_id].copy()
        self.metrics.approach_min_err = min(
            self.metrics.approach_min_err,
            float(np.linalg.norm(current_target.pos - ee_pos)),
        )
        ee_rot = self.data.xmat[self.arm_cfg.palm_body_id].reshape(3, 3).copy()
        tgt_rot = _quat_to_rot(current_target.quat_wxyz)
        rot_err = 0.5 * (
            np.cross(ee_rot[:, 0], tgt_rot[:, 0])
            + np.cross(ee_rot[:, 1], tgt_rot[:, 1])
            + np.cross(ee_rot[:, 2], tgt_rot[:, 2])
        )
        self.metrics.approach_min_rot_err_deg = min(
            self.metrics.approach_min_rot_err_deg,
            float(np.rad2deg(np.linalg.norm(rot_err))),
        )

        if self.cfg.task_name == TASK_WRAP_AND_LIFT:
            lifted = (float(object_pos[2]) - self.metrics.object_z_ref) >= self.cfg.lift_success_delta
            meets = _contact_meets(int(n_post), float(f_post), touched_post, self.cfg)
            self.metrics.wrap_hold_hits = self.metrics.wrap_hold_hits + 1 if (lifted and meets) else 0
            if self.metrics.wrap_hold_hits >= self.lift_hold_steps:
                self.metrics.success = True
        elif self.cfg.task_name == TASK_PUSH_OVER:
            tilt_now = _tilt_deg(object_rot)
            self.metrics.topple_angle_max = max(self.metrics.topple_angle_max, tilt_now)
            push_planar_disp = float(np.linalg.norm(object_pos[:2] - self.initial_xy))
            self.metrics.push_planar_disp_max = max(self.metrics.push_planar_disp_max, push_planar_disp)
            self.metrics.push_contact_seen = self.metrics.push_contact_seen or _push_contact_meets(
                int(n_post), float(f_post), touched_post, self.cfg
            )
        elif self.cfg.task_name == TASK_HOOK_AND_PULL:
            pull_dx = max(0.0, self.initial_x - float(object_pos[0]))
            self.metrics.pull_dx_max = max(self.metrics.pull_dx_max, pull_dx)
            self.metrics.hook_contact_seen = self.metrics.hook_contact_seen or _pull_contact_meets(
                int(n_post), float(f_post), touched_post, self.cfg
            )

    def _run_post_settle(self) -> None:
        if self._episode_done:
            return
        release_after_topple_hits = 0
        for _ in range(int(self.cfg.post_settle_steps)):
            self.data.ctrl[self.arm_cfg.act_ids] = self.q_arm_cmd
            self.data.ctrl[7:23] = self.q_hand_cmd
            mujoco.mj_step(self.model, self.data, nstep=self.control_nstep)
            self._maybe_sync_viewer()
            n_post, f_post, touched_post, _ = self._contact_stats()
            self._update_metrics(n_post, f_post, touched_post)
            object_pos = self.data.xpos[self.mustard_cfg.body_id].copy()
            object_rot = self.data.xmat[self.mustard_cfg.body_id].reshape(3, 3).copy()

            if self.cfg.task_name == TASK_PUSH_OVER:
                tilt_now = _tilt_deg(object_rot)
                self.metrics.topple_angle_max = max(self.metrics.topple_angle_max, tilt_now)
                toppled = tilt_now >= self.cfg.push_success_tilt_deg
                not_lifted = (self.metrics.object_z_max - self.metrics.object_z_ref) <= self.cfg.push_max_lift_dz
                released = int(n_post) == 0
                if toppled and not_lifted:
                    release_after_topple_hits = release_after_topple_hits + 1 if released else 0
                    self.metrics.release_after_topple = (
                        self.metrics.release_after_topple
                        or release_after_topple_hits >= max(int(self.cfg.push_release_steps), 1)
                    )
                else:
                    release_after_topple_hits = 0
            elif self.cfg.task_name == TASK_HOOK_AND_PULL:
                pull_dx = max(0.0, self.initial_x - float(object_pos[0]))
                self.metrics.pull_dx_max = max(self.metrics.pull_dx_max, pull_dx)

        if self.cfg.task_name == TASK_PUSH_OVER:
            self.metrics.success = (
                self.metrics.topple_angle_max >= self.cfg.push_success_tilt_deg
                and (self.metrics.object_z_max - self.metrics.object_z_ref) <= self.cfg.push_max_lift_dz
                and self.metrics.release_after_topple
            )
        elif self.cfg.task_name == TASK_HOOK_AND_PULL:
            self.metrics.success = (
                self.metrics.hook_contact_seen
                and self.metrics.pull_dx_max >= self.cfg.pull_success_dx
                and (self.metrics.object_z_max - self.metrics.object_z_ref) <= self.cfg.pull_max_lift_dz
            )

        self.metrics.reached = self.metrics.approach_min_err <= self.cfg.arm_reach_threshold
        self.metrics.success = bool(self.metrics.reached and self.metrics.success)
        self._episode_done = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        initial_state = self.model.key("initial_state").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, initial_state)
        mujoco.mj_forward(self.model, self.data)

        spawn_pos, spawn_quat = _sample_spawn_pose(
            self.base_spawn_pos,
            self.base_spawn_quat,
            self.rng,
            self.cfg,
        )
        _set_mustard_pose(self.data, self.mustard_cfg, spawn_pos, spawn_quat)
        mujoco.mj_forward(self.model, self.data)

        self.q_arm_cmd = self.data.qpos[self.arm_cfg.qpos_ids].astype(np.float32).copy()
        self.q_arm_cmd = np.clip(self.q_arm_cmd, self.arm_cfg.q_min, self.arm_cfg.q_max)
        self.q_hand_cmd = self.data.qpos[self.hand_cfg.qpos_ids].astype(np.float32).copy()
        self.q_hand_cmd = np.clip(self.q_hand_cmd, self.hand_cfg.q_min, self.hand_cfg.q_max)
        self.cached_arm_target = ArmTargetPose(
            pos=self.data.xpos[self.arm_cfg.palm_body_id].astype(np.float64).copy(),
            quat_wxyz=_rot_to_quat(
                self.data.xmat[self.arm_cfg.palm_body_id].reshape(3, 3).copy()
            ).astype(np.float64),
        )
        self.cached_hand_target = self.q_hand_cmd.copy()
        self.metrics = EpisodeMetrics()
        self.metrics.object_z_ref = float(self.data.xpos[self.mustard_cfg.body_id][2])
        self.initial_xy = self.data.xpos[self.mustard_cfg.body_id][:2].copy()
        self.initial_x = float(self.initial_xy[0])
        self.step_count = 0
        self._episode_done = False

        return self._current_observation(), {
            "task_name": self.cfg.task_name,
            "instruction": self.get_instruction(),
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN/Inf.")

        arm_pred = action[:6]
        hand_pred = action[6:]
        self.cached_arm_target = ArmTargetPose(
            pos=np.asarray(arm_pred[:3], dtype=np.float64).copy(),
            quat_wxyz=_euler_xyz_to_quat(arm_pred[3:6]).astype(np.float64),
        )
        self.cached_hand_target = self.mu + self.B @ hand_pred
        self.cached_hand_target = np.clip(
            self.cached_hand_target, self.hand_cfg.q_min, self.hand_cfg.q_max
        )

        q_arm_des, _, _ = _step_arm_ik(
            model=self.model,
            data=self.data,
            arm_cfg=self.arm_cfg,
            target=self.cached_arm_target,
            gain=float(self.cfg.ik_gain),
            rot_gain=float(self.cfg.ik_rot_gain),
            damping=float(self.cfg.ik_damping),
            rot_weight=float(self.cfg.ik_rot_weight),
            max_joint_step=float(self.cfg.ik_max_joint_step),
        )
        self.q_arm_cmd = (1.0 - self.alpha) * self.q_arm_cmd + self.alpha * q_arm_des
        self.q_hand_cmd = (1.0 - self.alpha) * self.q_hand_cmd + self.alpha * self.cached_hand_target
        self.q_arm_cmd = np.clip(self.q_arm_cmd, self.arm_cfg.q_min, self.arm_cfg.q_max)
        self.q_hand_cmd = np.clip(self.q_hand_cmd, self.hand_cfg.q_min, self.hand_cfg.q_max)

        self.metrics.steps += 1
        self.step_count += 1
        for _ in range(int(self.cfg.policy_repeat)):
            self.data.ctrl[self.arm_cfg.act_ids] = self.q_arm_cmd
            self.data.ctrl[7:23] = self.q_hand_cmd
            mujoco.mj_step(self.model, self.data, nstep=self.control_nstep)
            self._maybe_sync_viewer()
            n_post, f_post, touched_post, _ = self._contact_stats()
            self._update_metrics(n_post, f_post, touched_post)
            if self.metrics.success:
                break

        terminated = False
        truncated = False
        if self.metrics.success:
            self.metrics.reached = self.metrics.approach_min_err <= self.cfg.arm_reach_threshold
            self.metrics.success = bool(self.metrics.reached and self.metrics.success)
            terminated = bool(self.metrics.success)
            self._episode_done = True
        elif self.step_count >= int(self.cfg.max_episode_steps):
            self._run_post_settle()
            terminated = bool(self.metrics.success)
            truncated = not terminated

        obs = self._current_observation()
        reward = 1.0 if self.metrics.success else 0.0
        info = self.get_episode_summary()
        return obs, reward, terminated, truncated, info

    def get_episode_summary(self) -> dict[str, Any]:
        return {
            "task_name": self.cfg.task_name,
            "instruction": self.get_instruction(),
            "success": bool(self.metrics.success),
            "reached": bool(self.metrics.approach_min_err <= self.cfg.arm_reach_threshold),
            "approach_min_err": float(self.metrics.approach_min_err),
            "approach_min_rot_err_deg": float(self.metrics.approach_min_rot_err_deg),
            "best_contacts": int(self.metrics.best_contacts),
            "best_force": float(self.metrics.best_force),
            "best_fingers": sorted(self.metrics.best_fingers),
            "object_dz_max": float(self.metrics.object_z_max - self.metrics.object_z_ref),
            "wrap_hold_hits": int(self.metrics.wrap_hold_hits),
            "tilt_deg_max": float(self.metrics.topple_angle_max),
            "release_after_topple": bool(self.metrics.release_after_topple),
            "push_contact_seen": bool(self.metrics.push_contact_seen),
            "push_planar_disp_max": float(self.metrics.push_planar_disp_max),
            "hook_contact_seen": bool(self.metrics.hook_contact_seen),
            "pull_dx_max": float(self.metrics.pull_dx_max),
            "steps": int(self.metrics.steps),
        }

    def render(self):
        return _capture_frame(self.render_renderer, self.data, self.render_cam)

    def close(self):
        self.obs_renderer.close()
        self.render_renderer.close()
