#!/usr/bin/env python3
"""Interactive MuJoCo viewer for official-style mustard Octo rollout.

Controls:
  r : reset to a new episode seed
  p : pause / resume policy stepping
  s : single policy step while paused
  q : quit
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gym
import jax
import mujoco
from mujoco import viewer
import numpy as np
import tensorflow as tf

from env.viewer_utils import set_default_franka_allegro_camera
from scripts.eval.mustard_intent_gym_env import MustardIntentEnvConfig, MustardIntentGymEnv
from scripts.eval.rollout_mustard_intent_octo_official import (
    AddObservationPadMaskWrapper,
    TrainResizeImageWrapper,
    _build_task,
    _find_stats,
)

_DEFAULT_OCTO_SRC = "/home/minchan/Downloads/dual_arm_VLA/octo"
if importlib.util.find_spec("octo") is None:
    octo_src = Path(
        (_REPO_ROOT / ".octo_src_path").read_text(encoding="utf-8").strip()
        if (_REPO_ROOT / ".octo_src_path").exists()
        else _DEFAULT_OCTO_SRC
    )
    if octo_src.exists():
        sys.path.insert(0, str(octo_src))

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import (
    HistoryWrapper,
    NormalizeProprio,
    ResizeImageWrapper,
    RHCWrapper,
    TemporalEnsembleWrapper,
)


class TerminalKeyReader:
    def __init__(self) -> None:
        self.fd: int | None = None
        self.old_attrs = None
        self.enabled = False

    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            return self
        self.fd = sys.stdin.fileno()
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.fd is not None and self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
        self.enabled = False

    def read_keys(self) -> list[str]:
        if not self.enabled:
            return []
        keys: list[str] = []
        assert self.fd is not None
        while select.select([self.fd], [], [], 0.0)[0]:
            chunk = os.read(self.fd, 1024).decode("utf-8", errors="ignore")
            keys.extend(list(chunk))
        return keys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run interactive official-style Octo rollout in a MuJoCo viewer."
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="models/mustard_octo_official_wrap500_mix_k4_20260324/run_260324_182938/best_train_model",
    )
    p.add_argument(
        "--basis-path",
        type=str,
        default="dataset/mustard_intent_wrap500_mix_k4_basis_20260324/mustard_intent_hand_pca_k4.npz",
    )
    p.add_argument("--dataset-name", type=str, default="mustard_grasp_oxe")
    p.add_argument("--task", type=str, default="wrap_and_lift")
    p.add_argument("--max-policy-steps", type=int, default=60)
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--policy-repeat", type=int, default=20)
    p.add_argument("--action-smoothing", type=float, default=1.0)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--policy-image-size", type=int, default=256)
    p.add_argument("--render-width", type=int, default=1280)
    p.add_argument("--render-height", type=int, default=720)
    p.add_argument("--spawn-pos", type=float, nargs=3, default=(0.78, 0.12, 0.82))
    p.add_argument("--spawn-quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    p.add_argument("--spawn-jitter-xy", type=float, default=0.0)
    p.add_argument("--spawn-yaw-jitter-deg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wrapper", choices=("rhc", "temporal"), default="temporal")
    p.add_argument("--exec-horizon", type=int, default=2)
    p.add_argument(
        "--history-mode",
        choices=("per_env_step", "per_plan"),
        default="per_env_step",
    )
    p.add_argument("--argmax", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--conditioning",
        choices=("language", "goal", "language_goal"),
        default="language_goal",
    )
    p.add_argument(
        "--resize-mode",
        choices=("octo_avg_crop", "train_resize"),
        default="octo_avg_crop",
    )
    p.add_argument(
        "--task-builder",
        choices=("create_tasks", "manual"),
        default="create_tasks",
    )
    p.add_argument(
        "--goal-episode",
        type=str,
        default="dataset/mustard_intent_wrap500_mix_k4_20260324/raw/episode_00000.npz",
    )
    p.add_argument("--show-right-ui", action="store_true")
    p.add_argument(
        "--viewer-step-delay",
        type=float,
        default=0.0,
        help="Optional extra sleep after each viewer sync.",
    )
    p.add_argument(
        "--outer-step-seconds",
        type=float,
        default=0.2,
        help="Wall-clock cadence for high-level Octo policy steps.",
    )
    p.add_argument(
        "--sync-viewer-per-inner-step",
        action="store_true",
        default=True,
        help="Sync the viewer during each inner control step, not just once per policy step.",
    )
    p.add_argument(
        "--no-sync-viewer-per-inner-step",
        action="store_false",
        dest="sync_viewer_per_inner_step",
        help="Disable inner control-step viewer sync.",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop automatically after this many completed episodes. 0 means infinite.",
    )
    return p.parse_args()


def _wrap_env(
    base_env: MustardIntentGymEnv,
    window_size: int,
    pred_horizon: int,
    proprio_stats: dict | None,
    args: argparse.Namespace,
):
    env: gym.Env = base_env
    env = NormalizeProprio(env, {"proprio": proprio_stats} if proprio_stats is not None else {})
    if args.resize_mode == "octo_avg_crop":
        env = ResizeImageWrapper(
            env,
            resize_size={"primary": (args.policy_image_size, args.policy_image_size)},
        )
    else:
        env = TrainResizeImageWrapper(env, args.policy_image_size)

    if args.history_mode == "per_env_step":
        env = HistoryWrapper(env, window_size)
        env = AddObservationPadMaskWrapper(env, window_size)
        if args.wrapper == "temporal":
            env = TemporalEnsembleWrapper(env, pred_horizon)
        else:
            exec_horizon = max(1, min(int(args.exec_horizon), pred_horizon))
            env = RHCWrapper(env, exec_horizon)
    else:
        if args.wrapper == "temporal":
            env = TemporalEnsembleWrapper(env, pred_horizon)
        else:
            exec_horizon = max(1, min(int(args.exec_horizon), pred_horizon))
            env = RHCWrapper(env, exec_horizon)
        env = HistoryWrapper(env, window_size)
        env = AddObservationPadMaskWrapper(env, window_size)
    return env


def main() -> None:
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")

    model = OctoModel.load_pretrained(args.model_path)
    if model.text_processor is None and args.conditioning in ("language", "language_goal"):
        raise RuntimeError("Loaded model has no text processor.")

    action_stats, proprio_stats = _find_stats(model, args.dataset_name)
    if action_stats is None:
        raise RuntimeError(f"Could not find action stats for dataset '{args.dataset_name}'.")

    window_size = int(model.example_batch["observation"]["timestep_pad_mask"].shape[1])
    pred_horizon = int(model.example_batch["action"].shape[2])
    if pred_horizon <= 0:
        pred_horizon = 1

    env_cfg = MustardIntentEnvConfig(
        task_name=args.task,
        basis_path=args.basis_path,
        action_horizon=pred_horizon,
        max_episode_steps=args.max_policy_steps,
        control_hz=args.control_hz,
        policy_repeat=args.policy_repeat,
        action_smoothing=args.action_smoothing,
        image_width=args.image_width,
        image_height=args.image_height,
        render_width=args.render_width,
        render_height=args.render_height,
        policy_image_size=args.policy_image_size,
        spawn_pos=tuple(float(x) for x in args.spawn_pos),
        spawn_quat=tuple(float(x) for x in args.spawn_quat),
        spawn_jitter_xy=args.spawn_jitter_xy,
        spawn_yaw_jitter_deg=args.spawn_yaw_jitter_deg,
        seed=args.seed,
    )

    base_env = MustardIntentGymEnv(env_cfg)
    env = _wrap_env(base_env, window_size, pred_horizon, proprio_stats, args)
    task = _build_task(model, args, base_env)

    episode_idx = 0
    decision_idx = 0
    paused = False
    obs = None
    rng = None
    pending_action: np.ndarray | None = None

    def reset_episode(reason: str) -> None:
        nonlocal episode_idx, decision_idx, obs, rng, pending_action
        if episode_idx > 0 or reason != "initial":
            print(f"[reset] reason={reason}", flush=True)
        obs, _ = env.reset(seed=args.seed + episode_idx)
        rng = jax.random.PRNGKey(args.seed + episode_idx)
        decision_idx = 0
        pending_action = None
        print(
            f"[episode] idx={episode_idx} seed={args.seed + episode_idx} "
            f"instruction='{base_env.get_instruction()}'",
            flush=True,
        )
        episode_idx += 1

    def sample_action_once() -> np.ndarray:
        nonlocal rng, obs
        assert obs is not None
        assert rng is not None
        obs_batch = jax.tree_map(lambda x: x[None], obs)
        rng, sample_rng = jax.random.split(rng)
        action = model.sample_actions(
            obs_batch,
            task,
            unnormalization_statistics=action_stats,
            rng=sample_rng,
            argmax=bool(args.argmax),
            temperature=float(args.temperature),
        )[0]
        return np.asarray(action, dtype=np.float32)

    reset_episode("initial")
    print("[warmup] compiling first policy call...", flush=True)
    warmup_t0 = time.time()
    pending_action = sample_action_once()
    print(f"[warmup] done in {time.time() - warmup_t0:.2f}s", flush=True)

    print(
        "[controls] r=reset  p=pause/resume  s=single-step(when paused)  q=quit",
        flush=True,
    )
    if not sys.stdin.isatty():
        print("[controls] stdin is not a tty; keyboard controls are disabled.", flush=True)

    with TerminalKeyReader() as key_reader, viewer.launch_passive(
        base_env.model,
        base_env.data,
        show_left_ui=False,
        show_right_ui=args.show_right_ui,
    ) as passive_viewer:
        set_default_franka_allegro_camera(passive_viewer.cam)
        base_env.set_step_sync_callback(
            passive_viewer.sync if args.sync_viewer_per_inner_step else None,
            delay=float(args.viewer_step_delay),
        )

        while passive_viewer.is_running():
            single_step = False
            for key in key_reader.read_keys():
                key = key.lower()
                if key == "q":
                    print("[quit] requested from terminal.", flush=True)
                    base_env.close()
                    return
                if key == "r":
                    reset_episode("manual")
                elif key == "p":
                    paused = not paused
                    print(f"[pause] paused={paused}", flush=True)
                elif key == "s":
                    single_step = True

            if paused and not single_step:
                passive_viewer.sync()
                if args.viewer_step_delay > 0:
                    time.sleep(args.viewer_step_delay)
                continue

            if pending_action is not None:
                action = pending_action
                pending_action = None
            else:
                action = sample_action_once()

            wall_t0 = time.monotonic()
            obs, reward, terminated, truncated, info = env.step(action)
            decision_idx += 1
            summary = base_env.get_episode_summary()

            if decision_idx == 1 or decision_idx % 5 == 0 or terminated or truncated:
                print(
                    "[step] "
                    f"episode={episode_idx - 1} decision={decision_idx} "
                    f"reached={bool(summary['reached'])} "
                    f"contacts={int(summary['best_contacts'])} "
                    f"fingers={','.join(summary['best_fingers']) if summary['best_fingers'] else '-'} "
                    f"dz={float(summary['object_dz_max']):.4f} "
                    f"approach_err={float(summary['approach_min_err']):.4f}",
                    flush=True,
                )

            if not args.sync_viewer_per_inner_step:
                passive_viewer.sync()
                if args.viewer_step_delay > 0:
                    time.sleep(args.viewer_step_delay)

            remaining = float(args.outer_step_seconds) - (time.monotonic() - wall_t0)
            if remaining > 0.0:
                time.sleep(remaining)

            if terminated or truncated:
                print(
                    "[episode-end] "
                    f"episode={episode_idx - 1} success={bool(summary['success'])} "
                    f"reached={bool(summary['reached'])} "
                    f"contacts={int(summary['best_contacts'])} "
                    f"fingers={','.join(summary['best_fingers']) if summary['best_fingers'] else '-'} "
                    f"dz={float(summary['object_dz_max']):.4f} "
                    f"hold={int(summary['wrap_hold_hits'])}",
                    flush=True,
                )
                if args.max_episodes > 0 and episode_idx >= args.max_episodes:
                    print("[done] reached max_episodes.", flush=True)
                    break
                reset_episode("auto")

    base_env.close()


if __name__ == "__main__":
    main()
