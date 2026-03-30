#!/usr/bin/env python3
"""Official-style Octo rollout on mustard intent tasks using Gym wrappers."""

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

import imageio.v2 as imageio
import jax
import numpy as np
from PIL import Image
import tensorflow as tf
import gym

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
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, TemporalEnsembleWrapper, NormalizeProprio, ResizeImageWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a finetuned Octo model with official-style Gym wrappers."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--basis-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="mustard_grasp_oxe")
    parser.add_argument("--task", type=str, default="wrap_and_lift")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-policy-steps", type=int, default=60)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--policy-repeat", type=int, default=20)
    parser.add_argument("--action-smoothing", type=float, default=1.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--policy-image-size", type=int, default=256)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--spawn-pos", type=float, nargs=3, default=(0.78, 0.12, 0.82))
    parser.add_argument("--spawn-quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--spawn-jitter-xy", type=float, default=0.0)
    parser.add_argument("--spawn-yaw-jitter-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wrapper", choices=("rhc", "temporal"), default="temporal")
    parser.add_argument("--exec-horizon", type=int, default=2)
    parser.add_argument(
        "--history-mode",
        choices=("per_env_step", "per_plan"),
        default="per_env_step",
        help=(
            "per_env_step matches official HistoryWrapper->RHC ordering. "
            "per_plan stacks history only after each replanning step."
        ),
    )
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--conditioning",
        choices=("language", "goal", "language_goal"),
        default="language",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("octo_avg_crop", "train_resize"),
        default="octo_avg_crop",
        help="Image resize path for rollout observations.",
    )
    parser.add_argument(
        "--task-builder",
        choices=("create_tasks", "manual"),
        default="create_tasks",
        help="How to format language tasks for rollout diagnostics.",
    )
    parser.add_argument(
        "--goal-episode",
        type=str,
        default="",
        help="Raw episode npz used to source goal image (last frame).",
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-fps", type=int, default=5)
    parser.add_argument("--record-dir", type=str, default="codex/logs/videos")
    parser.add_argument(
        "--record-path",
        type=str,
        default="",
        help="Optional explicit mp4 path. If empty, a timestamped path under --record-dir is used.",
    )
    parser.add_argument(
        "--record-all-episodes",
        action="store_true",
        help="Record all episodes into one combined video instead of only episode 0.",
    )
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def _default_summary_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs/json") / f"mustard_intent_octo_official_eval_{ts}.json"


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
    bbox = tf.stack(
        [height_offset, width_offset, height_offset + new_height, width_offset + new_width]
    )
    image = tf.image.crop_and_resize(image[None], bbox[None], [0], (size, size))[0]
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
    return image


def _init_writer(path: Path, fps: int) -> imageio.Writer:
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )


def _resolve_record_path(args: argparse.Namespace) -> Path:
    if args.record_path:
        return Path(args.record_path)
    video_name = f"mustard_intent_octo_official_{args.task}_{datetime.now().strftime('%y%m%d_%H%M%S')}.mp4"
    return Path(args.record_dir) / video_name


def _build_task(model: OctoModel, args: argparse.Namespace, env: MustardIntentGymEnv):
    if args.task_builder == "manual":
        if args.conditioning != "language":
            raise SystemExit("--task-builder manual currently supports language-only rollout.")
        return {"language_instruction": model.text_processor.encode([env.get_instruction()])}
    texts = None
    goals = None
    if args.conditioning in ("language", "language_goal"):
        texts = [env.get_instruction()]
    if args.conditioning in ("goal", "language_goal"):
        if not args.goal_episode:
            raise SystemExit("--goal-episode is required for goal-conditioned rollout.")
        goal_image = MustardIntentGymEnv.goal_image_from_episode(args.goal_episode, which="last")
        goal_image = _resize_goal_like_octo(goal_image, args.policy_image_size)
        goals = {"image_primary": goal_image[None, ...]}
    return model.create_tasks(goals=goals, texts=texts)


class TrainResizeImageWrapper(gym.ObservationWrapper):
    """Resize observations the same way our current fine-tune pipeline does."""

    def __init__(self, env: gym.Env, size: int):
        super().__init__(env)
        spaces = self.observation_space.spaces
        spaces["image_primary"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 3),
            dtype=np.uint8,
        )
        self.size = size
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        image = np.asarray(
            Image.fromarray(observation["image_primary"]).resize(
                (self.size, self.size), Image.BILINEAR
            ),
            dtype=np.uint8,
        )
        observation["image_primary"] = image
        return observation


class AddObservationPadMaskWrapper(gym.ObservationWrapper):
    """Add Octo-style observation pad_mask_dict after history stacking."""

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


def run_eval(args: argparse.Namespace) -> dict:
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

    summary = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "basis_path": str(Path(args.basis_path).expanduser().resolve()),
        "dataset_name": args.dataset_name,
        "task": args.task,
        "conditioning": args.conditioning,
        "wrapper": args.wrapper,
        "window_size": window_size,
        "pred_horizon": pred_horizon,
        "episodes": [],
    }

    combined_writer = None
    combined_record_path = _resolve_record_path(args) if args.record and args.record_all_episodes else None
    if combined_record_path is not None:
        combined_writer = _init_writer(combined_record_path, args.record_fps)

    for ep in range(int(args.episodes)):
        base_env = MustardIntentGymEnv(env_cfg)
        env = NormalizeProprio(base_env, {"proprio": proprio_stats} if proprio_stats is not None else {})
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

        task = _build_task(model, args, base_env)
        obs, _ = env.reset(seed=args.seed + ep)
        writer = None
        if combined_writer is not None:
            writer = combined_writer
            if ep > 0:
                separator = np.zeros_like(base_env.render())
                for _ in range(max(1, int(args.record_fps // 2))):
                    writer.append_data(separator)
            writer.append_data(base_env.render())
        elif args.record and ep == 0:
            writer = _init_writer(_resolve_record_path(args), args.record_fps)
            writer.append_data(base_env.render())

        terminated = False
        truncated = False
        step = 0
        rng = jax.random.PRNGKey(args.seed + ep)
        while not (terminated or truncated):
            step += 1
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
            action = np.asarray(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if writer is not None:
                writer.append_data(base_env.render())
            if step >= int(args.max_policy_steps) + 2:
                truncated = True

        if writer is not None and writer is not combined_writer:
            writer.close()

        ep_summary = base_env.get_episode_summary()
        ep_summary["episode"] = ep
        summary["episodes"].append(ep_summary)
        base_env.close()

    if combined_writer is not None:
        combined_writer.close()

    total = len(summary["episodes"])
    success = sum(int(ep["success"]) for ep in summary["episodes"])
    summary["overall"] = {
        "episodes": total,
        "successes": success,
        "success_rate": float(success / max(total, 1)),
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = run_eval(args)
    out_path = Path(args.save_json) if args.save_json else _default_summary_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
