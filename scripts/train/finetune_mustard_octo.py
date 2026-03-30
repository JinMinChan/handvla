#!/usr/bin/env python3
"""Fine-tune Octo-base on mustard RLDS using the official Octo data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
import json
import os
import shutil
import subprocess
import traceback
from functools import partial
from typing import Any

import numpy as np

from octo_data.mustard import (
    make_mustard_dataset_kwargs,
    make_mustard_goal_trajectory_kwargs,
)

_DEFAULT_OCTO_SRC = "/home/minchan/Downloads/dual_arm_VLA/octo"


def _maybe_add_octo_src() -> None:
    override_path = (
        (Path.cwd() / ".octo_src_path").read_text(encoding="utf-8").strip()
        if (Path.cwd() / ".octo_src_path").exists()
        else ""
    )
    for candidate in [override_path, _DEFAULT_OCTO_SRC]:
        if not candidate:
            continue
        src = Path(candidate).expanduser().resolve()
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Octo-base on mustard RLDS using Octo's official dataset path: "
            "make_single_dataset -> trajectory transforms -> frame transforms."
        )
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="hf://rail-berkeley/octo-base-1.5",
        help="Pretrained Octo checkpoint path or HF id.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/mustard_intent_wrap_shortsettle_disturb_k4_20260317_oxe",
        help="TFDS root directory containing the official-RLDS mustard export.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mustard_grasp_oxe",
        help="TFDS dataset name.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/mustard_octo_official_wrap_k4",
        help="Directory for best models and traces.",
    )
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5000,
        help="Maximum number of optimization steps. Use <=0 to stop only by plateau.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1024,
        help="Frame-level shuffle buffer after trajectory transforms.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--val-patience", type=int, default=8)
    parser.add_argument("--val-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--train-check-every",
        type=int,
        default=50,
        help="Check train-plateau only every N steps instead of every step.",
    )
    parser.add_argument("--train-patience", type=int, default=20)
    parser.add_argument("--train-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--min-steps-before-stop",
        type=int,
        default=1000,
        help="Do not apply train-plateau early stop before this many steps.",
    )
    parser.add_argument("--rollout-every", type=int, default=1000)
    parser.add_argument("--rollout-min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--rollout-eval-script",
        type=str,
        default="scripts/eval/rollout_mustard_intent_octo_official.py",
        help="Official-style rollout evaluator used for best_rollout_model selection.",
    )
    parser.add_argument(
        "--rollout-basis-path",
        type=str,
        default="dataset/mustard_intent_wrap_shortsettle_disturb_k4_basis_20260317/mustard_intent_hand_pca_k4.npz",
        help="Synergy basis required by rollout evaluator.",
    )
    parser.add_argument("--rollout-task", type=str, default="wrap_and_lift")
    parser.add_argument("--rollout-episodes", type=int, default=1)
    parser.add_argument("--rollout-policy-repeat", type=int, default=20)
    parser.add_argument("--rollout-max-policy-steps", type=int, default=60)
    parser.add_argument(
        "--rollout-wrapper",
        type=str,
        default="temporal",
        choices=("temporal", "rhc"),
        help="Wrapper used for training-time rollout probes.",
    )
    parser.add_argument(
        "--rollout-exec-horizon",
        type=int,
        default=2,
        help="Execution horizon used only when rollout-wrapper=rhc.",
    )
    parser.add_argument(
        "--rollout-history-mode",
        type=str,
        default="per_env_step",
        help="History buffering mode passed to the rollout evaluator.",
    )
    parser.add_argument(
        "--rollout-conditioning",
        type=str,
        default="language_goal",
        help="Conditioning mode passed to the rollout evaluator.",
    )
    parser.add_argument(
        "--rollout-resize-mode",
        type=str,
        default="octo_avg_crop",
        help="Image resize mode passed to the rollout evaluator.",
    )
    parser.add_argument(
        "--rollout-task-builder",
        type=str,
        default="create_tasks",
        help="Task-builder name passed to the rollout evaluator.",
    )
    parser.add_argument(
        "--rollout-spawn-jitter-xy",
        type=float,
        default=0.0,
        help="Optional object XY jitter used for rollout selection probes.",
    )
    parser.add_argument(
        "--rollout-spawn-yaw-jitter-deg",
        type=float,
        default=0.0,
        help="Optional object yaw jitter used for rollout selection probes.",
    )
    parser.add_argument(
        "--rollout-goal-episode",
        type=str,
        default="dataset/mustard_intent_wrap_shortsettle_disturb_k4_20260317/raw/episode_00000.npz",
        help="Raw episode npz used to source the rollout goal image (last frame).",
    )
    parser.add_argument(
        "--disable-jit",
        action="store_true",
        help="Disable JAX JIT for debugging.",
    )
    return parser.parse_args()


def _flatten_scalar_metrics(tree: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flatten_scalar_metrics(value, child))
        return out
    arr = np.asarray(tree)
    if arr.size == 1:
        out[prefix or "metric"] = float(arr.reshape(()))
    return out


def _image_tokenizer_supported(tokenizer_spec: dict[str, Any], example_batch: dict[str, Any]) -> bool:
    """Keep pretrained image tokenizers only when their required keys exist.

    Octo names image tokenizers by camera alias (`primary`, `wrist`), not by the
    underlying observation key (`image_primary`, `image_wrist`). Filtering by
    tokenizer *name* drops valid image paths. We instead inspect the tokenizer's
    declared obs/task stack keys.
    """
    kwargs = tokenizer_spec.get("kwargs", {})
    obs_keys = list(kwargs.get("obs_stack_keys", []))
    task_keys = list(kwargs.get("task_stack_keys", []))
    return all(k in example_batch["observation"] for k in obs_keys) and all(
        k in example_batch["task"] for k in task_keys
    )


def _default_run_name() -> str:
    return datetime.now().strftime("run_%y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _episode_rollout_aux_score(ep: dict[str, Any]) -> float:
    success = 1.0 if bool(ep.get("success", False)) else 0.0
    reached = 1.0 if bool(ep.get("reached", False)) else 0.0
    best_contacts = float(ep.get("best_contacts", 0.0))
    approach_err = float(ep.get("approach_min_err", 1e6))
    approach_rot = float(ep.get("approach_min_rot_err_deg", 1e6))
    object_dz = max(0.0, float(ep.get("object_dz_max", 0.0)))
    proximity = max(0.0, 1.0 - min(approach_err, 1.0))
    rot_align = max(0.0, 1.0 - min(approach_rot / 180.0, 1.0))
    return (
        success * 1_000_000.0
        + object_dz * 10_000.0
        + best_contacts * 100.0
        + reached * 50.0
        + proximity * 10.0
        + rot_align
    )


def _run_rollout_eval(args: argparse.Namespace, run_dir: Path, model_dir: Path, step: int) -> dict[str, float] | None:
    if args.rollout_every <= 0 or not args.rollout_basis_path:
        return None

    eval_script = Path(args.rollout_eval_script).expanduser().resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"Rollout evaluator not found: {eval_script}")

    save_json = run_dir / f"rollout_eval_step_{step:06d}.json"
    cmd = [
        sys.executable,
        str(eval_script),
        "--model-path",
        str(model_dir),
        "--basis-path",
        str(Path(args.rollout_basis_path).expanduser().resolve()),
        "--task",
        args.rollout_task,
        "--episodes",
        str(int(args.rollout_episodes)),
        "--policy-repeat",
        str(int(args.rollout_policy_repeat)),
        "--max-policy-steps",
        str(int(args.rollout_max_policy_steps)),
        "--wrapper",
        str(args.rollout_wrapper),
        "--history-mode",
        str(args.rollout_history_mode),
        "--conditioning",
        str(args.rollout_conditioning),
        "--resize-mode",
        str(args.rollout_resize_mode),
        "--task-builder",
        str(args.rollout_task_builder),
        "--goal-episode",
        str(Path(args.rollout_goal_episode).expanduser().resolve()),
        "--save-json",
        str(save_json),
    ]
    if str(args.rollout_wrapper) == "rhc":
        cmd.extend(["--exec-horizon", str(int(args.rollout_exec_horizon))])
    if float(args.rollout_spawn_jitter_xy) > 0.0:
        cmd.extend(["--spawn-jitter-xy", str(float(args.rollout_spawn_jitter_xy))])
    if float(args.rollout_spawn_yaw_jitter_deg) > 0.0:
        cmd.extend(
            [
                "--spawn-yaw-jitter-deg",
                str(float(args.rollout_spawn_yaw_jitter_deg)),
            ]
        )
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Rollout eval failed.\n"
            + (f"stdout:\n{result.stdout}\n" if result.stdout else "")
            + (f"stderr:\n{result.stderr}" if result.stderr else "")
        )
    summary = json.loads(save_json.read_text(encoding="utf-8"))
    episodes = summary.get("episodes", [])
    overall = summary.get("overall", {})
    score = (
        float(np.mean([_episode_rollout_aux_score(ep) for ep in episodes]))
        if episodes
        else float(overall.get("success_rate", 0.0))
    )
    return {
        "success_rate": float(overall.get("success_rate", 0.0)),
        "episodes": float(overall.get("episodes", 0.0)),
        "successes": float(overall.get("successes", 0.0)),
        "score": score,
        "json_path": str(save_json),
    }


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise SystemExit("--window-size must be >= 1.")
    if args.action_horizon <= 0:
        raise SystemExit("--action-horizon must be >= 1.")

    _maybe_add_octo_src()

    try:
        import jax
        from jax.experimental import multihost_utils
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
        import optax
        import tensorflow as tf

        from octo.data.dataset import make_single_dataset
        from octo.model.components.action_heads import L1ActionHead
        from octo.model.components.tokenizers import LowdimObsTokenizer
        from octo.model.octo_model import OctoModel
        from octo.utils.spec import ModuleSpec
        from octo.utils.train_utils import TrainState, merge_params, process_text
    except ImportError as exc:
        raise SystemExit(
            "Missing Octo/JAX dependencies. Use an environment with octo+jax installed "
            "(e.g. conda activate octoketi)."
        ) from exc

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    devices = jax.devices()
    device_count = len(devices)
    if args.batch_size % device_count != 0:
        raise SystemExit(
            f"batch-size ({args.batch_size}) must be divisible by device count ({device_count})."
        )
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def shard(batch: dict[str, Any]) -> Any:
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

    save_root = Path(args.save_dir).expanduser().resolve()
    run_dir = save_root / _default_run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_args.json", vars(args))

    status_path = run_dir / "run_status.json"
    latest_metrics_path = run_dir / "latest_metrics.json"

    def update_status(
        *,
        state: str,
        step: int = 0,
        message: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "state": state,
            "pid": os.getpid(),
            "run_dir": str(run_dir),
            "updated_at": datetime.now().isoformat(),
            "step": int(step),
            "message": message,
        }
        if extra:
            payload.update(extra)
        _write_json(status_path, payload)

    update_status(state="starting", message="initializing official Octo training")

    print(
        f"[setup] devices={device_count} batch={args.batch_size} "
        f"({args.batch_size // device_count} per device)"
    )
    print(f"[setup] save_dir={run_dir}")

    dataset_kwargs = make_mustard_dataset_kwargs(
        name=args.dataset_name,
        data_dir=args.data_dir,
        load_camera_views=("primary", "goal"),
        load_proprio=True,
        load_language=True,
    )
    traj_transform_kwargs = {
        "window_size": int(args.window_size),
        "action_horizon": int(args.action_horizon),
        **make_mustard_goal_trajectory_kwargs(),
    }
    frame_transform_kwargs = {
        "resize_size": {"primary": (int(args.image_size), int(args.image_size))}
    }

    print("[data] loading train dataset through make_single_dataset(...)")
    train_dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=True,
    )
    print("[data] loading val dataset through make_single_dataset(...)")
    val_dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=False,
    )

    text_processor = None
    print("[load] pretrained octo-base...")
    pretrained_model = OctoModel.load_pretrained(args.pretrained_path)
    text_processor = pretrained_model.text_processor

    def process_batch_fn(batch: dict[str, Any]) -> dict[str, Any]:
        batch = process_text(batch, text_processor)
        if "dataset_name" in batch:
            del batch["dataset_name"]
        return batch

    train_iter = (
        train_dataset.repeat()
        .unbatch()
        .shuffle(int(args.shuffle_buffer_size))
        .batch(int(args.batch_size), drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .iterator()
    )
    train_iter = map(lambda batch: shard(process_batch_fn(batch)), train_iter)
    example_batch = process_batch_fn(
        next(
            train_dataset.repeat()
            .unbatch()
            .shuffle(int(args.shuffle_buffer_size))
            .batch(int(args.batch_size), drop_remainder=True)
            .take(1)
            .iterator()
        )
    )

    val_iter_factory = lambda: map(
        lambda batch: shard(process_batch_fn(batch)),
        val_dataset.unbatch()
        .batch(int(args.batch_size), drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .iterator(),
    )

    print("[model] adapting octo-base to primary-image + proprio + action-10")
    config = pretrained_model.config
    config["window_size"] = int(args.window_size)

    obs_tokenizers = config["model"]["observation_tokenizers"]
    for key in list(obs_tokenizers.keys()):
        if not _image_tokenizer_supported(obs_tokenizers[key], example_batch):
            del obs_tokenizers[key]
    obs_tokenizers["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=int(args.action_horizon),
        action_dim=int(args.action_dim),
        readout_key="readout_action",
    )

    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=False,
        dataset_statistics=train_dataset.dataset_statistics,
    )
    model = model.replace(params=merge_params(model.params, pretrained_model.params))
    del pretrained_model

    if args.warmup_steps > 0:
        lr_schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, args.learning_rate, args.warmup_steps),
                optax.constant_schedule(args.learning_rate),
            ],
            [args.warmup_steps],
        )
    else:
        lr_schedule = optax.constant_schedule(args.learning_rate)

    tx = optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(args.seed),
        model=model,
        tx=tx,
    )
    train_state = jax.device_put(train_state, replicated_sharding)
    print(
        f"[optim] full finetune lr={args.learning_rate} warmup={args.warmup_steps} "
        f"window={args.window_size} horizon={args.action_horizon}"
    )

    def loss_fn(params, batch, dropout_rng, train: bool = True):
        bound_module = train_state.model.module.bind(
            {"params": params}, rngs={"dropout": dropout_rng}
        )
        embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
        out_shardings=(replicated_sharding, replicated_sharding, replicated_sharding),
    )
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, True
        )
        state = state.apply_gradients(grads=grads, rng=rng)
        return state, loss, metrics

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
        out_shardings=(replicated_sharding, replicated_sharding),
    )
    def eval_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        loss, metrics = loss_fn(state.model.params, batch, dropout_rng, False)
        return loss, metrics

    def host_model_for_save(state: TrainState):
        host_params = jax.device_get(state.model.params)
        return state.model.replace(params=host_params)

    def evaluate(state) -> dict[str, float]:
        losses: list[float] = []
        metric_buckets: dict[str, list[float]] = {}
        batch_count = 0
        for batch in val_iter_factory():
            loss, metrics = eval_step(state, batch)
            losses.append(float(np.asarray(jax.device_get(loss))))
            flat = _flatten_scalar_metrics(jax.device_get(metrics))
            for key, value in flat.items():
                metric_buckets.setdefault(key, []).append(float(value))
            batch_count += 1
            if args.eval_max_batches > 0 and batch_count >= args.eval_max_batches:
                break
        if not losses:
            return {}
        out = {
            "loss": float(np.mean(losses)),
            "num_batches": float(batch_count),
        }
        for key, values in metric_buckets.items():
            out[key] = float(np.mean(values))
        return out

    print("[train] start")
    loss_trace: list[dict[str, float]] = []
    eval_trace: list[dict[str, float]] = []
    rollout_trace: list[dict[str, float]] = []

    best_train_loss = np.inf
    best_train_step = -1
    best_val_loss = np.inf
    best_val_step = -1
    best_rollout_score = -np.inf
    best_rollout_step = -1
    stale_train_checks = 0
    last_checked_best_train_step = -1
    stale_val_checks = 0
    stopped_early = False

    max_steps = int(args.num_steps) if int(args.num_steps) > 0 else None
    step = 0
    try:
        while True:
            step += 1
            if max_steps is not None and step > max_steps:
                step -= 1
                break

            batch = next(train_iter)
            train_state, loss, metrics = train_step(train_state, batch)
            loss_value = float(np.asarray(jax.device_get(loss)))
            loss_trace.append({"step": float(step), "loss": loss_value})

            if step % int(args.log_every) == 0 or step == 1:
                metrics_dict = _flatten_scalar_metrics(jax.device_get(metrics))
                preview = " ".join(
                    f"{k}={v:.5f}"
                    for k, v in list(sorted(metrics_dict.items()))[:3]
                )
                print(f"[train] step={step:06d} loss={loss_value:.6f}" + (f" {preview}" if preview else ""))
                _write_json(
                    latest_metrics_path,
                    {
                        "step": int(step),
                        "train_loss": float(loss_value),
                        "metrics_preview": metrics_dict,
                        "updated_at": datetime.now().isoformat(),
                    },
                )
                update_status(
                    state="running",
                    step=step,
                    message="training",
                    extra={
                        "best_train_step": int(best_train_step),
                        "best_train_loss": float(best_train_loss) if np.isfinite(best_train_loss) else None,
                        "best_val_step": int(best_val_step),
                        "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                        "best_rollout_step": int(best_rollout_step),
                        "best_rollout_score": float(best_rollout_score) if best_rollout_step >= 0 else None,
                    },
                )

            improved_train = loss_value + float(args.train_min_delta) < best_train_loss
            if improved_train:
                best_train_loss = loss_value
                best_train_step = step
                stale_train_checks = 0
            if (
                args.train_check_every > 0
                and step % int(args.train_check_every) == 0
            ):
                if best_train_step > last_checked_best_train_step:
                    stale_train_checks = 0
                    last_checked_best_train_step = best_train_step
                else:
                    stale_train_checks += 1

            if args.eval_every > 0 and step % int(args.eval_every) == 0:
                val_metrics = evaluate(train_state)
                if val_metrics:
                    val_metrics["step"] = float(step)
                    eval_trace.append(val_metrics)
                    print(f"[eval] step={step} val_loss={val_metrics['loss']:.6f}")
                    improved_val = val_metrics["loss"] + float(args.val_min_delta) < best_val_loss
                    if improved_val:
                        best_val_loss = float(val_metrics["loss"])
                        best_val_step = step
                        stale_val_checks = 0
                        best_dir = run_dir / "best_val_model"
                        if best_dir.exists():
                            shutil.rmtree(best_dir)
                        host_model_for_save(train_state).save_pretrained(step=step, checkpoint_path=str(best_dir))
                        _write_json(
                            run_dir / "best_val_model_info.json",
                            {
                                "step": int(step),
                                "val_loss": float(val_metrics["loss"]),
                                "saved_at": datetime.now().isoformat(),
                            },
                        )
                        print(f"[save] best_val_model step={step} val_loss={val_metrics['loss']:.6f}")
                    else:
                        stale_val_checks += 1
                    _write_json(
                        latest_metrics_path,
                        {
                            "step": int(step),
                            "train_loss": float(loss_value),
                            "val_metrics": val_metrics,
                            "best_val_step": int(best_val_step),
                            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                            "updated_at": datetime.now().isoformat(),
                        },
                    )

            if args.rollout_every > 0 and step % int(args.rollout_every) == 0:
                candidate_dir = run_dir / f"_tmp_rollout_step_{step:06d}"
                if candidate_dir.exists():
                    shutil.rmtree(candidate_dir)
                host_model_for_save(train_state).save_pretrained(step=step, checkpoint_path=str(candidate_dir))
                rollout_metrics = _run_rollout_eval(args, run_dir, candidate_dir, step)
                shutil.rmtree(candidate_dir, ignore_errors=True)
                if rollout_metrics is not None:
                    rollout_metrics["step"] = float(step)
                    rollout_trace.append(rollout_metrics)
                    print(
                        f"[rollout] step={step} success_rate={rollout_metrics['success_rate']:.3f} "
                        f"score={rollout_metrics['score']:.3f}"
                    )
                    if rollout_metrics["score"] > best_rollout_score + float(args.rollout_min_delta):
                        best_rollout_score = float(rollout_metrics["score"])
                        best_rollout_step = step
                        best_dir = run_dir / "best_rollout_model"
                        if best_dir.exists():
                            shutil.rmtree(best_dir)
                        host_model_for_save(train_state).save_pretrained(step=step, checkpoint_path=str(best_dir))
                        _write_json(
                            run_dir / "best_rollout_model_info.json",
                            {
                                "step": int(step),
                                "score": float(rollout_metrics["score"]),
                                "success_rate": float(rollout_metrics["success_rate"]),
                                "json_path": rollout_metrics["json_path"],
                                "saved_at": datetime.now().isoformat(),
                            },
                        )
                        print(
                            f"[save] best_rollout_model step={step} "
                            f"success_rate={rollout_metrics['success_rate']:.3f}"
                        )

            if (
                step >= int(args.min_steps_before_stop)
                and args.eval_every > 0
                and stale_val_checks >= int(args.val_patience)
            ):
                stopped_early = True
                step -= 0
                print(
                    f"[early-stop] no val improvement for {stale_val_checks} "
                    f"eval checks (every {args.eval_every} steps); stopping."
                )
                break
            if (
                step >= int(args.min_steps_before_stop)
                and args.eval_every <= 0
                and stale_train_checks >= int(args.train_patience)
            ):
                stopped_early = True
                print(
                    f"[early-stop] no train improvement for {stale_train_checks} "
                    f"train checks (every {args.train_check_every} steps); stopping."
                )
                break

        summary = {
            "finished": True,
            "num_steps_ran": int(step),
            "stopped_early": bool(stopped_early),
            "best_train_step": int(best_train_step),
            "best_train_loss": float(best_train_loss) if np.isfinite(best_train_loss) else None,
            "best_val_step": int(best_val_step),
            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
            "best_rollout_step": int(best_rollout_step),
            "best_rollout_score": float(best_rollout_score) if best_rollout_step >= 0 else None,
            "created_at": datetime.now().isoformat(),
        }
        _write_json(run_dir / "loss_trace.json", loss_trace)
        _write_json(run_dir / "eval_trace.json", eval_trace)
        _write_json(run_dir / "rollout_trace.json", rollout_trace)
        _write_json(run_dir / "train_summary.json", summary)
        update_status(
            state="finished",
            step=step,
            message="training finished",
            extra=summary,
        )
        print(f"[done] run_dir={run_dir}")
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        failure = {
            "finished": False,
            "num_steps_ran": int(step),
            "stopped_early": False,
            "best_train_step": int(best_train_step),
            "best_train_loss": float(best_train_loss) if np.isfinite(best_train_loss) else None,
            "best_val_step": int(best_val_step),
            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
            "best_rollout_step": int(best_rollout_step),
            "best_rollout_score": float(best_rollout_score) if best_rollout_step >= 0 else None,
            "failed_at": datetime.now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(run_dir / "loss_trace.json", loss_trace)
        _write_json(run_dir / "eval_trace.json", eval_trace)
        _write_json(run_dir / "rollout_trace.json", rollout_trace)
        _write_json(run_dir / "train_summary.json", failure)
        update_status(
            state="failed",
            step=step,
            message="training failed",
            extra={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "best_train_step": int(best_train_step),
                "best_train_loss": float(best_train_loss) if np.isfinite(best_train_loss) else None,
                "best_val_step": int(best_val_step),
                "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                "best_rollout_step": int(best_rollout_step),
                "best_rollout_score": float(best_rollout_score) if best_rollout_step >= 0 else None,
            },
        )
        raise


if __name__ == "__main__":
    main()
