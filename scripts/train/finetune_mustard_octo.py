#!/usr/bin/env python3
"""Fine-tune Octo-base on mustard OXE dataset (overfit-friendly defaults)."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Octo-base on mustard_grasp_oxe dataset."
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
        default="dataset/mustard_grasp_oxe",
        help="TFDS root directory containing mustard_grasp_oxe.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mustard_grasp_oxe",
        help="TFDS dataset name exported by scripts/data/convert_mustard_raw_to_oxe.py.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/mustard_octo_overfit",
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--shuffle-buffer-size", type=int, default=4000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--freeze-transformer",
        action="store_true",
        help="Freeze transformer blocks in addition to default frozen keys.",
    )
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="Ignore model default frozen_keys and train all params.",
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
        for k, v in tree.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(_flatten_scalar_metrics(v, key))
        return out
    arr = np.asarray(tree)
    if arr.size == 1:
        out[prefix or "metric"] = float(arr.reshape(()))
    return out


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _default_run_name() -> str:
    return datetime.now().strftime("run_%y%m%d_%H%M%S")


def _compute_stats(raw_episode_ds, action_dim: int) -> dict[str, Any]:
    action_sum = None
    action_sq_sum = None
    proprio_sum = None
    proprio_sq_sum = None
    n_steps = 0
    n_eps = 0
    proprio_dim = None

    for episode in raw_episode_ds.as_numpy_iterator():
        steps = episode["steps"]
        action = np.asarray(steps["action"], dtype=np.float32)
        proprio = np.asarray(steps["observation"]["state"], dtype=np.float32)
        if action.ndim != 2 or proprio.ndim != 2:
            continue
        if action.shape[0] == 0:
            continue
        if action.shape[1] != action_dim:
            raise ValueError(
                f"Action dim mismatch: dataset has {action.shape[1]}, but --action-dim={action_dim}."
            )

        if action_sum is None:
            action_sum = np.zeros((action_dim,), dtype=np.float64)
            action_sq_sum = np.zeros((action_dim,), dtype=np.float64)
            proprio_dim = int(proprio.shape[1])
            proprio_sum = np.zeros((proprio_dim,), dtype=np.float64)
            proprio_sq_sum = np.zeros((proprio_dim,), dtype=np.float64)

        action_sum += action.sum(axis=0)
        action_sq_sum += np.square(action).sum(axis=0)
        proprio_sum += proprio.sum(axis=0)
        proprio_sq_sum += np.square(proprio).sum(axis=0)
        n_steps += int(action.shape[0])
        n_eps += 1

    if action_sum is None or proprio_sum is None or n_steps == 0:
        raise ValueError("Dataset is empty. No transitions found for statistics.")

    action_mean = action_sum / n_steps
    action_var = np.maximum(action_sq_sum / n_steps - np.square(action_mean), 1e-12)
    action_std = np.sqrt(action_var)

    proprio_mean = proprio_sum / n_steps
    proprio_var = np.maximum(proprio_sq_sum / n_steps - np.square(proprio_mean), 1e-12)
    proprio_std = np.sqrt(proprio_var)

    return {
        "num_transitions": int(n_steps),
        "num_trajectories": int(n_eps),
        "action": {
            "mean": action_mean.astype(np.float32),
            "std": action_std.astype(np.float32),
            "mask": np.ones((action_dim,), dtype=bool),
        },
        "proprio": {
            "mean": proprio_mean.astype(np.float32),
            "std": proprio_std.astype(np.float32),
            "mask": np.ones((int(proprio_dim),), dtype=bool),
        },
    }


def _make_train_iterator(tf, builder, stats: dict[str, Any], args: argparse.Namespace):
    action_horizon = int(args.action_horizon)
    image_size = int(args.image_size)

    action_mean = tf.constant(stats["action"]["mean"], dtype=tf.float32)
    action_std = tf.constant(stats["action"]["std"], dtype=tf.float32)
    proprio_mean = tf.constant(stats["proprio"]["mean"], dtype=tf.float32)
    proprio_std = tf.constant(stats["proprio"]["std"], dtype=tf.float32)

    raw_ds = builder.as_dataset(split="train", shuffle_files=True)

    def episode_to_samples(ep):
        steps = ep["steps"]
        images = tf.cast(
            tf.image.resize(steps["observation"]["image_primary"], [image_size, image_size]),
            tf.uint8,
        )
        proprio = tf.cast(steps["observation"]["state"], tf.float32)
        action = tf.cast(steps["action"], tf.float32)
        language = ep["language_instruction"]

        action = (action - action_mean) / action_std
        proprio = (proprio - proprio_mean) / proprio_std

        total = tf.shape(action)[0] - action_horizon + 1
        total = tf.maximum(total, 0)
        idxs = tf.range(total)

        def make_sample(i):
            action_chunk = action[i : i + action_horizon]  # [horizon, action_dim]
            action_chunk = tf.expand_dims(action_chunk, axis=0)  # [window=1, horizon, action_dim]
            action_pad_mask = tf.ones_like(action_chunk, dtype=tf.bool)
            return {
                "observation": {
                    "image_primary": tf.expand_dims(images[i], axis=0),
                    "proprio": tf.expand_dims(proprio[i], axis=0),
                    "timestep_pad_mask": tf.ones((1,), dtype=tf.bool),
                },
                "action": action_chunk,
                "action_pad_mask": action_pad_mask,
                "task": {
                    "language_instruction": language,
                },
            }

        return tf.data.Dataset.from_tensor_slices(idxs).map(
            make_sample,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = raw_ds.flat_map(episode_to_samples)
    if args.shuffle_buffer_size > 0:
        train_ds = train_ds.shuffle(args.shuffle_buffer_size)
    train_ds = train_ds.repeat().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds.as_numpy_iterator()


def main() -> None:
    args = parse_args()
    if args.window_size != 1:
        raise SystemExit("This script currently supports --window-size 1 only.")
    if args.action_horizon <= 0:
        raise SystemExit("--action-horizon must be >= 1.")

    try:
        import jax
        import optax
        import tensorflow as tf
        import tensorflow_datasets as tfds

        from octo.model.components.action_heads import L1ActionHead
        from octo.model.components.tokenizers import LowdimObsTokenizer
        from octo.model.octo_model import OctoModel
        from octo.utils.jax_utils import initialize_compilation_cache
        from octo.utils.spec import ModuleSpec
        from octo.utils.train_utils import (
            TrainState,
            freeze_weights,
            merge_params,
            process_text,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing Octo/JAX dependencies. "
            "Use an environment with octo+jax installed (e.g., conda activate octoketi)."
        ) from exc

    initialize_compilation_cache()
    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    device_count = jax.device_count()
    if args.batch_size % device_count != 0:
        raise SystemExit(
            f"batch-size ({args.batch_size}) must be divisible by device count ({device_count})."
        )

    save_root = Path(args.save_dir).expanduser().resolve()
    run_dir = save_root / _default_run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"[setup] devices={device_count} batch={args.batch_size}")
    print(f"[setup] save_dir={run_dir}")

    builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)
    print("[data] computing normalization stats...")
    raw_for_stats = builder.as_dataset(split="train", shuffle_files=False)
    dataset_stats_single = _compute_stats(raw_for_stats, action_dim=args.action_dim)
    dataset_statistics = {args.dataset_name: dataset_stats_single}
    print(
        f"[data] transitions={dataset_stats_single['num_transitions']} "
        f"episodes={dataset_stats_single['num_trajectories']}"
    )

    print("[data] building train iterator...")
    train_iter = _make_train_iterator(tf, builder, dataset_stats_single, args)

    print("[load] pretrained model...")
    pretrained_model = OctoModel.load_pretrained(args.pretrained_path)
    text_processor = pretrained_model.text_processor

    def process_batch_fn(batch: dict[str, Any]) -> dict[str, Any]:
        return process_text(batch, text_processor)

    train_iter = map(process_batch_fn, train_iter)
    example_batch = next(train_iter)

    print("[model] adapting config for single image + state + custom action head...")
    config = pretrained_model.config
    obs_tokenizers = config["model"]["observation_tokenizers"]
    if "wrist" in obs_tokenizers:
        del obs_tokenizers["wrist"]
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
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        readout_key="readout_action",
    )

    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=False,
        dataset_statistics=dataset_statistics,
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
    if args.unfreeze_all:
        frozen_keys: list[str] = []
    else:
        frozen_keys = list(model.config["optimizer"].get("frozen_keys", []))
    if args.freeze_transformer and "BlockTransformer_0" not in frozen_keys:
        frozen_keys.append("BlockTransformer_0")
    if frozen_keys:
        tx = freeze_weights(tx, model.params, frozen_keys)
    print(f"[optim] lr={args.learning_rate} warmup={args.warmup_steps} frozen={frozen_keys}")

    train_state = TrainState.create(
        rng=jax.random.PRNGKey(args.seed),
        model=model,
        tx=tx,
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

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, True
        )
        state = state.apply_gradients(grads=grads, rng=rng)
        return state, loss, metrics

    print("[train] start")
    loss_trace: list[dict[str, float]] = []
    for step in range(1, args.num_steps + 1):
        batch = next(train_iter)
        train_state, loss, metrics = train_step(train_state, batch)
        loss_value = float(np.asarray(jax.device_get(loss)))

        if step % args.log_every == 0 or step == 1:
            metrics_dict = _flatten_scalar_metrics(jax.device_get(metrics))
            preview_items = []
            for k in sorted(metrics_dict.keys()):
                if k.lower().endswith("loss") or "mse" in k.lower() or "mae" in k.lower():
                    preview_items.append(f"{k}={metrics_dict[k]:.5f}")
                if len(preview_items) >= 3:
                    break
            preview = " ".join(preview_items)
            print(
                f"[train] step={step:06d}/{args.num_steps} loss={loss_value:.6f}"
                + (f" {preview}" if preview else "")
            )

        if step % args.save_every == 0:
            ckpt_dir = run_dir / f"checkpoint_{step:06d}"
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
            train_state.model.save_pretrained(step=step, checkpoint_path=str(ckpt_dir))
            print(f"[save] {ckpt_dir}")

        loss_trace.append({"step": float(step), "loss": loss_value})

    final_dir = run_dir / "final_model"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    train_state.model.save_pretrained(step=args.num_steps, checkpoint_path=str(final_dir))
    print(f"[done] final_model={final_dir}")

    with (run_dir / "loss_trace.json").open("w", encoding="utf-8") as f:
        json.dump(loss_trace, f, indent=2)
    with (run_dir / "dataset_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonify(dataset_statistics), f, indent=2)
    print(f"[done] logs={run_dir}")


if __name__ == "__main__":
    main()
