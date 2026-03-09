#!/usr/bin/env python3
"""Fine-tune Octo-base on Franka pick-and-lift arm-TCP + hand OXE dataset."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
import json
import shutil
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Octo-base on pick-and-lift arm_tcp6+hand_synergy_k dataset with "
            "validation-based early stopping and best-only checkpoint saving."
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
        default="dataset/franka_pickandlift_arm_tcp_hand_synergy_k4_owrap20_oxe_20260306",
        help="TFDS root directory.",
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
        default="models/pickandlift_arm_tcp_hand_octo_overfit_k4_owrap20_20260306",
        help="Directory for the best model and logs.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument(
        "--hand-action-start",
        type=int,
        default=6,
        help="First action index belonging to the hand branch.",
    )
    parser.add_argument(
        "--arm-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for arm action dimensions [0:hand-action-start).",
    )
    parser.add_argument(
        "--hand-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for hand action dimensions [hand-action-start:action_dim).",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--shuffle-buffer-size", type=int, default=4000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        default=True,
        help="Overwrite only the best_model directory and skip intermediate checkpoints.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        dest="save_best_only",
        action="store_false",
        help="Keep periodic checkpoints and final_model in addition to best_model.",
    )
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


def _weighted_continuous_metrics(
    pred: Any,
    target: Any,
    timestep_pad_mask: Any,
    action_pad_mask: Any,
    *,
    action_dim: int,
    hand_action_start: int,
    arm_loss_weight: float,
    hand_loss_weight: float,
    loss_type: str,
) -> tuple[Any, dict[str, Any]]:
    import jax.numpy as jnp

    mask = timestep_pad_mask[:, :, None, None] & action_pad_mask
    weights = jnp.ones((1, 1, 1, action_dim), dtype=jnp.float32) * float(hand_loss_weight)
    split = int(np.clip(hand_action_start, 0, action_dim))
    if split > 0:
        weights = weights.at[..., :split].set(float(arm_loss_weight))
    weighted_mask = mask.astype(jnp.float32) * weights

    if loss_type == "l1":
        err = jnp.abs(pred - target)
    elif loss_type == "mse":
        err = jnp.square(pred - target)
    else:
        raise ValueError(f"Unsupported loss type for weighted regression: {loss_type}")

    sq_err = jnp.square(pred - target)
    denom = jnp.clip(jnp.mean(weighted_mask), a_min=1e-5, a_max=None)
    loss = jnp.mean(err * weighted_mask) / denom
    mse = jnp.mean(sq_err * weighted_mask) / denom

    arm_mask = mask[..., :split]
    hand_mask = mask[..., split:]

    def _masked_mean(x, m):
        if x.shape[-1] == 0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        m = m.astype(jnp.float32)
        denom_local = jnp.clip(jnp.mean(m), a_min=1e-5, a_max=None)
        return jnp.mean(x * m) / denom_local

    arm_l1 = _masked_mean(jnp.abs(pred[..., :split] - target[..., :split]), arm_mask)
    hand_l1 = _masked_mean(jnp.abs(pred[..., split:] - target[..., split:]), hand_mask)
    arm_mse = _masked_mean(jnp.square(pred[..., :split] - target[..., :split]), arm_mask)
    hand_mse = _masked_mean(jnp.square(pred[..., split:] - target[..., split:]), hand_mask)

    loss = loss * action_dim
    mse = mse * action_dim
    return loss, {
        "loss": loss,
        "mse": mse,
        "arm_l1": arm_l1 * max(split, 1),
        "hand_l1": hand_l1 * max(action_dim - split, 1),
        "arm_mse": arm_mse * max(split, 1),
        "hand_mse": hand_mse * max(action_dim - split, 1),
    }


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


def _split_episode_dataset(tf, raw_ds, total_episodes: int, val_fraction: float):
    if total_episodes <= 1 or val_fraction <= 0.0:
        empty_ds = raw_ds.enumerate().filter(lambda i, _: i < 0).map(lambda _, ep: ep)
        return raw_ds, empty_ds, total_episodes, 0

    val_count = int(round(total_episodes * val_fraction))
    val_count = min(max(val_count, 1), total_episodes - 1)
    split_idx = total_episodes - val_count

    enumerated = raw_ds.enumerate()
    train_eps = enumerated.filter(lambda i, _: i < split_idx).map(lambda _, ep: ep)
    val_eps = enumerated.filter(lambda i, _: i >= split_idx).map(lambda _, ep: ep)
    return train_eps, val_eps, split_idx, val_count


def _make_sample_dataset(tf, raw_ds, stats: dict[str, Any], args: argparse.Namespace, *, shuffle: bool, repeat: bool):
    window_size = int(args.window_size)
    action_horizon = int(args.action_horizon)
    image_size = int(args.image_size)

    action_mean = tf.constant(stats["action"]["mean"], dtype=tf.float32)
    action_std = tf.constant(stats["action"]["std"], dtype=tf.float32)
    proprio_mean = tf.constant(stats["proprio"]["mean"], dtype=tf.float32)
    proprio_std = tf.constant(stats["proprio"]["std"], dtype=tf.float32)

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

        traj_len = tf.shape(action)[0]
        idxs = tf.range(traj_len)

        def make_sample(i):
            history_offsets = tf.range(-window_size + 1, 1)
            history_indices = i + history_offsets
            timestep_pad_mask = history_indices >= 0
            history_indices = tf.maximum(history_indices, 0)

            obs_images = tf.gather(images, history_indices)
            obs_proprio = tf.gather(proprio, history_indices)

            horizon_offsets = tf.range(action_horizon)[None, :]
            action_indices = history_indices[:, None] + horizon_offsets
            action_valid = action_indices < traj_len
            action_indices = tf.minimum(action_indices, traj_len - 1)
            action_chunk = tf.gather(action, action_indices)
            action_pad_mask = tf.logical_and(timestep_pad_mask[:, None], action_valid)
            action_pad_mask = tf.broadcast_to(
                action_pad_mask[:, :, None],
                tf.shape(action_chunk),
            )
            return {
                "observation": {
                    "image_primary": obs_images,
                    "proprio": obs_proprio,
                    "timestep_pad_mask": timestep_pad_mask,
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

    ds = raw_ds.flat_map(episode_to_samples)
    if shuffle and args.shuffle_buffer_size > 0:
        ds = ds.shuffle(args.shuffle_buffer_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise SystemExit("--window-size must be >= 1.")
    if args.action_horizon <= 0:
        raise SystemExit("--action-horizon must be >= 1.")
    if args.hand_action_start < 0 or args.hand_action_start > args.action_dim:
        raise SystemExit("--hand-action-start must be in [0, action-dim].")

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
    total_episodes = int(builder.info.splits["train"].num_examples)
    if total_episodes <= 0:
        raise SystemExit("Dataset has zero episodes.")

    print("[data] computing normalization stats...")
    raw_for_stats = builder.as_dataset(split="train", shuffle_files=False)
    dataset_stats_single = _compute_stats(raw_for_stats, action_dim=args.action_dim)
    dataset_statistics = {args.dataset_name: dataset_stats_single}
    print(
        f"[data] transitions={dataset_stats_single['num_transitions']} "
        f"episodes={dataset_stats_single['num_trajectories']}"
    )

    raw_all = builder.as_dataset(split="train", shuffle_files=False)
    train_eps, val_eps, train_count, val_count = _split_episode_dataset(
        tf,
        raw_all,
        total_episodes=total_episodes,
        val_fraction=float(args.val_fraction),
    )
    print(f"[data] split train_episodes={train_count} val_episodes={val_count}")

    train_ds = _make_sample_dataset(tf, train_eps, dataset_stats_single, args, shuffle=True, repeat=True)
    val_ds = _make_sample_dataset(tf, val_eps, dataset_stats_single, args, shuffle=False, repeat=False)

    print("[load] pretrained model...")
    pretrained_model = OctoModel.load_pretrained(args.pretrained_path)
    text_processor = pretrained_model.text_processor

    def process_batch_fn(batch: dict[str, Any]) -> dict[str, Any]:
        return process_text(batch, text_processor)

    train_iter = map(process_batch_fn, train_ds.as_numpy_iterator())
    example_batch = next(train_iter)

    print("[model] adapting config for image+state history + custom action head...")
    config = pretrained_model.config
    config["window_size"] = int(args.window_size)
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
        action_head = bound_module.heads["action"]
        use_weighted_loss = (
            abs(float(args.arm_loss_weight) - 1.0) > 1e-6
            or abs(float(args.hand_loss_weight) - 1.0) > 1e-6
        )
        if use_weighted_loss:
            pred = action_head(embeddings, train=train)
            action_loss, action_metrics = _weighted_continuous_metrics(
                pred,
                batch["action"],
                batch["observation"]["timestep_pad_mask"],
                batch["action_pad_mask"],
                action_dim=int(args.action_dim),
                hand_action_start=int(args.hand_action_start),
                arm_loss_weight=float(args.arm_loss_weight),
                hand_loss_weight=float(args.hand_loss_weight),
                loss_type=str(getattr(action_head, "loss_type", "l1")),
            )
        else:
            action_loss, action_metrics = action_head.loss(
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

    @jax.jit
    def eval_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        loss, metrics = loss_fn(state.model.params, batch, dropout_rng, False)
        return loss, metrics

    def evaluate(state) -> dict[str, float]:
        if val_count <= 0:
            return {}
        val_iter = map(process_batch_fn, val_ds.as_numpy_iterator())
        losses: list[float] = []
        metrics_acc: dict[str, list[float]] = {}
        batch_count = 0
        for batch in val_iter:
            loss, metrics = eval_step(state, batch)
            losses.append(float(np.asarray(jax.device_get(loss))))
            flat_metrics = _flatten_scalar_metrics(jax.device_get(metrics))
            for key, value in flat_metrics.items():
                metrics_acc.setdefault(key, []).append(float(value))
            batch_count += 1
            if args.eval_max_batches > 0 and batch_count >= args.eval_max_batches:
                break
        if not losses:
            return {}
        summary = {"loss": float(np.mean(losses)), "num_batches": float(batch_count)}
        for key, values in metrics_acc.items():
            summary[key] = float(np.mean(values))
        return summary

    best_val_loss = np.inf
    best_step = -1
    stale_evals = 0
    stopped_early = False
    loss_trace: list[dict[str, float]] = []
    eval_trace: list[dict[str, float]] = []

    print("[train] start")
    for step in range(1, args.num_steps + 1):
        batch = next(train_iter)
        train_state, loss, metrics = train_step(train_state, batch)
        loss_value = float(np.asarray(jax.device_get(loss)))
        loss_trace.append({"step": float(step), "loss": loss_value})

        if step % args.log_every == 0 or step == 1:
            metrics_dict = _flatten_scalar_metrics(jax.device_get(metrics))
            preview_items = []
            for key in sorted(metrics_dict.keys()):
                if key.lower().endswith("loss") or "mse" in key.lower() or "mae" in key.lower():
                    preview_items.append(f"{key}={metrics_dict[key]:.5f}")
                if len(preview_items) >= 3:
                    break
            preview = " ".join(preview_items)
            print(
                f"[train] step={step:06d}/{args.num_steps} loss={loss_value:.6f}"
                + (f" {preview}" if preview else "")
            )

        if (not args.save_best_only) and args.save_every > 0 and step % args.save_every == 0:
            ckpt_dir = run_dir / f"checkpoint_{step:06d}"
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
            train_state.model.save_pretrained(step=step, checkpoint_path=str(ckpt_dir))
            print(f"[save] {ckpt_dir}")

        if val_count > 0 and args.eval_every > 0 and (step % args.eval_every == 0 or step == 1):
            eval_summary = evaluate(train_state)
            val_loss = float(eval_summary.get("loss", np.inf))
            eval_record = {"step": float(step), "train_loss": loss_value, "val_loss": val_loss}
            for key, value in eval_summary.items():
                if key != "loss":
                    eval_record[f"val/{key}"] = float(value)
            eval_trace.append(eval_record)
            print(f"[eval] step={step:06d} val_loss={val_loss:.6f}")

            improved = val_loss + float(args.early_stop_min_delta) < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_step = step
                stale_evals = 0
                best_dir = run_dir / "best_model"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                train_state.model.save_pretrained(step=step, checkpoint_path=str(best_dir))
                best_info = {
                    "step": int(step),
                    "val_loss": float(val_loss),
                    "train_loss": float(loss_value),
                    "dataset_name": args.dataset_name,
                    "action_dim": int(args.action_dim),
                    "window_size": int(args.window_size),
                    "saved_at": datetime.now().isoformat(),
                }
                (run_dir / "best_model_info.json").write_text(
                    json.dumps(best_info, indent=2),
                    encoding="utf-8",
                )
                print(f"[save] best_model step={step} val_loss={val_loss:.6f}")
            else:
                stale_evals += 1
                print(
                    f"[early-stop] no improvement evals={stale_evals}/{args.early_stop_patience} "
                    f"(best_step={best_step}, best_val={best_val_loss:.6f})"
                )
                if stale_evals >= int(args.early_stop_patience):
                    stopped_early = True
                    print(f"[stop] early stop at step={step}")
                    break

    if best_step < 0:
        best_dir = run_dir / "best_model"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        train_state.model.save_pretrained(step=args.num_steps, checkpoint_path=str(best_dir))
        best_info = {
            "step": int(args.num_steps),
            "val_loss": None if val_count <= 0 else float(best_val_loss),
            "train_loss": float(loss_trace[-1]["loss"]) if loss_trace else None,
            "dataset_name": args.dataset_name,
            "action_dim": int(args.action_dim),
            "window_size": int(args.window_size),
            "saved_at": datetime.now().isoformat(),
        }
        (run_dir / "best_model_info.json").write_text(
            json.dumps(best_info, indent=2),
            encoding="utf-8",
        )
        print(f"[save] fallback best_model={best_dir}")

    if not args.save_best_only:
        final_dir = run_dir / "final_model"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        train_state.model.save_pretrained(step=min(args.num_steps, step), checkpoint_path=str(final_dir))
        print(f"[done] final_model={final_dir}")

    with (run_dir / "loss_trace.json").open("w", encoding="utf-8") as f:
        json.dump(loss_trace, f, indent=2)
    with (run_dir / "eval_trace.json").open("w", encoding="utf-8") as f:
        json.dump(eval_trace, f, indent=2)
    with (run_dir / "dataset_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonify(dataset_statistics), f, indent=2)
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": args.dataset_name,
                "num_steps_requested": int(args.num_steps),
                "num_steps_ran": int(step),
                "stopped_early": bool(stopped_early),
                "best_step": int(best_step),
                "best_val_loss": None if not np.isfinite(best_val_loss) else float(best_val_loss),
                "train_episodes": int(train_count),
                "val_episodes": int(val_count),
                "save_best_only": bool(args.save_best_only),
            },
            f,
            indent=2,
        )
    print(f"[done] logs={run_dir}")


if __name__ == "__main__":
    main()
