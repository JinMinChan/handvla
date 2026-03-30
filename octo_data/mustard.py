"""Official-style Octo dataset helpers for mustard RLDS exports.

This module intentionally mirrors Octo's dataset-config style without
depending on the fixed OXE action/proprio enums, since our action spaces
can vary across experiments.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import tensorflow as tf


def mustard_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize mustard RLDS trajectories into Octo-ready raw semantics.

    Input trajectory is expected to come from `dlimp.from_rlds(...)` over the
    TFDS builder produced by `scripts/data/convert_mustard_raw_to_oxe.py`.

    The exporter already writes:
    - `observation.image_primary`
    - `observation.state`
    - step-level `language_instruction`

    This transform keeps those conventions, but explicitly adds a canonical
    `observation.proprio` field so the downstream Octo loader can treat mustard
    like a normal dataset config.
    """

    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    traj_len = tf.shape(trajectory["action"])[0]

    trajectory["observation"]["proprio"] = tf.cast(
        trajectory["observation"]["state"], tf.float32
    )
    trajectory["language_instruction"] = tf.cast(
        trajectory["language_instruction"], tf.string
    )

    # Define the episode goal image as the last observation frame. This matches the
    # current mustard exporter semantics ("goal_image_primary = last_frame"), but
    # keeps the logic inside the official Octo dataset path rather than trainer code.
    if "image_primary" in trajectory["observation"]:
        goal_image = trajectory["observation"]["image_primary"][-1]
        goal_image = tf.repeat(goal_image[None], traj_len, axis=0)
        trajectory["observation"]["goal_image_primary"] = goal_image
    return trajectory


def attach_episode_goal_as_task_image(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the exported episode goal image to `task.image_primary`.

    This runs after chunking. `goal_image_primary` is constant across the trajectory,
    so taking the most recent history slot is equivalent to using any slot.
    """

    goal_obs_key = "image_goal"
    if goal_obs_key not in traj["observation"]:
        return traj

    goal_image = traj["observation"][goal_obs_key][:, -1]
    traj_len = tf.shape(traj["action"])[0]

    traj["task"]["image_primary"] = goal_image
    traj["task"]["pad_mask_dict"]["image_primary"] = tf.ones(
        [traj_len], dtype=tf.bool
    )

    del traj["observation"][goal_obs_key]
    if "pad_mask_dict" in traj["observation"]:
        traj["observation"]["pad_mask_dict"].pop(goal_obs_key, None)

    return traj


def make_mustard_dataset_kwargs(
    *,
    name: str,
    data_dir: str,
    load_camera_views: Sequence[str] = ("primary",),
    load_proprio: bool = True,
    load_language: bool = True,
    force_recompute_dataset_statistics: bool = False,
    action_proprio_normalization_type: Any = "normal",
) -> Dict[str, Any]:
    """Build official-style kwargs for `octo.data.dataset.make_dataset_from_rlds`.

    This is intentionally shaped like `octo.data.oxe.make_oxe_dataset_kwargs(...)`,
    but kept local because mustard uses custom action dimensions/interfaces.
    """
    from octo.data.utils.data_utils import NormalizationType
    from octo.utils.spec import ModuleSpec

    if action_proprio_normalization_type == "normal":
        action_proprio_normalization_type = NormalizationType.NORMAL

    supported_views = {"primary", "goal"}
    missing_views = set(load_camera_views) - supported_views
    if missing_views:
        raise ValueError(
            f"Mustard dataset only supports {sorted(supported_views)} views, "
            f"but received {sorted(missing_views)}."
        )

    image_obs_keys = {
        k: v
        for k, v in {
            "primary": "image_primary",
            "goal": "goal_image_primary",
        }.items()
        if k in load_camera_views
    }

    dataset_kwargs: Dict[str, Any] = {
        "name": name,
        "data_dir": data_dir,
        "image_obs_keys": image_obs_keys,
        "depth_obs_keys": {},
        "standardize_fn": ModuleSpec.create(mustard_dataset_transform),
        "action_proprio_normalization_type": action_proprio_normalization_type,
    }
    if load_proprio:
        dataset_kwargs["proprio_obs_key"] = "proprio"
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"
    if force_recompute_dataset_statistics:
        dataset_kwargs["force_recompute_dataset_statistics"] = True
    return dataset_kwargs


def make_mustard_goal_trajectory_kwargs() -> Dict[str, Any]:
    """Build trajectory kwargs that attach exported episode goal images to tasks.

    Octo's core pipeline expects goal images to live under `task.image_*`.
    Our exporter stores them as episode-level metadata, so this helper keeps the
    official loader/transform path intact and only adds a small dataset-specific
    trajectory transform that moves the exported goal image into `task.image_primary`.
    """
    from octo.utils.spec import ModuleSpec

    return {
        "post_chunk_transforms": [
            ModuleSpec.create(attach_episode_goal_as_task_image)
        ]
    }
