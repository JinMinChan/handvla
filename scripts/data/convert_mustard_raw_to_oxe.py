#!/usr/bin/env python3
"""Convert raw mustard grasp NPZ episodes to OXE-style RLDS (TFDS) dataset."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw mustard grasp episodes to OXE-style RLDS/TFDS dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset/mustard_grasp/raw",
        help="Directory containing raw episode_*.npz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/mustard_grasp_oxe",
        help="TFDS output root directory.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="1.0.0",
        help="TFDS dataset version.",
    )
    parser.add_argument(
        "--language-default",
        type=str,
        default="grasp the mustard bottle",
        help="Fallback instruction when raw episode has no language field.",
    )
    parser.add_argument(
        "--only-success",
        dest="only_success",
        action="store_true",
        default=True,
        help="Use only successful episodes (default: on).",
    )
    parser.add_argument(
        "--include-failures",
        dest="only_success",
        action="store_false",
        help="Include failure episodes too.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect raw files and print stats without building TFDS.",
    )
    parser.add_argument(
        "--goal-image-source",
        choices=("last_frame", "first_frame"),
        default="last_frame",
        help="Episode-level goal image source for optional goal-conditioned training.",
    )
    return parser.parse_args()


def _list_episode_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.glob("episode_*.npz") if p.is_file())


def _to_scalar_string(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
        return str(value.tolist())
    return str(value)


def _load_episode(path: Path, language_default: str, goal_image_source: str = "last_frame") -> dict:
    data = np.load(path, allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)
    states = np.asarray(data["state"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)

    steps = min(images.shape[0], states.shape[0], actions.shape[0])
    images = images[:steps]
    states = states[:steps]
    actions = actions[:steps]

    success = bool(data["success"]) if "success" in data else True
    language = language_default
    if "language_instruction" in data:
        language = _to_scalar_string(data["language_instruction"])

    object_qpos = (
        np.asarray(data["object_qpos"], dtype=np.float32)
        if "object_qpos" in data
        else np.zeros((7,), dtype=np.float32)
    )
    if images.shape[0] > 0:
        goal_image_primary = images[0].copy() if goal_image_source == "first_frame" else images[-1].copy()
    else:
        goal_image_primary = np.zeros((1, 1, 3), dtype=np.uint8)

    return {
        "images": images,
        "states": states,
        "actions": actions,
        "success": success,
        "language_instruction": language,
        "object_qpos": object_qpos,
        "goal_image_primary": goal_image_primary,
    }


def _inspect_files(files: list[Path], language_default: str) -> dict:
    if not files:
        return {
            "count": 0,
            "success_count": 0,
            "failure_count": 0,
            "image_shape": None,
            "state_dim": None,
            "action_dim": None,
        }

    success_count = 0
    failure_count = 0
    state_dim = None
    action_dim = None
    image_shape = None
    min_steps = 10**9
    max_steps = 0
    total_steps = 0

    for path in files:
        ep = _load_episode(path, language_default)
        steps = ep["images"].shape[0]
        min_steps = min(min_steps, steps)
        max_steps = max(max_steps, steps)
        total_steps += steps
        success_count += int(ep["success"])
        failure_count += int(not ep["success"])
        if image_shape is None:
            image_shape = tuple(ep["images"].shape[1:])
        if state_dim is None:
            state_dim = int(ep["states"].shape[1])
        if action_dim is None:
            action_dim = int(ep["actions"].shape[1])

    return {
        "count": len(files),
        "success_count": success_count,
        "failure_count": failure_count,
        "image_shape": image_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "min_steps": int(min_steps),
        "max_steps": int(max_steps),
        "mean_steps": float(total_steps / len(files)),
    }


def _build_tfds(args: argparse.Namespace, selected_files: list[Path], schema: dict) -> None:
    try:
        import tensorflow as tf  # noqa: F401
        import tensorflow_datasets as tfds
        from tensorflow_datasets.rlds import rlds_base
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "TensorFlow + TensorFlow Datasets are required.\n"
            "Install with: pip install tensorflow tensorflow-datasets"
        ) from exc

    image_h, image_w, image_c = schema["image_shape"]
    state_dim = schema["state_dim"]
    action_dim = schema["action_dim"]
    dataset_version = args.dataset_version
    raw_dir_path = Path(args.raw_dir).resolve()
    language_default = args.language_default

    class MustardGraspOxe(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(dataset_version)
        RELEASE_NOTES = {dataset_version: "Initial mustard grasp OXE/RLDS export."}

        def _info(self) -> tfds.core.DatasetInfo:
            description = (
                "Mustard fixed-air grasp dataset in OXE-style RLDS format "
                "(image_primary + state + action)."
            )
            return rlds_base.build_info(
                rlds_base.DatasetConfig(
                    name="mustard_grasp_oxe",
                    version=dataset_version,
                    overall_description=description,
                    observation_info=tfds.features.FeaturesDict(
                        {
                            "image_primary": tfds.features.Image(
                                shape=(image_h, image_w, image_c), dtype=np.uint8
                            ),
                            "state": tfds.features.Tensor(
                                shape=(state_dim,), dtype=np.float32
                            ),
                        }
                    ),
                    action_info=tfds.features.Tensor(
                        shape=(action_dim,), dtype=np.float32
                    ),
                    reward_info=tfds.features.Scalar(dtype=np.float32),
                    discount_info=tfds.features.Scalar(dtype=np.float32),
                    step_metadata_info={
                        "language_instruction": tfds.features.Text(),
                    },
                    episode_metadata_info={
                        "success": tfds.features.Scalar(dtype=np.bool_),
                        "object_qpos": tfds.features.Tensor(
                            shape=(7,), dtype=np.float32
                        ),
                        "goal_image_primary": tfds.features.Image(
                            shape=(image_h, image_w, image_c), dtype=np.uint8
                        ),
                        "source_file": tfds.features.Text(),
                    },
                ),
                self,
            )

        def _split_generators(self, dl_manager):  # noqa: ANN001
            del dl_manager
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            episode_id = 0
            for path in selected_files:
                ep = _load_episode(
                    path,
                    language_default=language_default,
                    goal_image_source=args.goal_image_source,
                )
                steps_count = int(ep["images"].shape[0])
                if steps_count == 0:
                    continue
                rewards = np.zeros((steps_count,), dtype=np.float32)
                rewards[-1] = 1.0 if ep["success"] else 0.0
                steps = []
                for i in range(steps_count):
                    steps.append(
                        {
                            "observation": {
                                "image_primary": ep["images"][i],
                                "state": ep["states"][i],
                            },
                            "action": ep["actions"][i],
                            "reward": rewards[i],
                            "discount": np.float32(1.0),
                            "is_first": np.bool_(i == 0),
                            "is_last": np.bool_(i == steps_count - 1),
                            "is_terminal": np.bool_(i == steps_count - 1),
                            "language_instruction": ep["language_instruction"],
                        }
                    )
                key = f"episode_{episode_id:05d}"
                example = {
                    "steps": steps,
                    "success": np.bool_(ep["success"]),
                    "object_qpos": ep["object_qpos"],
                    "goal_image_primary": ep["goal_image_primary"],
                    "source_file": str(path.relative_to(raw_dir_path)),
                }
                yield key, example
                episode_id += 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TFDS_DISABLE_GCS", "1")

    builder = MustardGraspOxe(data_dir=str(out_dir))
    builder.download_and_prepare()
    print(f"[done] TFDS built at: {out_dir}")
    print(f"[done] dataset id: {builder.info.full_name}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    files = _list_episode_files(raw_dir)
    if not files:
        raise SystemExit(f"No raw episodes found: {raw_dir}")

    stats = _inspect_files(files, language_default=args.language_default)
    print(
        "[scan] total={count} success={success_count} fail={failure_count} "
        "shape={image_shape} state_dim={state_dim} action_dim={action_dim} "
        "steps(min/mean/max)={min_steps}/{mean_steps:.1f}/{max_steps}".format(**stats)
    )

    selected_files: list[Path] = []
    for path in files:
        ep = _load_episode(
            path,
            language_default=args.language_default,
            goal_image_source=args.goal_image_source,
        )
        if args.only_success and not ep["success"]:
            continue
        selected_files.append(path)

    if not selected_files:
        raise SystemExit(
            "No episodes selected for export. "
            "Use --include-failures or collect successful episodes first."
        )

    print(f"[select] exporting {len(selected_files)} episodes to OXE/RLDS")
    if args.dry_run:
        print("[dry-run] conversion skipped")
        return

    _build_tfds(args=args, selected_files=selected_files, schema=stats)


if __name__ == "__main__":
    main()
