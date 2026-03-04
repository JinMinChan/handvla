#!/usr/bin/env python3
"""Launch Allegro hand-only MuJoCo viewer with joint control sliders."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import mujoco
from mujoco import viewer

from env import allegro_hand_mjcf
from env.viewer_utils import set_default_hand_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Allegro hand-only MuJoCo scene with UI control panel."
    )
    parser.add_argument(
        "--side",
        choices=("right", "left"),
        default="right",
        help="Which hand model to load (default: right).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record MP4 while simulation is running.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default="",
        help="Output mp4 path. Default: codex/logs/<timestamp>.mp4",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=1920,
        help="Recording width (default: 1920).",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=1080,
        help="Recording height (default: 1080).",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=60,
        help="Recording frame rate (default: 60).",
    )
    return parser.parse_args()


def _default_record_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"allegro_hand_{ts}.mp4"


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    if not (project_root / "env" / "allegro_hand_mjcf.py").exists():
        raise FileNotFoundError("Could not find env/allegro_hand_mjcf.py")

    mjcf = allegro_hand_mjcf.load(side=args.side)
    model = mjcf.compile()
    data = mujoco.MjData(model)
    initial_state = model.key("initial_state").id
    mujoco.mj_resetDataKeyframe(model, data, initial_state)
    mujoco.mj_forward(model, data)

    renderer = None
    writer = None
    next_frame_t = 0.0
    frame_dt = 1.0 / max(args.record_fps, 1)

    if args.record:
        record_path = Path(args.record_path) if args.record_path else _default_record_path()
        record_path.parent.mkdir(parents=True, exist_ok=True)

        renderer = mujoco.Renderer(
            model, width=args.record_width, height=args.record_height
        )
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

    with viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=True
    ) as passive_viewer:
        set_default_hand_camera(passive_viewer.cam)
        next_frame_t = data.time

        try:
            while passive_viewer.is_running():
                mujoco.mj_step(model, data)

                if renderer is not None and writer is not None and data.time >= next_frame_t:
                    renderer.update_scene(data, camera=passive_viewer.cam)
                    writer.append_data(renderer.render())
                    next_frame_t += frame_dt

                passive_viewer.sync()
        finally:
            if writer is not None:
                writer.close()
                print("[record] finished", flush=True)
            if renderer is not None:
                renderer.close()


if __name__ == "__main__":
    main()
