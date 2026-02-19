#!/usr/bin/env python3
"""Allegro finger IK feasibility experiment.

Each trial:
1) Sample a reachable target by randomizing one finger's joints.
2) Reset to initial pose.
3) Solve IK so finger TCP reaches that target.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from mujoco import viewer

from env import allegro_hand_mjcf
from env.viewer_utils import set_default_hand_camera


@dataclass(frozen=True)
class FingerConfig:
    name: str
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    tcp_site_id: int
    target_site_id: int | None
    target_mocap_id: int | None
    target_geom_id: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random-target Allegro finger IK test.")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--trials", type=int, default=3, help="Trials per finger")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol-mm", type=float, default=2.5, help="Success tolerance in mm")
    parser.add_argument("--max-iters", type=int, default=120)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--step-scale", type=float, default=0.75)
    parser.add_argument("--max-dq", type=float, default=0.08, help="Per-joint step clip (rad)")
    parser.add_argument(
        "--joint-margin",
        type=float,
        default=0.08,
        help="Fractional margin away from joint limits for random target sampling",
    )
    parser.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Visualize trials (default on).",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Run headless without opening viewer.",
    )
    parser.set_defaults(viewer=True)
    parser.add_argument(
        "--viewer-step",
        type=float,
        default=20.0,
        help="Viewer update frequency in Hz (default: 20).",
    )
    parser.add_argument(
        "--viewer-step-delay",
        type=float,
        default=None,
        help="Manual viewer delay in seconds. If set, overrides --viewer-step.",
    )
    parser.add_argument(
        "--viewer-trial-hold",
        type=float,
        default=0.20,
        help="Seconds to hold each trial result in viewer.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep viewer open after experiment until window is closed manually.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output path for JSON summary",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record MP4 while running the IK experiment viewer.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default="",
        help="Output mp4 path. Default: codex/logs/finger_ik_<timestamp>.mp4",
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


def build_finger_configs(model: mujoco.MjModel, side: str) -> dict[str, FingerConfig]:
    prefix = f"allegro_{side}"
    finger_names = ("ff", "mf", "rf", "th")

    cfgs: dict[str, FingerConfig] = {}
    for finger in finger_names:
        qpos_ids = []
        dof_ids = []
        q_min = []
        q_max = []
        for j in range(4):
            joint_name = f"{prefix}/{finger}j{j}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")

            qpos_ids.append(model.jnt_qposadr[joint_id])
            dof_ids.append(model.jnt_dofadr[joint_id])
            q_min.append(model.jnt_range[joint_id][0])
            q_max.append(model.jnt_range[joint_id][1])

        tcp_name = f"{prefix}/{finger}_tcp"
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
        if tcp_site_id < 0:
            raise RuntimeError(f"TCP site not found: {tcp_name}")

        target_name = f"{prefix}/{finger}_target"
        target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, target_name)
        if target_site_id < 0:
            target_site_id = None

        target_geom_name = f"{prefix}/{finger}_target_geom"
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name)
        if target_geom_id < 0:
            target_geom_id = None

        target_body_name = f"{prefix}/{finger}_target_body"
        target_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, target_body_name
        )
        target_mocap_id = None
        if target_body_id >= 0:
            mocap_id = model.body_mocapid[target_body_id]
            if mocap_id >= 0:
                target_mocap_id = int(mocap_id)

        cfgs[finger] = FingerConfig(
            name=finger,
            qpos_ids=np.asarray(qpos_ids, dtype=int),
            dof_ids=np.asarray(dof_ids, dtype=int),
            q_min=np.asarray(q_min, dtype=float),
            q_max=np.asarray(q_max, dtype=float),
            tcp_site_id=tcp_site_id,
            target_site_id=target_site_id,
            target_mocap_id=target_mocap_id,
            target_geom_id=target_geom_id,
        )

    return cfgs


def _default_record_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"finger_ik_{ts}.mp4"


def _set_target_visibility(
    model: mujoco.MjModel, cfgs: dict[str, FingerConfig], active_finger: str | None
) -> None:
    for finger, cfg in cfgs.items():
        if cfg.target_geom_id is None:
            continue
        alpha = 0.95 if active_finger is not None and finger == active_finger else 0.0
        model.geom_rgba[cfg.target_geom_id, 3] = alpha


def reset_to_initial(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    initial_id = model.key("initial_state").id
    mujoco.mj_resetDataKeyframe(model, data, initial_id)
    mujoco.mj_forward(model, data)


def sample_reachable_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: FingerConfig,
    rng: np.random.Generator,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Random joint pose => guaranteed reachable TCP target.
    span = cfg.q_max - cfg.q_min
    q_low = cfg.q_min + span * margin
    q_high = cfg.q_max - span * margin
    q_goal = rng.uniform(q_low, q_high)

    q_backup = data.qpos[cfg.qpos_ids].copy()
    data.qpos[cfg.qpos_ids] = q_goal
    mujoco.mj_forward(model, data)
    target = data.site_xpos[cfg.tcp_site_id].copy()

    data.qpos[cfg.qpos_ids] = q_backup
    mujoco.mj_forward(model, data)

    return target, q_goal


def solve_finger_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: FingerConfig,
    target: np.ndarray,
    tol: float,
    max_iters: int,
    damping: float,
    step_scale: float,
    max_dq: float,
    sync_fn=None,
) -> tuple[bool, float, int]:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    best_err = math.inf
    best_iter = 0

    for it in range(1, max_iters + 1):
        mujoco.mj_forward(model, data)
        tcp_pos = data.site_xpos[cfg.tcp_site_id]
        err = target - tcp_pos
        err_norm = float(np.linalg.norm(err))

        if err_norm < best_err:
            best_err = err_norm
            best_iter = it

        if err_norm <= tol:
            return True, err_norm, it

        mujoco.mj_jacSite(model, data, jacp, jacr, cfg.tcp_site_id)
        J = jacp[:, cfg.dof_ids]  # (3,4)

        # Damped least-squares inverse kinematics.
        lhs = J @ J.T + damping * np.eye(3)
        try:
            dq = J.T @ np.linalg.solve(lhs, err)
        except np.linalg.LinAlgError:
            dq = J.T @ np.linalg.lstsq(lhs, err, rcond=None)[0]

        dq = np.clip(dq, -max_dq, max_dq)
        data.qpos[cfg.qpos_ids] += step_scale * dq
        data.qpos[cfg.qpos_ids] = np.clip(data.qpos[cfg.qpos_ids], cfg.q_min, cfg.q_max)

        if sync_fn is not None:
            sync_fn()

    return False, best_err, best_iter


def summarize_errors(errors: list[float]) -> dict[str, float]:
    if not errors:
        return {"mean_mm": float("nan"), "p50_mm": float("nan"), "p90_mm": float("nan")}

    arr = np.asarray(errors) * 1000.0
    return {
        "mean_mm": float(np.mean(arr)),
        "p50_mm": float(np.percentile(arr, 50)),
        "p90_mm": float(np.percentile(arr, 90)),
    }


def run_experiment(args: argparse.Namespace) -> dict:
    mjcf = allegro_hand_mjcf.load(side=args.side, add_ik_targets=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)

    rng = np.random.default_rng(args.seed)
    cfgs = build_finger_configs(model, args.side)
    tol = args.tol_mm / 1000.0
    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )

    result = {
        "side": args.side,
        "seed": args.seed,
        "trials_per_finger": args.trials,
        "tol_mm": args.tol_mm,
        "max_iters": args.max_iters,
        "damping": args.damping,
        "step_scale": args.step_scale,
        "max_dq": args.max_dq,
        "joint_margin": args.joint_margin,
        "per_finger": {},
    }

    viewer_ctx = None
    renderer = None
    writer = None
    if args.viewer:
        viewer_ctx = viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=True
        )
        viewer_ctx.__enter__()
        set_default_hand_camera(viewer_ctx.cam)

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

    def _sync_and_record() -> None:
        if viewer_ctx is not None:
            viewer_ctx.sync()
        if renderer is not None and writer is not None:
            cam = viewer_ctx.cam if viewer_ctx is not None else None
            if cam is not None:
                renderer.update_scene(data, camera=cam)
            else:
                renderer.update_scene(data)
            writer.append_data(renderer.render())
        if viewer_delay > 0.0:
            time.sleep(viewer_delay)

    try:
        _set_target_visibility(model, cfgs, active_finger=None)
        mujoco.mj_forward(model, data)

        for finger, cfg in cfgs.items():
            _set_target_visibility(model, cfgs, active_finger=finger)
            errors = []
            iterations = []
            success_count = 0

            for _ in range(args.trials):
                reset_to_initial(model, data)
                target, _ = sample_reachable_target(model, data, cfg, rng, args.joint_margin)

                if cfg.target_mocap_id is not None:
                    data.mocap_pos[cfg.target_mocap_id] = target
                    data.mocap_quat[cfg.target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
                    mujoco.mj_forward(model, data)
                elif cfg.target_site_id is not None:
                    model.site_pos[cfg.target_site_id] = target
                    mujoco.mj_forward(model, data)

                sync_fn = _sync_and_record if (viewer_ctx is not None or writer is not None) else None

                ok, err, it_used = solve_finger_ik(
                    model=model,
                    data=data,
                    cfg=cfg,
                    target=target,
                    tol=tol,
                    max_iters=args.max_iters,
                    damping=args.damping,
                    step_scale=args.step_scale,
                    max_dq=args.max_dq,
                    sync_fn=sync_fn,
                )

                errors.append(err)
                iterations.append(it_used)
                success_count += int(ok)

                if viewer_ctx is not None or writer is not None:
                    hold_steps = max(1, int(args.viewer_trial_hold / 0.01))
                    for _ in range(hold_steps):
                        _sync_and_record()

            summary = summarize_errors(errors)
            summary["success_rate"] = success_count / args.trials
            summary["mean_iters"] = float(np.mean(np.asarray(iterations)))
            result["per_finger"][finger] = summary

        if viewer_ctx is not None and args.keep_open:
            print("Viewer is kept open. Close the MuJoCo window to finish.")
            while viewer_ctx.is_running():
                _sync_and_record()

    finally:
        if writer is not None:
            writer.close()
            print("[record] finished", flush=True)
        if renderer is not None:
            renderer.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}")

    return result


def print_result(result: dict) -> None:
    print("\n=== Allegro Finger IK Feasibility ===")
    print(
        f"side={result['side']} trials/finger={result['trials_per_finger']} "
        f"tol={result['tol_mm']}mm max_iters={result['max_iters']}"
    )
    print("finger  success   mean_err(mm)  p50(mm)  p90(mm)  mean_iters")

    for finger in ("ff", "mf", "rf", "th"):
        s = result["per_finger"][finger]
        print(
            f"{finger:<5}  {s['success_rate']*100:6.1f}%   "
            f"{s['mean_mm']:11.3f}  {s['p50_mm']:7.3f}  {s['p90_mm']:7.3f}  {s['mean_iters']:10.2f}"
        )


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print_result(result)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
