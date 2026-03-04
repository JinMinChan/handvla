#!/usr/bin/env python3
"""Run tcp12-only closed-loop rollout for finetuned Octo model in mustard scene."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import jax
import mujoco
from mujoco import viewer
import numpy as np
from PIL import Image
import tensorflow as tf

from scripts.data.collect_mustard_grasp import (
    build_contact_config,
    build_hand_config,
    build_mustard_config,
    compute_mustard_spawn_pose,
    detect_contact_with_target,
    reset_to_initial,
    set_mustard_pose,
)
from env import allegro_hand_mjcf
from env.viewer_utils import set_default_hand_camera
from rollout_mustard_octo import (
    _tcp_delta_local_to_world,
    _build_finger_ik_configs,
    _build_joint_indices,
    _find_stats,
    _make_state_vector,
    _solve_tcp12_to_joint_targets,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rollout finetuned Octo policy in tcp12 mode only."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mustard_octo_overfit/run_260226_133632/final_model",
        help="Path to tcp12 finetuned Octo checkpoint directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mustard_grasp_oxe",
        help="Key inside model.dataset_statistics for action/proprio normalization.",
    )
    parser.add_argument("--task", type=str, default="grasp the mustard bottle")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-policy-steps", type=int, default=100)
    parser.add_argument("--control-repeat", type=int, default=5)
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=1.0,
        help="EMA blend for joint targets from IK output. 1.0 means direct IK target.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--policy-image-size", type=int, default=256)

    parser.add_argument("--ik-max-iters", type=int, default=60)
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--ik-step-scale", type=float, default=0.9)
    parser.add_argument("--ik-max-dq", type=float, default=0.08)
    parser.add_argument("--ik-tol-mm", type=float, default=0.5)
    parser.add_argument(
        "--tcp12-action-type",
        choices=("delta", "absolute"),
        default="delta",
        help="delta: tcp_target=tcp_now+pred, absolute: tcp_target=pred.",
    )
    parser.add_argument(
        "--tcp12-frame",
        choices=("palm_local", "world"),
        default="palm_local",
        help="Frame for tcp12 delta actions. Use world for legacy tcp12 datasets.",
    )

    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=2)
    parser.add_argument(
        "--require-thumb-contact",
        action="store_true",
        default=True,
        help="Require thumb contact for success (default: on).",
    )
    parser.add_argument(
        "--no-require-thumb-contact",
        dest="require_thumb_contact",
        action="store_false",
        help="Disable thumb-contact requirement.",
    )
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)

    parser.add_argument("--debug-every", type=int, default=20)
    parser.add_argument(
        "--motion-warn-threshold",
        type=float,
        default=1e-4,
        help="Warn if per-step joint delta L2 is below this value on average.",
    )

    parser.add_argument("--viewer", dest="viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.set_defaults(viewer=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")

    parser.add_argument("--record", action="store_true", help="Record MP4.")
    parser.add_argument(
        "--record-path",
        type=str,
        default="",
        help="Output path (default: codex/logs/rollout_mustard_octo_tcp12_<timestamp>.mp4).",
    )
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-fps", type=int, default=60)
    return parser.parse_args()


def _default_record_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"rollout_mustard_octo_tcp12_{ts}.mp4"


def run_rollout(args: argparse.Namespace) -> dict:
    tf.config.set_visible_devices([], "GPU")
    rng = jax.random.PRNGKey(args.seed)

    mjcf = allegro_hand_mjcf.load(side=args.side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)

    hand_cfg = build_hand_config(model, args.side)
    hand_idx = _build_joint_indices(model, args.side)
    finger_ik_cfgs = _build_finger_ik_configs(model, args.side)
    mustard_cfg = build_mustard_config(model)
    contact_cfg = build_contact_config(model, args.side, mustard_cfg.body_id)
    force_buf = np.zeros(6, dtype=float)

    policy_model = OctoModel.load_pretrained(args.model_path)
    if policy_model.text_processor is None:
        raise RuntimeError("Loaded model has no text_processor.")
    task = {"language_instruction": policy_model.text_processor.encode([args.task])}
    action_stats, proprio_stats = _find_stats(policy_model, args.dataset_name)

    viewer_ctx = None
    if args.viewer:
        viewer_ctx = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
        viewer_ctx.__enter__()
        set_default_hand_camera(viewer_ctx.cam)

    renderer_obs = mujoco.Renderer(model, width=args.image_width, height=args.image_height)
    record_renderer = None
    record_fallback_cam = None
    writer = None
    if args.record:
        record_path = Path(args.record_path) if args.record_path else _default_record_path()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_renderer = mujoco.Renderer(
            model, width=args.record_width, height=args.record_height
        )
        record_fallback_cam = mujoco.MjvCamera()
        set_default_hand_camera(record_fallback_cam)
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

    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )

    episodes_data = []
    try:
        for ep in range(args.episodes):
            reset_to_initial(model, data)
            spawn_pos, spawn_quat = compute_mustard_spawn_pose(model, data, args.side)
            set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)

            q_cmd = data.qpos[hand_idx.qpos_ids].astype(np.float32).copy()
            q_cmd = np.clip(q_cmd, hand_cfg.q_min, hand_cfg.q_max)

            stable_hits = 0
            first_success_policy_step = -1
            best_contacts = 0
            best_fingers = 0
            best_force = 0.0
            best_finger_set: set[str] = set()
            nan_action = False
            action_dim_error = False

            q_delta_acc = 0.0
            tcp_delta_acc = 0.0
            decision_steps = 0
            sim_step_counter = 0

            for p_step in range(args.max_policy_steps):
                _, n_contacts, total_force, touched = detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                thumb_contact = 1.0 if "th" in touched else 0.0
                contact_stats = np.asarray(
                    [float(n_contacts), float(total_force), float(len(touched)), thumb_contact],
                    dtype=np.float32,
                )
                state = _make_state_vector(data, hand_idx, mustard_cfg, contact_stats)
                if proprio_stats is not None:
                    mean = np.asarray(proprio_stats["mean"], dtype=np.float32)
                    std = np.asarray(proprio_stats["std"], dtype=np.float32)
                    state = (state - mean) / np.maximum(std, 1e-6)

                renderer_obs.update_scene(data)
                obs_image = renderer_obs.render()
                obs_image = np.asarray(
                    Image.fromarray(obs_image).resize(
                        (args.policy_image_size, args.policy_image_size), Image.BILINEAR
                    ),
                    dtype=np.uint8,
                )

                obs = {
                    "image_primary": obs_image[None, None, ...],
                    "proprio": state[None, None, :],
                    "timestep_pad_mask": np.array([[True]], dtype=np.bool_),
                }

                rng, key = jax.random.split(rng)
                if action_stats is not None:
                    pred = policy_model.sample_actions(
                        obs, task, unnormalization_statistics=action_stats, rng=key
                    )[0, 0]
                else:
                    pred = policy_model.sample_actions(obs, task, rng=key)[0, 0]
                pred = np.asarray(pred, dtype=np.float32)

                if pred.shape[-1] != 12:
                    action_dim_error = True
                    print(
                        f"[episode {ep:03d}] expected action_dim=12, got {pred.shape[-1]}",
                        flush=True,
                    )
                    break

                if not np.all(np.isfinite(pred)):
                    nan_action = True
                    print(f"[episode {ep:03d}] non-finite action detected.", flush=True)
                    break

                tcp_now = data.site_xpos[hand_idx.tcp_site_ids].reshape(4, 3).astype(np.float32)
                pred_tcp = pred.reshape(4, 3)
                if args.tcp12_action_type == "delta":
                    if args.tcp12_frame == "palm_local":
                        pred_tcp_world = _tcp_delta_local_to_world(
                            data=data,
                            hand_idx=hand_idx,
                            tcp_delta_local=pred_tcp,
                        )
                    else:
                        pred_tcp_world = pred_tcp
                    tcp_targets = tcp_now + pred_tcp_world
                else:
                    tcp_targets = pred_tcp
                tcp_delta = float(np.linalg.norm((tcp_targets - tcp_now).reshape(-1)))
                pred_norm = float(np.linalg.norm(pred.reshape(-1)))

                q_prev = q_cmd.copy()
                q_target = _solve_tcp12_to_joint_targets(
                    model=model,
                    data=data,
                    hand_qpos_ids=hand_idx.qpos_ids,
                    q_seed=q_cmd,
                    finger_cfgs=finger_ik_cfgs,
                    tcp_targets=tcp_targets,
                    ik_max_iters=args.ik_max_iters,
                    ik_damping=args.ik_damping,
                    ik_step_scale=args.ik_step_scale,
                    ik_max_dq=args.ik_max_dq,
                    ik_tol_m=args.ik_tol_mm / 1000.0,
                )
                alpha = float(np.clip(args.action_smoothing, 0.0, 1.0))
                q_cmd = (1.0 - alpha) * q_cmd + alpha * q_target
                q_delta = float(np.linalg.norm(q_cmd - q_prev))

                q_delta_acc += q_delta
                tcp_delta_acc += tcp_delta
                decision_steps += 1

                if args.debug_every > 0 and (p_step % args.debug_every == 0):
                    print(
                        f"[episode {ep:03d} step {p_step:03d}] "
                        f"pred_norm={pred_norm:.6f} tcp_delta={tcp_delta:.6f} q_delta={q_delta:.6f}",
                        flush=True,
                    )

                for _ in range(args.control_repeat):
                    data.ctrl[:16] = q_cmd
                    set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_step(model, data)
                    set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)

                    _, n_c, f_c, touched_c = detect_contact_with_target(
                        model, data, contact_cfg, force_buf
                    )
                    best_contacts = max(best_contacts, int(n_c))
                    best_fingers = max(best_fingers, len(touched_c))
                    best_force = max(best_force, float(f_c))
                    if len(touched_c) >= len(best_finger_set):
                        best_finger_set = set(touched_c)

                    thumb_ok = (not args.require_thumb_contact) or ("th" in touched_c)
                    meets = (
                        n_c >= args.min_contacts
                        and len(touched_c) >= args.min_contact_fingers
                        and thumb_ok
                        and f_c >= args.min_force
                        and f_c <= args.max_force
                    )
                    stable_hits = stable_hits + 1 if meets else 0
                    if stable_hits >= args.stable_steps and first_success_policy_step < 0:
                        first_success_policy_step = p_step

                    if viewer_ctx is not None:
                        viewer_ctx.sync()
                    if writer is not None and record_renderer is not None:
                        cam = viewer_ctx.cam if viewer_ctx is not None else record_fallback_cam
                        if cam is not None:
                            record_renderer.update_scene(data, camera=cam)
                        else:
                            record_renderer.update_scene(data)
                        writer.append_data(record_renderer.render())
                    if viewer_delay > 0.0:
                        time.sleep(viewer_delay)

                    sim_step_counter += 1

                if first_success_policy_step >= 0:
                    break

            success = first_success_policy_step >= 0 and not nan_action and not action_dim_error
            mean_q_delta = q_delta_acc / max(decision_steps, 1)
            mean_tcp_delta = tcp_delta_acc / max(decision_steps, 1)
            if mean_q_delta < args.motion_warn_threshold:
                print(
                    f"[warn] episode {ep:03d} low motion: mean_q_delta={mean_q_delta:.6e} "
                    f"(threshold={args.motion_warn_threshold:.6e})",
                    flush=True,
                )

            ep_result = {
                "episode": ep,
                "success": success,
                "first_success_policy_step": first_success_policy_step,
                "best_contacts": best_contacts,
                "best_contact_fingers": best_fingers,
                "best_fingers": sorted(best_finger_set),
                "best_force": best_force,
                "sim_steps": sim_step_counter,
                "decision_steps": decision_steps,
                "mean_q_delta": mean_q_delta,
                "mean_tcp_delta": mean_tcp_delta,
                "nan_action": nan_action,
                "action_dim_error": action_dim_error,
            }
            episodes_data.append(ep_result)
            print(
                f"[episode {ep:03d}] success={int(success)} "
                f"contacts={best_contacts} fingers={best_fingers} "
                f"({','.join(sorted(best_finger_set))}) force={best_force:.3f}N "
                f"mean_q_delta={mean_q_delta:.6f} mean_tcp_delta={mean_tcp_delta:.6f}",
                flush=True,
            )

        if viewer_ctx is not None and args.keep_open:
            print("Viewer kept open. Close window to finish.", flush=True)
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)

    finally:
        if writer is not None:
            writer.close()
            print("[record] finished", flush=True)
        if record_renderer is not None:
            record_renderer.close()
        renderer_obs.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}", flush=True)

    success_rate = (
        float(np.mean([float(ep["success"]) for ep in episodes_data])) if episodes_data else 0.0
    )
    summary = {
        "model_path": args.model_path,
        "task": args.task,
        "episodes": args.episodes,
        "max_policy_steps": args.max_policy_steps,
        "control_repeat": args.control_repeat,
        "action_interface": "tcp12",
        "tcp12_action_type": args.tcp12_action_type,
        "tcp12_frame": args.tcp12_frame,
        "ik_tol_mm": args.ik_tol_mm,
        "success_rate": success_rate,
        "episodes_data": episodes_data,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = run_rollout(args)
    print(
        "\n=== Mustard Octo TCP12 Rollout Summary ===\n"
        f"episodes={summary['episodes']} success_rate={100.0 * summary['success_rate']:.1f}%\n"
        f"model={summary['model_path']}"
    )


if __name__ == "__main__":
    main()
