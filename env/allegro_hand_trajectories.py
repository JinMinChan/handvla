"""Shared Allegro hand grasp trajectories used across data and eval scripts."""

from __future__ import annotations

import numpy as np


# KETI-OCTO style cylindrical wrap used in earlier experiments.
KETI_HUMAN_CLOSE_QPOS = np.array(
    [
        0.00,
        1.55,
        1.45,
        1.05,
        0.00,
        1.55,
        1.45,
        1.05,
        0.00,
        1.50,
        1.40,
        1.00,
        1.10,
        0.95,
        1.35,
        0.95,
    ],
    dtype=np.float32,
)


POWER_GRASP_CLOSE_QPOS = np.array(
    [
        0.0920,
        1.6100,
        1.5500,
        0.9000,
        -0.0139,
        1.6100,
        1.5500,
        0.9000,
        -0.0885,
        1.6100,
        1.5500,
        0.9000,
        1.2679,
        0.9546,
        1.5218,
        0.6767,
    ],
    dtype=np.float32,
)


# First rotate the thumb inward, then flex it.
POWER_GRASP_PRESHAPE_QPOS = np.array(
    [
        0.0920,
        0.0000,
        0.0000,
        0.0000,
        -0.0139,
        0.0000,
        0.0000,
        0.0000,
        -0.0885,
        0.0000,
        0.0000,
        0.0000,
        1.2679,
        0.4296,
        0.8370,
        0.3722,
    ],
    dtype=np.float32,
)


# A wider thumb aperture for cylindrical O-shape grasps:
# keep the thumb base/opposition high, while delaying thumb curl.
THUMB_O_WRAP_PRESHAPE_QPOS = np.array(
    [
        0.0920,
        0.0000,
        0.0000,
        0.0000,
        -0.0139,
        0.0000,
        0.0000,
        0.0000,
        -0.0885,
        0.0000,
        0.0000,
        0.0000,
        1.3600,
        0.0800,
        0.3200,
        0.1200,
    ],
    dtype=np.float32,
)


THUMB_O_WRAP_CLOSE_QPOS = np.array(
    [
        0.0920,
        1.6100,
        1.5500,
        0.9000,
        -0.0139,
        1.6100,
        1.5500,
        0.9000,
        -0.0885,
        1.6100,
        1.5500,
        0.9000,
        1.3600,
        0.5800,
        1.0200,
        0.2600,
    ],
    dtype=np.float32,
)


# Narrow thumb-index style pinch. Middle/ring stay tucked away so only
# thumb and index meaningfully move during the task. For the mustard
# benchmark, middle/ring should stay extended and splayed away rather
# than curled around the object.
PINCH_OPEN_QPOS = np.array(
    [
        0.0200,
        0.0000,
        0.0000,
        0.0000,
        0.3500,
        0.0000,
        0.0000,
        0.0000,
        -0.3500,
        0.0000,
        0.0000,
        0.0000,
        1.2800,
        0.0000,
        0.1000,
        0.0000,
    ],
    dtype=np.float32,
)


PINCH_PRESHAPE_QPOS = np.array(
    [
        0.0200,
        0.2400,
        0.1800,
        0.0400,
        0.3500,
        0.0000,
        0.0000,
        0.0000,
        -0.3500,
        0.0000,
        0.0000,
        0.0000,
        1.3600,
        0.0600,
        0.5000,
        0.0200,
    ],
    dtype=np.float32,
)


PINCH_THUMB_SETTLE_QPOS = np.array(
    [
        0.0200,
        0.2600,
        0.2000,
        0.0500,
        0.3500,
        0.0000,
        0.0000,
        0.0000,
        -0.3500,
        0.0000,
        0.0000,
        0.0000,
        1.3900,
        0.1200,
        0.9500,
        0.0500,
    ],
    dtype=np.float32,
)


PINCH_CLOSE_QPOS = np.array(
    [
        0.0200,
        0.8500,
        0.7500,
        0.2800,
        0.3500,
        0.0000,
        0.0000,
        0.0000,
        -0.3500,
        0.0000,
        0.0000,
        0.0000,
        1.3900,
        0.1200,
        1.1200,
        0.0800,
    ],
    dtype=np.float32,
)


# Pointing pose for pushing with the index fingertip. Other fingers stay curled.
POINT_PUSH_OPEN_QPOS = np.array(
    [
        0.0200,
        0.0200,
        0.0200,
        0.0100,
        0.0200,
        1.3000,
        1.3200,
        0.8200,
        -0.0500,
        1.2600,
        1.3000,
        0.8000,
        1.1800,
        0.4200,
        0.7000,
        0.1800,
    ],
    dtype=np.float32,
)


POINT_PUSH_CONTACT_QPOS = np.array(
    [
        0.0600,
        0.1000,
        0.0800,
        0.0100,
        0.0200,
        1.3000,
        1.3200,
        0.8200,
        -0.0500,
        1.2600,
        1.3000,
        0.8000,
        1.1800,
        0.5200,
        0.8600,
        0.2200,
    ],
    dtype=np.float32,
)


# Hook pull should keep the same overall hand as push, but curl only the
# index fingertip so the distal link can catch the bottle neck.
HOOK_PULL_OPEN_QPOS = POINT_PUSH_OPEN_QPOS.copy()
HOOK_PULL_CONTACT_QPOS = POINT_PUSH_CONTACT_QPOS.copy()
HOOK_PULL_CONTACT_QPOS[2] = 0.1800
HOOK_PULL_CONTACT_QPOS[3] = 1.0500


# Wrap-grasp support for table-top rotation.
ROTATE_WRAP_PRESHAPE_QPOS = np.array(
    [
        0.0900,
        0.3000,
        0.2400,
        0.0800,
        -0.0100,
        0.2600,
        0.2200,
        0.0800,
        -0.0900,
        0.2600,
        0.2200,
        0.0800,
        1.2400,
        0.1400,
        0.4200,
        0.1400,
    ],
    dtype=np.float32,
)


ROTATE_WRAP_CLOSE_QPOS = np.array(
    [
        0.0920,
        1.3000,
        1.1800,
        0.5800,
        -0.0139,
        1.2800,
        1.1600,
        0.5600,
        -0.0885,
        1.2600,
        1.1200,
        0.5400,
        1.3000,
        0.5200,
        0.9800,
        0.2400,
    ],
    dtype=np.float32,
)


def _trajectory_open_pose(trajectory: str) -> np.ndarray:
    if trajectory == "pinch_precision":
        return PINCH_OPEN_QPOS
    if trajectory == "hook_pull":
        return HOOK_PULL_OPEN_QPOS
    if trajectory == "index_point_push":
        return POINT_PUSH_OPEN_QPOS
    return np.zeros(16, dtype=np.float32)


def interpolate_allegro_hand_pose(
    grasp_value: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    trajectory: str,
    preshape_pivot: float = 0.45,
) -> np.ndarray:
    g = float(np.clip(grasp_value, 0.0, 1.0))
    pivot = float(np.clip(preshape_pivot, 0.05, 0.95))
    open_q = _trajectory_open_pose(trajectory)

    if trajectory == "keti_human":
        q = (1.0 - g) * open_q + g * KETI_HUMAN_CLOSE_QPOS
    elif trajectory == "thumb_opposition":
        if g <= pivot:
            alpha = g / pivot
            q = (1.0 - alpha) * open_q + alpha * POWER_GRASP_PRESHAPE_QPOS
        else:
            beta = (g - pivot) / max(1.0 - pivot, 1e-6)
            q = (1.0 - beta) * POWER_GRASP_PRESHAPE_QPOS + beta * POWER_GRASP_CLOSE_QPOS
    elif trajectory == "thumb_o_wrap":
        if g <= pivot:
            alpha = g / pivot
            q = (1.0 - alpha) * open_q + alpha * THUMB_O_WRAP_PRESHAPE_QPOS
        else:
            beta = (g - pivot) / max(1.0 - pivot, 1e-6)
            q = (1.0 - beta) * THUMB_O_WRAP_PRESHAPE_QPOS + beta * THUMB_O_WRAP_CLOSE_QPOS
    elif trajectory == "pinch_precision":
        if g <= pivot:
            alpha = g / pivot
            q = (1.0 - alpha) * open_q + alpha * PINCH_PRESHAPE_QPOS
        elif g <= 0.80:
            beta = (g - pivot) / max(0.80 - pivot, 1e-6)
            q = (1.0 - beta) * PINCH_PRESHAPE_QPOS + beta * PINCH_THUMB_SETTLE_QPOS
        else:
            gamma = (g - 0.80) / max(1.0 - 0.80, 1e-6)
            q = (1.0 - gamma) * PINCH_THUMB_SETTLE_QPOS + gamma * PINCH_CLOSE_QPOS
    elif trajectory == "hook_pull":
        q = (1.0 - g) * HOOK_PULL_OPEN_QPOS + g * HOOK_PULL_CONTACT_QPOS
    elif trajectory == "index_point_push":
        q = (1.0 - g) * POINT_PUSH_OPEN_QPOS + g * POINT_PUSH_CONTACT_QPOS
    elif trajectory == "rotate_wrap":
        if g <= pivot:
            alpha = g / pivot
            q = (1.0 - alpha) * open_q + alpha * ROTATE_WRAP_PRESHAPE_QPOS
        else:
            beta = (g - pivot) / max(1.0 - pivot, 1e-6)
            q = (1.0 - beta) * ROTATE_WRAP_PRESHAPE_QPOS + beta * ROTATE_WRAP_CLOSE_QPOS
    elif trajectory == "power_linear":
        q = (1.0 - g) * open_q + g * POWER_GRASP_CLOSE_QPOS
    else:
        raise ValueError(f"Unsupported hand trajectory: {trajectory}")

    return np.clip(q, q_min, q_max).astype(np.float32)
