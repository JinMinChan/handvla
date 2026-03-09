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


def interpolate_allegro_hand_pose(
    grasp_value: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    trajectory: str,
    preshape_pivot: float = 0.45,
) -> np.ndarray:
    g = float(np.clip(grasp_value, 0.0, 1.0))
    pivot = float(np.clip(preshape_pivot, 0.05, 0.95))
    open_q = np.zeros(16, dtype=np.float32)

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
    elif trajectory == "power_linear":
        q = (1.0 - g) * open_q + g * POWER_GRASP_CLOSE_QPOS
    else:
        raise ValueError(f"Unsupported hand trajectory: {trajectory}")

    return np.clip(q, q_min, q_max).astype(np.float32)
