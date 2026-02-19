import mujoco
import numpy as np


def set_default_hand_camera(cam: mujoco.MjvCamera) -> None:
    """Apply the default close frontal camera used in handvla viewers."""
    cam.lookat[:] = np.array([-0.1, -0.05, 0.24], dtype=float)
    cam.distance = 0.43
    cam.azimuth = 25.0
    cam.elevation = -12.0
