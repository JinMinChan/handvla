import os

import mujoco
import numpy as np


FINGER_KEYS = ("ff", "mf", "rf", "th")


def _add_tcp_markers(hand_spec: mujoco.MjSpec) -> None:
    # TCP markers near fingertip pads, shifted slightly toward the palm.
    marker_specs = {
        # White fingertip visuals are around z=0.0147; move 6 mm palm-ward.
        "ff_tip": ("ff_tcp", [0.02, 0.0, 0.028]),
        "mf_tip": ("mf_tcp", [0.02, 0.0, 0.028]),
        "rf_tip": ("rf_tcp", [0.02, 0.0, 0.028]),
        # White thumb visual is around z=0.0303; move 10 mm palm-ward.
        "th_tip": ("th_tcp", [0.02, 0.0, 0.043]),
    }

    for body_name, (marker_name, marker_pos) in marker_specs.items():
        body = hand_spec.body(body_name)

        body.add_geom(
            name=f"{marker_name}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0065, 0, 0],
            pos=marker_pos,
            rgba=[1.0, 0.0, 0.0, 1.0],
            contype=0,
            conaffinity=0,
            group=1,
            mass=0.0,
        )
        body.add_site(
            name=marker_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0065, 0, 0],
            pos=marker_pos,
            rgba=[1.0, 0.1, 0.1, 1.0],
            group=5,
        )


def _add_ik_target_markers(scene: mujoco.MjSpec, side: str) -> None:
    prefix = f"allegro_{side}"
    for finger in FINGER_KEYS:
        target_body = scene.worldbody.add_body(
            name=f"{prefix}/{finger}_target_body", mocap=True, pos=[0.0, 0.0, 0.0]
        )
        target_body.add_geom(
            name=f"{prefix}/{finger}_target_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[0.0, 0.0, 0.0],
            size=[0.008, 0, 0],
            rgba=[0.15, 1.0, 0.2, 0.95],
            contype=0,
            conaffinity=0,
            group=1,
            mass=0.0,
        )
        target_body.add_site(
            name=f"{prefix}/{finger}_target",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[0.0, 0.0, 0.0],
            size=[0.008, 0, 0],
            rgba=[0.15, 1.0, 0.2, 1.0],
            group=5,
        )


def load(side: str = "right", add_ik_targets: bool = False) -> mujoco.MjSpec:
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported side: {side}")

    env_dir = os.path.dirname(os.path.abspath(__file__))

    scene_xml_path = os.path.join(env_dir, "assets", "hand_scene.xml")
    scene = mujoco.MjSpec.from_file(scene_xml_path)

    hand_xml_path = os.path.join(env_dir, "assets", "wonik_allegro", f"{side}_hand.xml")
    hand = mujoco.MjSpec.from_file(hand_xml_path)
    _add_tcp_markers(hand)

    # Make the wrist stay lower and fingers point upward (+z in world frame).
    hand_attachment_frame = scene.worldbody.add_frame(
        pos=[0.0, 0.0, 0.18], euler=[0.0, -np.pi / 2, 0.0]
    )
    scene.attach(child=hand, prefix=hand.modelname + "/", frame=hand_attachment_frame)
    if add_ik_targets:
        _add_ik_target_markers(scene, side)

    initial_qpos = [
        0.0,
        0.35,
        0.15,
        0.10,
        0.0,
        0.35,
        0.15,
        0.10,
        0.0,
        0.35,
        0.15,
        0.10,
        0.60,
        0.10,
        0.20,
        0.20,
    ]
    scene.add_key(name="initial_state", qpos=initial_qpos, ctrl=initial_qpos)

    scene.compile()
    return scene
