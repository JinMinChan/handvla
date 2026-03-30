import os

import mujoco
import numpy as np


FINGER_KEYS = ("ff", "mf", "rf", "th")
MUSTARD_PREFIX = "mustard/"


def _add_tcp_markers(hand_spec: mujoco.MjSpec) -> None:
    marker_specs = {
        "ff_tip": ("ff_tcp", [0.02, 0.0, 0.028]),
        "mf_tip": ("mf_tcp", [0.02, 0.0, 0.028]),
        "rf_tip": ("rf_tcp", [0.02, 0.0, 0.028]),
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


def _add_table(scene: mujoco.MjSpec) -> None:
    # Simple table under the manipulation area.
    table_center = np.array([0.62, 0.0, 0.70], dtype=float)
    table_half = np.array([0.36, 0.48, 0.04], dtype=float)
    leg_half = np.array([0.04, 0.04, 0.33], dtype=float)

    scene.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=table_half.tolist(),
        pos=table_center.tolist(),
        rgba=[0.68, 0.68, 0.68, 1.0],
        friction=[1.0, 0.05, 0.01],
    )

    leg_dx = table_half[0] - leg_half[0] - 0.03
    leg_dy = table_half[1] - leg_half[1] - 0.03
    leg_z = table_center[2] - table_half[2] - leg_half[2]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            scene.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=leg_half.tolist(),
                pos=[table_center[0] + sx * leg_dx, table_center[1] + sy * leg_dy, leg_z],
                rgba=[0.38, 0.38, 0.38, 1.0],
                friction=[1.0, 0.05, 0.01],
            )


def _attach_mustard(scene: mujoco.MjSpec, env_dir: str) -> None:
    mustard_xml_path = os.path.join(
        env_dir, "assets", "ycb", "006_mustard_bottle", "006_mustard_bottle.xml"
    )
    mustard = mujoco.MjSpec.from_file(mustard_xml_path)

    # Mustard on table, close to reachable pre-grasp region.
    mustard_frame = scene.worldbody.add_frame(
        pos=[0.62, 0.06, 0.82],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    scene.attach(child=mustard, prefix=MUSTARD_PREFIX, frame=mustard_frame)


def _add_frame_axes(
    scene: mujoco.MjSpec,
    body_name: str,
    name_prefix: str,
    axis_len: float = 0.08,
    axis_radius: float = 0.003,
) -> None:
    body = scene.body(body_name)
    axes = (
        ("x", [axis_len, 0.0, 0.0], [1.0, 0.15, 0.15, 0.95]),
        ("y", [0.0, axis_len, 0.0], [0.15, 1.0, 0.15, 0.95]),
        ("z", [0.0, 0.0, axis_len], [0.20, 0.40, 1.0, 0.95]),
    )
    for axis_name, endpoint, rgba in axes:
        body.add_geom(
            name=f"{name_prefix}_{axis_name}_axis",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, endpoint[0], endpoint[1], endpoint[2]],
            size=[axis_radius, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=2,
            mass=0.0,
        )


def load(
    side: str = "right",
    add_mustard: bool = True,
    add_frame_axes: bool = False,
) -> mujoco.MjSpec:
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported side: {side}")

    env_dir = os.path.dirname(os.path.abspath(__file__))

    scene_xml_path = os.path.join(env_dir, "assets", "franka_scene.xml")
    scene = mujoco.MjSpec.from_file(scene_xml_path)
    _add_table(scene)

    franka_xml_path = os.path.join(env_dir, "assets", "franka_emika_panda", "panda_nohand.xml")
    franka = mujoco.MjSpec.from_file(franka_xml_path)
    franka.modelname = "franka"

    hand_xml_path = os.path.join(env_dir, "assets", "wonik_allegro", f"{side}_hand.xml")
    hand = mujoco.MjSpec.from_file(hand_xml_path)
    _add_tcp_markers(hand)

    # Attach Allegro to the Franka flange frame.
    flange_body = franka.body("attachment")
    adaptor = flange_body.add_body(name="allegro_mount", pos=[0.0, 0.0, 0.0])
    adaptor.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.03773072, 0.005, 0],
        rgba=[0.2, 0.2, 0.2, 1.0],
        pos=[0.0, 0.0, -0.005],
    )
    hand_attachment_frame = adaptor.add_frame(pos=[0.0, 0.0, 0.095], euler=[0.0, -np.pi / 2, 0.0])
    franka.attach(child=hand, prefix=f"allegro_{side}/", frame=hand_attachment_frame)

    # Place Franka base on the tabletop near the rear-left side.
    franka_attachment_frame = scene.worldbody.add_frame(
        pos=[0.30, -0.10, 0.74],
        euler=[0.0, 0.0, 0.0],
    )
    scene.attach(child=franka, prefix=f"{franka.modelname}/", frame=franka_attachment_frame)

    if add_mustard:
        _attach_mustard(scene, env_dir)
    if add_frame_axes:
        _add_frame_axes(scene, f"franka/allegro_{side}/palm", "palm", axis_len=0.12, axis_radius=0.004)
        _add_frame_axes(
            scene,
            f"franka/allegro_{side}/ff_distal",
            "ff_distal",
            axis_len=0.05,
            axis_radius=0.0025,
        )
        if add_mustard:
            _add_frame_axes(
                scene,
                f"{MUSTARD_PREFIX}006_mustard_bottle",
                "mustard",
                axis_len=0.10,
                axis_radius=0.004,
            )

    franka_qpos = [0.0, -0.6, 0.0, -2.05, 0.0, 1.55, 0.7]
    hand_qpos = [
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
    initial_qpos = franka_qpos + hand_qpos
    if add_mustard:
        # mustard freejoint qpos: [x, y, z, qw, qx, qy, qz]
        initial_qpos += [0.62, 0.06, 0.82, 1.0, 0.0, 0.0, 0.0]

    initial_ctrl = franka_qpos + hand_qpos
    scene.add_key(name="initial_state", qpos=initial_qpos, ctrl=initial_ctrl)

    scene.compile()
    return scene
