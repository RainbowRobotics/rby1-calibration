import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rby1_sdk as rby
import yaml

from marker_detection import Marker_Transform
from homeoffset_core import reset_home_offsets
from robot_motion import check_calibration_state


np.set_printoptions(suppress=True, precision=6)

BASE_DIR = Path(__file__).resolve().parent.parent
SETTING_PATH = BASE_DIR / "config" / "setting.yaml"
DEFAULT_LAMBDA_CAM_POS = 1.0
DEFAULT_LAMBDA_CAM_ROT = 1.0

ARM_SIDES = ("right", "left")
D2R = np.pi / 180.0


# ============================================================
# Auto data-collection motion
# ============================================================

# (Auto motion logic has been moved to core.robot_motion)


from core.calibration_optimizer import (
    adjoint, make_transform, so3_exp, se3_exp, so3_log, se3_log, rot_to_euler_zyx,
    CalibrationOptimizer, QPCalibrationOptimizer, compute_fk, prepare_q_full,
    DEFAULT_LAMBDA_CAM_POS, DEFAULT_LAMBDA_CAM_ROT
)
# ============================================================
# Config / helpers
# ============================================================
def create_robot(ip, model_name="a", power_regex=".*", servo_regex=".*"):
    robot = rby.create_robot(ip, model_name)
    robot.connect()
    time.sleep(1)
    if not robot.is_power_on(power_regex):
        robot.power_on(power_regex)
        time.sleep(1)
    if not robot.is_servo_on(servo_regex):
        robot.servo_on(servo_regex)
    time.sleep(1) 
    robot.reset_fault_control_manager()
    robot.enable_control_manager(False)
    return robot

def load_npz_dataset(path):
    data = np.load(path)
    q_arm = data["q_arm"] if "q_arm" in data else data["q"]
    q_head = data["q_head"] if "q_head" in data else None
    return q_arm, q_head, data["marker"]

def save_npz_dataset(path, q_arm, T_meas, q_head=None):
    save_kwargs = {
        "q": q_arm,
        "q_arm": q_arm,
        "marker": T_meas,
    }
    if q_head is not None:
        save_kwargs["q_head"] = q_head
    np.savez_compressed(path, **save_kwargs)


def validate_dataset(q_arm, q_head, T_meas, optimize_head, active_arms):
    if len(q_arm) != len(T_meas):
        raise RuntimeError(
            f"Dataset size mismatch: q_arm={len(q_arm)}, marker={len(T_meas)}"
        )

    if q_head is not None and len(q_head) != len(q_arm):
        raise RuntimeError(
            f"Dataset size mismatch: q_head={len(q_head)}, q_arm={len(q_arm)}"
        )

    if q_head is None and optimize_head:
        raise RuntimeError(
            "Head-mounted camera calibration requires `q_head`, but the loaded npz does not contain it."
        )

    if q_arm.ndim != 2:
        raise RuntimeError(f"Expected q_arm to be a 2D array, got shape {q_arm.shape}")

    expected_q_arm_len = 7 * len(active_arms)
    if q_arm.shape[1] != expected_q_arm_len:
        raise RuntimeError(
            f"Unsupported q_arm width {q_arm.shape[1]}. Expected {expected_q_arm_len} for arms: {active_arms}."
        )

    if len(active_arms) > 1:
        if T_meas.ndim != 4 or T_meas.shape[1:] != (len(active_arms), 4, 4):
            raise RuntimeError(
                f"Expected marker measurements with shape (N, {len(active_arms)}, 4, 4), got {T_meas.shape}"
            )
    else:
        if T_meas.ndim != 3 or T_meas.shape[1:] != (4, 4):
            raise RuntimeError(
                f"Expected marker measurements with shape (N, 4, 4), got {T_meas.shape}"
            )

def split_arm_offsets(q_offset):
    q_offset = np.asarray(q_offset, dtype=np.float64).reshape(-1)
    if len(q_offset) == 14:
        return q_offset[:7], q_offset[7:]
    return q_offset, None

def load_camera_nominals(version="1.2"):
    with open(SETTING_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    camera_cfg = config.get("camera", {})
    mount_to_cam_nom = camera_cfg.get("mount_to_cam")
    head_base_to_cam_nom = camera_cfg.get("head_base_to_cam")

    if str(version) == "1.3":
        ee_to_marker_left = camera_cfg.get("Tf_to_marker_left_v13", camera_cfg.get("Tf_to_marker_left"))
        ee_to_marker_right = camera_cfg.get("Tf_to_marker_right_v13", camera_cfg.get("Tf_to_marker_right"))
    else:
        ee_to_marker_left = camera_cfg.get("Tf_to_marker_left", camera_cfg.get("Tf_to_marker_left_v12"))
        ee_to_marker_right = camera_cfg.get("Tf_to_marker_right", camera_cfg.get("Tf_to_marker_right_v12"))

    return {
        "mount_to_cam_nom": mount_to_cam_nom,
        "head_base_to_cam_nom": head_base_to_cam_nom,
        "camera_mount_link": camera_cfg.get("camera_mount_link", "link_head_2"),
        "ee_to_marker_left": ee_to_marker_left,
        "ee_to_marker_right": ee_to_marker_right,
    }

def get_arm_config(model, arm, version="1.2"):
    camera_nominals = load_camera_nominals(version=version)
    base_config = {
        "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
        "head_base_to_cam_nom": camera_nominals["head_base_to_cam_nom"],
        "camera_mount_link": camera_nominals["camera_mount_link"],
    }

    if arm == "right":
        base_config.update({
            "arm_idx": model.right_arm_idx[:7],
            "ee_link": "ee_right",
            "ee_to_marker_nom": camera_nominals["ee_to_marker_right"],
        })
    else:
        base_config.update({
            "arm_idx": model.left_arm_idx[:7],
            "ee_link": "ee_left",
            "ee_to_marker_nom": camera_nominals["ee_to_marker_left"],
        })
    return base_config

def get_both_arm_config(model, version="1.2"):
    camera_nominals = load_camera_nominals(version=version)
    return {
        "arm_idx": np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]]),
        "ee_links": {
            "right": "ee_right",
            "left": "ee_left",
        },
        "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
        "head_base_to_cam_nom": camera_nominals["head_base_to_cam_nom"],
        "camera_mount_link": camera_nominals["camera_mount_link"],
        "ee_to_marker_nom": {
            "right": camera_nominals["ee_to_marker_right"],
            "left": camera_nominals["ee_to_marker_left"],
        },
    }

def get_head_config(model):
    camera_nominals = load_camera_nominals()
    head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
    return {
        "head_idx": head_idx,
        "camera_link": camera_nominals["camera_mount_link"],
    }


# ============================================================
# Capture dataset
# ============================================================

def create_live_marker_transform():
    marker_transform = Marker_Transform(
        serial_number=None
    )
    marker_transform.marker_detection.set_marker_type("plate")
    return marker_transform


def capture_one_sample(robot, arm_idx, marker_transform, sampling_time=1, side="all", head_idx=None):
    state = robot.get_state()
    q_full = state.position.copy()
    q_arm = q_full[arm_idx].copy()
    q_head = q_full[head_idx].copy() if head_idx is not None else None

    result = marker_transform.get_marker_transform(sampling_time=sampling_time, side=side)
    if result is None:
        return None, None, None

    # side="all" returns [right, left] where each entry is a flattened 4x4.
    # If either side is missing, skip this sample gracefully.
    if side == "all":
        if len(result) < 2 or result[0] is None or result[1] is None:
            return None, None, None

        def _to_tf(flat_tf):
            arr = np.asarray(flat_tf, dtype=np.float64).reshape(-1)
            if arr.size != 16:
                raise RuntimeError(
                    f"Expected one marker transform to contain 16 values, got shape {np.asarray(flat_tf).shape}"
                )
            return arr.reshape(4, 4)

        T_right = _to_tf(result[0])
        T_left = _to_tf(result[1])
        return q_arm, q_head, np.stack([T_right, T_left], axis=0)

    T_meas = np.asarray(result, dtype=np.float64).reshape(-1)
    if T_meas.size != 16:
        raise RuntimeError(
            f"Expected marker transform with 16 values for side='{side}', got shape {np.asarray(result).shape}"
        )
    return q_arm, q_head, T_meas.reshape(4, 4)



# ============================================================
# Optimizer
# ============================================================

def generate_sim_measurements(
    robot,
    dyn_model,
    q_arm_list,
    q_head_list,
    arm_idx,
    head_idx,
    q_nominal,
    optimize_arm,
    optimize_head,
    optimize_camera,
    active_arms,
    ee_links,
    mount_to_cam_nom,
    head_base_to_cam_nom,
    ee_to_marker_nom,
    camera_link="link_head_2",
    camera_position_noise_std_m=0.0005,
    camera_orientation_noise_std_deg=0.5,
):
    q_offset_true = np.deg2rad([3, 0, 1, 2, -3, 2, 1, -2, -1, 3, 2, -4, 2, -2])
    if len(active_arms) == 1:
        q_offset_true = q_offset_true[:7]
        
    q_head_offset_true = np.deg2rad([2.0, -1.5])
    xi_t5_cam_true = np.array([0.01, -0.02, 0.03, 0.04, 0.05, -0.06])

    optimize_head = optimize_head and q_head_list is not None and head_idx is not None
    use_head_kinematics = optimize_head


    if use_head_kinematics:
        base_link = camera_link
        # In head mode, the camera extrinsic is a static transform from the
        # head mount link to the camera, so it must not depend on the current q.
        T_mount_to_cam_nom = make_transform(mount_to_cam_nom)
        T_mount_to_cam_true = (
            T_mount_to_cam_nom @ se3_exp(xi_t5_cam_true)
            if optimize_camera else T_mount_to_cam_nom
        )
    else:
        base_link = "link_head_0"
        T_mount_to_cam_nom = make_transform(head_base_to_cam_nom)
        T_mount_to_cam_true = (
            T_mount_to_cam_nom @ se3_exp(xi_t5_cam_true)
            if optimize_camera else T_mount_to_cam_nom
        )
    T_list = []

    if q_head_list is None:
        q_head_iter = [None] * len(q_arm_list)
    else:
        q_head_iter = q_head_list

    for q_arm, q_head in zip(q_arm_list, q_head_iter):
        q_full = prepare_q_full(
            q_nominal=q_nominal,
            arm_idx=arm_idx,
            q_cmd=q_arm,
            q_offset=q_offset_true if optimize_arm else None,
            head_idx=head_idx if use_head_kinematics else None,
            q_head=q_head,
            q_head_offset=q_head_offset_true if optimize_head else None,
        )

        T_pair = []
        for arm_side in active_arms:
            _, T_fk = compute_fk(robot, dyn_model, q_full, ee_links[arm_side], base_link=base_link)
            T_ee_to_marker = make_transform(ee_to_marker_nom[arm_side])
            T_meas = np.linalg.inv(T_mount_to_cam_true) @ T_fk @ T_ee_to_marker

            if camera_orientation_noise_std_deg > 0.0:
                rot_noise_rad = np.deg2rad(np.random.normal(
                    0.0,
                    camera_orientation_noise_std_deg,
                    size=3,
                ))
                R_noise = se3_exp(np.concatenate([rot_noise_rad, np.zeros(3)]))[:3, :3]
                T_meas[:3, :3] = R_noise @ T_meas[:3, :3]

            if camera_position_noise_std_m > 0.0:
                T_meas[:3, 3] += np.random.normal(
                    0.0,
                    camera_position_noise_std_m,
                    size=3,
                )

            T_pair.append(T_meas)
            
        if len(active_arms) == 1:
            T_list.append(T_pair[0])
        else:
            T_list.append(np.stack(T_pair, axis=0))

    return np.array(T_list)



