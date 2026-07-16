import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rby1_sdk as rby


def load_offset_from_json(filename="calibration_result.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    if "joint_offset_deg" in data:
        arm_offset_deg = np.array(data["joint_offset_deg"], dtype=np.float64)
    elif (
        data.get("right_arm_joint_offset_deg") is not None
        and data.get("left_arm_joint_offset_deg") is not None
    ):
        arm_offset_deg = np.concatenate([
            np.array(data["right_arm_joint_offset_deg"], dtype=np.float64),
            np.array(data["left_arm_joint_offset_deg"], dtype=np.float64),
        ])
    else:
        raise KeyError("joint_offset_deg is required in calibration result JSON")

    head_offset_deg = data.get("head_joint_offset_deg")
    head_offset_rad = None
    if head_offset_deg is not None:
        head_offset_rad = np.deg2rad(np.array(head_offset_deg, dtype=np.float64))
    return np.deg2rad(arm_offset_deg), head_offset_rad


def _split_arm_offset(model, arm, offset_rad):
    right_arm_dof = len(model.right_arm_idx)
    left_arm_dof = len(model.left_arm_idx)

    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    offset_rad = np.array(offset_rad, dtype=np.float64).reshape(-1)
    if len(offset_rad) == right_arm_dof + left_arm_dof:
        return (
            "both",
            offset_rad[:right_arm_dof],
            offset_rad[right_arm_dof:],
        )
    if arm == "right" and len(offset_rad) == right_arm_dof:
        return "right", offset_rad, np.zeros(left_arm_dof)
    if arm == "left" and len(offset_rad) == left_arm_dof:
        return "left", np.zeros(right_arm_dof), offset_rad

    expected = f"{right_arm_dof + left_arm_dof} (both arms)"
    if arm == "right":
        expected += f" or {right_arm_dof} (right arm)"
    elif arm == "left":
        expected += f" or {left_arm_dof} (left arm)"
    raise RuntimeError(
        f"Offset size mismatch: expected {expected}, got {len(offset_rad)}"
    )


def _normalize_head_offset(model, head_offset_rad, include_head):
    head_dof = len(model.head_idx)
    if not include_head or head_offset_rad is None or head_dof == 0:
        return None, 0

    head_offset_rad = np.array(head_offset_rad, dtype=np.float64).reshape(-1)
    if len(head_offset_rad) > head_dof:
        raise RuntimeError(
            f"Head offset size mismatch: expected up to {head_dof}, got {len(head_offset_rad)}"
        )

    head_offset_full = np.zeros(head_dof, dtype=np.float64)
    head_offset_full[:len(head_offset_rad)] = head_offset_rad
    return head_offset_full, len(head_offset_rad)


def build_home_reset_baseline_data(robot, model, model_name=None, include_head=True):
    state = robot.get_state()
    q_full = np.array(state.position, dtype=np.float64).reshape(-1)

    right_offset_rad = q_full[model.right_arm_idx].copy()
    left_offset_rad = q_full[model.left_arm_idx].copy()
    head_offset_rad = None
    if include_head and len(model.head_idx) > 0:
        head_offset_rad = q_full[model.head_idx].copy()

    data = {
        "joint_offset_deg": np.rad2deg(
            np.concatenate([right_offset_rad, left_offset_rad])
        ).tolist(),
        "right_arm_joint_offset_deg": np.rad2deg(right_offset_rad).tolist(),
        "left_arm_joint_offset_deg": np.rad2deg(left_offset_rad).tolist(),
        "head_joint_offset_deg": (
            None if head_offset_rad is None else np.rad2deg(head_offset_rad).tolist()
        ),
        "metadata": {
            "type": "home_reset_baseline",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "include_head": bool(include_head),
        },
    }
    return data


def save_home_reset_baseline_json(
    robot,
    model,
    output_dir,
    model_name=None,
    include_head=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "home_reset_baseline.json"

    data = build_home_reset_baseline_data(
        robot,
        model,
        model_name=model_name,
        include_head=include_head,
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return path, data


def movej(robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=5):
    rc = rby.BodyComponentBasedCommandBuilder()

    if right_arm is not None:
        rc.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(right_arm)
        )

    if left_arm is not None:
        rc.set_left_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(left_arm)
        )

    rc.set_torso_command(
        rby.JointPositionCommandBuilder()
        .set_minimum_time(minimum_time)
        .set_position(np.zeros(6))
    )

    cmd = rby.ComponentBasedCommandBuilder().set_body_command(rc)
    if head is not None:
        cmd.set_head_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time)
            .set_position(head)
        )

    rv = robot.send_command(
        rby.RobotCommandBuilder().set_command(cmd),
        1,
    ).get()

    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        logging.error("Failed to conduct movej.")
        return False

    return True


def initialize_robot(address, model, power=".*", servo=".*"):
    robot = rby.create_robot(address, model)

    if not robot.connect():
        raise RuntimeError(f"Failed to connect robot: {address}")

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            raise RuntimeError("Power on failed")

    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            raise RuntimeError("Servo on failed")

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        robot.reset_fault_control_manager()

    robot.enable_control_manager()
    return robot

def move_robot_to_zero_pose(address, model_name, arm, power=".*", servo=".*", include_head=True):
    robot = initialize_robot(address, model_name, power, servo)
    model = robot.model()

    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if include_head else None

    ok = movej(
        robot,
        right_arm=right_zero_pose,
        left_arm=left_zero_pose,
        head=head_zero_pose,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose")

    return {
        "status": "success",
        "arm": arm,
        "message": "Robot moved to zero pose. Please compare it with the reference image.",
    }


def move_to_offset_candidate(
    robot,
    model,
    arm,
    offset_rad,
    head_offset_rad=None,
    include_head=True,
    minimum_time=10,
    move_zero_first=True,
    settle_time=2.0,
):
    apply_mode, right_offset_rad, left_offset_rad = _split_arm_offset(
        model,
        arm,
        offset_rad,
    )

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if include_head else None
    head_offset_full, head_offset_size = _normalize_head_offset(
        model,
        head_offset_rad,
        include_head,
    )

    right_offset_to_apply = -right_offset_rad
    left_offset_to_apply = -left_offset_rad
    head_offset_to_apply = None if head_offset_full is None else -head_offset_full

    if move_zero_first:
        ok = movej(
            robot,
            right_arm=right_zero_pose,
            left_arm=left_zero_pose,
            head=head_zero_pose,
            minimum_time=5,
        )
        if not ok:
            raise RuntimeError("Failed to move robot to zero pose")
        time.sleep(settle_time)

    head_target_pose = None
    if include_head:
        head_target_pose = (
            head_zero_pose
            if head_offset_to_apply is None
            else head_zero_pose + head_offset_to_apply
        )

    ok = movej(
        robot,
        right_arm=right_zero_pose + right_offset_to_apply,
        left_arm=left_zero_pose + left_offset_to_apply,
        head=head_target_pose if include_head else None,
        minimum_time=minimum_time,
    )
    if not ok:
        raise RuntimeError("Failed to move robot with offset pose")

    time.sleep(settle_time)

    offset_to_apply = np.concatenate([right_offset_to_apply, left_offset_to_apply])
    head_offset_deg = None
    if head_offset_to_apply is not None:
        head_offset_deg = np.rad2deg(head_offset_to_apply[:head_offset_size]).tolist()

    return {
        "status": "success",
        "arm": apply_mode,
        "offset_deg": np.rad2deg(offset_to_apply).tolist(),
        "right_offset_deg": np.rad2deg(right_offset_to_apply).tolist(),
        "left_offset_deg": np.rad2deg(left_offset_to_apply).tolist(),
        "head_offset_deg": head_offset_deg,
    }


def move_to_offset_candidate_from_json(
    robot,
    model,
    arm,
    json_path,
    include_head=True,
    minimum_time=10,
    move_zero_first=True,
):
    offset_rad, head_offset_rad = load_offset_from_json(json_path)
    result = move_to_offset_candidate(
        robot=robot,
        model=model,
        arm=arm,
        offset_rad=offset_rad,
        head_offset_rad=head_offset_rad,
        include_head=include_head,
        minimum_time=minimum_time,
        move_zero_first=move_zero_first,
    )
    result["source"] = "json"
    result["json_path"] = str(json_path)
    return result


def reset_current_pose_home_offsets(
    robot,
    model,
    arm="both",
    include_head=True,
    log_cb=None,
    power_cycle=True,
):
    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    if log_cb is not None:
        log_cb("Starting Home Offset Reset from current pose...")

    robot.disable_control_manager()
    time.sleep(0.5)

    failed_joints = []

    def reset_joint_group(prefix, dof, label):
        for i in range(dof):
            joint_name = f"{prefix}_{i}"
            success = robot.home_offset_reset(joint_name)
            if not success:
                failed_joints.append(joint_name)
                if log_cb is not None:
                    log_cb(f"Failed to reset {label} joint: {joint_name}")
            elif log_cb is not None:
                log_cb(f"Reset {label} joint OK: {joint_name}")

    if arm in ("right", "both"):
        reset_joint_group("right_arm", len(model.right_arm_idx), "right arm")
    if arm in ("left", "both"):
        reset_joint_group("left_arm", len(model.left_arm_idx), "left arm")
    if include_head:
        reset_joint_group("head", len(model.head_idx), "head")

    all_success = len(failed_joints) == 0
    if log_cb is not None:
        if all_success:
            log_cb("All selected joints reset successfully!")
        else:
            log_cb("Some joints failed to reset. Proceeding with power cycle...")

    if power_cycle:
        if log_cb is not None:
            log_cb("Disabling control manager and waiting 2 seconds...")
        robot.disable_control_manager()
        time.sleep(2.0)

        if log_cb is not None:
            log_cb("Powering off 48V power...")
        robot.power_off("48v")
        time.sleep(1.5)

    return {
        "status": "success" if all_success else "partial_failure",
        "success": all_success,
        "arm": arm,
        "include_head": bool(include_head),
        "failed_joints": failed_joints,
    }


def apply_home_offset(
    address,
    model_name,
    arm,
    offset_rad,
    head_offset_rad=None,
    power=".*",
    servo=".*",
    include_head=True,
):
    robot = initialize_robot(address, model_name, power, servo)
    model = robot.model()

    move_result = move_to_offset_candidate(
        robot=robot,
        model=model,
        arm=arm,
        offset_rad=offset_rad,
        head_offset_rad=head_offset_rad,
        include_head=include_head,
        minimum_time=10,
        move_zero_first=True,
    )

    reset_result = reset_current_pose_home_offsets(
        robot=robot,
        model=model,
        arm=move_result["arm"],
        include_head=include_head and head_offset_rad is not None,
    )
    if not reset_result["success"]:
        raise RuntimeError(f"Failed to reset joints: {reset_result['failed_joints']}")

    robot = initialize_robot(address, model_name, power=power, servo=servo)

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if include_head else None
    ok = movej(
        robot,
        right_arm=right_zero_pose,
        left_arm=left_zero_pose,
        head=head_zero_pose,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose after reset")

    move_result["reset_result"] = reset_result
    return move_result


def apply_home_offset_from_json(
    address,
    model_name,
    arm="right",
    json_path="calibration_result.json",
    power=".*",
    servo=".*",
    include_head=True,
):
    offset_rad, head_offset_rad = load_offset_from_json(json_path)

    result = apply_home_offset(
        address=address,
        model_name=model_name,
        arm=arm,
        offset_rad=offset_rad,
        head_offset_rad=head_offset_rad,
        power=power,
        servo=servo,
        include_head=include_head,
    )

    result["source"] = "json"
    result["json_path"] = json_path
    return result


def reset_home_offsets(robot, model, log_cb=None):
    result = reset_current_pose_home_offsets(
        robot,
        model,
        arm="both",
        include_head=True,
        log_cb=log_cb,
    )
    return result["success"]
