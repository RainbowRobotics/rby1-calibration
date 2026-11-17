from core.robot.robot_core import check_motion_cancelled, wait_motion
from core.language import tr
import os
from core.storage import StoragePaths, CONFIG_PATHS
from core.storage import ResultStorage, ConfigStorage
from core.storage import FileStorage
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import rby1_sdk as rby


class HomeOffsetError(RuntimeError):
    """A reset failed after some joints may already have changed physically."""
    def __init__(self, message, progress):
        super().__init__(message)
        self.partial = {"home": progress}


def load_offset_from_json(filename="calibration_result.json"):
    data = ResultStorage.load(filename)

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


def split_arm_offset(model, arm, offset_rad):
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


def normalize_head_offset(model, head_offset_rad, include_head):
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
        head_offset_rad = np.array([float(q_full[i]) for i in list(model.head_idx)], dtype=np.float64)

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


def home_reset_pending(path):
    """(pending, created_at) for a home_reset_baseline.json. Pending = the file still holds arm offsets:
    a reset was done but its calibration was not applied yet (Apply writes the offsets back to 0)."""
    path = Path(path)
    if not path.is_file():
        return False, None
    data = ResultStorage.load(path) or {}
    offsets = list(data.get("right_arm_joint_offset_deg") or []) + list(data.get("left_arm_joint_offset_deg") or [])
    pending = any(abs(float(value)) > 1e-6 for value in offsets)
    return pending, (data.get("metadata") or {}).get("created_at")


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
    ResultStorage.save(path, data)

    return path, data


def movej(robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=5):
    if head is not None:
        model = robot.model()
        has_head = hasattr(model, 'head_idx') and len(model.head_idx) > 0
        if not has_head:
            head = None

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

    check_motion_cancelled()
    rv = robot.send_command(
        rby.RobotCommandBuilder().set_command(cmd),
        1,
    ).get()

    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        logging.error("Failed to conduct movej.")
        return False

    return True


def initialize_robot(address, model, power="48v", servo="^(?!.*wheel).*$"):
    robot = rby.create_robot(address, model)

    if not robot.connect():
        raise RuntimeError(f"Failed to connect robot: {address}")

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            raise RuntimeError("Power on failed")
        wait_motion(1.0)

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        robot.reset_fault_control_manager()
        wait_motion(0.5)

    is_servo_ok = robot.is_servo_on(servo)
    cm_state = robot.get_control_manager_state().state
    is_cm_enabled = (cm_state == rby.ControlManagerState.State.Enabled)

    if not is_servo_ok:
        if is_cm_enabled:
            robot.disable_control_manager()
            wait_motion(0.5)
        if not robot.servo_on(servo):
            raise RuntimeError("Servo on failed")
        wait_motion(0.5)

    if not robot.is_control_manager_enabled():
        robot.enable_control_manager()
        wait_motion(0.5)

    return robot

def move_robot_to_zero_pose(address, model_name, arm, power="48v", servo=None, include_head=True):
    if servo is None:
        servo = "^(?!.*wheel).*$" if include_head else "^(?!.*(head|wheel)).*$"
    robot = initialize_robot(address, model_name, power, servo)
    model = robot.model()

    if arm not in ("right", "left", "both"):
        raise ValueError("arm must be 'right', 'left', or 'both'")

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if (include_head and hasattr(model, 'head_idx') and model.head_idx is not None and len(model.head_idx) >= 2) else None

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
    apply_mode, right_offset_rad, left_offset_rad = split_arm_offset(
        model,
        arm,
        offset_rad,
    )

    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = np.zeros(len(model.head_idx)) if include_head else None
    head_offset_full, head_offset_size = normalize_head_offset(
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
        wait_motion(settle_time)

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

    wait_motion(settle_time)

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


def validate_home_offset_joint_limits(robot, model, arm="both", include_head=True, log_cb=None):
    """
    Validates the arm posture signs for a Home Offset Reset from the current pose.
    Returns (is_valid, error_message).

    The ready-pose joint-limit check that also lived here was removed at the user's request
    (2026-10-16): it judged the reset against the old zero, so an elbow pushed back past +1.0 deg
    on purpose (offsets one way only) was refused for an elbow ready pose that is that very spot.
    """
    if robot is None or model is None:
        return True, ""

    try:
        state = robot.get_state()
        q_current = np.array(state.position, dtype=np.float64).reshape(-1)

        # Reset posture: shoulder roll in toward the body (right J1 +, left J1 -), elbows back (J3 +). Signs were
        # reversed until 2026-10-27; real resets (J3 up to +5.64 deg) had passed only inside the 5 deg buffer.
        tol = np.radians(5.0)  # 5° buffer
        invalid_reasons = []

        def reason(side, joint, value_rad):
            return tr(f"dialogs.home_reset_posture.{joint}_{side}", value=f"{np.degrees(value_rad):+.1f}")

        if arm in ("right", "both"):
            r_j1 = q_current[model.right_arm_idx[1]]
            r_j3 = q_current[model.right_arm_idx[3]]
            if r_j1 < -tol:
                invalid_reasons.append(reason("right", "j1", r_j1))
            if r_j3 < -tol:
                invalid_reasons.append(reason("right", "j3", r_j3))

        if arm in ("left", "both"):
            l_j1 = q_current[model.left_arm_idx[1]]
            l_j3 = q_current[model.left_arm_idx[3]]
            if l_j1 > tol:
                invalid_reasons.append(reason("left", "j1", l_j1))
            if l_j3 < -tol:
                invalid_reasons.append(reason("left", "j3", l_j3))

        if invalid_reasons:
            err_msg = (
                tr("dialogs.home_reset_posture.title") + "\n\n"
                + "\n".join(f"- {r}" for r in invalid_reasons)
                + "\n\n" + tr("dialogs.home_reset_posture.guide")
            )
            if log_cb:
                log_cb(f"[ERROR] {err_msg}")
            return False, err_msg

        return True, ""
    except Exception as e:
        if log_cb:
            log_cb(f"[WARN] Reset posture validation exception: {e}")
        return True, ""


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

    # Safety Check: Validate the posture signs before performing reset
    is_valid, err_msg = validate_home_offset_joint_limits(
        robot, model, arm=arm, include_head=include_head, log_cb=log_cb
    )
    if not is_valid:
        if log_cb is not None:
            log_cb(f"[ERROR] Home Offset Reset aborted due to the reset posture: {err_msg}")
        return {
            "status": "failed",
            "success": False,
            "arm": arm,
            "include_head": bool(include_head),
            "failed_joints": [],
            "error": err_msg
        }

    robot.disable_control_manager()
    wait_motion(0.5)

    failed_joints = []
    reset_joints = []

    def reset_joint_group(prefix, dof, label):
        for i in range(dof):
            check_motion_cancelled({"home": {"reset_joints": reset_joints.copy(), "failed_joints": failed_joints.copy()}})
            joint_name = f"{prefix}_{i}"
            try:
                success = robot.home_offset_reset(joint_name)
            except Exception as error:
                raise HomeOffsetError(f"Home Offset response failed for {joint_name}: {error}", {
                    "reset_joints": reset_joints.copy(), "failed_joints": failed_joints.copy(),
                    "uncertain_joints": [joint_name],
                }) from error
            if success:
                reset_joints.append(joint_name)
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
    check_motion_cancelled({"home": {"reset_joints": reset_joints.copy(), "failed_joints": failed_joints.copy()}})
    if log_cb is not None:
        if all_success:
            log_cb("All selected joints reset successfully!")
        else:
            log_cb("Some joints failed to reset. Proceeding with power cycle...")

    if power_cycle:
        if log_cb is not None:
            log_cb("Disabling control manager and waiting 2 seconds...")
        robot.disable_control_manager()
        wait_motion(2.0)

        if log_cb is not None:
            log_cb("Powering off overall power (.*)...")
        robot.power_off(".*")
        wait_motion(1.5)

    return {
        "status": "success" if all_success else "partial_failure",
        "success": all_success,
        "arm": arm,
        "include_head": bool(include_head),
        "failed_joints": failed_joints,
        "reset_joints": reset_joints,
    }


def apply_home_offset(
    address,
    model_name,
    arm,
    offset_rad,
    head_offset_rad=None,
    power="48v",
    servo=None,
    include_head=True,
):
    if servo is None:
        servo = "^(?!.*wheel).*$" if include_head else "^(?!.*(head|wheel)).*$"
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


class HomeOffsetController:
    """Robot preview/application workflows, with values supplied by the UI core."""
    def __init__(self, robot, log_callback=None):
        self.robot = robot
        self.model = robot.model() if robot is not None else None
        self.log_msg = log_callback or (lambda message: None)

    def ensure_home_offset_robot(self):
        if self.robot is None or self.model is None:
            raise RuntimeError("Robot is not connected")

    def move_home_offset_candidate_path(self, json_path, label, arm, include_head):
        self.ensure_home_offset_robot()
        result = move_to_offset_candidate_from_json(
            robot=self.robot,
            model=self.model,
            arm=arm,
            json_path=str(json_path),
            include_head=include_head,
            minimum_time=10,
            move_zero_first=True,
        )

        self.log_msg(f"\n===== HOME OFFSET PREVIEW: {label} =====")
        self.log_msg(f"JSON: {json_path}")
        self.log_msg(f"Arm: {result['arm']}")
        if result.get("right_offset_deg") is not None:
            self.log_msg(f"Right move offset (deg): {result['right_offset_deg']}")
        if result.get("left_offset_deg") is not None:
            self.log_msg(f"Left move offset (deg): {result['left_offset_deg']}")
        if result.get("head_offset_deg") is not None:
            self.log_msg(f"Head move offset (deg): {result['head_offset_deg']}")
        self.log_msg("Preview move complete. Inspect the robot pose before applying.")
        return result


    def move_to_check_position_candidate_path(self, json_path, label, arm, include_head, skip_init_pose=False):
        self.ensure_home_offset_robot()
        
        # Load offsets from json
        offset_rad, head_offset_rad = load_offset_from_json(str(json_path))
        
        apply_mode, right_offset_rad, left_offset_rad = split_arm_offset(
            self.model,
            arm,
            offset_rad,
        )
        
        right_offset_to_apply = -right_offset_rad
        left_offset_to_apply = -left_offset_rad
        head_offset_full, head_offset_size = normalize_head_offset(
            self.model,
            head_offset_rad,
            include_head,
        )
        head_offset_to_apply = None if head_offset_full is None else -head_offset_full
        
        # Determine version key
        version_key = "v1.2"
        if self.robot is not None:
            try:
                robot_info = self.robot.get_robot_info()
                raw_version = robot_info.robot_model_version
                if "1.3" in raw_version:
                    version_key = "v1.3"
            except Exception:
                pass
                
        # Load check_calib ready poses
        ready_poses_path = StoragePaths.root / "config" / "ready_poses.yaml"
        check_calib_joints = {
            "right_arm": [-90.0, -45.0, 73.0, -107.0, 90.0, 90.0, 0.0],
            "left_arm": [-90.0, 45.0, -73.0, -107.0, -90.0, 90.0, 0.0]
        }
        if ready_poses_path.exists():
            try:
                ready_poses = ConfigStorage.load(ready_poses_path)
                if version_key in ready_poses and "check_calib" in ready_poses[version_key]:
                    check_calib_joints = ready_poses[version_key]["check_calib"]
            except Exception as e:
                self.log_msg(f"[WARNING] Failed to parse ready_poses.yaml: {e}")
                
        # 1. Move to 1st ready pose (like check_calibration_state)
        if not skip_init_pose:
            active_arms = []
            if arm in ("right", "both"):
                active_arms.append("right")
            if arm in ("left", "both"):
                active_arms.append("left")
                
            self.log_msg(f"\n[Check Position] Step 1: Moving to Joint Ready Pose...")
            ok = movej(
                self.robot,
                torso=np.deg2rad([0, 30, -60, 30, 0, 0]),
                right_arm=np.deg2rad([-45, -30, 0, -90, 0, 45, 0]) if "right" in active_arms else np.deg2rad([0, 0, 0, -90, 0, 0, 0]),
                left_arm=np.deg2rad([-45, 30, 0, -90, 0, 45, 0]) if "left" in active_arms else np.deg2rad([0, 0, 0, -90, 0, 0, 0]),
                minimum_time=5
            )
            if not ok:
                raise RuntimeError("Failed to move robot to Step 1 Ready Pose")
            wait_motion(2.0)
        else:
            self.log_msg("[Check Position] Step 1: Skipping Ready Pose move (already initialized in check session)")
        
        # 2. Move to Check Position with offsets added
        self.log_msg(f"[Check Position] Step 2: Moving to Check Pose with Offsets...")
        
        right_target = np.deg2rad(check_calib_joints["right_arm"]) + right_offset_to_apply
        left_target = np.deg2rad(check_calib_joints["left_arm"]) + left_offset_to_apply
        
        head_zero_pose = np.zeros(len(self.model.head_idx)) if include_head else None
        head_target_pose = None
        if include_head:
            head_target_pose = (
                head_zero_pose
                if head_offset_to_apply is None
                else head_zero_pose + head_offset_to_apply
            )
            
        ok = movej(
            self.robot,
            torso=np.deg2rad([0, 30, -60, 30, 0, 0]),
            right_arm=right_target,
            left_arm=left_target,
            head=head_target_pose if include_head else None,
            minimum_time=10
        )
        if not ok:
            raise RuntimeError("Failed to move robot to Step 2 Check Pose")
        wait_motion(2.0)
        
        offset_to_apply = np.concatenate([right_offset_to_apply, left_offset_to_apply])
        head_offset_deg = None
        if head_offset_to_apply is not None:
            head_offset_deg = np.rad2deg(head_offset_to_apply[:head_offset_size]).tolist()
            
        result = {
            "status": "success",
            "arm": apply_mode,
            "offset_deg": np.rad2deg(offset_to_apply).tolist(),
            "right_offset_deg": np.rad2deg(right_offset_to_apply).tolist(),
            "left_offset_deg": np.rad2deg(left_offset_to_apply).tolist(),
            "head_offset_deg": head_offset_deg,
        }
        
        self.log_msg(f"\n===== HOME OFFSET PREVIEW: {label} =====")
        self.log_msg(f"JSON: {json_path}")
        self.log_msg(f"Arm: {result['arm']}")
        if result.get("right_offset_deg") is not None:
            self.log_msg(f"Right move offset (deg): {result['right_offset_deg']}")
        if result.get("left_offset_deg") is not None:
            self.log_msg(f"Left move offset (deg): {result['left_offset_deg']}")
        if result.get("head_offset_deg") is not None:
            self.log_msg(f"Head move offset (deg): {result['head_offset_deg']}")
        self.log_msg("Preview move complete. Inspect the robot pose before applying.")
        return result


    def apply_current_pose_home_offset(self, arm, include_head, json_path=None):
        self.ensure_home_offset_robot()
        result = reset_current_pose_home_offsets(
            self.robot,
            self.model,
            arm=arm,
            include_head=include_head,
        )

        if not result["success"]:
            return result

        # If camera was optimized, save the new camera pose to setting.yaml
        if json_path is not None and os.path.exists(json_path):
            try:
                data = ResultStorage.load(json_path)
                
                mount_to_cam_new = data.get("mount_to_cam_new")
                head_base_to_cam_new = data.get("head_base_to_cam_new")
                
                if mount_to_cam_new or head_base_to_cam_new:
                    config_path = CONFIG_PATHS["setting_yaml"]
                    changes = {}
                    if include_head and mount_to_cam_new:
                        self.log_msg(f"[APPLY] Saving optimized mount_to_cam to setting.yaml: {mount_to_cam_new}")
                        changes[("camera", "mount_to_cam")] = list(mount_to_cam_new)
                    elif not include_head and head_base_to_cam_new:
                        self.log_msg(f"[APPLY] Saving optimized head_base_to_cam to setting.yaml: {head_base_to_cam_new}")
                        changes[("camera", "head_base_to_cam")] = list(head_base_to_cam_new)
                    if changes:
                        duplicates = ConfigStorage.update_values(config_path, changes)
                        if duplicates:
                            self.log_msg(f"[WARN] setting.yaml had duplicate keys {sorted(set(duplicates))}; "
                                         "only the last of each was kept.")
            except Exception as e:
                self.log_msg(f"[ERROR] Failed to save optimized camera pose to setting.yaml: {e}")

        # Robot reconnection should be done in the main thread to avoid GUI thread safety issues.
        # We will signal the caller to handle the reconnection.
        result['needs_reconnect'] = True
        return result
