import time
import numpy as np
import rby1_sdk as rby
from dataclasses import dataclass

D2R = np.pi / 180.0

@dataclass
class AutoCollectionConfig:
    angle_step_deg: float = 5.0
    position_step_m: float = 0.03
    step_x_m: float = 0.03
    max_x: float = 0.5
    move_time: float = 2.4
    settle_time: float = 0.6
    hold_time: float = 0.5
    priority: int = 10

def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def make_T(R, p):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = np.array(p, dtype=np.float32)
    return T

def apply_cartesian_offset(T, dx=0.0, dy=0.0, dz=0.0, droll_deg=0.0, dpitch_deg=0.0, dyaw_deg=0.0):
    T_new = T.copy()
    T_new[0, 3] += dx
    T_new[1, 3] += dy
    T_new[2, 3] += dz
    
    R_off = rot_z(np.deg2rad(dyaw_deg)) @ rot_y(np.deg2rad(dpitch_deg)) @ rot_x(np.deg2rad(droll_deg))
    # Apply rotation in tool frame (right-multiply) to keep marker in view more easily
    T_new[:3, :3] = T_new[:3, :3] @ R_off
    return T_new

def compute_fk(robot, dyn_model, q_full, ee_link, base_link="link_torso_5"):
    state = dyn_model.make_state(
        [base_link, ee_link],
        robot.model().robot_joint_names
    )
    state.set_q(q_full)
    dyn_model.compute_forward_kinematics(state)
    return state, dyn_model.compute_transformation(state, 0, 1)

def compute_head_tracking_q(T_right, T_left, active_arms, p_neck, q_head_0, p_marker_0):
    if q_head_0 is None or p_neck is None or p_marker_0 is None:
        return None
        
    pts = []
    if "right" in active_arms and T_right is not None:
        pts.append(T_right[:3, 3])
    if "left" in active_arms and T_left is not None:
        pts.append(T_left[:3, 3])
        
    if len(pts) == 0:
        return q_head_0.copy()
        
    p_marker = np.mean(pts, axis=0)
    
    v_0 = p_marker_0 - p_neck
    v_i = p_marker - p_neck
    
    yaw_geo_0 = np.arctan2(v_0[1], v_0[0])
    pitch_geo_0 = np.arctan2(v_0[2], np.sqrt(v_0[0]**2 + v_0[1]**2))
    
    yaw_geo_i = np.arctan2(v_i[1], v_i[0])
    pitch_geo_i = np.arctan2(v_i[2], np.sqrt(v_i[0]**2 + v_i[1]**2))
    
    yaw_diff = yaw_geo_i - yaw_geo_0
    pitch_diff = pitch_geo_i - pitch_geo_0
    
    yaw_target = q_head_0[0] + yaw_diff
    # Pitch joint sign convention: positive pitch rotates head downward (looking down),
    # so we subtract pitch_diff to look upward.
    pitch_target = q_head_0[1] - pitch_diff
    
    # Clip head angles to safe ranges (Yaw: ±25 deg, Pitch: ±20 deg relative to zero)
    yaw_target = np.clip(yaw_target, -25.0 * D2R, 25.0 * D2R)
    pitch_target = np.clip(pitch_target, -20.0 * D2R, 20.0 * D2R)
    
    return np.array([yaw_target, pitch_target], dtype=np.float64)
_motion_state = {
    "q_right_baseline": None,
    "q_left_baseline": None,
    "q_head_baseline": None,
    "p_neck": None,
    "q_head_0": None,
    "p_marker_0": None,
}

def reset_motion_state():
    global _motion_state
    _motion_state = {
        "q_right_baseline": None,
        "q_left_baseline": None,
        "q_head_baseline": None,
        "p_neck": None,
        "q_head_0": None,
        "p_marker_0": None,
    }

def build_incremental_motion_plan(robot, dyn_model, config: AutoCollectionConfig, active_arms=["right", "left"]):
    """
    현재 자세를 읽어서 X축으로 전진하며 RPY/YZ 오프셋 타겟들과 헤드 트래킹 타겟 각도들을 생성합니다.
    """
    reset_motion_state()
    state = robot.get_state()
    if state is None or getattr(state, 'position', None) is None:
        raise RuntimeError("Robot state position is None. Please check connection.")
    q_full = np.array(state.position)
    _, T_base_right = compute_fk(robot, dyn_model, q_full, "ee_right", "link_torso_5")
    _, T_base_left = compute_fk(robot, dyn_model, q_full, "ee_left", "link_torso_5")
    
    model = robot.model()
    head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
    q_head_0 = q_full[head_idx].copy() if head_idx is not None else None
    
    try:
        _, T_head_0 = compute_fk(robot, dyn_model, q_full, "link_head_2", "link_torso_5")
        p_neck = T_head_0[:3, 3]
    except Exception:
        p_neck = None
        
    def get_marker_midpoint(tr, tl):
        pts = []
        if "right" in active_arms and tr is not None:
            pts.append(tr[:3, 3])
        if "left" in active_arms and tl is not None:
            pts.append(tl[:3, 3])
        if len(pts) == 0:
            return None
        return np.mean(pts, axis=0)
        
    p_marker_0 = get_marker_midpoint(T_base_right, T_base_left)
    
    plan = []
    T_curr_right = T_base_right.copy()
    T_curr_left = T_base_left.copy()
    
    while True:
        curr_x = T_curr_right[0, 3]
        if curr_x > config.max_x:
            break
            
        half_ang = config.angle_step_deg / 2.0
        full_ang = config.angle_step_deg

        # 1. Joint steps for joint 0, 1, and 4
        joint_offsets = [-half_ang, -full_ang, half_ang, full_ang]
        for joint_idx in [0, 1, 4]:
            for offset in joint_offsets:
                plan.append({
                    "type": "joint",
                    "joint_idx": joint_idx,
                    "offset_deg": offset,
                    "T_right": T_curr_right.copy(),
                    "T_left": T_curr_left.copy(),
                    "desc": f"Joint {joint_idx} Offset: {offset:.1f}deg"
                })
        plan.append({
            "type": "restore_baseline",
            "T_right": T_curr_right.copy(),
            "T_left": T_curr_left.copy(),
            "desc": "Restore Baseline Pose"
        })

        rpy_targets = [
            (-half_ang, 0.0, 0.0), (-full_ang, 0.0, 0.0), (half_ang, 0.0, 0.0), (full_ang, 0.0, 0.0),
            (0.0, -half_ang, 0.0), (0.0, -full_ang, 0.0), (0.0, half_ang, 0.0), (0.0, full_ang, 0.0),
            (0.0, 0.0, -half_ang), (0.0, 0.0, -full_ang), (0.0, 0.0, half_ang), (0.0, 0.0, full_ang)
        ]
        for dr, dp, dy in rpy_targets:
            tr = apply_cartesian_offset(T_curr_right, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            tl = apply_cartesian_offset(T_curr_left, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            head_q = compute_head_tracking_q(tr, tl, active_arms, p_neck, q_head_0, p_marker_0)
            plan.append({
                "T_right": tr, "T_left": tl,
                "head_q": head_q,
                "desc": f"RPY: ({dr:.2f},{dp:.2f},{dy:.2f})"
            })
            
        half_pos = config.position_step_m / 2.0
        full_pos = config.position_step_m
        yz_targets = [
            (0.0, -half_pos, 0.0), (0.0, -full_pos, 0.0), (0.0, half_pos, 0.0), (0.0, full_pos, 0.0),
            (0.0, 0.0, -half_pos), (0.0, 0.0, -full_pos), (0.0, 0.0, half_pos), (0.0, 0.0, full_pos)
        ]
        for dx, dy, dz in yz_targets:
            tr = apply_cartesian_offset(T_curr_right, dx=dx, dy=dy, dz=dz)
            tl = apply_cartesian_offset(T_curr_left, dx=dx, dy=dy, dz=dz)
            head_q = compute_head_tracking_q(tr, tl, active_arms, p_neck, q_head_0, p_marker_0)
            plan.append({
                "T_right": tr, "T_left": tl,
                "head_q": head_q,
                "desc": f"Pos: ({dx:.3f},{dy:.3f},{dz:.3f})"
            })
            
        # 4. Independent head motions (Pan Left/Right, Tilt Up/Down)
        if q_head_0 is not None:
            ang_rad = np.radians(config.angle_step_deg)
            head_targets = [
                (-ang_rad, 0.0, f"Head Pan: {-config.angle_step_deg:.1f}deg"),
                (ang_rad, 0.0, f"Head Pan: {config.angle_step_deg:.1f}deg"),
                (0.0, -ang_rad, f"Head Tilt: {-config.angle_step_deg:.1f}deg"),
                (0.0, ang_rad, f"Head Tilt: {config.angle_step_deg:.1f}deg"),
            ]
            for d_pan, d_tilt, desc in head_targets:
                hq = np.array([q_head_0[0] + d_pan, q_head_0[1] + d_tilt], dtype=np.float64)
                plan.append({
                    "T_right": T_curr_right.copy(),
                    "T_left": T_curr_left.copy(),
                    "head_q": hq,
                    "desc": desc
                })
            
        T_curr_right = apply_cartesian_offset(T_curr_right, dx=config.step_x_m)
        T_curr_left = apply_cartesian_offset(T_curr_left, dx=config.step_x_m)
        
    return plan

def move_to_auto_ready_pose(robot, active_arms, minimum_time=5.0, priority=10):
    # Step 1: Joint Ready Pose (go_to_ready_pose 기준)
    q_torso = np.array([0, 30, -60, 30, 0, 0], dtype=np.float64) * D2R
    
    if "right" in active_arms:
        q_right = np.array([-45, -30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
    else:
        q_right = np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R
        
    if "left" in active_arms:
        q_left = np.array([-45, 30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
    else:
        q_left = np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R
        
    q_ready = np.concatenate([q_torso, q_right, q_left])
    
    print("Step 1: Moving to Joint Ready Pose...")
    cmd1 = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.JointPositionCommandBuilder()
            .set_position(q_ready)
            .set_minimum_time(minimum_time)
        )
    )
    rv1 = robot.send_command(cmd1, priority).get()
    if rv1.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError("Failed to move to Step 1: Joint Ready Pose.")

    # Step 2: Cartesian Checking Pose (go_to_calibration_checking_pose 기준, offset=0.2)
    # Raising Z-axis to 0.2m (down 20cm from previous 0.4m) and rotating 6th axis (wrist) by 180 degrees (@ rot_z(180))
    T_right = make_T(rot_z(0*D2R) @ rot_y(-90*D2R) @ rot_x(90*D2R), [0.3, -0.15, 0.25])
    T_right[:3, :3] = T_right[:3, :3] @ rot_z(180*D2R)
    
    T_left = make_T(rot_z(0*D2R) @ rot_y(-90*D2R) @ rot_x(-90*D2R), [0.3, 0.15, 0.25])
    T_left[:3, :3] = T_left[:3, :3] @ rot_z(180*D2R)

    body2 = rby.BodyComponentBasedCommandBuilder()

    if "right" in active_arms:
        header_right = rby.CommandHeaderBuilder()
        header_right.set_control_hold_time(0.5)
        
        right_cmd = rby.CartesianCommandBuilder()
        right_cmd.add_target("link_torso_5", "ee_right", T_right.astype(np.float32), 0.5, 1.0, 0.3)
        right_cmd.set_stop_position_tracking_error(0.005)
        right_cmd.set_stop_orientation_tracking_error(0.02)
        right_cmd.set_minimum_time(minimum_time)
        right_cmd.set_command_header(header_right)
        
        body2.set_right_arm_command(right_cmd)
    else:
        right_joint = rby.JointPositionCommandBuilder()
        right_joint.set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
        right_joint.set_minimum_time(minimum_time)
        body2.set_right_arm_command(right_joint)

    if "left" in active_arms:
        header_left = rby.CommandHeaderBuilder()
        header_left.set_control_hold_time(0.5)
        
        left_cmd = rby.CartesianCommandBuilder()
        left_cmd.add_target("link_torso_5", "ee_left", T_left.astype(np.float32), 0.5, 1.0, 0.3)
        left_cmd.set_stop_position_tracking_error(0.005)
        left_cmd.set_stop_orientation_tracking_error(0.02)
        left_cmd.set_minimum_time(minimum_time)
        left_cmd.set_command_header(header_left)
        
        body2.set_left_arm_command(left_cmd)
    else:
        left_joint = rby.JointPositionCommandBuilder()
        left_joint.set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
        left_joint.set_minimum_time(minimum_time)
        body2.set_left_arm_command(left_joint)

    print("Step 2: Moving to Cartesian Checking Pose...")
    cmd2 = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body2)
    )
    rv2 = robot.send_command(cmd2, priority).get()
    if rv2.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError("Failed to move to Step 2: Cartesian Checking Pose.")

def make_dual_arm_head_cmd(T_right, T_left, active_arms, head_position=None, min_time=1.2, hold_time=0.5, q_right=None, q_left=None):
    body = rby.BodyComponentBasedCommandBuilder()

    header_right = None
    if "right" in active_arms:
        if q_right is not None:
            header_right = rby.CommandHeaderBuilder()
            header_right.set_control_hold_time(hold_time)
            
            right_joint = rby.JointPositionCommandBuilder()
            right_joint.set_position(q_right)
            right_joint.set_minimum_time(min_time)
            right_joint.set_command_header(header_right)
            body.set_right_arm_command(right_joint)
        elif T_right is not None:
            header_right = rby.CommandHeaderBuilder()
            header_right.set_control_hold_time(hold_time)
            
            right_cart = rby.CartesianCommandBuilder()
            right_cart.add_target("link_torso_5", "ee_right", T_right.astype(np.float32), 0.2, 0.5, 0.3)
            right_cart.set_stop_position_tracking_error(0.001)
            right_cart.set_stop_orientation_tracking_error(0.005)
            right_cart.set_command_header(header_right)
            right_cart.set_minimum_time(min_time)
            body.set_right_arm_command(right_cart)
    else:
        # Lock inactive right arm
        right_joint = rby.JointPositionCommandBuilder()
        right_joint.set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
        right_joint.set_minimum_time(min_time)
        body.set_right_arm_command(right_joint)

    header_left = None
    if "left" in active_arms:
        if q_left is not None:
            header_left = rby.CommandHeaderBuilder()
            header_left.set_control_hold_time(hold_time)
            
            left_joint = rby.JointPositionCommandBuilder()
            left_joint.set_position(q_left)
            left_joint.set_minimum_time(min_time)
            left_joint.set_command_header(header_left)
            body.set_left_arm_command(left_joint)
        elif T_left is not None:
            header_left = rby.CommandHeaderBuilder()
            header_left.set_control_hold_time(hold_time)
            
            left_cart = rby.CartesianCommandBuilder()
            left_cart.add_target("link_torso_5", "ee_left", T_left.astype(np.float32), 0.2, 0.5, 0.3)
            left_cart.set_stop_position_tracking_error(0.001)
            left_cart.set_stop_orientation_tracking_error(0.005)
            left_cart.set_command_header(header_left)
            left_cart.set_minimum_time(min_time)
            body.set_left_arm_command(left_cart)
    else:
        # Lock inactive left arm
        left_joint = rby.JointPositionCommandBuilder()
        left_joint.set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
        left_joint.set_minimum_time(min_time)
        body.set_left_arm_command(left_joint)

    cmd = rby.ComponentBasedCommandBuilder().set_body_command(body)
    if head_position is not None:
        cmd.set_head_command(
            rby.JointPositionCommandBuilder()
            .set_position(head_position)
            .set_minimum_time(min_time)
        )
    return rby.RobotCommandBuilder().set_command(cmd)

def execute_auto_motion_step(robot, config, motion_plan_step, active_arms, include_head_motion=True):
    global _motion_state

    step_type = motion_plan_step.get("type")

    if step_type == "joint":
        state = robot.get_state()
        if state is None or getattr(state, 'position', None) is None:
            raise RuntimeError("Robot state position is None. Please check connection.")
        q_full = np.array(state.position)
        model = robot.model()
        dyn_model = robot.get_dynamics()

        # Save baseline configurations if not already saved
        if _motion_state["q_right_baseline"] is None:
            _motion_state["q_right_baseline"] = q_full[model.right_arm_idx[:7]].copy()
            _motion_state["q_left_baseline"] = q_full[model.left_arm_idx[:7]].copy()
            head_idx = model.head_idx[:2] if len(model.head_idx) >= 2 else None
            _motion_state["q_head_baseline"] = q_full[head_idx].copy() if head_idx is not None else None
            _motion_state["q_head_0"] = _motion_state["q_head_baseline"].copy() if head_idx is not None else None

            _, T_base_right = compute_fk(robot, dyn_model, q_full, "ee_right", "link_torso_5")
            _, T_base_left = compute_fk(robot, dyn_model, q_full, "ee_left", "link_torso_5")

            try:
                _, T_head_0 = compute_fk(robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                _motion_state["p_neck"] = T_head_0[:3, 3]
            except Exception:
                _motion_state["p_neck"] = None

            pts = []
            if "right" in active_arms and T_base_right is not None:
                pts.append(T_base_right[:3, 3])
            if "left" in active_arms and T_base_left is not None:
                pts.append(T_base_left[:3, 3])
            if len(pts) > 0:
                _motion_state["p_marker_0"] = np.mean(pts, axis=0)
            else:
                _motion_state["p_marker_0"] = None

        q_right_target = _motion_state["q_right_baseline"].copy()
        q_left_target = _motion_state["q_left_baseline"].copy()

        j_idx = motion_plan_step["joint_idx"]
        offset_deg = motion_plan_step["offset_deg"]

        if "right" in active_arms:
            q_right_target[j_idx] += np.deg2rad(offset_deg)
        if "left" in active_arms:
            q_left_target[j_idx] += np.deg2rad(offset_deg)

        head_q = None
        if j_idx == 0 and include_head_motion:
            q_full_temp = q_full.copy()
            q_full_temp[model.right_arm_idx[:7]] = q_right_target
            q_full_temp[model.left_arm_idx[:7]] = q_left_target

            _, T_right_fk = compute_fk(robot, dyn_model, q_full_temp, "ee_right", "link_torso_5")
            _, T_left_fk = compute_fk(robot, dyn_model, q_full_temp, "ee_left", "link_torso_5")

            head_q = compute_head_tracking_q(
                T_right_fk,
                T_left_fk,
                active_arms,
                _motion_state["p_neck"],
                _motion_state["q_head_0"],
                _motion_state["p_marker_0"]
            )
        elif include_head_motion:
            head_q = _motion_state["q_head_baseline"]

        cmd = make_dual_arm_head_cmd(
            T_right=None,
            T_left=None,
            active_arms=active_arms,
            head_position=head_q,
            min_time=config.move_time,
            hold_time=config.hold_time,
            q_right=q_right_target if "right" in active_arms else None,
            q_left=q_left_target if "left" in active_arms else None,
        )
        rv = robot.send_command(cmd, config.priority).get()
        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"Auto motion joint command failed: {rv.finish_code}")

        time.sleep(config.settle_time)
        return motion_plan_step

    elif step_type == "restore_baseline":
        if _motion_state["q_right_baseline"] is not None:
            cmd = make_dual_arm_head_cmd(
                T_right=None,
                T_left=None,
                active_arms=active_arms,
                head_position=_motion_state["q_head_baseline"] if include_head_motion else None,
                min_time=config.move_time,
                hold_time=config.hold_time,
                q_right=_motion_state["q_right_baseline"],
                q_left=_motion_state["q_left_baseline"]
            )
            rv = robot.send_command(cmd, config.priority).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                raise RuntimeError(f"Restore baseline command failed: {rv.finish_code}")

            # Clear the baseline state
            reset_motion_state()

        time.sleep(config.settle_time)
        return motion_plan_step

    else:
        # Standard Cartesian step
        T_right = motion_plan_step["T_right"]
        T_left = motion_plan_step["T_left"]
        head_q = motion_plan_step.get("head_q", None) if include_head_motion else None

        cmd = make_dual_arm_head_cmd(
            T_right=T_right,
            T_left=T_left,
            active_arms=active_arms,
            head_position=head_q,
            min_time=config.move_time,
            hold_time=config.hold_time,
        )
        rv = robot.send_command(cmd, config.priority).get()
        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"Auto motion command failed: {rv.finish_code}")

        time.sleep(config.settle_time)
        return motion_plan_step


def check_calibration_state(robot, model_name, active_arms, data, offset, log_cb=None, skip_ready=False):
    # Ensure Control Manager is enabled
    cm_state = robot.get_control_manager_state()
    if cm_state.state in [rby.ControlManagerState.State.MinorFault, rby.ControlManagerState.State.MajorFault]:
        if log_cb is not None:
            log_cb("[ControlManager] Control manager in fault state. Resetting...")
        robot.reset_fault_control_manager()
        time.sleep(1.0)
        
    cm_state = robot.get_control_manager_state()
    if cm_state.state != rby.ControlManagerState.State.Enabled:
        if log_cb is not None:
            log_cb("[ControlManager] Enabling control manager...")
        robot.enable_control_manager()
        time.sleep(1.0)

    q_torso = np.array([0, 30, -60, 30, 0, 0], dtype=np.float64) * D2R
    if not skip_ready:
        if log_cb is not None:
            log_cb("Step 1: Moving to Joint Ready Pose...")
        
        # 1. Joint Ready Pose
        if "right" in active_arms:
            q_right = np.array([-45, -30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
        else:
            q_right = np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R
            
        if "left" in active_arms:
            q_left = np.array([-45, 30, 0, -90, 0, 45, 0], dtype=np.float64) * D2R
        else:
            q_left = np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R
            
        q_ready = np.concatenate([q_torso, q_right, q_left])
        
        cmd1 = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.JointPositionCommandBuilder()
                .set_position(q_ready)
                .set_minimum_time(5.0)
            )
        )
        rv1 = robot.send_command(cmd1, 10).get()
        if rv1.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"Failed to move to Joint Ready Pose: {rv1.finish_code}")
            
        time.sleep(1.0)
    else:
        if log_cb is not None:
            log_cb("Skipping Joint Ready Pose (Subsequent Move)...")
        
    if log_cb is not None:
        log_cb("Step 2: Moving to Cartesian Symmetrical Checking Pose...")
        
    # 2. Cartesian Symmetrical Pose
    # Compute transformations
    import math
    roll_r = 90 * math.pi / 180
    pitch_r = -90 * math.pi / 180
    yaw_r = 0.0
    
    # Right transform
    cr_r = math.cos(roll_r); sr_r = math.sin(roll_r)
    cp_r = math.cos(pitch_r); sp_r = math.sin(pitch_r)
    cy_r = math.cos(yaw_r); sy_r = math.sin(yaw_r)
    
    T_right = np.eye(4, dtype=np.float64)
    T_right[0, 0] = cy_r * cp_r
    T_right[0, 1] = sr_r * sp_r * cy_r - cr_r * sy_r
    T_right[0, 2] = cr_r * sp_r * cy_r + sr_r * sy_r
    T_right[0, 3] = data[0]
    
    T_right[1, 0] = sy_r * cp_r
    T_right[1, 1] = sr_r * sp_r * sy_r + cr_r * cy_r
    T_right[1, 2] = cr_r * sp_r * sy_r - sr_r * cy_r
    T_right[1, 3] = data[1] - offset
    
    T_right[2, 0] = -sp_r
    T_right[2, 1] = cp_r * sr_r
    T_right[2, 2] = cp_r * cr_r
    T_right[2, 3] = data[2]

    roll_l = -90 * math.pi / 180
    pitch_l = -90 * math.pi / 180
    yaw_l = 0.0
    
    # Left transform
    cr_l = math.cos(roll_l); sr_l = math.sin(roll_l)
    cp_l = math.cos(pitch_l); sp_l = math.sin(pitch_l)
    cy_l = math.cos(yaw_l); sy_l = math.sin(yaw_l)
    
    T_left = np.eye(4, dtype=np.float64)
    T_left[0, 0] = cy_l * cp_l
    T_left[0, 1] = sr_l * sp_l * cy_l - cr_l * sy_l
    T_left[0, 2] = cr_l * sp_l * cy_l + sr_l * sy_l
    T_left[0, 3] = data[0]
    
    T_left[1, 0] = sy_l * cp_l
    T_left[1, 1] = sr_l * sp_l * sy_l + cr_l * cy_l
    T_left[1, 2] = cr_l * sp_l * sy_l - sr_l * cy_l
    T_left[1, 3] = data[1] + offset
    
    T_left[2, 0] = -sp_l
    T_left[2, 1] = cp_l * sr_l
    T_left[2, 2] = cp_l * cr_l
    T_left[2, 3] = data[2]

    MINIMUM_TIME = 5.0
    LINEAR_VELOCITY_LIMIT = 1.5
    ANGULAR_VELOCITY_LIMIT = math.pi * 1.5
    ACCELERATION_LIMIT = 1.0
    STOP_ORIENTATION_TRACKING_ERROR = 1e-4
    STOP_POSITION_TRACKING_ERROR = 1e-3

    body = rby.BodyComponentBasedCommandBuilder()

    if "right" in active_arms:
        header_right = rby.CommandHeaderBuilder()
        header_right.set_control_hold_time(0.5)
        
        right_cart = rby.CartesianCommandBuilder()
        right_cart.add_target("link_torso_5", "ee_right", T_right.astype(np.float32), LINEAR_VELOCITY_LIMIT, ANGULAR_VELOCITY_LIMIT, ACCELERATION_LIMIT)
        right_cart.set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
        right_cart.set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
        right_cart.set_minimum_time(MINIMUM_TIME)
        right_cart.set_command_header(header_right)
        
        body.set_right_arm_command(right_cart)
    else:
        right_joint = rby.JointPositionCommandBuilder()
        right_joint.set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
        right_joint.set_minimum_time(MINIMUM_TIME)
        body.set_right_arm_command(right_joint)

    if "left" in active_arms:
        header_left = rby.CommandHeaderBuilder()
        header_left.set_control_hold_time(0.5)
        
        left_cart = rby.CartesianCommandBuilder()
        left_cart.add_target("link_torso_5", "ee_left", T_left.astype(np.float32), LINEAR_VELOCITY_LIMIT, ANGULAR_VELOCITY_LIMIT, ACCELERATION_LIMIT)
        left_cart.set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
        left_cart.set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
        left_cart.set_minimum_time(MINIMUM_TIME)
        left_cart.set_command_header(header_left)
        
        body.set_left_arm_command(left_cart)
    else:
        left_joint = rby.JointPositionCommandBuilder()
        left_joint.set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
        left_joint.set_minimum_time(MINIMUM_TIME)
        body.set_left_arm_command(left_joint)

    cmd2 = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body)
    )
    rv2 = robot.send_command(cmd2, 10).get()
    if rv2.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Failed to move to Cartesian Checking Pose. FinishCode: {rv2.finish_code}")
