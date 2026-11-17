from core.robot.robot_core import check_motion_cancelled, wait_motion
import numpy as np
import rby1_sdk as rby
from dataclasses import dataclass

D2R = np.pi / 180.0

# Forward reach of the Step 2 ready pose when there is no head (see move_to_auto_ready_pose).
HEADLESS_X_REACH_M = 0.34

@dataclass
class AutoCollectionConfig:
    angle_step_deg: float = 5.0
    position_step_m: float = 0.03
    step_x_m: float = 0.03
    max_x: float = 0.5
    max_loops: int = 1
    move_time: float = 1.8
    settle_time: float = 0.4
    hold_time: float = 0.4
    priority: int = 10

# Optional plan steps (wrist-diversity poses) are skipped when a joint target comes this close to its limit.
OPTIONAL_POSE_LIMIT_MARGIN_DEG = 5.0


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

def build_incremental_motion_plan(robot, dyn_model, config: AutoCollectionConfig, active_arms=["right", "left"], include_head_motion=True,
                                  wrist_diversity=None, skip_half_steps=False):
    """
    현재 자세를 읽어서 X축으로 전진하며 RPY/YZ 오프셋 타겟들과 헤드 트래킹 타겟 각도들을 생성합니다. (최대 2루프, 총 82개 포즈로 구성)

    wrist_diversity: extra poses for head robots (load_step2_wrist_diversity), each
    {"right": [dJ4, dJ5, dJ6], "left": [...], "head": [dpan, dtilt]} in degrees from the baseline.
    They are appended once, marked optional (a pose that is out of limits or loses a marker is
    skipped instead of aborting Step 2), and followed by a return to the baseline.

    skip_half_steps (setting.yaml step2.skip_half_steps): leave out every half-size step whose
    full-size step in the same direction is also in the plan (head 64 -> 44 poses, headless 54 -> 36).
    """
    reset_motion_state()
    if robot is None:
        raise RuntimeError("Robot is not connected.")
    state = robot.get_state()
    if state is None or getattr(state, 'position', None) is None:
        raise RuntimeError("Failed to get robot joint states.")
    q_full = np.array(state.position)
    _, T_base_right = compute_fk(robot, dyn_model, q_full, "ee_right", "link_torso_5")
    _, T_base_left = compute_fk(robot, dyn_model, q_full, "ee_left", "link_torso_5")
    
    model = robot.model()
    head_idx = list(model.head_idx[:2]) if (len(model.head_idx) >= 2 and include_head_motion) else None
    has_head = head_idx is not None
    q_head_0 = np.array([float(q_full[i]) for i in head_idx], dtype=np.float64) if has_head else None
    
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
    T_curr_right = T_base_right.copy() if T_base_right is not None else None
    T_curr_left = T_base_left.copy() if T_base_left is not None else None
    
    loop_count = 0
    max_loops = getattr(config, 'max_loops', 1)

    while loop_count < max_loops:
        curr_x = T_curr_right[0, 3]
        if curr_x > config.max_x + 1e-4:
            break
        loop_count += 1
        full_ang = getattr(config, 'angle_step_deg', 5.0)
        half_ang = full_ang / 2.0

        if has_head:
            # 1. 2D Decoupled Cross-Grid: Varies J0 and Head Tilt independently across FOV regions (Top, Center, Bottom)
            # Guarantees markers stay safely within camera FOV (+/- 12 deg) while completely breaking collinearity
            j0_tilt_grid = [
                (-3.5, -5.0, "J0 (-3.5deg) + Head Tilt (-5.0deg) [Center FOV]"),
                (-3.5, -2.5, "J0 (-3.5deg) + Head Tilt (-2.5deg) [Upper FOV]"),
                (+3.5, +5.0, "J0 (+3.5deg) + Head Tilt (+5.0deg) [Center FOV]"),
                (+3.5, +2.5, "J0 (+3.5deg) + Head Tilt (+2.5deg) [Lower FOV]"),
                ( 0.0, -3.0, "J0 ( 0.0deg) + Head Tilt (-3.0deg) [Upper FOV]"),
                ( 0.0, +3.0, "J0 ( 0.0deg) + Head Tilt (+3.0deg) [Lower FOV]"),
            ]
            for j0_off, tilt_off, desc in j0_tilt_grid:
                plan.append({
                    "type": "joint",
                    "joint_idx": 0,
                    "offset_deg": j0_off,
                    "head_tilt_offset_deg": tilt_off,
                    "T_right": T_curr_right.copy() if T_curr_right is not None else None,
                    "T_left": T_curr_left.copy() if T_curr_left is not None else None,
                    "desc": desc
                })
        else:
            # Fixed camera (no head): safe small angle sweeps within optical window
            j0_safe = min(getattr(config, 'angle_step_deg', 5.0), 3.0)
            j0_offsets = [-j0_safe/2.0, -j0_safe, j0_safe/2.0, j0_safe]
            for j0_off in (j0_offsets[1::2] if skip_half_steps else j0_offsets):
                plan.append({
                    "type": "joint",
                    "joint_idx": 0,
                    "offset_deg": j0_off,
                    "T_right": T_curr_right.copy() if T_curr_right is not None else None,
                    "T_left": T_curr_left.copy() if T_curr_left is not None else None,
                    "desc": f"Joint 0 Offset: {j0_off:+.1f}deg"
                })

        # Restore baseline pose after J0 sweeps so next joint steps start from neutral pose
        plan.append({
            "type": "restore_baseline",
            "T_right": T_curr_right.copy() if T_curr_right is not None else None,
            "T_left": T_curr_left.copy() if T_curr_left is not None else None,
            "desc": "Restore Baseline Pose"
        })

        # 1.2 Other Joint steps for joint 1, 2, and 4
        for joint_idx in [1, 2, 4]:
            if joint_idx == 2:
                # Joint 2 (Shoulder Yaw) swings the marker across a large 3D arc.
                # Use a safe angle range (±1.5°, ±3.0°) so it never clips FOV even with head offsets
                j2_half = min(half_ang, 1.5)
                j2_full = min(full_ang * 0.6, 3.0)
                j_offsets = [-j2_half, -j2_full, j2_half, j2_full]
            else:
                j_offsets = [-half_ang, -full_ang, half_ang, full_ang]
            for offset in (j_offsets[1::2] if skip_half_steps else j_offsets):
                plan.append({
                    "type": "joint",
                    "joint_idx": joint_idx,
                    "offset_deg": offset,
                    "T_right": T_curr_right.copy() if T_curr_right is not None else None,
                    "T_left": T_curr_left.copy() if T_curr_left is not None else None,
                    "desc": f"Joint {joint_idx} Offset: {offset:+.1f}deg"
                })

        # 1.5 Multi-joint diagonal sweeps for J1 and J4/J2 coupling decoupling
        j2_diag = min(full_ang * 0.6, 3.0)
        multi_joint_targets = [
            ({1:  full_ang, 4:  full_ang}, f"Joint 1+4 (+{full_ang:.1f},+{full_ang:.1f})deg"),
            ({1:  full_ang, 4: -full_ang}, f"Joint 1+4 (+{full_ang:.1f},-{full_ang:.1f})deg"),
            ({1: -full_ang, 4:  full_ang}, f"Joint 1+4 (-{full_ang:.1f},+{full_ang:.1f})deg"),
            ({1: -full_ang, 4: -full_ang}, f"Joint 1+4 (-{full_ang:.1f},-{full_ang:.1f})deg"),
            ({1:  full_ang, 2:  j2_diag},  f"Joint 1+2 (+{full_ang:.1f},+{j2_diag:.1f})deg"),
            ({1: -full_ang, 2: -j2_diag},  f"Joint 1+2 (-{full_ang:.1f},-{j2_diag:.1f})deg"),
            ({2:  j2_diag,  4: -j2_diag},  f"Joint 2-4 Decouple (+{j2_diag:.1f},-{j2_diag:.1f})deg"),
            ({2: -j2_diag,  4:  j2_diag},  f"Joint 2-4 Decouple (-{j2_diag:.1f},+{j2_diag:.1f})deg"),
        ]
        for off_dict, desc in multi_joint_targets:
            plan.append({
                "type": "joint",
                "joint_idx": None,
                "offsets_dict": off_dict,
                "T_right": T_curr_right.copy() if T_curr_right is not None else None,
                "T_left": T_curr_left.copy() if T_curr_left is not None else None,
                "desc": desc
            })

        plan.append({
            "type": "restore_baseline",
            "T_right": T_curr_right.copy() if T_curr_right is not None else None,
            "T_left": T_curr_left.copy() if T_curr_left is not None else None,
            "desc": "Restore Baseline Pose"
        })

        # 2. Elbow depth sweeps (both Extension and safe Flexion)
        elbow_joint_targets = [
            ({3:  2.0, 5: -2.0}, "Elbow Extension Low (J3 +2deg, J5 -2deg)"),
            ({3:  4.0, 5: -4.0}, "Elbow Extension Mid (J3 +4deg, J5 -4deg)"),
            ({3: -3.0, 5:  3.0}, "Elbow Flexion Low (J3 -3deg, J5 +3deg)"),
            ({3:  2.0, 2:  3.0, 4: -3.0}, "Elbow Extension + Outward Yaw (+3deg)"),
            ({3:  2.0, 2:  6.0, 4: -6.0}, "Elbow Extension + Outward Wide Yaw (+6deg)"),
        ]
        for off_dict, desc in elbow_joint_targets:
            plan.append({
                "type": "joint",
                "joint_idx": None,
                "offsets_dict": off_dict,
                "T_right": T_curr_right.copy() if T_curr_right is not None else None,
                "T_left": T_curr_left.copy() if T_curr_left is not None else None,
                "desc": desc
            })

        plan.append({
            "type": "restore_baseline",
            "T_right": T_curr_right.copy() if T_curr_right is not None else None,
            "T_left": T_curr_left.copy() if T_curr_left is not None else None,
            "desc": "Restore Baseline Pose"
        })

        rpy_targets = [
            (-half_ang, 0.0, 0.0), (-full_ang, 0.0, 0.0), (half_ang, 0.0, 0.0), (full_ang, 0.0, 0.0),
            (0.0, -half_ang, 0.0), (0.0, -full_ang, 0.0), (0.0, half_ang, 0.0), (0.0, full_ang, 0.0),
            (0.0, 0.0, -half_ang), (0.0, 0.0, -full_ang), (0.0, 0.0, half_ang), (0.0, 0.0, full_ang)
        ]
        for dr, dp, dy in (rpy_targets[1::2] if skip_half_steps else rpy_targets):
            tr = apply_cartesian_offset(T_curr_right, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            tl = apply_cartesian_offset(T_curr_left, droll_deg=dr, dpitch_deg=dp, dyaw_deg=dy)
            head_q = compute_head_tracking_q(tr, tl, active_arms, p_neck, q_head_0, p_marker_0) if has_head else None
            plan.append({
                "T_right": tr, "T_left": tl,
                "head_q": head_q,
                "desc": f"RPY: ({dr:.2f},{dp:.2f},{dy:.2f})"
            })
            
        half_pos = config.position_step_m / 2.0
        full_pos = config.position_step_m
        dx_step = min(getattr(config, 'step_x_m', 0.04), 0.04)
        xyz_targets = [
            (-dx_step, 0.0, 0.0), (+dx_step, 0.0, 0.0),  # Direct X-axis depth variation!
            (0.0, -half_pos, 0.0), (0.0, -full_pos, 0.0), (0.0, half_pos, 0.0), (0.0, full_pos, 0.0),
            (0.0, 0.0, -half_pos), (0.0, 0.0, -full_pos), (0.0, 0.0, half_pos), (0.0, 0.0, full_pos)
        ]
        if skip_half_steps:
            xyz_targets = xyz_targets[:2] + xyz_targets[2:][1::2]
        for dx, dy, dz in xyz_targets:
            tr = apply_cartesian_offset(T_curr_right, dx=dx, dy=dy, dz=dz)
            tl = apply_cartesian_offset(T_curr_left, dx=dx, dy=dy, dz=dz)
            head_q = compute_head_tracking_q(tr, tl, active_arms, p_neck, q_head_0, p_marker_0) if has_head else None
            plan.append({
                "T_right": tr, "T_left": tl,
                "head_q": head_q,
                "desc": f"Pos: ({dx:.3f},{dy:.3f},{dz:.3f})"
            })
            
        # 4. Independent head motions (Pan Left/Right, Tilt Up/Down) with denser steps and optimized angle range
        if has_head and q_head_0 is not None:
            head_sweep_range_deg = 3.5
            steps_deg = [-head_sweep_range_deg, -head_sweep_range_deg / 2.0, head_sweep_range_deg / 2.0, head_sweep_range_deg]
            if skip_half_steps:
                steps_deg = [-head_sweep_range_deg, head_sweep_range_deg]

            head_targets = []
            for ang in steps_deg:
                ang_rad = np.radians(ang)
                head_targets.append((ang_rad, 0.0, f"Head Pan: {ang:+.2f}deg"))
            for ang in steps_deg:
                ang_rad = np.radians(ang)
                head_targets.append((0.0, ang_rad, f"Head Tilt: {ang:+.2f}deg"))

            for d_pan, d_tilt, desc in head_targets:
                hq = np.array([q_head_0[0] + d_pan, q_head_0[1] + d_tilt], dtype=np.float64)
                plan.append({
                    "T_right": T_curr_right.copy() if T_curr_right is not None else None,
                    "T_left": T_curr_left.copy() if T_curr_left is not None else None,
                    "head_q": hq,
                    "desc": desc
                })
            
        T_curr_right = apply_cartesian_offset(T_curr_right, dx=config.step_x_m)
        T_curr_left = apply_cartesian_offset(T_curr_left, dx=config.step_x_m)

    if has_head and wrist_diversity:
        n = len(wrist_diversity)
        for k, pose in enumerate(wrist_diversity, start=1):
            plan.append({
                "type": "joint",
                "joint_idx": None,
                "offsets_by_arm": {side: {4: pose[side][0], 5: pose[side][1], 6: pose[side][2]}
                                   for side in ("right", "left")},
                "head_pan_offset_deg": pose["head"][0],
                "head_tilt_offset_deg": pose["head"][1],
                "optional": True,
                "T_right": T_base_right.copy() if T_base_right is not None else None,
                "T_left": T_base_left.copy() if T_base_left is not None else None,
                "desc": f"Wrist diversity {k}/{n}: R J4-6 {pose['right']} L J4-6 {pose['left']} head {pose['head']} deg",
            })
        plan.append({
            "type": "restore_baseline",
            "T_right": T_base_right.copy() if T_base_right is not None else None,
            "T_left": T_base_left.copy() if T_base_left is not None else None,
            "desc": "Restore Baseline Pose",
        })

    return plan

def move_to_auto_ready_pose(robot, active_arms, minimum_time=5.0, priority=10, include_head_motion=True, robot_version=None,
                           pos_tolerance_m=0.005, ori_tolerance_rad=0.02, skip_joint_pose=False):
    model = robot.model() if robot else None
    has_head = (include_head_motion) and (model is not None and hasattr(model, 'head_idx') and len(getattr(model, 'head_idx', [])) >= 2)

    # Step 1: Joint Ready Pose (go_to_ready_pose 기준)
    # It exists to put the arms in a known configuration before the Cartesian IK runs. A caller
    # that knows the arms are already standing at this same ready pose -- Step 2 right after
    # Step 1.5, which used this very function -- can skip it instead of travelling back and forth.
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
    
    comp1 = rby.ComponentBasedCommandBuilder().set_body_command(
        rby.JointPositionCommandBuilder()
        .set_position(q_ready)
        .set_minimum_time(minimum_time)
    )
    if has_head:
        comp1.set_head_command(
            rby.JointPositionCommandBuilder()
            .set_position(np.zeros(2, dtype=np.float64))
            .set_minimum_time(minimum_time)
        )
    if skip_joint_pose:
        print("Step 1: Joint Ready Pose skipped (arms are already there).")
    else:
        print("Step 1: Moving to Joint Ready Pose...")
        cmd1 = rby.RobotCommandBuilder().set_command(comp1)
        check_motion_cancelled()
        rv1 = robot.send_command(cmd1, priority).get()
        if rv1.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError("Failed to move to Step 1: Joint Ready Pose.")

    # Determine whether v1.3 behavior is requested (UI setting takes precedence over hardware model)
    is_v13 = False
    if robot_version is not None:
        is_v13 = (str(robot_version).replace("v", "").strip() == "1.3")
    elif model is not None:
        model_name = getattr(model, 'robot_model_name', '').lower()
        joint_cnt = len(getattr(model, 'robot_joint_names', []))
        is_v13 = (model_name == 'm' or joint_cnt == 26)

    # Step 2: Cartesian Checking Pose (Lower Z to 0.15m for lowered fixed chest camera vs 0.27m for head)
    z_height = 0.15 if not has_head else 0.27
    y_val = 0.11 if is_v13 else 0.13
    # The fixed chest camera sits ~130 mm lower than the head one, so at the same reach it sees the
    # markers from 157 mm at a 35 deg incidence -- close and steeply angled, which is where the
    # headless runs lost their measurement precision (0.38 mm vs 0.22 mm noise). Reaching further
    # forward restores the head configuration's geometry (191 mm at 28 deg, vs 203 mm at 25 deg);
    # raising z instead makes the incidence angle worse, not better.
    x_reach = HEADLESS_X_REACH_M if not has_head else 0.3

    T_right = make_T(rot_z(0*D2R) @ rot_y(-90*D2R) @ rot_x(90*D2R), [x_reach, -y_val, z_height])
    T_right[:3, :3] = T_right[:3, :3] @ rot_z(180*D2R)
    
    T_left = make_T(rot_z(0*D2R) @ rot_y(-90*D2R) @ rot_x(-90*D2R), [x_reach, y_val, z_height])
    T_left[:3, :3] = T_left[:3, :3] @ rot_z(180*D2R)

    # v1.3 (Model M) branch: Rotate +90 deg around base frame Pitch (Y) axis
    if is_v13:
        R_pitch_90 = rot_y(90 * D2R)
        T_right[:3, :3] = R_pitch_90 @ T_right[:3, :3]
        T_left[:3, :3]  = R_pitch_90 @ T_left[:3, :3]

    body2 = rby.BodyComponentBasedCommandBuilder()

    if "right" in active_arms:
        header_right = rby.CommandHeaderBuilder()
        header_right.set_control_hold_time(0.5)
        
        right_cmd = rby.CartesianCommandBuilder()
        right_cmd.add_target("link_torso_5", "ee_right", T_right.astype(np.float32), 0.5, 1.0, 0.3)
        right_cmd.set_stop_position_tracking_error(pos_tolerance_m)
        right_cmd.set_stop_orientation_tracking_error(ori_tolerance_rad)
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
        left_cmd.set_stop_position_tracking_error(pos_tolerance_m)
        left_cmd.set_stop_orientation_tracking_error(ori_tolerance_rad)
        left_cmd.set_minimum_time(minimum_time)
        left_cmd.set_command_header(header_left)
        
        body2.set_left_arm_command(left_cmd)
    else:
        left_joint = rby.JointPositionCommandBuilder()
        left_joint.set_position(np.array([0, 0, 0, -90, 0, 0, 0], dtype=np.float64) * D2R)
        left_joint.set_minimum_time(minimum_time)
        body2.set_left_arm_command(left_joint)

    print("Step 2: Moving to Cartesian Checking Pose...")
    comp2 = rby.ComponentBasedCommandBuilder().set_body_command(body2)
    if has_head:
        comp2.set_head_command(
            rby.JointPositionCommandBuilder()
            .set_position(np.zeros(2, dtype=np.float64))
            .set_minimum_time(minimum_time)
        )
    cmd2 = rby.RobotCommandBuilder().set_command(comp2)
    check_motion_cancelled()
    rv2 = robot.send_command(cmd2, priority).get()
    if rv2.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError("Failed to move to Step 2: Cartesian Checking Pose.")

def make_dual_arm_head_cmd(T_right, T_left, active_arms, head_position=None, min_time=1.2, hold_time=0.5, q_right=None, q_left=None, elbow_angle_deg=None,
                           pos_tolerance_m=0.005, ori_tolerance_rad=0.02):
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
            if elbow_angle_deg is not None:
                right_cart.add_joint_position_target("right_arm_3", float(np.radians(elbow_angle_deg)))
            right_cart.set_stop_position_tracking_error(pos_tolerance_m)
            right_cart.set_stop_orientation_tracking_error(ori_tolerance_rad)
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
            if elbow_angle_deg is not None:
                left_cart.add_joint_position_target("left_arm_3", float(np.radians(elbow_angle_deg)))
            left_cart.set_stop_position_tracking_error(pos_tolerance_m)
            left_cart.set_stop_orientation_tracking_error(ori_tolerance_rad)
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

def send_auto_motion_cmd(
    robot,
    config,
    active_arms,
    T_right=None,
    T_left=None,
    q_right=None,
    q_left=None,
    head_position=None,
    elbow_angle_deg=None,
    pos_tolerance_m=0.005,
    ori_tolerance_rad=0.02,
):
    """
    Sends a unified motion command moving Arm and Head simultaneously in parallel.
    Data capture occurs cleanly after both arm and head arrive and settle at the target pose.
    """
    cmd = make_dual_arm_head_cmd(
        T_right=T_right,
        T_left=T_left,
        active_arms=active_arms,
        head_position=head_position,
        min_time=config.move_time,
        hold_time=config.hold_time,
        q_right=q_right,
        q_left=q_left,
        elbow_angle_deg=elbow_angle_deg,
        pos_tolerance_m=pos_tolerance_m,
        ori_tolerance_rad=ori_tolerance_rad,
    )
    check_motion_cancelled()
    rv = robot.send_command(cmd, config.priority).get()
    if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Auto motion command failed: {rv.finish_code}")

def execute_auto_motion_step(robot, config, motion_plan_step, active_arms, include_head_motion=True):
    """Moves to one plan step and returns it, or None when an optional step was skipped (its joint
    targets fall within OPTIONAL_POSE_LIMIT_MARGIN_DEG of a joint limit) -- then take no sample."""
    global _motion_state

    step_type = motion_plan_step.get("type")

    if step_type == "joint":
        state = robot.get_state()
        if state is None or getattr(state, 'position', None) is None:
            raise RuntimeError("Robot state position is None. Please check connection.")
        q_full = np.array(state.position)
        model = robot.model()
        dyn_model = robot.get_dynamics()

        # Save baseline configurations independently if not already saved
        if _motion_state["q_right_baseline"] is None:
            _motion_state["q_right_baseline"] = q_full[model.right_arm_idx[:7]].copy()
            _motion_state["q_left_baseline"] = q_full[model.left_arm_idx[:7]].copy()

        if include_head_motion and _motion_state["q_head_baseline"] is None:
            head_idx = list(model.head_idx[:2]) if len(model.head_idx) >= 2 else None
            _motion_state["q_head_baseline"] = np.array([float(q_full[i]) for i in head_idx], dtype=np.float64) if head_idx is not None else None
            _motion_state["q_head_0"] = np.array([float(q_full[i]) for i in head_idx], dtype=np.float64) if head_idx is not None else None

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

        if _motion_state["q_right_baseline"] is None:
            _motion_state["q_right_baseline"] = q_full[model.right_arm_idx[:7]].copy()
        if _motion_state["q_left_baseline"] is None:
            _motion_state["q_left_baseline"] = q_full[model.left_arm_idx[:7]].copy()

        q_right_target = _motion_state["q_right_baseline"].copy()
        q_left_target = _motion_state["q_left_baseline"].copy()

        if motion_plan_step.get("offsets_by_arm") is not None:
            # Explicit per-arm offsets (no mirroring), e.g. the wrist-diversity poses.
            targets = {"right": q_right_target, "left": q_left_target}
            for side, offsets in motion_plan_step["offsets_by_arm"].items():
                if side in active_arms:
                    for j_i, off_deg in offsets.items():
                        targets[side][int(j_i)] += np.deg2rad(off_deg)
            if motion_plan_step.get("optional"):
                names = model.robot_joint_names
                limit_state = dyn_model.make_state(["link_torso_5", "ee_right"], names)
                lower = np.asarray(dyn_model.get_limit_q_lower(limit_state))
                upper = np.asarray(dyn_model.get_limit_q_upper(limit_state))
                margin = np.deg2rad(OPTIONAL_POSE_LIMIT_MARGIN_DEG)
                for side, idx, q_t in (("right", model.right_arm_idx[:7], q_right_target),
                                       ("left", model.left_arm_idx[:7], q_left_target)):
                    if side in active_arms and (np.any(q_t < lower[idx] + margin) or np.any(q_t > upper[idx] - margin)):
                        return None      # skipped: the caller takes no sample
        elif "offsets_dict" in motion_plan_step and motion_plan_step["offsets_dict"] is not None:
            for j_i, off_deg in motion_plan_step["offsets_dict"].items():
                if j_i == 2:
                    if "right" in active_arms:
                        q_right_target[j_i] += np.deg2rad(off_deg)
                    if "left" in active_arms:
                        q_left_target[j_i] += np.deg2rad(-off_deg)
                else:
                    if "right" in active_arms:
                        q_right_target[j_i] += np.deg2rad(off_deg)
                    if "left" in active_arms:
                        q_left_target[j_i] += np.deg2rad(off_deg)
            j_idx = list(motion_plan_step["offsets_dict"].keys())[0]
        else:
            j_idx = motion_plan_step["joint_idx"]
            offset_deg = motion_plan_step["offset_deg"]

            if j_idx == 2:
                # Joint 2 (Shoulder Yaw): Opposing rotation to keep both markers in camera FOV
                if "right" in active_arms:
                    q_right_target[j_idx] += np.deg2rad(offset_deg)
                if "left" in active_arms:
                    q_left_target[j_idx] += np.deg2rad(-offset_deg)
            else:
                if "right" in active_arms:
                    q_right_target[j_idx] += np.deg2rad(offset_deg)
                if "left" in active_arms:
                    q_left_target[j_idx] += np.deg2rad(offset_deg)

        head_q = None
        if include_head_motion:
            if "head_q" in motion_plan_step and motion_plan_step["head_q"] is not None:
                head_q = motion_plan_step["head_q"]
            elif "head_tilt_offset_deg" in motion_plan_step or "head_pan_offset_deg" in motion_plan_step:
                d_pan = np.radians(motion_plan_step.get("head_pan_offset_deg", 0.0))
                d_tilt = np.radians(motion_plan_step.get("head_tilt_offset_deg", 0.0))
                base_head = _motion_state["q_head_0"] if _motion_state["q_head_0"] is not None else np.zeros(2, dtype=np.float64)
                head_q = np.array([base_head[0] + d_pan, base_head[1] + d_tilt], dtype=np.float64)
            else:
                # For regular joint sweeps (Joint 1, 2, 4, diagonal sweeps, elbow sweeps),
                # keep head strictly at the baseline aligned orientation
                head_q = _motion_state["q_head_baseline"].copy() if _motion_state["q_head_baseline"] is not None else np.zeros(2, dtype=np.float64)

        send_auto_motion_cmd(
            robot=robot,
            config=config,
            active_arms=active_arms,
            q_right=q_right_target if "right" in active_arms else None,
            q_left=q_left_target if "left" in active_arms else None,
            head_position=head_q,
        )

        wait_motion(config.settle_time)
        return motion_plan_step

    elif step_type == "restore_baseline":
        if _motion_state["q_right_baseline"] is not None:
            base_head = _motion_state["q_head_baseline"].copy() if (include_head_motion and _motion_state["q_head_baseline"] is not None) else None
            send_auto_motion_cmd(
                robot=robot,
                config=config,
                active_arms=active_arms,
                q_right=_motion_state["q_right_baseline"] if "right" in active_arms else None,
                q_left=_motion_state["q_left_baseline"] if "left" in active_arms else None,
                head_position=base_head,
            )

        wait_motion(config.settle_time)
        return motion_plan_step

    else:
        # Standard Cartesian step (with optional elbow bias target)
        T_right = motion_plan_step["T_right"]
        T_left = motion_plan_step["T_left"]
        head_q = motion_plan_step.get("head_q", None) if include_head_motion else None
        elbow_bias_deg = motion_plan_step.get("elbow_bias_deg", None)

        send_auto_motion_cmd(
            robot=robot,
            config=config,
            active_arms=active_arms,
            T_right=T_right,
            T_left=T_left,
            head_position=head_q,
            elbow_angle_deg=elbow_bias_deg,
        )

        wait_motion(config.settle_time)
        return motion_plan_step


def check_calibration_state(robot, model_name, active_arms, data, offset, log_cb=None, skip_ready=False):
    # Ensure Control Manager is enabled
    cm_state = robot.get_control_manager_state()
    if cm_state.state in [rby.ControlManagerState.State.MinorFault, rby.ControlManagerState.State.MajorFault]:
        if log_cb is not None:
            log_cb("[ControlManager] Control manager in fault state. Resetting...")
        robot.reset_fault_control_manager()
        wait_motion(1.0)
        
    cm_state = robot.get_control_manager_state()
    if cm_state.state != rby.ControlManagerState.State.Enabled:
        if log_cb is not None:
            log_cb("[ControlManager] Enabling control manager...")
        robot.enable_control_manager()
        wait_motion(1.0)

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
        check_motion_cancelled()
        rv1 = robot.send_command(cmd1, 10).get()
        if rv1.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"Failed to move to Joint Ready Pose: {rv1.finish_code}")
            
        wait_motion(1.0)
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
    check_motion_cancelled()
    rv2 = robot.send_command(cmd2, 10).get()
    if rv2.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Failed to move to Cartesian Checking Pose. FinishCode: {rv2.finish_code}")


def execute_draw_square_trajectory(robot, active_arms, offset=0.0, log_cb=None):
    """Draw square trajectory with dual arms to verify calibration status."""
    def _log(msg):
        if log_cb:
            log_cb(msg)

    square_points = [
        [0.35, 0.07, 0.0],
        [0.35, 0.0, 0.07],
        [0.35, -0.07, 0.0],
        [0.35, 0.0, -0.07],
    ]
    
    _log("Starting square drawing sequence (2 loops)...")
    for loop_idx in range(2):
        _log(f"Loop {loop_idx + 1} / 2")
        for pt_idx, pt in enumerate(square_points):
            _log(f"  Target Point {pt_idx + 1}: X={pt[0]}, Y={pt[1]}, Z={pt[2]}")
            
            roll_r = 90.0 * D2R
            pitch_r = -90.0 * D2R
            yaw_r = 0.0
            cr_r = np.cos(roll_r); sr_r = np.sin(roll_r)
            cp_r = np.cos(pitch_r); sp_r = np.sin(pitch_r)
            cy_r = np.cos(yaw_r); sy_r = np.sin(yaw_r)
            
            T_right = np.eye(4, dtype=np.float64)
            T_right[0, 0] = cy_r * cp_r
            T_right[0, 1] = sr_r * sp_r * cy_r - cr_r * sy_r
            T_right[0, 2] = cr_r * sp_r * cy_r + sr_r * sy_r
            T_right[0, 3] = pt[0]
            T_right[1, 0] = sy_r * cp_r
            T_right[1, 1] = sr_r * sp_r * sy_r + cr_r * cy_r
            T_right[1, 2] = cr_r * sp_r * sy_r - sr_r * cy_r
            T_right[1, 3] = pt[1] - offset
            T_right[2, 0] = -sp_r
            T_right[2, 1] = cp_r * sr_r
            T_right[2, 2] = cp_r * cr_r
            T_right[2, 3] = pt[2]
            
            roll_l = -90.0 * D2R
            pitch_l = -90.0 * D2R
            yaw_l = 0.0
            cr_l = np.cos(roll_l); sr_l = np.sin(roll_l)
            cp_l = np.cos(pitch_l); sp_l = np.sin(pitch_l)
            cy_l = np.cos(yaw_l); sy_l = np.sin(yaw_l)
            
            T_left = np.eye(4, dtype=np.float64)
            T_left[0, 0] = cy_l * cp_l
            T_left[0, 1] = sr_l * sp_l * cy_l - cr_l * sy_l
            T_left[0, 2] = cr_l * sp_l * cy_l + sr_l * sy_l
            T_left[0, 3] = pt[0]
            T_left[1, 0] = sy_l * cp_l
            T_left[1, 1] = sr_l * sp_l * sy_l + cr_l * cy_l
            T_left[1, 2] = cr_l * sp_l * sy_l - sr_l * cy_l
            T_left[1, 3] = pt[1] + offset
            T_left[2, 0] = -sp_l
            T_left[2, 1] = cp_l * sr_l
            T_left[2, 2] = cp_l * cr_l
            T_left[2, 3] = pt[2]
            
            cmd = make_dual_arm_head_cmd(
                T_right=T_right,
                T_left=T_left,
                active_arms=active_arms,
                head_position=None,
                min_time=2.0,
                hold_time=0.2
            )
            check_motion_cancelled()
            rv = robot.send_command(cmd, 10).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                raise RuntimeError(f"Draw point move failed: {rv.finish_code}")
            wait_motion(0.5)
            
    _log("Square drawing sequence completed successfully.")
    return True


def move_to_zero_pose(robot, model, arm="both", include_head=True, log_cb=None):
    """Move robot arms and head to zero joint angle pose."""
    def _log(msg):
        if log_cb:
            log_cb(msg)

    if robot is None or model is None:
        raise RuntimeError("Robot or model is not available.")

    try:
        from .home_offset import movej
    except ImportError:
        from core.robot.home_offset import movej
    right_zero_pose = np.zeros(len(model.right_arm_idx))
    left_zero_pose = np.zeros(len(model.left_arm_idx))
    head_zero_pose = (
        np.zeros(len(model.head_idx))
        if (include_head and hasattr(model, 'head_idx') and model.head_idx is not None and len(model.head_idx) >= 2)
        else None
    )

    _log("Moving robot to zero pose...")
    ok = movej(
        robot,
        right_arm=right_zero_pose if arm in ("right", "both") else None,
        left_arm=left_zero_pose if arm in ("left", "both") else None,
        head=head_zero_pose,
        minimum_time=5,
    )
    if not ok:
        raise RuntimeError("Failed to move robot to zero pose")
    return True


def verify_and_align_head_at_ready_pose(
    robot,
    marker_st,
    model,
    active_arms,
    priority=10,
    include_head_motion=True,
    prompt_teaching_cb=None,
    log_cb=None,
    on_head_aligned_cb=None,
):
    """Verifies marker visibility at ready pose, prompting user teaching if needed,
    and automatically centers the head camera to align markers."""
    def _log(msg):
        if log_cb:
            log_cb(msg)

    if marker_st is None:
        return True

    _log("[Step2] Verifying marker visibility at the initial ready pose...")
    wait_motion(1.5)

    # Check right arm
    right_check = marker_st.get_marker_transform(sampling_time=1.5, side="right")
    if right_check is None:
        _log("[INFO] Right arm marker not visible at Init Pose. Showing teaching dialog...")
        resolved = prompt_teaching_cb("right") if prompt_teaching_cb else False
        if not resolved:
            raise RuntimeError("Right arm posture teaching canceled by user.")

    # Check left arm
    left_check = marker_st.get_marker_transform(sampling_time=1.5, side="left")
    if left_check is None:
        _log("[INFO] Left arm marker not visible at Init Pose. Showing teaching dialog...")
        resolved = prompt_teaching_cb("left") if prompt_teaching_cb else False
        if not resolved:
            raise RuntimeError("Left arm posture teaching canceled by user.")

    # Re-verify visibility
    _log("[INFO] Re-verifying marker visibility at the new posture...")
    right_check = marker_st.get_marker_transform(sampling_time=1.5, side="right")
    left_check = marker_st.get_marker_transform(sampling_time=1.5, side="left")
    if right_check is None or left_check is None:
        raise RuntimeError("Marker still not detected at Init Pose after teaching.")

    # Auto-center head if markers are vertically or horizontally offset from camera optical center
    if include_head_motion and model is not None:
        try:
            head_idx = getattr(model, "head_idx", None)
            if head_idx is not None and len(head_idx) >= 2:
                pts = []
                if right_check is not None and len(right_check) > 0:
                    T_r = np.array(right_check[0]) if (isinstance(right_check, list) and isinstance(right_check[0], list)) else np.array(right_check)
                    if T_r.ndim == 2 and T_r.shape == (4, 4):
                        pts.append(T_r[:3, 3])
                if left_check is not None and len(left_check) > 0:
                    T_l = np.array(left_check[0]) if (isinstance(left_check, list) and isinstance(left_check[0], list)) else np.array(left_check)
                    if T_l.ndim == 2 and T_l.shape == (4, 4):
                        pts.append(T_l[:3, 3])
                if len(pts) > 0:
                    p_mid = np.mean(pts, axis=0)  # [X_cam, Y_cam, Z_cam] in camera frame
                    pitch_err_rad = np.arctan2(p_mid[1], p_mid[2])
                    yaw_err_rad = np.arctan2(p_mid[0], p_mid[2])

                    if abs(pitch_err_rad) > np.deg2rad(1.0) or abs(yaw_err_rad) > np.deg2rad(1.5):
                        _log(f"[INFO] Auto-centering head: aligning camera optical center (Pitch: {np.rad2deg(pitch_err_rad):+.2f}°, Yaw: {np.rad2deg(yaw_err_rad):+.2f}°)...")
                        state = robot.get_state()
                        if state is not None and getattr(state, 'position', None) is not None:
                            q_full = np.array(state.position)
                            h_idx = list(head_idx)
                            q_head_curr = np.array([float(q_full[i]) for i in h_idx], dtype=np.float64)
                            q_head_target = q_head_curr + np.array([yaw_err_rad, pitch_err_rad], dtype=np.float64)

                            # Clip to safe head limits
                            q_head_target[0] = np.clip(q_head_target[0], -25.0 * D2R, 25.0 * D2R)
                            q_head_target[1] = np.clip(q_head_target[1], -20.0 * D2R, 20.0 * D2R)

                            comp = rby.ComponentBasedCommandBuilder().set_head_command(
                                rby.JointPositionCommandBuilder()
                                .set_position(q_head_target)
                                .set_minimum_time(2.0)
                            )
                            cmd = rby.RobotCommandBuilder().set_command(comp)
                            check_motion_cancelled()
                            robot.send_command(cmd, priority).get()
                            wait_motion(1.0)

                            if on_head_aligned_cb:
                                on_head_aligned_cb(q_head_target)
                            _log(f"[SUCCESS] Head auto-centered successfully to (Pan: {np.rad2deg(q_head_target[0]):+.2f}°, Tilt: {np.rad2deg(q_head_target[1]):+.2f}°).")
        except Exception as ex:
            _log(f"[WARNING] Auto-centering head encountered a minor issue: {ex}")

    _log("[SUCCESS] Marker visibility verified successfully at the ready pose.")
    return True
