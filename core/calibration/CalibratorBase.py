import sys
import time
import logging
import os
import yaml
import numpy as np
import rby1_sdk as rby
import matplotlib
import threading
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class BaseCalibrator:
    JOINT_CONFIGS = {
        "wrist_roll_v13":  {"cand_joint": 6, "sweep_joint_A": 6, "sweep_joint_B": 5, "offset_key": "wrist_roll",  "offset_range": (-30.0, 30.0), "sweep_range_A": 20.0, "sweep_range_B": 20.0},
        "wrist_pitch_v13": {"cand_joint": 5, "sweep_joint_A": 5, "sweep_joint_B": 3, "offset_key": "wrist_pitch", "offset_range": (-30.0, 30.0), "sweep_range_A": 20.0, "sweep_range_B": 15.0},
        "wrist_yaw2":      {"cand_joint": 6, "sweep_joint_A": 6, "sweep_joint_B": 5, "offset_key": "wrist_yaw2",  "offset_range": (-30.0, 30.0), "sweep_range_A": 20.0, "sweep_range_B": 20.0},
        "wrist_pitch":     {"cand_joint": 5, "sweep_joint_A": 4, "sweep_joint_B": 6, "offset_key": "wrist_pitch", "offset_range": (-30.0, 30.0), "sweep_range_A": 20.0, "sweep_range_B": 20.0},
        "elbow":           {"cand_joint": 3, "sweep_joint_A": 2, "sweep_joint_B": 4, "offset_key": "elbow",       "offset_range": (-5.0, 0.0),   "sweep_range_A": 20.0, "sweep_range_B": 20.0},
    }
    MARKER_CONFIGS = {
        "axis_4": {"joint_i": 4, "start_deg": -15.0, "end_deg": 15.0, "n_nom_v12": [0.0, 0.0, 1.0], "n_nom_v13": [0.0, 0.0, 1.0]},
        "axis_5": {"joint_i": 5, "start_deg": -25.0, "end_deg": 25.0, "n_nom_v12": [0.0, 1.0, 0.0], "n_nom_v13": [0.0, 1.0, 0.0]},
        "axis_6": {"joint_i": 6, "start_deg": -20.0, "end_deg": 20.0, "n_nom_v12": [0.0, 0.0, 1.0], "n_nom_v13": [1.0, 0.0, 0.0]},
    }
    MOCK_GT_OFFSETS = {
        "right": {
            "joint0": 0.5,
            "joint1": 2.5,
            "joint2": 1.2,
            "joint3": 0.5,
            "joint4": -1.5,
            "joint5_v13": -2.1,
            "joint5_v12": 5.4,
            "joint6": 2.3,
            "bracket_pos": [0.0005, 0.0, 0.002],  # meters
            "bracket_rpy": [-0.1, -0.1, 0.05]        # degrees
        },
        "left": {
            "joint0": -0.4,
            "joint1": -1.6,
            "joint2": -1.0,
            "joint3": 0.7,
            "joint4": 1.1,
            "joint5_v13": 3.6,
            "joint5_v12": -3.0,
            "joint6": 3.5,
            "bracket_pos": [0.001, 0.0005, -0.002], # meters
            "bracket_rpy": [0.1, 0.1, 0.0]        # degrees
        }
    }
    NOMINAL_BRACKET_TEMPLATES = {
        "1.3": {
            "left":  [0.097, 0.0, -0.005, 90.0, 0.0, -90.0],
            "right": [0.097, 0.0, -0.005, 90.0, 0.0, -90.0]
        },
        "1.2": {
            "left":  [0.0, 0.054, -0.048, 90.0, 0.0, 0.0],
            "right": [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
        }
    }
    def __init__(self, marker_st=None, robot=None):
        self.marker_st = marker_st
        self.robot = robot
        self.robot_version = "1.2"
        
        # Load camera setting config if available
        self.camera_config = {}
        self.load_camera_config()
        
        self.ready_poses = {}
        self.load_ready_poses()
        
        # Active joint home offsets to apply to commanded trajectories
        self.joint_offsets = {"wrist_pitch": 0.0, "elbow": 0.0}
        self.stop_requested = False

    def load_ready_poses(self):
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
        yaml_path = os.path.join(config_dir, "ready_poses.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    self.ready_poses = yaml.safe_load(f) or {}
                logging.info(f"Loaded ready poses from {yaml_path}")
            except Exception as e:
                logging.error(f"Failed to load ready_poses.yaml: {e}")
                sys.exit(f"[CRITICAL ERROR] Failed to parse ready_poses.yaml: {e}")
        else:
            logging.error(f"ready_poses.yaml not found at {yaml_path}!")
            sys.exit(f"[CRITICAL ERROR] ready_poses.yaml not found at {yaml_path}!")

    def get_robot_version(self) -> str:
        """Returns the robot version as a string: '1.0', '1.1', '1.2', or '1.3'."""
        return str(getattr(self, "robot_version", "1.2"))

    @property
    def is_mock(self) -> bool:
        return self.marker_st is None or type(self.marker_st).__name__ == "SimulatedMarkerTransform"

    def is_v13(self) -> bool:
        """Returns True only for model-m v1.3 robots."""
        return self.get_robot_version() == "1.3"

    def get_ready_pose(self, version_key, type_key, mode_key, arm_side):
        if not self.ready_poses:
            raise RuntimeError("Ready poses are not loaded or the configuration file is empty.")
        
        try:
            val = self.ready_poses[version_key]
            if type_key == "joint":
                val = val["joint"][mode_key][f"{arm_side}_arm"]
            elif type_key == "check_calib":
                val = val["check_calib"][f"{arm_side}_arm"]
            else:
                val = val["marker"][f"{arm_side}_arm"]
            return np.deg2rad(val)
        except (KeyError, TypeError) as e:
            raise KeyError(
                f"[ERROR] Failed to get ready pose for version='{version_key}', type='{type_key}', mode='{mode_key}', arm='{arm_side}_arm'. "
                f"Please check your ready_poses.yaml file. Details: {e}"
            )


    def load_camera_config(self):
        # Locate setting.yaml
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
        yaml_path = os.path.join(config_dir, "setting.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
                    self.camera_config = config_data.get("camera", {})
                logging.info(f"Loaded config from setting.yaml")
            except Exception as e:
                logging.error(f"Failed to load setting.yaml: {e}")
        else:
            logging.warning(f"setting.yaml not found at {yaml_path}")

    def save_debug_points(self, arm_side, axis_num, dataset, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, type_key, log_callback=None):
        try:
            if T_mount_to_cam is None:
                mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                T_mount_to_cam = self.make_transform(mount_to_cam)
            from core.paths import CONFIG_PATHS
            result_txt_dir = CONFIG_PATHS["txt_dir"]
            os.makedirs(result_txt_dir, exist_ok=True)
            if not self.robot:
                raise RuntimeError("Robot instance is not initialized")
            arm_idx = self.robot.model().left_arm_idx if arm_side == "left" else self.robot.model().right_arm_idx
            
            filename = os.path.join(result_txt_dir, f"sweep_points_{arm_side}_{type_key}_axis_{axis_num}.txt")
            
            # Determine prefix for header
            if type_key == "joint_A":
                angle_header_name = "Joint_A"
            elif type_key == "joint_B":
                angle_header_name = "Joint_B"
            else:
                angle_header_name = f"Joint_{axis_num}"
            file_exists = os.path.exists(filename)
            with open(filename, "a") as f:
                if file_exists:
                    f.write("\n=== NEW ITERATION ===\n")
                f.write(f"# {angle_header_name}_Angle(deg), Cam_X(mm), Cam_Y(mm), Cam_Z(mm), Torso_X(mm), Torso_Y(mm), Torso_Z(mm), EE_X(mm), EE_Y(mm), EE_Z(mm), "
                        "T_cam2marker_flat(16), T_torso2marker_flat(16), T_ee2marker_flat(16)\n")
                for q_full, pose in dataset:
                    q_val = q_full[arm_idx[axis_num]]
                    s_deg = np.degrees(q_val - initial_joint_pos[axis_num])
                    p_cam = pose[:3, 3]
                    
                    T_cam_to_marker = pose
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, ee_name)
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                    T_t5_to_marker = T_t5_to_cam @ T_cam_to_marker
                    T_ee_to_marker = np.linalg.inv(T_t5_to_ee) @ T_t5_to_marker
                    
                    T_cam_flat_str = ", ".join(f"{v:.6f}" for v in T_cam_to_marker.flatten())
                    T_t5_flat_str = ", ".join(f"{v:.6f}" for v in T_t5_to_marker.flatten())
                    T_ee_flat_str = ", ".join(f"{v:.6f}" for v in T_ee_to_marker.flatten())
                    
                    f.write(f"{s_deg:.4f}, {p_cam[0]*1000.0:.4f}, {p_cam[1]*1000.0:.4f}, {p_cam[2]*1000.0:.4f}, "
                            f"{p_meas_t5[0]*1000.0:.4f}, {p_meas_t5[1]*1000.0:.4f}, {p_meas_t5[2]*1000.0:.4f}, "
                            f"{p_ee[0]*1000.0:.4f}, {p_ee[1]*1000.0:.4f}, {p_ee[2]*1000.0:.4f}, "
                            f"{T_cam_flat_str}, {T_t5_flat_str}, {T_ee_flat_str}\n")
            if log_callback:
                if type_key == "marker":
                    log_callback(f"[DEBUG] Saved Axis {axis_num} marker sweep debug points to {os.path.basename(filename)}")
                else:
                    log_callback(f"[DEBUG] Saved Axis {axis_num} debug points to {os.path.basename(filename)}")
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to save debug points: {e}")


    @staticmethod
    def initialize_robot(address, model, power=".*", servo=".*"):
        robot = rby.create_robot(address, model)
        if not robot.connect():
            logging.error(f"Failed to connect robot {address}")
            return None
        
        # Safety check: Verify actual connected robot model matches expected model
        try:
            robot_info = robot.get_robot_info()
            actual_model = robot_info.robot_model_name.lower()
            expected_model = model.lower()
            if actual_model != expected_model:
                logging.warning(f"Model mismatch! UI selected model: {model}, but actual robot model is: {robot_info.robot_model_name}. Auto-reconnecting with actual model...")
                robot.disconnect()
                robot = rby.create_robot(address, robot_info.robot_model_name)
                if not robot.connect():
                    logging.error(f"Failed to connect robot {address} with actual model {robot_info.robot_model_name}")
                    return None
        except Exception as e:
            logging.error(f"Failed to verify robot model: {e}")
            robot.disconnect()
            return None

        # Check if overall power is ON; if not, turn on overall power
        try:
            if not robot.is_power_on(".*"):
                logging.info("Power is not ON. Turning overall power on...")
                if not robot.power_on(".*"):
                    logging.error("Failed to turn power on.")
                    robot.disconnect()
                    return None
            else:
                logging.info("Power is already ON.")
        except Exception as e:
            logging.error(f"Failed to check or set power status: {e}")
            robot.disconnect()
            return None

        # Wait 1 second
        time.sleep(1.0)

        # Check control manager status
        try:
            cm_state = robot.get_control_manager_state().state
            is_cm_enabled = (cm_state == rby.ControlManagerState.State.Enabled)
        except Exception as e:
            logging.warning(f"Failed to check control manager state: {e}")
            is_cm_enabled = False

        # Check if both arms' servos are ON
        try:
            is_servo_ok = robot.is_servo_on(".*")
        except Exception as e:
            logging.warning(f"Failed to check servo status: {e}")
            is_servo_ok = False

        def enable_cm_helper(r):
            try:
                cm_state_post = r.get_control_manager_state()
                if cm_state_post.state in [
                    rby.ControlManagerState.State.MajorFault,
                    rby.ControlManagerState.State.MinorFault,
                ]:
                    logging.warning(f"Control manager is in fault state: {cm_state_post.state}. Resetting...")
                    if not r.reset_fault_control_manager():
                        logging.error("Failed to reset control manager")
                
                cm_state_post = r.get_control_manager_state()
                if cm_state_post.state == rby.ControlManagerState.State.Enabled:
                    logging.info("Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...")
                    try:
                        r.disable_control_manager()
                        time.sleep(0.5)
                    except Exception as ex:
                        logging.warning(f"Failed to disable control manager: {ex}")
                
                logging.info("Enabling control manager with unlimited_mode_enabled=True...")
                if not r.enable_control_manager(unlimited_mode_enabled=True):
                    logging.error("Failed to enable control manager with unlimited_mode_enabled=True")
                else:
                    time.sleep(1.0)
            except Exception as ex:
                logging.error(f"Failed to configure control manager: {ex}")

        if is_servo_ok:
            logging.info("Servos are ON. Ensuring Control Manager is enabled with unlimited mode...")
            enable_cm_helper(robot)
        else:
            # 아니면, 일단 disable한다음에 양팔 서보 키고 enable
            logging.info("Servos are not ON. Disabling Control Manager first to turn on servos...")
            if is_cm_enabled:
                try:
                    robot.disable_control_manager()
                    time.sleep(0.5)
                except Exception as e:
                    logging.warning(f"Failed to disable control manager: {e}")
            
            logging.info("Turning servos on...")
            if not robot.servo_on(".*"):
                logging.error("Failed to turn servos on.")
            else:
                time.sleep(0.5)
            
            enable_cm_helper(robot)

        return robot

    @staticmethod
    def terminate_robot(robot):
        if robot:
            try:
                robot.disconnect()
                return True
            except Exception as e:
                logging.error(f"Failed to disconnect robot: {e}")
        return False

    @staticmethod
    def compute_fk(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
        model = robot.model()
        state = dyn_model.make_state([base_link, ee_link], model.robot_joint_names)
        state.set_q(q)
        dyn_model.compute_forward_kinematics(state)
        T = dyn_model.compute_transformation(state, 0, 1)
        return T

    @staticmethod
    def make_transform(data):
        """
        Creates a 4x4 transformation matrix from [x, y, z, roll, pitch, yaw].
        Coordinates in meters, angles in degrees (ZYX Euler).
        """
        T = np.eye(4)
        T[:3, 3] = data[:3]
        yaw = data[5]
        pitch = data[4]
        roll = data[3]
        T[:3, :3] = R_scipy.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
        return T



    def get_simulated_marker_pose(self, arm_side, sweep_joint=None, current_offset_deg=0.0, cand_joint=None, q_actual=None):
        is_v13 = self.is_v13()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        ee_name = f"ee_{arm_side}"
        
        mock_gt = self.MOCK_GT_OFFSETS[arm_side]
        bracket_pos_gt = mock_gt["bracket_pos"]
        bracket_rpy_gt = mock_gt["bracket_rpy"]
            
        injected_joint_offsets_deg = [0.0] * 7
        injected_joint_offsets_deg[0] = mock_gt.get("joint0", 0.0)
        injected_joint_offsets_deg[1] = mock_gt.get("joint1", 0.0)
        injected_joint_offsets_deg[2] = mock_gt.get("joint2", 0.0)
        injected_joint_offsets_deg[3] = mock_gt.get("joint3", 0.0)
        injected_joint_offsets_deg[4] = mock_gt.get("joint4", 0.0)
        injected_joint_offsets_deg[5] = mock_gt.get("joint5_v13" if is_v13 else "joint5_v12", 0.0)
        injected_joint_offsets_deg[6] = mock_gt.get("joint6", 0.0)

        
        if q_actual is None:
            state = self.robot.get_state()
            q_actual = np.array(state.position)
        else:
            q_actual = np.array(q_actual)
        
        for i in range(7):
            q_actual[arm_idx[i]] += np.radians(injected_joint_offsets_deg[i])
            

        dyn_model = self.robot.get_dynamics()
        T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_actual, ee_name)
        
        # Always use the version-specific nominal design values as the baseline for simulation data generation
        version_suffix = "_v13" if is_v13 else "_v12"
        tf_key = f"Tf_to_marker_{arm_side}{version_suffix}"
        tf_vec = self.camera_config.get(tf_key)
        if tf_vec is None:
            ver_key = "1.3" if is_v13 else "1.2"
            tf_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
            
        nominal_pos = tf_vec[:3]
        nominal_rpy = tf_vec[3:6]
                
        marker_pos_gt = np.array(nominal_pos) + np.array(bracket_pos_gt)
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
        R_bracket_offset = R_scipy.from_euler('ZYX', [bracket_rpy_gt[2], bracket_rpy_gt[1], bracket_rpy_gt[0]], degrees=True).as_matrix()
        R_ee_m_gt = R_bracket_offset @ R_ee_m_ideal
        
        T_ee_to_marker_gt = np.eye(4)
        T_ee_to_marker_gt[:3, :3] = R_ee_m_gt
        T_ee_to_marker_gt[:3, 3] = marker_pos_gt
        
        T_t5_to_marker = T_t5_to_ee @ T_ee_to_marker_gt
        
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        T_head_to_cam_gt = self.make_transform(mount_to_cam)
        
        T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_actual, "link_head_2", "link_torso_5")
        T_t5_to_cam = T_t5_to_head @ T_head_to_cam_gt
        
        T_cam_to_marker = np.linalg.inv(T_t5_to_cam) @ T_t5_to_marker
        
        return T_cam_to_marker

    def movej(self, robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=0, apply_offsets=True, priority=10):
        if getattr(self, 'stop_requested', False):
            return False
        if not robot:
            return False
            
        if apply_offsets and hasattr(self, 'joint_offsets') and self.joint_offsets is not None:
            # Offset mapping: Joint 3 (index 3) is elbow
            # For v1.3:
            # - Joint 5 (index 5) is wrist pitch
            # - Joint 6 (index 6) is wrist roll
            # For v1.2:
            # - Joint 5 (index 5) is wrist pitch
            is_v13 = self.is_v13()
            
            # Support both flat and nested left/right dictionary structures
            if "left" in self.joint_offsets and "right" in self.joint_offsets:
                left_offsets = self.joint_offsets["left"]
                right_offsets = self.joint_offsets["right"]
            else:
                left_offsets = self.joint_offsets
                right_offsets = self.joint_offsets
                
            if right_arm is not None:
                right_arm = list(right_arm)
                r_j6_offset = right_offsets.get("wrist_roll", 0.0) if is_v13 else right_offsets.get("wrist_yaw2", 0.0)
                right_arm[6] += np.radians(r_j6_offset)
                right_arm[5] += np.radians(right_offsets.get("wrist_pitch", 0.0))
                right_arm[3] += np.radians(right_offsets.get("elbow", 0.0))
            if left_arm is not None:
                left_arm = list(left_arm)
                l_j6_offset = left_offsets.get("wrist_roll", 0.0) if is_v13 else left_offsets.get("wrist_yaw2", 0.0)
                left_arm[6] += np.radians(l_j6_offset)
                left_arm[5] += np.radians(left_offsets.get("wrist_pitch", 0.0))
                left_arm[3] += np.radians(left_offsets.get("elbow", 0.0))

        comp_cmd = rby.ComponentBasedCommandBuilder()
        
        has_body = False
        body_cmd = rby.BodyComponentBasedCommandBuilder()
        if torso is not None:
            body_cmd.set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(torso)
            )
            has_body = True
        if right_arm is not None:
            body_cmd.set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(right_arm)
            )
            has_body = True
        if left_arm is not None:
            body_cmd.set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(left_arm)
            )
            has_body = True
        
        if has_body:
            comp_cmd.set_body_command(body_cmd)

        if head is not None:
            comp_cmd.set_head_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(head)
            )
        
        cmd = rby.RobotCommandBuilder().set_command(comp_cmd)
        
        try:
            rv = robot.send_command(cmd, priority).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                print(f"[DEBUG MOVEJ ERROR] Failed to conduct movej. Finish code: {rv.finish_code}", flush=True)
                logging.error(f"Failed to conduct movej. Finish code: {rv.finish_code}")
                return False
            return True
        except Exception as e:
            print(f"[DEBUG MOVEJ EXCEPTION] movej exception: {e}", flush=True)
            logging.error(f"movej exception: {e}")
            return False

    @staticmethod
    def fit_circle_3d(points, robust=True):
        """
        Fits a 3D circle to points.
        If robust is True, applies robust worst-inlier outlier rejection and moving median filter.
        Otherwise, performs a smooth closed-form algebraic fit for noise-free kinematics.
        Returns (center_3d, R_circle, radius, rmse, pts_2d, uc, vc)
        """
        points = np.array(points)
        
        if not robust:
            centroid = np.mean(points, axis=0)
            pts_centered = points - centroid
            _, _, vh = np.linalg.svd(pts_centered)
            normal = vh[2, :]
            ex = vh[0, :]
            ey = vh[1, :]
            pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)
            A = np.c_[2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(len(pts_2d))]
            b = pts_2d[:, 0]**2 + pts_2d[:, 1]**2
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            uc, vc = res[0], res[1]
            center_3d = centroid + uc * ex + vc * ey
            R_circle = np.column_stack((ex, ey, normal))
            radius_3d = np.mean(np.linalg.norm(points - center_3d, axis=1))
            return center_3d, R_circle, radius_3d, 0.0, pts_2d, uc, vc
        
        # Apply 3D Moving Median Filter (window size 5) to smooth out camera sensor jitter
        if len(points) >= 5:
            smoothed = np.copy(points)
            for i in range(2, len(points) - 2):
                smoothed[i] = np.median(points[i - 2 : i + 3], axis=0)
            points = smoothed
            
        inlier_mask = np.ones(len(points), dtype=bool)
        
        for out_iter in range(15):
            pts_in = points[inlier_mask]
            if len(pts_in) < 10:
                break
                
            centroid = np.mean(pts_in, axis=0)
            pts_centered = pts_in - centroid
            
            _, _, vh = np.linalg.svd(pts_centered)
            normal = vh[2, :]
            ex = vh[0, :]
            ey = vh[1, :]
            pts_2d_in = np.dot(pts_centered, np.vstack((ex, ey)).T)

            A = np.c_[2 * pts_2d_in[:, 0], 2 * pts_2d_in[:, 1], np.ones(len(pts_2d_in))]
            b = pts_2d_in[:, 0]**2 + pts_2d_in[:, 1]**2
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            uc, vc = res[0], res[1]
            radius = np.sqrt(max(0.001, res[2] + uc**2 + vc**2))
            
            def residuals(params):
                u, v, R = params
                return np.sqrt((pts_2d_in[:, 0] - u)**2 + (pts_2d_in[:, 1] - v)**2) - R
                
            opt = least_squares(residuals, [uc, vc, radius], loss='huber')
            uc_opt, vc_opt, R_opt = opt.x
            
            # Recompute errors for all original points
            all_errors = []
            for pt in points:
                pt_centered_all = pt - centroid
                u_all = np.dot(pt_centered_all, ex)
                v_all = np.dot(pt_centered_all, ey)
                dist_to_center = np.sqrt((u_all - uc_opt)**2 + (v_all - vc_opt)**2)
                err = abs(dist_to_center - R_opt)
                all_errors.append(err)
            all_errors = np.array(all_errors)
            
            # Find worst point among inliers
            inlier_indices = np.where(inlier_mask)[0]
            inlier_errors = all_errors[inlier_mask]
            worst_inlier_idx_in_inliers = np.argmax(inlier_errors)
            worst_global_idx = inlier_indices[worst_inlier_idx_in_inliers]
            worst_error = inlier_errors[worst_inlier_idx_in_inliers]
            
            if worst_error > 0.1:
                inlier_mask[worst_global_idx] = False
            else:
                break
                
        # Final fit on clean inliers
        pts_in = points[inlier_mask]
        centroid = np.mean(pts_in, axis=0)
        pts_centered = pts_in - centroid
        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[2, :]
        ex = vh[0, :]
        ey = vh[1, :]
        pts_2d_in = np.dot(pts_centered, np.vstack((ex, ey)).T)
        A = np.c_[2 * pts_2d_in[:, 0], 2 * pts_2d_in[:, 1], np.ones(len(pts_2d_in))]
        b = pts_2d_in[:, 0]**2 + pts_2d_in[:, 1]**2
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        uc, vc = res[0], res[1]
        radius = np.sqrt(max(0.001, res[2] + uc**2 + vc**2))
        
        def residuals_final(params):
            u, v, R = params
            return np.sqrt((pts_2d_in[:, 0] - u)**2 + (pts_2d_in[:, 1] - v)**2) - R
            
        opt = least_squares(residuals_final, [uc, vc, radius], loss='huber')
        uc_opt, vc_opt, R_opt = opt.x
        rmse = np.sqrt(np.mean(opt.fun**2))
        center_3d = centroid + uc_opt * ex + vc_opt * ey
        
        # 2D representation for all points
        pts_centered_all = points - centroid
        pts_2d_all = np.dot(pts_centered_all, np.vstack((ex, ey)).T)
        
        R_circle = np.column_stack((ex, ey, normal))
        
        radius_3d = np.mean(np.linalg.norm(pts_in - center_3d, axis=1))
        
        return center_3d, R_circle, radius_3d, rmse, pts_2d_all, uc_opt, vc_opt

    @staticmethod
    def fit_circle_3d_and_6dof_misalignment(relative_poses, captured_angles, axis_prior=None, return_plot_data=False, robust=True):
        points = np.array([T[:3, 3] * 1000.0 for T in relative_poses])
        angles_rad_base = np.radians(captured_angles)
        
        # Robust check for NaN or Inf in inputs
        if len(points) == 0:
            raise ValueError("fit_circle_3d_and_6dof_misalignment: Input points list is empty.")
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Input points contain NaN or Inf values! points={points}")
        if np.any(np.isnan(angles_rad_base)) or np.any(np.isinf(angles_rad_base)):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Input captured_angles contain NaN or Inf values! angles={captured_angles}")
            
        # Initial Center and Normal estimation using unified circle fit
        c_fit, R_fit, radius_fit, rmse_fit, _, _, _ = BaseCalibrator.fit_circle_3d(points, robust=robust)
        
        # Check if initial circle fit yielded valid numbers
        if np.any(np.isnan(c_fit)) or np.any(np.isinf(c_fit)) or np.isnan(radius_fit):
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: Initial circle fit (fit_circle_3d) returned NaN or Inf values! c_fit={c_fit}, radius_fit={radius_fit}")
        
        # Nominal Normal
        if axis_prior is not None:
            n_nominal = np.array(axis_prior, dtype=float)
            n_nominal /= np.linalg.norm(n_nominal)
        else:
            n_nominal = np.array([0.0, 0.0, 1.0])
            
        # Initial Normal
        if axis_prior is not None:
            best_normal = n_nominal.copy()
        else:
            best_normal = R_fit[:, 2] # Normal is the Z-axis of R_fit
            
        centroid = np.mean(points, axis=0)
        ex = R_fit[:, 0]
        ey = R_fit[:, 1]

        # Sagitta formula
        C_chord_vec = points[-1] - points[0]
        C_chord = np.linalg.norm(C_chord_vec)
        p_mid_chord = (points[0] + points[-1]) / 2.0
        p_mid_arc = points[len(points) // 2]
        v_sag = p_mid_arc - p_mid_chord
        H_sag = np.linalg.norm(v_sag)
        
        if H_sag > 0.05 and C_chord > 1.0:
            R_geom = (C_chord ** 2) / (8.0 * H_sag) + H_sag / 2.0
        else:
            R_geom = 280.0 if (axis_prior is not None and abs(axis_prior[2]) > 0.8) else 75.0
            
        R_init = np.clip(R_geom, 50.0, 800.0)
        
        if 50.0 <= R_geom <= 800.0 and H_sag > 0.05:
            u_sag = v_sag / H_sag
            c_init = p_mid_arc - R_init * u_sag
        else:
            R_init = np.clip(radius_fit, 50.0, 800.0)
            c_init = c_fit
        
        best_opt = None
        best_rmse = float('inf')
        best_sign = 1
        
        for sign in [1, -1]:
            angles_rad = angles_rad_base * sign
            r_dir_init = points[0] - c_init
            r_dir_init -= np.dot(r_dir_init, best_normal) * best_normal
            if np.linalg.norm(r_dir_init) > 1e-6:
                r_dir_init /= np.linalg.norm(r_dir_init)
            
            init_params = np.hstack([c_init, best_normal, r_dir_init, [R_init]])
            lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
            # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
            init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
            
            def total_residuals(params):
                c = params[0:3]
                axis = params[3:6]
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    
                r_init = params[6:9]
                r_init -= np.dot(r_init, axis) * axis
                r_init_norm = np.linalg.norm(r_init)
                if r_init_norm > 1e-6:
                    r_init = r_init / r_init_norm
                R = params[9]
                
                cos_t = np.cos(angles_rad)[:, None]
                sin_t = np.sin(angles_rad)[:, None]
                cross_term = np.cross(axis, r_init)
                dot_term = np.dot(axis, r_init)
                
                pred_pts = c + R * (r_init[None, :] * cos_t + 
                                   cross_term[None, :] * sin_t + 
                                   (axis * dot_term)[None, :] * (1.0 - cos_t))
                return (points - pred_pts).ravel()
                
            try:
                opt_res = least_squares(total_residuals, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            except ValueError as e:
                raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 1 failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
            rmse = np.sqrt(np.mean(opt_res.fun**2))
            if rmse < best_rmse:
                # [FIX] axis_prior가 주어진 경우, 피팅된 축이 선험 방향과 같은 반공간에 있을 때만 채택
                # → 노이즈 과적합으로 인해 잘못된 sign(-1)이 수학적으로 더 낮은 RMSE를 갖는 상황 방지
                if axis_prior is not None:
                    axis_candidate = opt_res.x[3:6]
                    axis_candidate_norm = axis_candidate / (np.linalg.norm(axis_candidate) + 1e-9)
                    if np.dot(axis_candidate_norm, n_nominal) > 0:
                        best_rmse = rmse
                        best_opt = opt_res
                        best_sign = sign
                else:
                    best_rmse = rmse
                    best_opt = opt_res
                    best_sign = sign

        # Fallback: if axis_prior direction check rejected all candidates (edge case),
        # fall back to the lowest-RMSE result to avoid best_opt being None
        if best_opt is None:
            for sign in [1, -1]:
                angles_rad = angles_rad_base * sign
                r_dir_init = points[0] - c_init
                r_dir_init -= np.dot(r_dir_init, best_normal) * best_normal
                if np.linalg.norm(r_dir_init) > 1e-6:
                    r_dir_init /= np.linalg.norm(r_dir_init)
                init_params = np.hstack([c_init, best_normal, r_dir_init, [R_init]])
                lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
                upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
                init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
                try:
                    opt_res = least_squares(total_residuals, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
                    rmse = np.sqrt(np.mean(opt_res.fun**2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_opt = opt_res
                        best_sign = sign
                except Exception:
                    pass

        # Extract optimal
        c_init = best_opt.x[0:3]
        best_normal = best_opt.x[3:6]
        best_normal /= np.linalg.norm(best_normal)
        r_final_dir = best_opt.x[6:9]
        r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
        if np.linalg.norm(r_final_dir) > 1e-6:
            r_final_dir /= np.linalg.norm(r_final_dir)
        R_init = best_opt.x[9]
        
        # Worst-outlier rejection
        inlier_mask = np.ones(len(points), dtype=bool)
        for out_iter in range(3):
            angles_rad = angles_rad_base * best_sign
            pts_in = points[inlier_mask]
            rad_in = angles_rad[inlier_mask]
            
            if len(pts_in) < 6:
                break
                
            init_params = np.hstack([c_init, best_normal, r_final_dir, [R_init]])
            lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
            upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
            # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
            init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
            
            def total_residuals_in(params):
                c = params[0:3]
                axis = params[3:6]
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                r_init = params[6:9]
                r_init -= np.dot(r_init, axis) * axis
                r_init_norm = np.linalg.norm(r_init)
                if r_init_norm > 1e-6:
                    r_init = r_init / r_init_norm
                R = params[9]
                
                cos_t = np.cos(rad_in)[:, None]
                sin_t = np.sin(rad_in)[:, None]
                cross_term = np.cross(axis, r_init)
                dot_term = np.dot(axis, r_init)
                
                pred_pts = c + R * (r_init[None, :] * cos_t + 
                                   cross_term[None, :] * sin_t + 
                                   (axis * dot_term)[None, :] * (1.0 - cos_t))
                return (pts_in - pred_pts).ravel()
                
            try:
                opt_res = least_squares(total_residuals_in, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
            except ValueError as e:
                raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 2 (outlier loop) failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
            c_init = opt_res.x[0:3]
            best_normal = opt_res.x[3:6]
            best_normal /= np.linalg.norm(best_normal)
            r_final_dir = opt_res.x[6:9]
            r_final_dir -= np.dot(r_final_dir, best_normal) * best_normal
            if np.linalg.norm(r_final_dir) > 1e-6:
                r_final_dir /= np.linalg.norm(r_final_dir)
            R_init = opt_res.x[9]
            
            cos_t = np.cos(angles_rad)[:, None]
            sin_t = np.sin(angles_rad)[:, None]
            cross_term = np.cross(best_normal, r_final_dir)
            dot_term = np.dot(best_normal, r_final_dir)
            
            pred_pts = c_init + R_init * (r_final_dir[None, :] * cos_t + 
                                         cross_term[None, :] * sin_t + 
                                         (best_normal * dot_term)[None, :] * (1.0 - cos_t))
            all_errors = np.linalg.norm(points - pred_pts, axis=1)
            
            inlier_indices = np.where(inlier_mask)[0]
            inlier_errors = all_errors[inlier_mask]
            worst_inlier_idx_in_inliers = np.argmax(inlier_errors)
            worst_global_idx = inlier_indices[worst_inlier_idx_in_inliers]
            worst_error = inlier_errors[worst_inlier_idx_in_inliers]
            
            if worst_error > 0.5:
                inlier_mask[worst_global_idx] = False
            else:
                break
                
        # Final Optimization
        pts_in = points[inlier_mask]
        rad_in = angles_rad_base[inlier_mask] * best_sign
        init_params = np.hstack([c_init, best_normal, r_final_dir, [R_init]])
        
        def total_residuals_final(params):
            c = params[0:3]
            axis = params[3:6]
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
            r_init = params[6:9]
            r_init -= np.dot(r_init, axis) * axis
            r_init_norm = np.linalg.norm(r_init)
            if r_init_norm > 1e-6:
                r_init = r_init / r_init_norm
            R = params[9]
            
            cos_t = np.cos(rad_in)[:, None]
            sin_t = np.sin(rad_in)[:, None]
            cross_term = np.cross(axis, r_init)
            dot_term = np.dot(axis, r_init)
            
            pred_pts = c + R * (r_init[None, :] * cos_t + 
                               cross_term[None, :] * sin_t + 
                               (axis * dot_term)[None, :] * (1.0 - cos_t))
            return (pts_in - pred_pts).ravel()
            
        lower_bounds = np.hstack([c_init - 200.0, [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [50.0]])
        upper_bounds = np.hstack([c_init + 200.0, [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [800.0]])
        # Ensure init_params strictly respects bound constraints to prevent SciPy's x0 bound violation error
        init_params = np.clip(init_params, lower_bounds + 1e-5, upper_bounds - 1e-5)
        
        try:
            opt_res = least_squares(total_residuals_final, init_params, bounds=(lower_bounds, upper_bounds), loss='huber', diff_step=1e-4)
        except ValueError as e:
            raise ValueError(f"fit_circle_3d_and_6dof_misalignment: least_squares stage 3 (final) failed: {e}\n  init_params: {init_params}\n  lower_bounds: {lower_bounds}\n  upper_bounds: {upper_bounds}")
        c_opt = opt_res.x[0:3]
        axis_opt = opt_res.x[3:6]
        axis_opt /= np.linalg.norm(axis_opt)
        
        r_init_opt = opt_res.x[6:9]
        r_init_opt -= np.dot(r_init_opt, axis_opt) * axis_opt
        r_init_opt /= np.linalg.norm(r_init_opt)
        radius_opt = opt_res.x[9]
        
        rmse = np.sqrt(np.mean(opt_res.fun**2))
        
        # Coordinate frames for plotting
        if axis_opt[0] < 0.9:
            ex = np.cross(axis_opt, [1, 0, 0])
        else:
            ex = np.cross(axis_opt, [0, 1, 0])
        ex /= np.linalg.norm(ex)
        ey = np.cross(axis_opt, ex)
        
        pts_centered = points - c_opt
        pts_2d = np.dot(pts_centered, np.vstack((ex, ey)).T)
        uc_opt = 0.0
        vc_opt = 0.0
        
        # Calculate 6-DOF misalignment
        tilt_angle = np.rad2deg(np.arccos(np.clip(np.dot(axis_opt, n_nominal), -1.0, 1.0)))
        
        # Projection for yaw
        n_proj = axis_opt - np.dot(axis_opt, n_nominal) * n_nominal
        if np.linalg.norm(n_proj) > 1e-6:
            n_proj /= np.linalg.norm(n_proj)
            if n_nominal[2] > 0.8:
                yaw_angle = np.rad2deg(np.arctan2(n_proj[1], n_proj[0]))
            else:
                yaw_angle = 0.0
        else:
            yaw_angle = 0.0
            
        # Calculate individual tilt angles for each pose to compute jitter/stddev
        tilt_list = []
        for T in relative_poses:
            if axis_prior is not None:
                if abs(axis_prior[0]) > 0.8:
                    axis_i = T[:3, 0]
                elif abs(axis_prior[1]) > 0.8:
                    axis_i = T[:3, 1]
                else:
                    axis_i = T[:3, 2]
            else:
                axis_i = T[:3, 2]
            axis_norm = np.linalg.norm(axis_i)
            if axis_norm > 1e-6:
                axis_i /= axis_norm
            tilt_i = np.rad2deg(np.arccos(np.clip(np.dot(axis_i, n_nominal), -1.0, 1.0)))
            tilt_list.append(tilt_i)
            
        res_dict = {
            'c_opt': c_opt,
            'axis_opt': axis_opt,
            'radius': radius_opt,
            'rmse': rmse,
            'tilt': tilt_angle,
            'yaw': yaw_angle,
            'pts_2d': pts_2d,
            'uc_opt': uc_opt,
            'vc_opt': vc_opt,
            'inlier_mask': inlier_mask,
            'tilt_list': tilt_list
        }
        return res_dict



    def perform_move_to_ready_pose(self, arm_side, mode="marker", log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to {mode} Ready Pose...")
        torso = [0, 0, 0, 0, 0, 0]
        
        # 1. First move the inactive arm to zero pose to avoid collision
        if log_callback: log_callback("[INFO] Moving inactive arm to zero pose first...")
        if arm_side == "right":
            success_other = self.movej(self.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
        else:
            success_other = self.movej(self.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
            
        if not success_other:
            if log_callback: log_callback("[ERROR] Failed to move inactive arm to zero pose.")
            return False
            
        # 2. Move active arm and head/torso to ready pose
        if log_callback: log_callback("[INFO] Moving active arm, torso, and head to ready pose...")
        
        version_key = "v1.3" if self.is_v13() else "v1.2"
        
        if mode == "marker":
            type_key = "marker"
            ready_mode = None
        else:
            type_key = "joint"
            if mode == "elbow":
                ready_mode = "elbow"
            elif mode == "wrist_yaw2":
                ready_mode = "wrist_yaw2"
            elif mode == "wrist_roll_v13":
                ready_mode = "wrist_roll_v13"
            else:
                ready_mode = "wrist_pitch"
            
        if arm_side == "right":
            right_arm = self.get_ready_pose(version_key, type_key, ready_mode, "right")
            left_arm = None
        else:
            right_arm = None
            left_arm = self.get_ready_pose(version_key, type_key, ready_mode, "left")

        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, head=[0, 0], minimum_time=5.0)
        if success and log_callback:
            log_callback("[INFO] Ready Pose Reached.")
        return success

    def save_calibration_comparison_plot(self, arm_side, mode, first_res, final_res, log_callback=None):
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            def plot_column(res, col_idx, stage_name):
                pts_a = res['pts_a_cam']
                pts_b = res['pts_b_cam']
                c_A = res['c_A']
                c_B = res['c_B']
                n_A = res['n_A']
                n_B = res['n_B']
                r_A = res['r_A']
                r_B = res['r_B']
                angle_error = res['angle_between_normals']
                center_dist = res['center_dist']

                # Compute local frames algebraically from normals (Z axes)
                def get_local_vectors(n):
                    n = n / np.linalg.norm(n)
                    if abs(n[0]) < 0.9:
                        u = np.cross(n, [1, 0, 0])
                    else:
                        u = np.cross(n, [0, 1, 0])
                    u = u / np.linalg.norm(u)
                    v = np.cross(n, u)
                    v = v / np.linalg.norm(v)
                    return u, v

                u_A, v_A = get_local_vectors(n_A)
                u_B, v_B = get_local_vectors(n_B)

                theta = np.linspace(0, 2 * np.pi, 200)
                circle_pts_a = c_A + r_A * (np.cos(theta)[:, None] * u_A + np.sin(theta)[:, None] * v_A)
                circle_pts_b = c_B + r_B * (np.cos(theta)[:, None] * u_B + np.sin(theta)[:, None] * v_B)

                # --- 1. TOP VIEW (Row 0, Col col_idx): X-Y Projection ---
                ax_top = axes[0, col_idx]
                ax_top.scatter(pts_a[:, 0], pts_a[:, 1], c='red', s=15, alpha=0.5, label='Sweep A Raw')
                ax_top.scatter(pts_b[:, 0], pts_b[:, 1], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
                ax_top.plot(circle_pts_a[:, 0], circle_pts_a[:, 1], 'r-', linewidth=1.5, label='Sweep A Fit')
                ax_top.plot(circle_pts_b[:, 0], circle_pts_b[:, 1], 'b-', linewidth=1.5, label='Sweep B Fit')
                ax_top.scatter([c_A[0]], [c_A[1]], c='darkred', marker='X', s=100, label='Center A')
                ax_top.scatter([c_B[0]], [c_B[1]], c='darkblue', marker='X', s=100, label='Center B')
                ax_top.plot([c_A[0], c_B[0]], [c_A[1], c_B[1]], color='purple', linestyle=':', linewidth=2, label='Center Shift')
                
                # Normal vector arrows on X-Y
                scale = min(r_A, r_B) * 0.4
                ax_top.arrow(c_A[0], c_A[1], n_A[0]*scale, n_A[1]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
                ax_top.arrow(c_B[0], c_B[1], n_B[0]*scale, n_B[1]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')

                ax_top.set_xlabel('X (mm)')
                ax_top.set_ylabel('Y (mm)')
                ax_top.set_title(f'[{stage_name}] Top View (X-Y Projection)', fontsize=15, fontweight='bold')
                ax_top.set_aspect('equal')
                ax_top.grid(True)
                if col_idx == 0:
                    ax_top.legend(loc='upper right', fontsize=10)

                # --- 2. SIDE VIEW (Row 1, Col col_idx): Y-Z Projection ---
                ax_side = axes[1, col_idx]
                ax_side.scatter(pts_a[:, 1], pts_a[:, 2], c='red', s=15, alpha=0.5, label='Sweep A Raw')
                ax_side.scatter(pts_b[:, 1], pts_b[:, 2], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
                ax_side.plot(circle_pts_a[:, 1], circle_pts_a[:, 2], 'r-', linewidth=1.5, label='Sweep A Fit')
                ax_side.plot(circle_pts_b[:, 1], circle_pts_b[:, 2], 'b-', linewidth=1.5, label='Sweep B Fit')
                ax_side.scatter([c_A[1]], [c_A[2]], c='darkred', marker='X', s=100, label='Center A')
                ax_side.scatter([c_B[1]], [c_B[2]], c='darkblue', marker='X', s=100, label='Center B')
                ax_side.plot([c_A[1], c_B[1]], [c_A[2], c_B[2]], color='purple', linestyle=':', linewidth=2, label='Center Shift')

                # Normal vector arrows on Y-Z
                ax_side.arrow(c_A[1], c_A[2], n_A[1]*scale, n_A[2]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
                ax_side.arrow(c_B[1], c_B[2], n_B[1]*scale, n_B[2]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')

                ax_side.set_xlabel('Y (mm)')
                ax_side.set_ylabel('Z (mm)')
                ax_side.set_title(f'[{stage_name}] Side View (Y-Z Projection)\nAngle Dev: {angle_error:.3f}° | Center Dist: {center_dist:.2f}mm', fontsize=15, fontweight='bold')
                ax_side.set_aspect('equal')
                ax_side.grid(True)

            def compute_shortest_distance_between_lines(cA, nA, cB, nB):
                nA_norm = nA / np.linalg.norm(nA)
                nB_norm = nB / np.linalg.norm(nB)
                cross = np.cross(nA_norm, nB_norm)
                cross_norm = np.linalg.norm(cross)
                diff = cB - cA
                if cross_norm > 1e-4:
                    return abs(np.dot(diff, cross)) / cross_norm
                else:
                    return np.linalg.norm(diff - np.dot(diff, nA_norm) * nA_norm)

            nominal_dist_35 = None
            if mode == "wrist_pitch_v13" and self.robot:
                try:
                    dyn_model = self.robot.get_dynamics()
                    names = self.robot.model().robot_joint_names
                    state_3_5 = dyn_model.make_state(
                        [f"link_{arm_side}_arm_3", f"link_{arm_side}_arm_5"],
                        names
                    )
                    state_3_5.set_q(np.zeros(len(names)))
                    dyn_model.compute_forward_kinematics(state_3_5)
                    T_3_5 = dyn_model.compute_transformation(state_3_5, 0, 1)
                    nominal_dist_35 = np.linalg.norm(T_3_5[:3, 3]) * 1000.0
                except Exception as e:
                    pass

            plot_column(first_res, 0, "BEFORE")
            plot_column(final_res, 1, "AFTER")

            before_dist_str = ""
            after_dist_str = ""
            if mode == "wrist_pitch_v13":
                dist_before = compute_shortest_distance_between_lines(
                    first_res['c_A'], first_res['n_A'], first_res['c_B'], first_res['n_B']
                )
                dist_after = compute_shortest_distance_between_lines(
                    final_res['c_A'], final_res['n_A'], final_res['c_B'], final_res['n_B']
                )
                before_dist_str = f" | Axis 3-5 Dist = {dist_before:.2f} mm"
                after_dist_str = f" | Axis 3-5 Dist = {dist_after:.2f} mm"
                if nominal_dist_35 is not None:
                    after_dist_str += f" (Nom: {nominal_dist_35:.2f} mm)"

            fig.suptitle(
                f"Joint Calibration: {arm_side.upper()} Arm - {mode.upper()}\n"
                f"Before: Angle Dev = {first_res['angle_between_normals']:.3f}°, Center Dist = {first_res['center_dist']:.2f} mm{before_dist_str}\n"
                f"After : Angle Dev = {final_res['angle_between_normals']:.3f}°, Center Dist = {final_res['center_dist']:.2f} mm{after_dist_str}",
                fontsize=16, fontweight='bold'
            )
            plt.tight_layout()

            from core.paths import CONFIG_PATHS
            result_dir = CONFIG_PATHS["plot_dir"]
            os.makedirs(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"circle_fit_{arm_side}_{mode}_joint_calib.png"))
            plt.savefig(plot_save_path, dpi=150)
            plt.close()

            if log_callback:
                log_callback(f"[SUCCESS] Saved combined calibration comparison plot to: {plot_save_path}")
            return plot_save_path
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to save combined calibration comparison plot: {e}")
            import traceback
            if log_callback:
                log_callback(traceback.format_exc())
            return None

    def compute_calibration_results(self, arm_side, mode, dataset_A, dataset_B, initial_joint_pos, current_offset_deg=0.0, use_angle_based_fitting=None, save_debug=False, log_callback=None, cand_joint=None, sweep_joint_A=None, sweep_joint_B=None):
        raise NotImplementedError("compute_calibration_results must be implemented in subclasses.")

    def perform_single_joint_sweep(self, arm_side, sweep_joint, q_center, start_deg, end_deg, sweep_duration, q_head=None, label="Joint Sweep", log_callback=None, **kwargs):
        if getattr(self, 'stop_requested', False):
            return None

        class MoveThread(threading.Thread):
            def __init__(self, calibrator, robot, torso, right_arm, left_arm, head, minimum_time):
                super().__init__()
                self.calibrator = calibrator
                self.robot = robot
                self.torso = torso
                self.right_arm = right_arm
                self.left_arm = left_arm
                self.head = head
                self.minimum_time = minimum_time
                self.success = False

            def run(self):
                self.success = self.calibrator.movej(
                    self.robot, torso=self.torso, 
                    right_arm=self.right_arm, left_arm=self.left_arm, 
                    head=self.head, minimum_time=self.minimum_time,
                    apply_offsets=False
                )

        is_camera_mock = self.is_mock
        model = self.robot.model() if self.robot else None
        arm_idx = (model.left_arm_idx if arm_side == "left" else model.right_arm_idx) if model else list(range(7))

        # Retrieve joint limits with safety clamping
        dyn_model = self.robot.get_dynamics() if self.robot else None
        q_min = -np.inf
        q_max = np.inf
        if dyn_model and model:
            try:
                state_lim = dyn_model.make_state([f"ee_{arm_side}"], model.robot_joint_names)
                q_lower_all = np.array(dyn_model.get_limit_q_lower(state_lim))
                q_upper_all = np.array(dyn_model.get_limit_q_upper(state_lim))
                global_joint_idx = arm_idx[sweep_joint]
                q_min = q_lower_all[global_joint_idx]
                q_max = q_upper_all[global_joint_idx]
            except Exception as e:
                if log_callback: log_callback(f"[WARN] Failed to retrieve joint limits in base: {e}")

        safety_margin = np.radians(0.5)
        q_start_val = q_center[sweep_joint] + np.radians(start_deg)
        q_end_val = q_center[sweep_joint] + np.radians(end_deg)

        if q_start_val < q_min + safety_margin:
            q_start_val_new = q_min + safety_margin
            if log_callback:
                log_callback(f"[WARN] Start angle ({np.degrees(q_start_val):.2f}°) exceeds min limit ({np.degrees(q_min):.2f}°). Clamping to {np.degrees(q_start_val_new):.2f}°.")
            q_start_val = q_start_val_new

        if q_end_val > q_max - safety_margin:
            q_end_val_new = q_max - safety_margin
            if log_callback:
                log_callback(f"[WARN] End angle ({np.degrees(q_end_val):.2f}°) exceeds max limit ({np.degrees(q_max):.2f}°). Clamping to {np.degrees(q_end_val_new):.2f}°.")
            q_end_val = q_end_val_new

        q_start = list(q_center)
        q_start[sweep_joint] = q_start_val
        q_end = list(q_center)
        q_end[sweep_joint] = q_end_val

        # 1. Move to start position
        logging.info(f"[INFO] Moving {label} to start sweep position...")
        if arm_side == "left":
            ok = self.movej(self.robot, left_arm=q_start, head=q_head, minimum_time=2.5, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=q_start, head=q_head, minimum_time=2.5, apply_offsets=False)

        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback(f"[ERROR] Failed to move {label} to start pose or stop requested.")
            return None

        if self.robot:
            time.sleep(1.0)
        else:
            time.sleep(0.01)

        # 2. Continuous sweep from start to end position
        logging.info(f"[INFO] Commencing continuous sweep on {label} (duration={sweep_duration}s)...")
        if getattr(self, 'stop_requested', False):
            return None

        move_thread = MoveThread(
            self, self.robot, torso=None,
            right_arm=q_end if arm_side == "right" else None,
            left_arm=q_end if arm_side == "left" else None,
            head=q_head, minimum_time=sweep_duration
        )

        dataset = []
        move_thread.start()

        # Capture poses and joint positions at high frequency
        while move_thread.is_alive():
            if getattr(self, 'stop_requested', False):
                if self.robot:
                    self.robot.cancel_control()
                move_thread.join()
                return None

            if self.robot:
                q_full_captured = np.array(self.robot.get_state().position)
            else:
                q_full_captured = np.zeros(20)

            if is_camera_mock:
                if hasattr(self, 'get_simulated_marker_pose'):
                    try:
                        pose = self.get_simulated_marker_pose(
                            arm_side, sweep_joint=sweep_joint, 
                            current_offset_deg=kwargs.get('current_offset_deg', 0.0), 
                            cand_joint=kwargs.get('cand_joint', None),
                            q_actual=q_full_captured
                        )
                    except TypeError:
                        pose = self.get_simulated_marker_pose(arm_side, sweep_joint=sweep_joint, q_actual=q_full_captured)
                else:
                    pose = np.eye(4)
                res = [pose.tolist()] if pose is not None else None
            else:
                res = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False)

            if res:
                pose_flat = res[0] if isinstance(res, list) else list(res.values())[0]
                pose_mat = np.array(pose_flat).reshape(4, 4)

                if np.linalg.norm(pose_mat[:3, 3]) > 0.01:
                    # Deduplicate: Only append if the pose is actually new (camera updated)
                    if len(dataset) == 0 or not np.allclose(dataset[-1][1], pose_mat, atol=1e-5):
                        dataset.append((q_full_captured, pose_mat))

            if is_camera_mock:
                time.sleep(0.002)
            else:
                time.sleep(0.01)

        move_thread.join()
        if not move_thread.success:
            if log_callback: log_callback(f"[ERROR] {label} sweep motion failed or was cancelled.")
            return None

        if len(dataset) < 10:
            if log_callback: log_callback(f"[ERROR] Too few valid captured points for {label}. Calibration failed.")
            return None

        logging.info(f"    -> Swept {len(dataset)} dense raw coordinate frames during {label} motion.")
        return dataset

