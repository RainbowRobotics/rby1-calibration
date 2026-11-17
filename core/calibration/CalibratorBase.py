from core.storage import ConfigStorage
from core.storage import FileStorage, ArtifactStorage
from core.robot.robot_core import RobotOperations
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

class BaseCalibrator(RobotOperations):
    # v1.3 (2026-10-14/15): wrist_pitch_v13 +-15/+-15 -> +-30/+-30 (J5 scatter 0.43 -> ~0.10 deg per pair);
    # wrist_roll_v13 +-20/+-15 -> +-30/+-25 once its axes came from the circle fits (0.12 -> 0.04 deg).
    JOINT_CONFIGS = {
        "wrist_roll_v13":  {"cand_joint": 6, "sweep_joint_A": 6, "sweep_joint_B": 5, "offset_key": "wrist_roll",  "offset_range": (-30.0, 30.0), "sweep_range_A": 30.0, "sweep_range_B": 25.0},
        "wrist_pitch_v13": {"cand_joint": 5, "sweep_joint_A": 6, "sweep_joint_B": 4, "offset_key": "wrist_pitch", "offset_range": (-30.0, 30.0), "sweep_range_A": 30.0, "sweep_range_B": 30.0},
        "wrist_yaw2":      {"cand_joint": 6, "sweep_joint_A": 6, "sweep_joint_B": 5, "offset_key": "wrist_yaw2",  "offset_range": (-30.0, 30.0), "sweep_range_A": 20.0, "sweep_range_B": 15.0},
        "wrist_pitch":     {"cand_joint": 5, "sweep_joint_A": 4, "sweep_joint_B": 6, "offset_key": "wrist_pitch", "offset_range": (-30.0, 30.0), "sweep_range_A": 15.0, "sweep_range_B": 15.0},
        "elbow":           {"cand_joint": 3, "sweep_joint_A": 2, "sweep_joint_B": 4, "offset_key": "elbow",       "offset_range": (-5.0, 0.0),   "sweep_range_A": 20.0, "sweep_range_B": 20.0},
    }
    MARKER_CONFIGS = {
        "axis_4": {"joint_i": 4, "start_deg": -20.0, "end_deg": 20.0, "n_nom_v12": [0.0, 0.0, 1.0], "n_nom_v13": [0.0, 0.0, 1.0]},
        "axis_5": {"joint_i": 5, "start_deg": 0.0, "end_deg": -40.0, "n_nom_v12": [0.0, 1.0, 0.0], "n_nom_v13": [0.0, 1.0, 0.0]},
        # v1.2 J6 sweep widened to +-30 deg (2026-09-22). The 45 deg arc of the 54 mm circle left the
        # J6 circle centre/axis poorly defined (41 mm chord). The J6 axis is ~85 deg from the camera
        # axis, so the plate tilts as it sweeps: up to 46 deg off the view ray at +-30 vs 63 at +-45.
        # The duration keeps the angular speed of the +-22.5 deg / 10 s sweep. v1.3 is unchanged.
        "axis_6": {"joint_i": 6, "start_deg": -22.5, "end_deg": 22.5, "n_nom_v12": [0.0, 0.0, 1.0], "n_nom_v13": [1.0, 0.0, 0.0],
                   "start_deg_v12": -30.0, "end_deg_v12": 30.0, "sweep_duration_s_v12": 13.5},
    }
    NOMINAL_BRACKET_TEMPLATES = {
        "1.3": {
            "left":  [0.067, 0.0, 0.0, 90.0, 0.0, -90.0],
            "right": [0.067, 0.0, 0.0, 90.0, 0.0, -90.0]
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
        self.markers_config = {}
        self.load_camera_config()
        
        self.ready_poses = {}
        self.load_ready_poses()
        
        # Active joint home offsets to apply to commanded trajectories
        self.joint_offsets = {
            "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
            "left":  {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
        }
        self.partial_data = {}
        self.include_head_motion = True
        self.user_taught_ready_poses = {}
        self.stop_requested = False

    def clear_user_taught_ready_poses(self, arm_side=None, mode=None):
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            if arm_side and mode:
                norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].pop(norm_mode, None)
            elif arm_side:
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].clear()
            else:
                self.user_taught_ready_poses.clear()

    def load_ready_poses(self):
        from core.storage import CONFIG_PATHS
        yaml_path = CONFIG_PATHS["ready_poses_yaml"]
        if os.path.exists(yaml_path):
            try:
                self.ready_poses = ConfigStorage.load(yaml_path)
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

    def is_v13(self) -> bool:
        """Returns True only for model-m v1.3 robots."""
        return self.get_robot_version() == "1.3"

    def is_head_active(self) -> bool:
        if getattr(self, 'include_head_motion', None) is False:
            return False
        if getattr(self, 'head_enabled', None) is False:
            return False
        model = getattr(self.robot, 'model', lambda: None)() if hasattr(self, 'robot') and self.robot else None
        if model is not None:
            return hasattr(model, 'head_idx') and len(getattr(model, 'head_idx', [])) >= 2
        return getattr(self, 'include_head_motion', True)

    def get_ready_pose(self, version_key, type_key, mode_key, arm_side):
        if not self.ready_poses:
            raise RuntimeError("Ready poses are not loaded or the configuration file is empty.")
        
        try:
            ver_clean = str(version_key).replace("v", "")
            val = self.ready_poses.get(version_key) or self.ready_poses.get(f"v{ver_clean}") or self.ready_poses.get(ver_clean)
            if val is None:
                raise KeyError(f"Version key {version_key} not in ready_poses (available: {list(self.ready_poses.keys())})")
            if type_key == "joint":
                lookup_key = mode_key
                if lookup_key == "wrist_pitch_v13":
                    lookup_key = "wrist_pitch"
                elif lookup_key == "wrist_roll_v13":
                    lookup_key = "wrist_roll"
                val = val["joint"][lookup_key][f"{arm_side}_arm"]
            elif type_key == "check_calib":
                val = val["check_calib"][f"{arm_side}_arm"]
            else:
                val = val["marker"][f"{arm_side}_arm"]
            val_arr = np.array(val, dtype=np.float64).copy()
            # If Head is disabled or robot has no 2-DOF head (fixed chest camera), lower Shoulder Pitch (Joint 0)
            # and adjust Elbow (Joint 3) in negative direction to keep marker perpendicular to camera FOV without tilting backward.
            if not self.is_head_active() and len(val_arr) >= 7:
                j0_delta = 19.0
                elbow_delta = -4.0 if (type_key == "marker" or mode_key in ["wrist_pitch", "wrist_yaw2", "wrist_roll", "wrist_pitch_v13", "wrist_roll_v13"]) else 0.0
                val_arr[0] += j0_delta  # Joint 0 positive pitch lowers the arm down into fixed FOV (-55 -> -36 deg)
                val_arr[3] += elbow_delta  # Joint 3 negative offset flexes elbow to prevent marker from tilting backwards
                msg = f"[READY POSE] Head disabled (Fixed Chest Camera): Joint 0 lowered by +{j0_delta:.1f}° ({val_arr[0]:.1f}°), Elbow adjusted by {elbow_delta:+.1f}° ({val_arr[3]:.1f}°) for {arm_side}_arm ({type_key}/{mode_key})"
                print(msg)
                logging.info(msg)

            return np.deg2rad(val_arr)
        except (KeyError, TypeError) as e:
            raise KeyError(
                f"[ERROR] Failed to get ready pose for version='{version_key}', type='{type_key}', mode='{mode_key}', arm='{arm_side}_arm'. "
                f"Please check your ready_poses.yaml file. Details: {e}"
            )


    def load_camera_config(self):
        # Locate setting.yaml
        from core.storage import CONFIG_PATHS
        yaml_path = CONFIG_PATHS["setting_yaml"]
        if not os.path.exists(yaml_path):
            logging.error(f"[CRITICAL ERROR] setting.yaml not found at {yaml_path}!")
            raise FileNotFoundError(f"[CRITICAL ERROR] setting.yaml not found at {yaml_path}!")

        try:
            config_data = ConfigStorage.load(yaml_path)
            self.camera_config = config_data.get("camera", {})
            self.markers_config = config_data.get("marker", {}) or {}
            
            # Legacy compatibility: if marker keys are under camera section, copy to markers_config
            legacy_keys = [
                "Tf_to_marker_left", "Tf_to_marker_right",
                "Tf_to_marker_left_v12", "Tf_to_marker_right_v12",
                "Tf_to_marker_left_v13", "Tf_to_marker_right_v13"
            ]
            for k in legacy_keys:
                if k not in self.markers_config and k in self.camera_config:
                    self.markers_config[k] = self.camera_config[k]

            # Strict validation for marker configurations in setting.yaml
            required_keys = ["Tf_to_marker_left_v13", "Tf_to_marker_right_v13", "Tf_to_marker_left_v12", "Tf_to_marker_right_v12"]
            missing_keys = [k for k in required_keys if k not in self.markers_config]
            if missing_keys:
                raise KeyError(f"[CRITICAL ERROR] Missing required marker configuration keys in setting.yaml: {missing_keys}")

            # For compatibility, merge Tf_to_marker keys from markers_config to camera_config
            for k, v in self.markers_config.items():
                if k.startswith("Tf_to_marker_"):
                    self.camera_config[k] = v
            
            # Dynamically synchronize NOMINAL_BRACKET_TEMPLATES with values from setting.yaml
            self.NOMINAL_BRACKET_TEMPLATES["1.3"]["left"] = list(self.markers_config["Tf_to_marker_left_v13"])
            self.NOMINAL_BRACKET_TEMPLATES["1.3"]["right"] = list(self.markers_config["Tf_to_marker_right_v13"])
            self.NOMINAL_BRACKET_TEMPLATES["1.2"]["left"] = list(self.markers_config["Tf_to_marker_left_v12"])
            self.NOMINAL_BRACKET_TEMPLATES["1.2"]["right"] = list(self.markers_config["Tf_to_marker_right_v12"])
            
            logging.info(f"Loaded config from setting.yaml successfully.")
        except Exception as e:
            logging.error(f"[CRITICAL ERROR] Failed to load setting.yaml: {e}")
            raise RuntimeError(f"[CRITICAL ERROR] Failed to load setting.yaml: {e}")

    def camera_exposure_record(self):
        """Exposure the camera is set to right now, for the debug logs and the Step 2 result.
        None without a real camera (simulation, tests): no value is made up."""
        source = self.marker_st
        if source is None or getattr(source, "sim", False) or getattr(source, "camera", None) is None:
            return None
        auto, setpoint = source.get_camera_exposure()
        return {"auto": bool(auto), "setpoint": float(setpoint),
                "setpoint_unit_us": float(source.get_camera_exposure_unit_us()),
                "last_frame_us": float(source.get_actual_exposure())}

    def save_debug_points(self, arm_side, axis_num, dataset, initial_joint_pos, ee_name, dyn_model, T_mount_to_cam, type_key, log_callback=None, note=None, file_tag=None):
        try:
            if T_mount_to_cam is None:
                if self.is_head_active():
                    mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                    T_mount_to_cam = self.make_transform(mount_to_cam)
                else:
                    head_base_to_cam = self.camera_config.get("head_base_to_cam", [0.098, 0.009, 0.012, -90.0, 0.0, -90.0])
                    T_mount_to_cam = self.make_transform(head_base_to_cam)
            from core.storage import CONFIG_PATHS
            result_txt_dir = CONFIG_PATHS["txt_dir"]
            FileStorage.ensure_dir(result_txt_dir, exist_ok=True)
            if not self.robot:
                raise RuntimeError("Robot instance is not initialized")
            arm_idx = self.robot.model().left_arm_idx if arm_side == "left" else self.robot.model().right_arm_idx
            
            filename = os.path.join(result_txt_dir, f"sweep_points_{arm_side}_{file_tag or type_key}_axis_{axis_num}.txt")

            # Determine prefix for header
            if type_key == "joint_A":
                angle_header_name = "Joint_A"
            elif type_key == "joint_B":
                angle_header_name = "Joint_B"
            else:
                angle_header_name = f"Joint_{axis_num}"
            file_exists = os.path.exists(filename)
            with FileStorage.open(filename, "a") as f:
                if file_exists:
                    f.write("\n=== NEW ITERATION ===\n")
                f.write(f"# {angle_header_name}_Angle(deg), Cam_X(mm), Cam_Y(mm), Cam_Z(mm), Torso_X(mm), Torso_Y(mm), Torso_Z(mm), EE_X(mm), EE_Y(mm), EE_Z(mm), "
                        "T_cam2marker_flat(16), T_torso2marker_flat(16), T_ee2marker_flat(16)\n")
                if note:
                    f.write(f"# {note}\n")
                for q_full, pose in dataset:
                    q_val = q_full[arm_idx[axis_num]]
                    s_deg = np.degrees(q_val - initial_joint_pos[axis_num])
                    p_cam = pose[:3, 3]
                    
                    T_cam_to_marker = pose
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, ee_name)
                    if self.is_head_active():
                        T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                        T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    else:
                        try:
                            T_t5_to_head_0 = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_0", "link_torso_5")
                        except Exception:
                            T_t5_to_head_0 = np.eye(4)
                        T_t5_to_cam = T_t5_to_head_0 @ T_mount_to_cam
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

    @staticmethod
    def filter_sweep_dataset(dataset, arm_idx, sweep_joint, mismatch_threshold_deg=3.0, log_callback=None):
        """
        Filters out IPPE planar ambiguity flip spikes from a single joint sweep dataset.
        dataset: list of (q_full, pose_mat)
        """
        if not dataset or len(dataset) < 4:
            return dataset

        angles_deg = [np.degrees(q_full[arm_idx[sweep_joint]]) for q_full, _ in dataset]
        poses = [pose for _, pose in dataset]
        n = len(dataset)

        bad_indices = set()
        for i in range(1, n):
            R_prev = poses[i-1][:3, :3]
            R_curr = poses[i][:3, :3]
            R_rel = R_curr @ R_prev.T
            tr = np.trace(R_rel)
            cos_val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
            dR = np.degrees(np.arccos(cos_val))
            dq = abs(angles_deg[i] - angles_deg[i-1])
            if abs(dR - dq) > mismatch_threshold_deg:
                bad_indices.add(i)

        if not bad_indices:
            return dataset

        confirmed_spikes = set()
        for i in bad_indices:
            prev_idx = i - 1
            while prev_idx in bad_indices and prev_idx > 0:
                prev_idx -= 1
            next_idx = i + 1
            while next_idx in bad_indices and next_idx < n - 1:
                next_idx += 1
                
            if prev_idx >= 0 and next_idx < n:
                R_p = poses[prev_idx][:3, :3]
                R_n = poses[next_idx][:3, :3]
                R_i = poses[i][:3, :3]
                
                dR_skip = np.degrees(np.arccos(np.clip((np.trace(R_n @ R_p.T) - 1.0) / 2.0, -1.0, 1.0)))
                dq_skip = abs(angles_deg[next_idx] - angles_deg[prev_idx])
                
                dR_i_p = np.degrees(np.arccos(np.clip((np.trace(R_i @ R_p.T) - 1.0) / 2.0, -1.0, 1.0)))
                dq_i_p = abs(angles_deg[i] - angles_deg[prev_idx])
                
                if abs(dR_skip - dq_skip) < abs(dR_i_p - dq_i_p):
                    confirmed_spikes.add(i)
            else:
                confirmed_spikes.add(i)

        if confirmed_spikes:
            msg = f"[FILTER] Filtered {len(confirmed_spikes)} IPPE flip spike frame(s) at indices: {sorted(list(confirmed_spikes))}"
            if log_callback:
                log_callback(msg)
            logging.info(msg)

        clean_dataset = [item for idx, item in enumerate(dataset) if idx not in confirmed_spikes]
        return clean_dataset




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
                # [FIX] If axis_prior is given, accept the fitted axis only if it lies in the same half-space as the prior direction.
                # -> Prevents cases where noise overfitting results in a mathematically lower RMSE with the wrong sign (-1).
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



    @staticmethod
    def refine_bracket_axes_constrained(points_by_axis, axes_init, centers_init, radii_init, log_callback=None):
        """Jointly refit the three bracket sweep circles under the physical wrist constraints.

        At the bracket ready pose (J5 at 90 deg) the J4/J5/J6 rotation axes all pass through one
        wrist point, and the J5 axis is perpendicular to both the J4 and the J6 axis (structural,
        independent of the J5 value). The J4-J6 angle is left FREE because that angle carries the
        J5 residual signal. Fitting each circle independently leaves that structure unused, so the
        axes came out 1-8 mm apart and up to 3 deg off orthogonal on the real robot (2026-09-15)
        even though the fit residual stayed at 0.08 mm -- and that slack leaks into the bracket
        pose (and from there into Step 2's J0).

        points_by_axis / axes_init / centers_init / radii_init: sequence ordered (J4, J6, J5),
        points in mm in the camera frame. Returns a dict with refined 'axes', 'centers', 'radii',
        the fit rms before/after and the per-axis direction change, or None if the fit fails.
        """
        from scipy.optimize import least_squares

        pts = [np.asarray(P, dtype=float) for P in points_by_axis]
        if any(len(P) < 10 for P in pts):
            return None
        n_init = [np.asarray(n, dtype=float) / np.linalg.norm(n) for n in axes_init]
        c_init = [np.asarray(c, dtype=float) for c in centers_init]
        r_init = np.asarray(radii_init, dtype=float)

        def basis(n):
            helper_vec = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(n, helper_vec)
            e1 /= np.linalg.norm(e1)
            return e1, np.cross(n, e1)

        def circle_residual(P, c, n, r):
            v = P - c
            axial = v @ n
            radial = np.linalg.norm(v - np.outer(axial, n), axis=1)
            return np.concatenate([radial - r, axial])

        n5 = n_init[2]
        e1, e2 = basis(n5)
        p0 = np.mean(c_init, axis=0)
        x0 = np.concatenate([
            p0, [0.0, 0.0],
            [np.arctan2(n_init[0] @ e2, n_init[0] @ e1), np.arctan2(n_init[1] @ e2, n_init[1] @ e1)],
            [(c_init[i] - p0) @ n_init[i] for i in range(3)],
            r_init,
        ])

        def unpack(x):
            wrist = x[:3]
            n5_ref = n5 + x[3] * e1 + x[4] * e2
            n5_ref /= np.linalg.norm(n5_ref)
            f1, f2 = basis(n5_ref)
            n4_ref = np.cos(x[5]) * f1 + np.sin(x[5]) * f2
            n6_ref = np.cos(x[6]) * f1 + np.sin(x[6]) * f2
            axes = [n4_ref, n6_ref, n5_ref]
            centers = [wrist + x[7 + i] * axes[i] for i in range(3)]
            return axes, centers, x[10:13], wrist

        def residual(x):
            axes, centers, radii, _ = unpack(x)
            return np.concatenate([circle_residual(pts[i], centers[i], axes[i], radii[i]) for i in range(3)])

        rms_before = float(np.sqrt(np.mean(np.concatenate(
            [circle_residual(pts[i], c_init[i], n_init[i], r_init[i]) for i in range(3)]) ** 2)))
        try:
            sol = least_squares(residual, x0, method="lm", xtol=1e-12, ftol=1e-12)
        except Exception as error:
            if log_callback:
                log_callback(f"[WARN] Constrained bracket axis fit failed ({error}); keeping independent circle fits.")
            return None
        if not sol.success:
            if log_callback:
                log_callback("[WARN] Constrained bracket axis fit did not converge; keeping independent circle fits.")
            return None

        axes, centers, radii, wrist = unpack(sol.x)
        axes = [a if np.dot(a, n_init[i]) >= 0 else -a for i, a in enumerate(axes)]
        centers = [centers[i] if np.dot(axes[i], unpack(sol.x)[0][i]) >= 0 else centers[i] for i in range(3)]
        rms_after = float(np.sqrt(np.mean(sol.fun ** 2)))
        change_deg = [float(np.degrees(np.arccos(np.clip(abs(np.dot(axes[i], n_init[i])), -1.0, 1.0)))) for i in range(3)]
        wrist_gap_before = [float(np.linalg.norm((np.eye(3) - np.outer(n_init[i], n_init[i])) @ (wrist - c_init[i]))) for i in range(3)]
        result = {
            "axes": axes, "centers": centers, "radii": [float(v) for v in radii], "wrist_point": wrist,
            "rms_before_mm": rms_before, "rms_after_mm": rms_after,
            "axis_change_deg": change_deg, "axis_gap_before_mm": wrist_gap_before,
        }
        if log_callback:
            log_callback(
                f"[INFO] Bracket axes refit with wrist-concurrency + J5 orthogonality: "
                f"fit rms {rms_before:.3f} -> {rms_after:.3f} mm, axis change (J4/J6/J5) "
                f"{change_deg[0]:.2f}/{change_deg[1]:.2f}/{change_deg[2]:.2f} deg, "
                f"axis-to-wrist gap before {wrist_gap_before[0]:.2f}/{wrist_gap_before[1]:.2f}/{wrist_gap_before[2]:.2f} mm")
        return result

    # How a bracket on one arm maps onto the other arm's convention. This is NOT a fixed
    # rotation: v1.2 mounts the two flange frames 180 deg apart about z, while v1.3 mounts
    # both brackets identically, so the map is Rz(180) for v1.2 and the identity for v1.3.
    # Deriving it from NOMINAL_BRACKET_TEMPLATES keeps both versions correct and encodes only
    # the frame convention, not the URDF. Hard-coding Rz(180) reported two identical v1.3
    # brackets as 134 mm apart and collapsed their position to the origin.
    @staticmethod
    def bracket_mirror_transform(version="1.2"):
        key = "1.3" if str(version).replace("v", "").strip() == "1.3" else "1.2"
        nominal = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES[key]
        T = BaseCalibrator.make_transform(nominal["left"]) @ np.linalg.inv(
            BaseCalibrator.make_transform(nominal["right"]))
        if not np.allclose(T @ T, np.eye(4), atol=1e-9):
            raise ValueError(f"Bracket mirror map for v{key} is not its own inverse; "
                             "the nominal templates are not a mirrored pair.")
        return T

    @staticmethod
    def mirror_bracket_vector(vec, version="1.2"):
        """Map a marker bracket [x, y, z, roll, pitch, yaw] to the other arm's convention."""
        T_out = BaseCalibrator.bracket_mirror_transform(version) @ BaseCalibrator.make_transform(list(vec))
        rpy = R_scipy.from_matrix(T_out[:3, :3]).as_euler('ZYX', degrees=True)[::-1]
        return [float(T_out[0, 3]), float(T_out[1, 3]), float(T_out[2, 3]),
                float(rpy[0]), float(rpy[1]), float(rpy[2])]

    @staticmethod
    def bracket_asymmetry(right_vec, left_vec, version="1.2"):
        """How far the two fitted brackets are from being mirror images of each other."""
        T_r = BaseCalibrator.make_transform(list(right_vec))
        T_l_as_r = BaseCalibrator.make_transform(BaseCalibrator.mirror_bracket_vector(left_vec, version))
        d_pos_mm = (T_l_as_r[:3, 3] - T_r[:3, 3]) * 1000.0
        rotvec = R_scipy.from_matrix(T_r[:3, :3].T @ T_l_as_r[:3, :3]).as_rotvec(degrees=True)
        return d_pos_mm, rotvec, float(np.linalg.norm(rotvec))

    @staticmethod
    def symmetrize_bracket_pair(right_vec, left_vec, log_callback=None, version="1.2"):
        """Replace both brackets with the mirror-symmetric average of the two fits.

        The brackets are rigid parts fitted to nominally symmetric flanges, so the only real
        left/right difference should be a small mounting skew. Averaging halves the random part
        of each arm's estimate; it does NOT help if one arm is genuinely mounted differently,
        which is why the measured asymmetry is logged before it is averaged away.
        """
        d_pos_mm, d_rot_vec, d_rot_deg = BaseCalibrator.bracket_asymmetry(right_vec, left_vec, version)
        T_r = BaseCalibrator.make_transform(list(right_vec))
        T_l_as_r = BaseCalibrator.make_transform(BaseCalibrator.mirror_bracket_vector(left_vec, version))
        pos_avg = 0.5 * (T_r[:3, 3] + T_l_as_r[:3, 3])
        rel = R_scipy.from_matrix(T_r[:3, :3].T @ T_l_as_r[:3, :3]).as_rotvec()
        R_avg = T_r[:3, :3] @ R_scipy.from_rotvec(0.5 * rel).as_matrix()
        rpy = R_scipy.from_matrix(R_avg).as_euler('ZYX', degrees=True)[::-1]
        right_out = [float(pos_avg[0]), float(pos_avg[1]), float(pos_avg[2]),
                     float(rpy[0]), float(rpy[1]), float(rpy[2])]
        left_out = BaseCalibrator.mirror_bracket_vector(right_out, version)
        if log_callback:
            log_callback("[INFO] Bracket left/right symmetry enforced.")
            log_callback(f"   measured asymmetry (left mirrored onto right): "
                         f"dX {d_pos_mm[0]:+.2f}, dY {d_pos_mm[1]:+.2f}, dZ {d_pos_mm[2]:+.2f} mm | "
                         f"rotation {d_rot_deg:.3f} deg "
                         f"[{d_rot_vec[0]:+.3f} {d_rot_vec[1]:+.3f} {d_rot_vec[2]:+.3f}]")
            for side, before, after in (("right", right_vec, right_out), ("left", left_vec, left_out)):
                moved = np.linalg.norm((np.array(after[:3]) - np.array(before[:3]))) * 1000.0
                log_callback(f"   {side:5s} before [{before[0]*1000:+7.2f} {before[1]*1000:+7.2f} {before[2]*1000:+7.2f}] mm "
                             f"[{before[3]:+6.2f} {before[4]:+6.2f} {before[5]:+7.2f}] deg")
                log_callback(f"   {side:5s} after  [{after[0]*1000:+7.2f} {after[1]*1000:+7.2f} {after[2]*1000:+7.2f}] mm "
                             f"[{after[3]:+6.2f} {after[4]:+6.2f} {after[5]:+7.2f}] deg  (moved {moved:.2f} mm)")
        return right_out, left_out, {"d_pos_mm": [float(v) for v in d_pos_mm],
                                     "d_rot_deg": d_rot_deg}

    def perform_move_to_ready_pose(self, arm_side, mode="marker", log_callback=None):
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        self.current_calib_mode = mode
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
            elif mode in ("wrist_roll", "wrist_roll_v13"):
                ready_mode = "wrist_roll"
            else:
                ready_mode = "wrist_pitch"
            
        norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
        taught_pose = None
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            arm_dict = self.user_taught_ready_poses.get(arm_side)
            if isinstance(arm_dict, dict):
                taught_pose = arm_dict.get(norm_mode)

        apply_offsets_flag = True
        if taught_pose is not None:
            if log_callback:
                log_callback(f"[INFO] Preserved user-taught ready pose detected for {arm_side} arm ({norm_mode}). Using taught posture.")
            if arm_side == "right":
                right_arm = list(taught_pose)
                left_arm = None
            else:
                right_arm = None
                left_arm = list(taught_pose)
            apply_offsets_flag = False
        else:
            if arm_side == "right":
                right_arm = self.get_ready_pose(version_key, type_key, ready_mode, "right")
                left_arm = None
            else:
                right_arm = None
                left_arm = self.get_ready_pose(version_key, type_key, ready_mode, "left")

        success = self.movej(self.robot, torso=torso, right_arm=right_arm, left_arm=left_arm, head=None, minimum_time=5.0, apply_offsets=apply_offsets_flag)
        if success and log_callback:
            log_callback("[INFO] Ready Pose Reached.")
        return success

    def save_calibration_comparison_plot(self, arm_side, mode, first_res, final_res, log_callback=None):
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            def extract_plot_dict(res_obj, default_stage="first"):
                if not res_obj or not isinstance(res_obj, dict):
                    return {}
                if '_plot_data' in res_obj:
                    return res_obj
                if default_stage == "first":
                    if 'first_res' in res_obj and isinstance(res_obj['first_res'], dict):
                        return extract_plot_dict(res_obj['first_res'], "first")
                    if 'final_res' in res_obj and isinstance(res_obj['final_res'], dict):
                        return extract_plot_dict(res_obj['final_res'], "final")
                else:
                    if 'final_res' in res_obj and isinstance(res_obj['final_res'], dict):
                        return extract_plot_dict(res_obj['final_res'], "final")
                    if 'first_res' in res_obj and isinstance(res_obj['first_res'], dict):
                        return extract_plot_dict(res_obj['first_res'], "first")
                return res_obj

            first_res_actual = extract_plot_dict(first_res, "first")
            final_res_actual = extract_plot_dict(final_res, "final")

            def plot_column(res, col_idx, stage_name):
                plot_data = res.get('_plot_data', res)
                pts_a = plot_data.get('pts_a_cam')
                pts_b = plot_data.get('pts_b_cam')
                c_A = plot_data.get('c_A')
                c_B = plot_data.get('c_B')
                n_A = plot_data.get('n_A')
                n_B = plot_data.get('n_B')
                r_A = plot_data.get('r_A', res.get('r_A', 1.0))
                r_B = plot_data.get('r_B', res.get('r_B', 1.0))
                angle_error = plot_data.get('angle_between_normals', res.get('angle_between_normals', 0.0))
                center_dist = plot_data.get('center_dist', res.get('center_dist', 0.0))

                if pts_a is None or c_A is None or n_A is None:
                    for row in range(2):
                        axes[row, col_idx].set_title(f'[{stage_name}] No plot data')
                        axes[row, col_idx].axis('off')
                    return

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
                except Exception:
                    pass

            plot_column(first_res_actual, 0, "BEFORE")
            plot_column(final_res_actual, 1, "AFTER")

            before_dist_str = ""
            after_dist_str = ""
            if mode == "wrist_pitch_v13":
                first_pd = first_res_actual.get('_plot_data', first_res_actual)
                final_pd = final_res_actual.get('_plot_data', final_res_actual)
                if all(k in first_pd for k in ('c_A', 'n_A', 'c_B', 'n_B')):
                    dist_before = compute_shortest_distance_between_lines(
                        first_pd['c_A'], first_pd['n_A'], first_pd['c_B'], first_pd['n_B']
                    )
                    before_dist_str = f" | Axis 3-5 Dist = {dist_before:.2f} mm"
                if all(k in final_pd for k in ('c_A', 'n_A', 'c_B', 'n_B')):
                    dist_after = compute_shortest_distance_between_lines(
                        final_pd['c_A'], final_pd['n_A'], final_pd['c_B'], final_pd['n_B']
                    )
                    after_dist_str = f" | Axis 3-5 Dist = {dist_after:.2f} mm"
                    if nominal_dist_35 is not None:
                        after_dist_str += f" (Nom: {nominal_dist_35:.2f} mm)"

            first_pd = first_res_actual.get('_plot_data', first_res_actual)
            final_pd = final_res_actual.get('_plot_data', final_res_actual)
            fig.suptitle(
                f"Joint Calibration: {arm_side.upper()} Arm - {mode.upper()}\n"
                f"Before: Angle Dev = {first_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {first_pd.get('center_dist', 0.0):.2f} mm{before_dist_str}\n"
                f"After : Angle Dev = {final_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {final_pd.get('center_dist', 0.0):.2f} mm{after_dist_str}",
                fontsize=16, fontweight='bold'
            )
            plt.tight_layout()

            from core.storage import CONFIG_PATHS
            result_dir = CONFIG_PATHS["plot_dir"]
            FileStorage.ensure_dir(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"circle_fit_{arm_side}_{mode}_joint_calib.png"))
            ArtifactStorage.save_figure(plot_save_path, dpi=150)
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

        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot is not connected.")
            return None

        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx

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
            ok = self.movej(self.robot, left_arm=q_start, head=q_head, minimum_time=1.2, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=q_start, head=q_head, minimum_time=1.2, apply_offsets=False)

        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback(f"[ERROR] Failed to move {label} to start pose or stop requested.")
            return None

        if self.robot:
            time.sleep(0.5)

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
        self.partial_data[label] = dataset
        t_start = time.time()
        move_thread.start()

        try:
            # Capture poses and joint positions at high frequency
            while move_thread.is_alive():
                if getattr(self, 'stop_requested', False):
                    self.robot.cancel_control()
                    move_thread.join()
                    return None

                q_full_captured = None
                for retry in range(3):
                    try:
                        state_obj = self.robot.get_state()
                        if state_obj is not None and getattr(state_obj, 'position', None) is not None:
                            q_full_captured = np.array(state_obj.position)
                            break
                    except Exception as e:
                        if retry == 2:
                            self.logger.warning(f"get_state() failed after 3 retries: {e}")
                        time.sleep(0.005)
                if q_full_captured is None:
                    if len(dataset) > 0:
                        q_full_captured = dataset[-1][0].copy()
                    else:
                        q_full_captured = np.zeros(26)

                res = self.marker_st.get_marker_transform(sampling_time=0, side=arm_side, use_filter=False, q_encoder=q_full_captured)

                if res:
                    pose_flat = res[0] if isinstance(res, list) else list(res.values())[0]
                    pose_mat = np.array(pose_flat).reshape(4, 4)

                    if np.linalg.norm(pose_mat[:3, 3]) > 0.01:
                        # Deduplicate: Only append if the pose is actually new (camera updated)
                        if len(dataset) == 0 or not np.allclose(dataset[-1][1], pose_mat, atol=1e-5):
                            dataset.append((q_full_captured, pose_mat))

                time.sleep(0.01)

        finally:
            if move_thread.is_alive():
                self.robot.cancel_control()
            move_thread.join()

        if not move_thread.success:
            if log_callback: log_callback(f"[ERROR] {label} sweep motion failed or was cancelled.")
            return None

        # Defense-in-depth: Filter out any IPPE planar ambiguity flip spikes before circle/axis fitting
        dataset = self.filter_sweep_dataset(dataset, arm_idx, sweep_joint, mismatch_threshold_deg=3.0, log_callback=log_callback)

        if len(dataset) < 10:
            if log_callback: log_callback(f"[ERROR] Too few valid captured points for {label} ({len(dataset)} points). Returning to ready pose and prompting posture adjustment...")
            try:
                sweep_mode = kwargs.get('mode', "marker" if "Marker" in label else "joint")
                self.perform_move_to_ready_pose(arm_side, mode=sweep_mode, log_callback=log_callback)
            except Exception as e:
                if log_callback: log_callback(f"[WARN] Failed to return to ready pose: {e}")
            if hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                # Pass the mode: the teaching dialog stores the taught posture under it, and the
                # re-sweep below looks it up by the same key. Without it the posture was stored
                # under None and the re-sweep ran from the old posture as if nothing was taught.
                sweep_mode = kwargs.get('mode', "marker" if "Marker" in label else "joint")
                resolved = self.marker_problem_callback(arm_side, mode=sweep_mode)
                if resolved:
                    norm_mode = "wrist_pitch" if sweep_mode == "wrist_pitch_v13" else ("wrist_roll" if sweep_mode == "wrist_roll_v13" else sweep_mode)
                    if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
                        arm_dict = self.user_taught_ready_poses.get(arm_side)
                        if isinstance(arm_dict, dict) and arm_dict.get(norm_mode) is not None:
                            q_center = arm_dict.get(norm_mode)
                    return self.perform_single_joint_sweep(
                        arm_side, sweep_joint, q_center, start_deg, end_deg, sweep_duration,
                        q_head=q_head, label=label, log_callback=log_callback, **kwargs
                    )
            return None

        logging.info(f"    -> Swept {len(dataset)} dense raw coordinate frames during {label} motion.")
        return dataset

