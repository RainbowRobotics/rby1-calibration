from core.storage import FileStorage
from core.storage import ArtifactStorage
import time
import logging
import os
import numpy as np
import rby1_sdk as rby
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator

class MarkerCalibrator(BaseCalibrator):

    @staticmethod
    def rodrigues_rotation(vector, axis, theta_rad):
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return vector * cos_t + np.cross(axis, vector) * sin_t + axis * np.dot(axis, vector) * (1 - cos_t)

    def perform_move_to_center(self, arm_side, log_callback=None, stop_event=None, target_dist=300.0, max_attempts=3):
        if not self.marker_st:
            if log_callback: log_callback("[ERROR] Camera system not initialized.")
            return False
        if not self.robot:
            if log_callback: log_callback("[ERROR] Robot not connected.")
            return False

        if log_callback: log_callback(f"[INFO] Moving {arm_side} arm to camera center (target: {target_dist}mm, max_attempts: {max_attempts})...")
        
        # Get rotation only from mount_to_cam
        mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        R_cam_to_rob = R_scipy.from_euler('ZYX', [mount_to_cam[5], mount_to_cam[4], mount_to_cam[3]], degrees=True).as_matrix()
        p_target_cam = np.array([0.0, 0.0, target_dist / 1000.0])

        for attempt in range(max_attempts):
            if stop_event and stop_event.is_set():
                if log_callback: log_callback("[INFO] Move canceled by user.")
                self.robot.cancel_control()
                return False
                
            if log_callback: log_callback(f"[Attempt {attempt + 1}/{max_attempts}] Capturing marker pose...")
            time.sleep(1.0)
            res = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not res:
                if log_callback: log_callback("  [ERROR] Marker not visible.")
                return False
            
            if isinstance(res, list):
                T_cam_to_marker = np.array(res[0]).reshape(4, 4)
            else:
                T_cam_to_marker = np.array(list(res.values())[0]).reshape(4, 4)
                
            cam_pos = T_cam_to_marker[:3, 3]
            cam_rot = T_cam_to_marker[:3, :3]
            
            pos_err_mm = np.linalg.norm(cam_pos - p_target_cam) * 1000.0
            rot_err_mat = cam_rot.T
            rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
            err_norm = np.linalg.norm([pos_err_mm, rot_err_deg])
 
            if log_callback:
                log_callback(f"  Current: X={cam_pos[0]*1000:.1f}, Y={cam_pos[1]*1000:.1f}, Z={cam_pos[2]*1000:.1f} mm")
                log_callback(f"  Error Norm: {err_norm:.2f} (Pos:{pos_err_mm:.1f}mm, Ang:{rot_err_deg:.1f}deg)")
 
            if err_norm <= 0.5:
                if log_callback: log_callback(f"  [SUCCESS] Reached center aligned pose! (Norm: {err_norm:.2f})")
                break
 
            if log_callback: log_callback("  Calculating joint command and moving...")
            
            dp_cam = p_target_cam - cam_pos
            dR_cam = cam_rot.T  # relative rotation error to identity
            
            # Rotate errors to robot frame (using only rotation R_cam_to_rob)
            dp_rob = R_cam_to_rob @ dp_cam
            dR_rob = R_cam_to_rob @ dR_cam @ R_cam_to_rob.T
            
            ee_name = f"ee_{arm_side}"
            T_rob_to_ee = self.compute_fk(self.robot, self.robot.get_dynamics(), self.robot.get_state().position, ee_name, "link_torso_5")
            p_ee = T_rob_to_ee[:3, 3]
            R_ee = T_rob_to_ee[:3, :3]
            
            T_rob_to_ee_new = np.eye(4)
            T_rob_to_ee_new[:3, :3] = dR_rob @ R_ee
            T_rob_to_ee_new[:3, 3] = p_ee + dp_rob
            
            cb = rby.CartesianCommandBuilder().set_minimum_time(3.0)
            cb.add_target("link_torso_5", ee_name, T_rob_to_ee_new.astype(np.float32), 0.2, 0.5, 1.0)
            cb.set_stop_orientation_tracking_error(1e-4)
            cb.set_stop_position_tracking_error(1e-3)
            
            body_cmd = rby.BodyComponentBasedCommandBuilder()
            if arm_side == "right":
                body_cmd.set_right_arm_command(cb)
            else:
                body_cmd.set_left_arm_command(cb)
                
            rc = rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
            )
            rv = self.robot.send_command(rc, 10).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                if log_callback: log_callback(f"  [ERROR] Failed to move: {rv.finish_code}")
                return False
            time.sleep(0.5)
        return True

    def adopt_taught_marker_pose(self, arm_side, log_callback=None):
        """Take the posture the operator just taught as the marker sweep posture, with J5 back at
        its ready-pose value (nominal + the J5 calibration offset, as movej applies it).

        The bracket orientation is solved as if J5 sits at 90 deg. Hand-adjusting the arm to bring
        the marker back into view moves J5 as well, and the taught pose is replayed as raw encoder
        values: on 2026-09-21 a right-arm re-teach left J5 2.7 deg off, the measured J4-J6 axes came
        out 86.8 deg apart, the bracket roll read 90.4 deg and Step 2 moved both J0 offsets ~0.5 deg.
        The other joints stay where the operator put them. Returns the stored pose (radians).
        """
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        pose = list(np.array(self.robot.get_state().position)[arm_idx])
        try:
            version_key = "v1.3" if self.is_v13() else "v1.2"
            nominal = self.get_ready_pose(version_key, "marker", None, arm_side)
            offsets = getattr(self, "joint_offsets", None) or {}
            arm_offsets = offsets.get(arm_side, offsets) if isinstance(offsets, dict) else {}
            target = float(nominal[5]) + np.radians(float(arm_offsets.get("wrist_pitch", 0.0)))
            if log_callback and abs(pose[5] - target) > np.radians(0.05):
                log_callback(f"[INFO] Re-taught {arm_side} marker posture: J5 {np.degrees(pose[5]):.2f} -> "
                             f"{np.degrees(target):.2f} deg (ready-pose value incl. calibration offset).")
            pose[5] = target
        except Exception as error:
            if log_callback:
                log_callback(f"[WARN] Could not restore J5 on the taught marker posture: {error}")
        if not isinstance(getattr(self, "user_taught_ready_poses", None), dict):
            self.user_taught_ready_poses = {}
        self.user_taught_ready_poses.setdefault(arm_side, {})["marker"] = list(pose)
        return pose

    def perform_calibration_sweep(self, arm_side, axis_mode, log_callback=None, status_callback=None, use_head_tracking=True, save_debug=False, initial_joint_pos=None, pass_idx=1, sweep_duration=10.0):
        try:
            if getattr(self, 'stop_requested', False):
                return None

            self.current_calib_mode = "marker"
            max_readjust_retries = 2
            readjust_retry_count = 0

            while True:
                if getattr(self, 'stop_requested', False):
                    return None

                if save_debug and pass_idx == 1:
                    from core.storage import CONFIG_PATHS
                    result_txt_dir = CONFIG_PATHS["txt_dir"]
                    fname = os.path.join(result_txt_dir, f"sweep_points_{arm_side}_marker_axis_{axis_mode}.txt")
                    if os.path.exists(fname):
                        try: FileStorage.remove(fname)
                        except: pass

                if log_callback:
                    log_callback("\n" + "="*50)
                    log_callback(f"   STARTING {str(axis_mode).upper()} CONTINUOUS MARKER SWEEP")
                    log_callback("="*50)
                    
                if not getattr(self.marker_st, 'sim', False):
                    # Pre-check marker visibility
                    initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
                    if not initial_check:
                        if log_callback: log_callback("[ERROR] Marker is not visible in ready pose.")
                        if readjust_retry_count < max_readjust_retries and hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                            readjust_retry_count += 1
                            if log_callback: log_callback(f"[INFO] Prompting user for manual teaching due to marker visibility error (Attempt {readjust_retry_count}/{max_readjust_retries})...")
                            # Back to the ready pose before asking, as every other re-teach path
                            # does: the arm may still be where the previous sweep ended, and the
                            # operator should adjust from the posture the sweep will start from.
                            self.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=log_callback)
                            resolved = self.marker_problem_callback(arm_side, mode="marker")
                            if resolved:
                                self.adopt_taught_marker_pose(arm_side, log_callback)
                                self.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=log_callback)
                                initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
                                if initial_check:
                                    continue
                        if not initial_check:
                            if status_callback: status_callback(False)
                            return None
                    if status_callback: status_callback(True)
                else:
                    if status_callback: status_callback(True)

                if not self.robot:
                    if log_callback: log_callback("[ERROR] Robot is not connected.")
                    return None

                state = self.robot.get_state()
                model = self.robot.model()
                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                
                # Check if user-taught ready pose exists, prioritizing taught pose
                taught_pose = None
                if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
                    arm_dict = self.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_dict, dict) and "marker" in arm_dict and arm_dict["marker"] is not None:
                        taught_pose = list(arm_dict["marker"])

                if taught_pose is not None:
                    cur_initial_pos = list(taught_pose)
                elif initial_joint_pos is not None:
                    cur_initial_pos = list(initial_joint_pos)
                else:
                    cur_initial_pos = list(np.array(state.position)[arm_idx])

                # Sweep configuration from MARKER_CONFIGS
                axis_str = str(axis_mode).lower()
                mcfg = None
                for key in self.MARKER_CONFIGS:
                    if key in axis_str or key.split("_")[-1] in axis_str:
                        mcfg = self.MARKER_CONFIGS[key]
                        break
                if mcfg is None:
                    raise ValueError(f"Unknown marker sweep axis mode: {axis_mode}")
                
                ver = "v13" if self.is_v13() else "v12"
                start_deg = mcfg.get(f"start_deg_{ver}", mcfg["start_deg"])
                end_deg = mcfg.get(f"end_deg_{ver}", mcfg["end_deg"])
                axis_duration = mcfg.get(f"sweep_duration_s_{ver}", sweep_duration)
                joint_i = mcfg["joint_i"]

                head_idx = list(model.head_idx[:2]) if len(model.head_idx) >= 2 else None
                q_head_0 = np.array([float(state.position[i]) for i in head_idx], dtype=np.float64) if head_idx is not None else None
                dyn_model = self.robot.get_dynamics()
                
                q_head_start = None
                if use_head_tracking and self.is_head_active() and head_idx is not None and q_head_0 is not None:
                    q_head_start = q_head_0

                dataset = self.perform_single_joint_sweep(
                    arm_side, joint_i, cur_initial_pos, start_deg, end_deg, axis_duration,
                    q_head=q_head_start, label=f"Marker Axis {axis_mode}", log_callback=log_callback, mode="marker"
                )
                if dataset is None:
                    if readjust_retry_count < max_readjust_retries and hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                        readjust_retry_count += 1
                        if log_callback:
                            log_callback(f"\n[WARNING] Marker Axis {axis_mode} sweep failed (marker lost or movement aborted).")
                            log_callback(f"[INFO] Prompting posture readjustment (Attempt {readjust_retry_count}/{max_readjust_retries})...")
                        self.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=log_callback)
                        resolved = self.marker_problem_callback(arm_side, mode="marker")
                        if resolved:
                            new_pose = self.adopt_taught_marker_pose(arm_side, log_callback)
                            cur_initial_pos = list(new_pose)
                            initial_joint_pos = list(new_pose)
                            if log_callback:
                                log_callback(f"[INFO] Posture readjusted and preserved. Restarting Marker Axis {axis_mode} sweep...")
                            time.sleep(1.0)
                            continue
                        else:
                            if log_callback: log_callback("[ERROR] Posture readjustment cancelled by user. Aborting marker sweep.")
                            return None
                    else:
                        if log_callback: log_callback(f"[ERROR] Marker Axis {axis_mode} sweep failed (consecutive retry limit reached). Aborting.")
                        return None

                if dataset:
                    cur_initial_pos = list(np.array(dataset[0][0])[arm_idx])

                captured_poses = [pose for _, pose in dataset]
                captured_angles = [np.degrees(np.array(q_full)[arm_idx[joint_i]] - cur_initial_pos[joint_i]) for q_full, _ in dataset]
                captured_q_full = [q_full for q_full, _ in dataset]

                if getattr(self, 'stop_requested', False):
                    if log_callback: log_callback("[INFO] Stop requested during marker sweep.")
                    return None

                if len(captured_poses) < 20:
                    if log_callback: log_callback(f"[ERROR] Too few valid marker poses ({len(captured_poses)} < 20).")
                    if readjust_retry_count < max_readjust_retries and hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                        readjust_retry_count += 1
                        if log_callback: log_callback(f"[INFO] Prompting posture readjustment (Attempt {readjust_retry_count}/{max_readjust_retries})...")
                        self.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=log_callback)
                        resolved = self.marker_problem_callback(arm_side, mode="marker")
                        if resolved:
                            new_pose = self.adopt_taught_marker_pose(arm_side, log_callback)
                            cur_initial_pos = list(new_pose)
                            initial_joint_pos = list(new_pose)
                            if log_callback:
                                log_callback(f"[INFO] Posture readjusted and preserved. Restarting Marker Axis {axis_mode} sweep...")
                            time.sleep(1.0)
                            continue
                        else:
                            if log_callback: log_callback("[ERROR] Posture readjustment cancelled by user. Aborting marker sweep.")
                            return None
                    else:
                        if log_callback: log_callback("[ERROR] Too few valid marker poses and retry limit reached. Aborting.")
                        return None

                # Solve Circle Fitting
                n_nom = mcfg["n_nom_v13"] if self.is_v13() else mcfg["n_nom_v12"]
                res = self.fit_circle_3d_and_6dof_misalignment(captured_poses, captured_angles, axis_prior=n_nom, robust=True)

                # Anomaly detection on fitted circle and rotation axis in marker coordinate system:
                ver_key = "1.3" if self.is_v13() else "1.2"
                nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
                nominal_rpy = nominal_vec[3:6]
                R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

                x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
                y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
                z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

                # Account for intermediate wrist angles between sweeping joint axis and flange
                q_5 = float(cur_initial_pos[5]) if len(cur_initial_pos) > 5 else 0.0
                q_6 = float(cur_initial_pos[6]) if len(cur_initial_pos) > 6 else 0.0
                if hasattr(self, 'joint_offsets') and self.joint_offsets:
                    offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                    q_6_eff = q_6 - np.radians(offsets.get("wrist_roll" if self.is_v13() else "wrist_yaw2", 0.0))
                    q_5_eff = q_5 - np.radians(offsets.get("wrist_pitch", 0.0))
                else:
                    q_6_eff = q_6
                    q_5_eff = q_5

                axis_str = str(axis_mode).lower()
                if "6" in axis_str:
                    target_ideal = x_ee_m_ideal if self.is_v13() else z_ee_m_ideal
                elif "5" in axis_str:
                    # In marker frame, Joint 5 axis (Link 5 Y) is rotated by Joint 6 around Flange Z (v1.2) or Flange X (v1.3)
                    if self.is_v13():
                        target_ideal = self.rodrigues_rotation(y_ee_m_ideal, x_ee_m_ideal, q_6_eff)
                    else:
                        target_ideal = self.rodrigues_rotation(y_ee_m_ideal, z_ee_m_ideal, q_6_eff)
                else: # Axis 4
                    if self.is_v13():
                        v4_ee = R_scipy.from_euler('X', q_6_eff).as_matrix() @ R_scipy.from_euler('Y', q_5_eff).as_matrix() @ np.array([0.0, 0.0, 1.0])
                    else:
                        v4_ee = R_scipy.from_euler('Z', q_6_eff).as_matrix() @ R_scipy.from_euler('Y', q_5_eff).as_matrix() @ np.array([0.0, 0.0, 1.0])
                    target_ideal = R_ee_m_ideal.T @ v4_ee

                target_ideal = target_ideal / np.linalg.norm(target_ideal)

                n_marker_actual = self.extract_axis_from_rotations(captured_poses, target_ideal)
                dot_val = np.clip(abs(np.dot(n_marker_actual, target_ideal)), -1.0, 1.0)
                axis_dev_deg = float(np.degrees(np.arccos(dot_val)))
                rmse = res.get('rmse', 0.0)

                is_anomalous = False
                anomaly_reasons = []
                if axis_dev_deg > 35.0:
                    is_anomalous = True
                    anomaly_reasons.append(f"Fitted rotation axis in marker frame deviated {axis_dev_deg:.2f}° > 35.0° from nominal axis")
                if rmse > 20.0:
                    is_anomalous = True
                    anomaly_reasons.append(f"Circle fitting RMSE {rmse:.2f} mm > 20.0 mm")

                if is_anomalous:
                    if log_callback:
                        log_callback(f"\n[ALERT] Runtime measurement anomaly detected for Marker Axis {axis_mode}:")
                        for r in anomaly_reasons:
                            log_callback(f"  - {r}")
                    if readjust_retry_count < max_readjust_retries and hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                        readjust_retry_count += 1
                        if log_callback:
                            log_callback(f"[INFO] Moving arm to ready pose and prompting user for posture readjustment (Attempt {readjust_retry_count}/{max_readjust_retries})...")
                        self.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=log_callback)
                        resolved = self.marker_problem_callback(arm_side, mode="marker")
                        if resolved:
                            new_pose = self.adopt_taught_marker_pose(arm_side, log_callback)
                            cur_initial_pos = list(new_pose)
                            initial_joint_pos = list(new_pose)
                            if log_callback:
                                log_callback(f"[INFO] Posture readjusted and preserved. Restarting Marker Axis {axis_mode} sweep...")
                            time.sleep(1.0)
                            continue
                        else:
                            if log_callback: log_callback("[ERROR] Posture readjustment cancelled by user. Aborting marker sweep.")
                            return None
                    else:
                        if log_callback:
                            log_callback(f"[ERROR] Consecutive anomaly retry limit reached ({readjust_retry_count}/{max_readjust_retries}). Aborting marker sweep.")
                        return None

                # Normal success: break out of retry loop
                break
            
            # Load camera transform relative to mount link
            if self.is_head_active():
                mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                mount_to_cam_rot_only = [0.0, 0.0, 0.0] + list(mount_to_cam[3:])
                T_cam_fixed = self.make_transform(mount_to_cam_rot_only)
            else:
                head_base_to_cam = self.camera_config.get("head_base_to_cam", [0.098, 0.009, 0.012, -90.0, 0.0, -90.0])
                head_base_rot_only = [0.0, 0.0, 0.0] + list(head_base_to_cam[3:])
                T_cam_fixed = self.make_transform(head_base_rot_only)
            
            ee_name = f"ee_{arm_side}"
            pts_ee = []
            is_v13 = self.is_v13()
            for q_full, pose_cam_to_marker in zip(captured_q_full, captured_poses):
                try:
                    # q_physical = q_full - joint_offset (apply joint offsets to reconstruct actual physical angle)
                    q_mod = np.array(q_full)
                    if hasattr(self, 'joint_offsets') and self.joint_offsets:
                        offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                        q_mod[arm_idx[3]] -= np.radians(offsets.get("elbow", 0.0))
                        q_mod[arm_idx[5]] -= np.radians(offsets.get("wrist_pitch", 0.0))
                        if is_v13:
                            q_mod[arm_idx[6]] -= np.radians(offsets.get("wrist_roll", 0.0))
                        else:
                            q_mod[arm_idx[6]] -= np.radians(offsets.get("wrist_yaw2", 0.0))
                    
                    if self.is_head_active():
                        T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_mod, "link_head_2", "link_torso_5")
                        T_t5_to_cam = T_t5_to_head @ T_cam_fixed
                    else:
                        try:
                            T_t5_to_head_0 = self.compute_fk(self.robot, dyn_model, q_mod, "link_head_0", "link_torso_5")
                        except Exception:
                            T_t5_to_head_0 = np.eye(4)
                        T_t5_to_cam = T_t5_to_head_0 @ T_cam_fixed
                    
                    T_t5_to_marker = T_t5_to_cam @ pose_cam_to_marker
                    T_t5_to_ee = self.compute_fk(self.robot, dyn_model, q_mod, ee_name, "link_torso_5")
                    p_ee = np.linalg.inv(T_t5_to_ee) @ T_t5_to_marker @ np.array([0, 0, 0, 1])
                    pts_ee.append(p_ee[:3] * 1000.0) # in mm
                except Exception as e:
                    pass
            
            if len(pts_ee) > 0:
                res['pts_ee'] = np.array(pts_ee)
            else:
                res['pts_ee'] = np.zeros((0, 3))
                
            res['captured_poses'] = captured_poses
            res['captured_q_full'] = captured_q_full
            if save_debug:
                dataset = list(zip(captured_q_full, captured_poses))
                self.save_debug_points(
                    arm_side, axis_mode, dataset, cur_initial_pos, ee_name, dyn_model, T_cam_fixed, "marker", log_callback
                )
            return res
        finally:
            self.current_calib_mode = None

    def get_link_length(self, arm_side):
        try:
            if not self.robot:
                raise RuntimeError("Robot instance is not initialized")
            dyn_model = self.robot.get_dynamics()
            q = np.array(self.robot.get_state().position)
            T = BaseCalibrator.compute_fk(self.robot, dyn_model, q, f"ee_{arm_side}", f"link_{arm_side}_arm_5")
            return np.linalg.norm(T[:3, 3]) * 1000.0 # m to mm
        except Exception as e:
            logging.error(f"Failed to get link kinematics: {e}")
            raise e

    def get_z_sign(self, arm_side):
        """
        Dynamically determines the link Z-translation direction between link_5 (Wrist Pitch) and ee (Flange).
        
        Geometric Derivation:
        - In forward kinematics, T = compute_fk(robot, dyn_model, q, 'ee_{arm_side}', 'link_{arm_side}_arm_5').
        - T[2, 3] represents the signed Z translation from link_5 to the end-effector.
        - For v1.2: The end-effector is located along the negative Z direction of link_5 (T[2, 3] ≈ -0.133 m = -133 mm).
          Because get_link_length() computes the Euclidean norm (always positive, +133 mm), z_sign (-1.0)
          restores the true physical vector: z_sign * L_5_ee = -133 mm.
        - For v1.3: The link arrangement is along positive/zero Z, so z_sign is +1.0.
        """
        if self.robot and hasattr(self.robot, "get_dynamics"):
            try:
                dyn_model = self.robot.get_dynamics()
                if dyn_model is not None:
                    q = np.array(self.robot.get_state().position)
                    T = BaseCalibrator.compute_fk(self.robot, dyn_model, q, f"ee_{arm_side}", f"link_{arm_side}_arm_5")
                    # Dynamically evaluate the actual sign of the Z-translation vector from forward kinematics
                    return -1.0 if T[2, 3] < 0.0 else 1.0
            except Exception as e:
                logging.warning(f"Could not dynamically query link kinematics in get_z_sign: {e}. Falling back to CAD nominal.")
        
        # Nominal fallback when robot instance is not connected (e.g. offline testing / simulation)
        return 1.0 if self.is_v13() else -1.0


    def extract_axis_from_rotations(self, poses, ideal_axis):
        if len(poses) < 2:
            return np.array(ideal_axis, dtype=float) / np.linalg.norm(ideal_axis)
        mid_idx = len(poses) // 2
        R_ref = poses[mid_idx][:3, :3]
        axes = []
        for i, T in enumerate(poses):
            if i == mid_idx: continue
            R_rel = R_ref.T @ T[:3, :3] 
            rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
            angle = np.linalg.norm(rotvec)
            if angle > np.radians(1.0):
                axis = rotvec / angle
                if np.dot(axis, ideal_axis) < 0:
                    axis = -axis
                axes.append(axis)
        if len(axes) > 0:
            avg_axis = np.mean(axes, axis=0)
            return avg_axis / np.linalg.norm(avg_axis)
        return np.array(ideal_axis, dtype=float) / np.linalg.norm(ideal_axis)

    def compute_wrist_joints_from_3axis_sweeps(self, marker_data_4, marker_data_5, marker_data_6, arm_side, calib_pitch_deg=None, calib_roll_deg=None):
        """
        Phase 1: Calculates Joint 5 (Pitch) and Joint 6 (Roll) offsets from 3-axis sweep normals,
        and evaluates mutual orthogonality (residual deviation from 90.0 deg).
        """
        ver_key = "1.3" if self.is_v13() else "1.2"
        nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
        nominal_rpy = nominal_vec[3:6]
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

        x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

        poses_4 = marker_data_4.get('captured_poses', []) if marker_data_4 else []
        poses_5 = marker_data_5.get('captured_poses', []) if marker_data_5 else []
        poses_6 = marker_data_6.get('captured_poses', []) if marker_data_6 else []

        n6_marker_actual = self.extract_axis_from_rotations(poses_6, x_ee_m_ideal)
        n5_marker_actual = self.extract_axis_from_rotations(poses_5, y_ee_m_ideal)
        n4_marker_actual = self.extract_axis_from_rotations(poses_4, z_ee_m_ideal) if len(poses_4) > 0 else z_ee_m_ideal

        # Compute mutual angles between normal vectors
        ang_45 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_marker_actual, n5_marker_actual)), -1.0, 1.0)))
        ang_56 = np.degrees(np.arccos(np.clip(abs(np.dot(n5_marker_actual, n6_marker_actual)), -1.0, 1.0)))
        ang_46 = np.degrees(np.arccos(np.clip(abs(np.dot(n4_marker_actual, n6_marker_actual)), -1.0, 1.0)))
        ortho_err = max(abs(ang_45 - 90.0), abs(ang_56 - 90.0), abs(ang_46 - 90.0))

        # Joint 6 Roll Offset Calculation (relative to Pitch axis)
        x_col = n6_marker_actual / np.linalg.norm(n6_marker_actual)
        ref_y = y_ee_m_ideal - np.dot(y_ee_m_ideal, x_col) * x_col
        ref_y /= np.linalg.norm(ref_y)
        ref_z = np.cross(x_col, ref_y)
        diff_angle_6 = np.arctan2(np.dot(n5_marker_actual, ref_z), np.dot(n5_marker_actual, ref_y))
        roll_corr = float(np.degrees(diff_angle_6))
        opt_delta_6 = (calib_roll_deg if calib_roll_deg is not None else 0.0) + roll_corr

        # Joint 5 Pitch Offset Calculation (orthogonal deviation between J4 and J6)
        cross_64 = np.cross(n6_marker_actual, n4_marker_actual)
        sign_5 = np.sign(np.dot(n5_marker_actual, cross_64)) if np.linalg.norm(cross_64) > 1e-4 else 1.0
        pitch_corr = float((ang_46 - 90.0) * sign_5)
        opt_delta_5 = (calib_pitch_deg if calib_pitch_deg is not None else 0.0) + pitch_corr

        converged = (ortho_err < 0.35)

        return {
            'converged': bool(converged),
            'd5_opt_deg': opt_delta_5,
            'd6_opt_deg': opt_delta_6,
            'opt_delta_5': opt_delta_5,
            'opt_delta_6': opt_delta_6,
            'recommended_joint_offset_5': opt_delta_5,
            'recommended_joint_offset_6': opt_delta_6,
            'ortho_err': float(ortho_err),
            'ang_45': float(ang_45),
            'ang_56': float(ang_56),
            'ang_46': float(ang_46),
            'n4_marker_actual': n4_marker_actual,
            'n5_marker_actual': n5_marker_actual,
            'n6_marker_actual': n6_marker_actual,
            'x_col': x_col,
            'ref_y': ref_y,
            'y_ee_m_ideal': y_ee_m_ideal
        }

    def compute_marker_bracket_from_orthogonal_sweeps(self, marker_data_4, marker_data_5, marker_data_6, arm_side):
        """
        Phase 2: Once Joint 5 & 6 are confirmed orthogonal, extracts pure 6-DOF marker bracket
        transform (Tf_to_marker) using unbiased forward kinematics and SVD orientation projection.
        """
        ver_key = "1.3" if self.is_v13() else "1.2"
        nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
        x_nom = nominal_vec[0] * 1000.0
        y_nom = 0.0 if self.is_v13() else nominal_vec[1] * 1000.0
        z_nom = nominal_vec[2] * 1000.0
        
        nominal_rpy = nominal_vec[3:6]
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()

        x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

        poses_4 = marker_data_4.get('captured_poses', []) if marker_data_4 else []
        poses_5 = marker_data_5.get('captured_poses', []) if marker_data_5 else []
        poses_6 = marker_data_6.get('captured_poses', []) if marker_data_6 else []

        # J6/J5 axes from the circle fit mapped with every frame, as on v1.2: the rotation-only extraction
        # measures every frame against the middle one and its yaw erred up to 0.48 deg at 0.37 deg noise (2026-10-16).
        def axis_in_marker_frame(marker_data, poses, ideal_axis):
            n_cam = marker_data.get('axis_opt') if marker_data else None
            if n_cam is None or len(poses) == 0:
                return self.extract_axis_from_rotations(poses, ideal_axis)
            return self.axis_in_marker_frame_robust(n_cam, poses, ideal_axis)

        n6_marker_actual = axis_in_marker_frame(marker_data_6, poses_6, x_ee_m_ideal)
        n5_marker_actual = axis_in_marker_frame(marker_data_5, poses_5, y_ee_m_ideal)
        n4_marker_actual = self.extract_axis_from_rotations(poses_4, z_ee_m_ideal) if len(poses_4) > 0 else z_ee_m_ideal

        radius_6 = marker_data_6.get('radius', 0.0) if marker_data_6 else 0.0
        radius_5 = marker_data_5.get('radius', 0.0) if marker_data_5 else 0.0
        radius_4 = marker_data_4.get('radius', 0.0) if marker_data_4 is not None else 0.0

        # Build orthogonal frame from 3 axes via SVD (Spherical wrist decoupled basis)
        x_col = n6_marker_actual / np.linalg.norm(n6_marker_actual)
        y_proj = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
        y_col = y_proj / np.linalg.norm(y_proj)
        z_col = np.cross(x_col, y_col)
        z_col /= np.linalg.norm(z_col)

        M = np.column_stack((x_col, y_col, z_col))
        U, S, Vt = np.linalg.svd(M)
        R_m_ee_actual = U @ Vt
        if np.linalg.det(R_m_ee_actual) < 0:
            U[:, 2] *= -1
            R_m_ee_actual = U @ Vt
        R_ee_m_actual = R_m_ee_actual.T

        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        if arm_side == "right" and yaw_e < 0 and abs(yaw_e - 270.0) < 45.0:
            yaw_e += 360.0

        # v1.3: In ZYX Euler angle representation with Yaw = -90 deg:
        # - pitch_e rotates around Flange X_ee (co-axial with Joint 6 Roll).
        # - roll_e rotates around Flange -Y_ee (co-axial with Joint 5 Pitch).
        # Lock pitch_e to nominal CAD (0.0 deg) and roll_e to nominal CAD (90.0 deg)
        # so that 100% of the physical joint offsets are assigned to Joint 6 and Joint 5.
        pitch_e = float(nominal_rpy[1])
        roll_e = float(nominal_rpy[0])
        # v1.3 yaw is held at the design too (user decision, 2026-10-23): the real robot's J6 circle normal sat
        # 5-7 deg off the marker rotation axis and its yaw (-92.35 / -88.34) did not fit Step 2; only logged.
        yaw_measured_e = float(yaw_e)
        if self.is_v13():
            yaw_e = float(nominal_rpy[2])
            if arm_side == "right" and yaw_e < 0 and abs(yaw_e - 270.0) < 45.0:
                yaw_e += 360.0

        rot_err_mat = R_ee_m_actual.T @ R_ee_m_ideal
        rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))

        # Translation from the encoder-angle sweep radii, as on v1.2 (2026-10-15). The median-of-FK
        # path it replaces sat > 40 mm from the design and silently returned the design every run.
        solved = self.v13_bracket_position_from_radii(marker_data_5, marker_data_6, arm_side, (x_nom, y_nom, z_nom))
        if solved is None:
            (xe_opt, ye_opt, ze_opt), position_at_limit = (x_nom, y_nom, z_nom), False
            position_source = "design value (sweep radii or joint angles missing)"
        else:
            (xe_opt, ye_opt, ze_opt), position_at_limit = solved
            position_source = "J5/J6 sweep radii, y held at the design value"

        pos_diff_mm = float(np.linalg.norm([xe_opt - x_nom, ye_opt - y_nom, ze_opt - z_nom]))
        warn_large_pos = pos_diff_mm > 40.0

        return {
            'converged': True,
            'x_e': float(xe_opt), 'y_e': float(ye_opt), 'z_e': float(ze_opt),
            'roll_e': float(roll_e), 'pitch_e': float(pitch_e), 'yaw_e': float(yaw_e),
            'yaw_measured_e': yaw_measured_e,
            'rot_err_deg': float(rot_err_deg),
            'pos_diff_mm': float(pos_diff_mm),
            'warn_large_angle': rot_err_deg > 15.0,
            'warn_large_pos': warn_large_pos,
            'radius_6': radius_6, 'radius_5': radius_5, 'radius_4': radius_4,
            'n6_marker_actual': n6_marker_actual,
            'n5_marker_actual': n5_marker_actual,
            'n4_marker_actual': n4_marker_actual,
            'y_ee_m_ideal': y_ee_m_ideal,
            'position_source': position_source,
            'position_at_limit': bool(position_at_limit),
        }

    def v13_bracket_position_from_radii(self, marker_data_5, marker_data_6, arm_side, nominal_mm):
        """v1.3 bracket position (mm, ee frame) from the J5 and J6 bracket sweeps' radii.

        The wrist axes meet at the wrist centre w, so the marker origin p sits |(p - w) x a_k| from swept
        axis k (axes and w from FK at each sweep's first frame, known J6 offset removed): J6 gives z, J5 x.
        y moves neither to first order, so as on v1.2 it is held at the design value; x/z are fitted within
        bracket_offset_tolerance_mm. The J4 radius is left out: it tilts with the J5 offset, which is not
        calibrated yet at this stage (~2.1 mm per deg; simulator J5 -2.1 deg read 72.0 mm for 67.5,
        2026-10-16). Returns ((x, y, z), at_limit) or None."""
        sweeps = {5: marker_data_5, 6: marker_data_6}
        if not self.robot or any(not d or d.get('radius') is None or not d.get('captured_q_full')
                                 for d in sweeps.values()):
            return None
        dyn = self.robot.get_dynamics()
        model = self.robot.model()
        arm_idx = list(model.left_arm_idx if arm_side == "left" else model.right_arm_idx)
        offsets = getattr(self, "joint_offsets", None) or {}
        arm_offsets = offsets.get(arm_side, offsets) if isinstance(offsets, dict) else {}
        axis_in_joint_frame = {5: ([0.0, 1.0, 0.0], 4), 6: ([1.0, 0.0, 0.0], 5)}
        lines = []
        for k, d in sweeps.items():
            # Only J6 turns the J5 axis in the ee frame; its offset is known from pass 2 on.
            q = np.array(d['captured_q_full'][0], dtype=float)
            q[arm_idx[6]] -= np.radians(float(arm_offsets.get("wrist_roll", 0.0)))
            to_ee = np.linalg.inv(self.compute_fk(self.robot, dyn, q, f"ee_{arm_side}"))
            local, link = axis_in_joint_frame[k]
            T_joint = self.compute_fk(self.robot, dyn, q, f"link_{arm_side}_arm_{link}")
            axis = to_ee[:3, :3] @ T_joint[:3, :3] @ np.array(local)
            lines.append((axis / np.linalg.norm(axis), (to_ee @ T_joint[:, 3])[:3] * 1000.0, float(d['radius'])))
        x0, y0, z0 = nominal_mm
        tol = self.bracket_offset_tolerance_mm()

        def residuals(v):
            p = np.array([v[0], y0, v[1]])
            return [np.linalg.norm(np.cross(p - centre, axis)) - radius for axis, centre, radius in lines]

        fit = least_squares(residuals, [x0, z0], bounds=([x0 - tol, z0 - tol], [x0 + tol, z0 + tol]))
        x, z = (float(v) for v in fit.x)
        at_limit = max(abs(x - x0), abs(z - z0)) >= tol - 1e-3
        return (x, float(y0), z), at_limit

    def compute_unified_bracket_calibration_v1_3(self, marker_data_5, marker_data_6, arm_side, tolerance=0.5, marker_data_4=None, calib_roll_deg=None, calib_pitch_deg=None, calib_roll_or_yaw_deg=None, lock_bracket=False):
        if calib_roll_or_yaw_deg is not None:
            calib_roll_deg = calib_roll_or_yaw_deg
        
        # 1. Joint calculation
        wrist_res = self.compute_wrist_joints_from_3axis_sweeps(
            marker_data_4, marker_data_5, marker_data_6, arm_side,
            calib_pitch_deg=calib_pitch_deg, calib_roll_deg=calib_roll_deg
        )
        # 2. Bracket calculation
        bracket_res = self.compute_marker_bracket_from_orthogonal_sweeps(
            marker_data_4, marker_data_5, marker_data_6, arm_side
        )
        
        # Merge dictionary
        combined = {**wrist_res, **bracket_res}
        combined['converged'] = True
        return combined

    @staticmethod
    def axis_in_marker_frame_robust(n_cam, poses, ideal_axis, outlier_floor_deg=1.0):
        """A joint axis measured in the camera frame, expressed in the marker frame.

        The marker sits beyond the swept joint, so the axis is fixed in the marker frame and every
        frame of the sweep gives an estimate R_k^T n. This used to take the middle frame alone,
        which made the result as noisy as one frame's marker orientation: on 2026-09-22 (D405)
        picking a different frame moved the right J6 estimate by 0.68 deg std (-1.8..+1.4 deg),
        the ~1 deg scatter that kept J6 from converging, and the same mapping feeds the bracket
        orientation. Frames further than max(outlier_floor_deg, 3x the median deviation) from the
        median direction (IPPE near-flips) are dropped before averaging.
        """
        n_cam = np.asarray(n_cam, dtype=float)
        n_cam = n_cam / np.linalg.norm(n_cam)
        ideal_axis = np.asarray(ideal_axis, dtype=float)
        ns = np.array([np.asarray(T, dtype=float)[:3, :3].T @ n_cam for T in poses])
        ns *= np.where(ns @ ideal_axis >= 0, 1.0, -1.0)[:, None]
        med = np.median(ns, axis=0)
        med /= np.linalg.norm(med)
        dev = np.degrees(np.arccos(np.clip(ns @ med, -1.0, 1.0)))
        keep = dev <= max(outlier_floor_deg, 3.0 * float(np.median(dev)))
        n_marker = ns[keep].mean(axis=0) if np.any(keep) else med
        return n_marker / np.linalg.norm(n_marker)

    def bracket_y_locked(self):
        """v1.2: hold the bracket y at the design value (setting.yaml marker.lock_bracket_y, default
        on). Both brackets measure 54 mm; the J6 sweep radius cannot resolve that."""
        if self.is_v13():
            return False
        markers = getattr(self, "markers_config", None) or {}
        return bool(markers.get("lock_bracket_y", True))

    def bracket_roll_locked(self):
        """v1.2: hold the bracket roll at the design value (setting.yaml marker.lock_bracket_roll,
        default on). The bracket joint seats at its design angle on the flange."""
        if self.is_v13():
            return False
        markers = getattr(self, "markers_config", None) or {}
        return bool(markers.get("lock_bracket_roll", True))

    def bracket_offset_tolerance_mm(self):
        """How far x/z may sit from the design values (assembly and marker placement, ~1-2 mm)."""
        markers = getattr(self, "markers_config", None) or {}
        return float(markers.get("bracket_offset_tolerance_mm", 2.0))

    # Frames further than max(floor, factor x median) from a sweep's encoder-angle circle are left
    # out of the constrained axis refit. Clean frames sit within ~0.7 mm on the real sweeps.
    REFIT_OUTLIER_FLOOR_MM = 1.0
    REFIT_OUTLIER_MEDIAN_FACTOR = 5.0

    def drop_refit_outliers(self, points, marker_data, axis_num, arm_side, log_callback=None):
        """Sweep points without the frames that sit off the sweep's encoder-angle circle.

        The encoder-angle fit (fit_circle_3d_and_6dof_misalignment) is robust, but the constrained
        refit is a plain least-squares fit, so one false detection bends all three axes. On
        2026-09-22 (left arm, pass 2) the marker left the view at the end of the J5 sweep and the
        last frame landed 26 mm off the circle: the refit moved the bracket pitch 0.09 -> 0.33 deg
        and the roll fit 90.2 -> 88.5 deg. Returns the points to use (all of them when the circle is
        unknown).
        """
        c, n, r = marker_data.get('c_opt'), marker_data.get('axis_opt'), marker_data.get('radius_encoder')
        if c is None or n is None or r is None or len(points) == 0:
            return points
        n = np.asarray(n, dtype=float) / np.linalg.norm(n)
        v = points - np.asarray(c, dtype=float)
        axial = v @ n
        radial = np.linalg.norm(v - np.outer(axial, n), axis=1)
        dist = np.hypot(radial - r, axial)
        limit = max(self.REFIT_OUTLIER_FLOOR_MM, self.REFIT_OUTLIER_MEDIAN_FACTOR * float(np.median(dist)))
        keep = dist <= limit
        if not np.all(keep) and log_callback:
            dropped = np.where(~keep)[0]
            log_callback(f"[INFO] {arm_side} bracket axis refit: left out {len(dropped)} of {len(points)} frame(s) of "
                         f"the axis {axis_num} sweep more than {limit:.1f} mm off its circle "
                         f"(frame {', '.join(str(i) for i in dropped[:5])}; up to {dist.max():.1f} mm).")
        return points[keep]

    J5_OFF_NOMINAL_WARN_DEG = 1.0
    # How far the bracket y solved from the sweep radii may sit from the design value before a
    # WARN (v1.2). Both brackets measure 54 mm; the encoder-angle radii put it within 0.5 mm.
    BRACKET_Y_WARN_MM = 1.0

    def warn_if_j5_off_nominal(self, marker_data_4, marker_data_6, arm_side, log_callback=None):
        """Warn when the measured J4 and J6 sweep axes are not perpendicular.

        With J5 at 90 deg they should be; the bracket roll absorbs whatever they are off. On
        2026-09-21 a re-taught right-arm posture (J5 moved 2.7 deg by hand) measured 86.8 deg,
        and the bracket roll (90.4 deg) moved Step 2's J0 by ~0.5 deg.
        Returns the deviation in degrees (or None).
        """
        n4, n6 = marker_data_4.get('axis_opt'), marker_data_6.get('axis_opt')
        if n4 is None or n6 is None:
            return None
        n4 = np.asarray(n4, dtype=float) / np.linalg.norm(n4)
        n6 = np.asarray(n6, dtype=float) / np.linalg.norm(n6)
        deviation = 90.0 - float(np.degrees(np.arccos(min(1.0, abs(float(n4 @ n6))))))
        if abs(deviation) > self.J5_OFF_NOMINAL_WARN_DEG and log_callback:
            log_callback(f"[WARN] {arm_side} bracket sweeps: measured J4-J6 axis angle is {90.0 - deviation:.2f} deg "
                         f"({abs(deviation):.1f} deg from perpendicular). The bracket roll absorbs this and Step 2 "
                         f"then shifts J0; check the sweep plots and re-run the bracket sweeps from the ready pose.")
        return deviation

    def compute_unified_bracket_calibration(self, marker_data_5, marker_data_6, arm_side, tolerance=0.5, marker_data_4=None, calib_roll_deg=None, calib_pitch_deg=None, calib_roll_or_yaw_deg=None, lock_bracket=False, log_callback=None):
        if calib_roll_or_yaw_deg is not None:
            calib_roll_deg = calib_roll_or_yaw_deg

        # 2026-09-15: refit the three sweep circles together under the physical wrist constraints
        # (all axes through one wrist point, J5 axis perpendicular to J4 and J6) before the bracket
        # pose is derived. Independent circle fits left the axes 1-8 mm apart and up to 3 deg off
        # orthogonal on the real robot while the fit residual stayed at 0.08 mm, and that slack went
        # straight into the bracket pose (and from there into Step 2's J0).
        # The bracket translation is solved from the encoder-angle circle fits each sweep arrives
        # with (fit_circle_3d_and_6dof_misalignment: the circle is walked by the measured joint
        # angle, so chord = 2 r sin(dtheta/2) pins the radius). The refit below replaces 'radius'
        # with a position-only radius, which a short arc cannot resolve: on the 2026-09-22 13:30
        # sweeps it put the J6 radius at 50.8 mm (right) and 57.3 mm (left), while the encoder fits
        # gave 54.0 / 54.2 against the 54 mm both brackets measure.
        for d in (marker_data_4, marker_data_5, marker_data_6):
            if d is not None and 'radius_encoder' not in d:
                d['radius_encoder'] = d.get('radius', 0.0)

        bracket_axis_refit = None
        if marker_data_4 is not None and not self.is_v13():
            self.warn_if_j5_off_nominal(marker_data_4, marker_data_6, arm_side, log_callback)
            try:
                sweeps = [marker_data_4, marker_data_6, marker_data_5]
                pts = [np.array([np.asarray(T)[:3, 3] * 1000.0 for T in d.get('captured_poses', [])]) for d in sweeps]
                pts = [self.drop_refit_outliers(P, d, axis_num, arm_side, log_callback)
                       for P, d, axis_num in zip(pts, sweeps, (4, 6, 5))]
                if all(len(P) >= 10 for P in pts) and all(d.get('axis_opt') is not None for d in sweeps):
                    bracket_axis_refit = self.refine_bracket_axes_constrained(
                        pts,
                        [d['axis_opt'] for d in sweeps],
                        [d.get('c_opt', np.mean(P, axis=0)) for d, P in zip(sweeps, pts)],
                        [d.get('radius', 0.0) for d in sweeps],
                        log_callback=log_callback,
                    )
                    if bracket_axis_refit is not None:
                        for d, axis, centre, radius in zip(sweeps, bracket_axis_refit['axes'],
                                                           bracket_axis_refit['centers'],
                                                           bracket_axis_refit['radii']):
                            d['axis_opt_independent'] = d['axis_opt']
                            d['axis_opt'] = axis
                            d['axis'] = axis
                            d['c_opt'] = centre
                            d['radius'] = radius
            except Exception as error:
                if log_callback:
                    log_callback(f"[WARN] Constrained bracket axis refit skipped: {error}")

        L_5_ee = self.get_link_length(arm_side)

        # 1. 이상적인 마커 오일러 각도 (ZYX 기준)
        version_suffix = "_v13" if self.is_v13() else "_v12"
        tf_key = f"Tf_to_marker_{arm_side}{version_suffix}"
        tf_vec = self.camera_config.get(tf_key)
        if tf_vec is None:
            tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}")
            
        if tf_vec is not None and len(tf_vec) >= 6:
            nominal_rpy = [tf_vec[3], tf_vec[4], tf_vec[5]]
        else:
            ver_key = "1.3" if self.is_v13() else "1.2"
            nominal_rpy = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side][3:6]
            
        R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
        
        z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])
        y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
        x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])

        def extract_axis_from_rotations(poses, ideal_axis):
            if len(poses) < 2: return ideal_axis
            mid_idx = len(poses) // 2
            R_ref = poses[mid_idx][:3, :3]
            axes = []
            for i, T in enumerate(poses):
                if i == mid_idx: continue
                R_rel = R_ref.T @ T[:3, :3] 
                rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
                angle = np.linalg.norm(rotvec)
                if angle > np.radians(1.0):
                    axis = rotvec / angle
                    if np.dot(axis, ideal_axis) < 0: axis = -axis
                    axes.append(axis)
            if len(axes) > 0:
                avg_axis = np.mean(axes, axis=0)
                return avg_axis / np.linalg.norm(avg_axis)
            return ideal_axis

        # Joint 6 angle correction for Joint 5 sweep
        theta_6 = marker_data_5.get('theta_6', None)
        if theta_6 is None:
            q_full_5 = marker_data_5.get('captured_q_full', [])
            if len(q_full_5) > 0:
                if not self.robot:
                    raise RuntimeError("Robot instance is not initialized")
                model = self.robot.model()
                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                q_idx = arm_idx[6]
                theta_6 = np.mean([q[q_idx] for q in q_full_5])
            else:
                theta_6 = 0.0

        # Correct J6 joint offset if available
        if hasattr(self, 'joint_offsets') and self.joint_offsets:
            offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
            offset_val = offsets.get("wrist_roll" if self.is_v13() else "wrist_yaw2", 0.0)
            theta_6 -= np.radians(offset_val)

        # 2. 정밀 회전축 벡터 산출
        # v1.2: joint axis from the circle fit of the marker POSITION trajectory (axis_opt, camera
        # frame) mapped into the marker frame by the mid-sweep marker orientation -- the method
        # used through the 2026-09-10 real-robot runs and dropped in 1a30d1c (2026-09-13).
        # Rotation-only axis extraction biased the right-arm bracket pitch by ~3 deg on the
        # real robot (2026-09-15) versus ~0.3 deg from the position trajectory; it remains the
        # fallback when no circle-fit axis is available (and the v1.3 path, validated with it).
        # v1.3 took the rotation-only path until 2026-10-15; with 0.37 deg marker-orientation noise its J6
        # stage scattered 0.5-0.9 deg per sweep in the simulator and never converged, so it now uses v1.2's.
        def axis_in_marker_frame(marker_data, poses, ideal_axis):
            n_cam = marker_data.get('axis_opt')
            if n_cam is None or len(poses) == 0:
                return extract_axis_from_rotations(poses, ideal_axis)
            return self.axis_in_marker_frame_robust(n_cam, poses, ideal_axis)

        poses_6 = marker_data_6.get('captured_poses', [])
        target_ideal_6 = x_ee_m_ideal if self.is_v13() else z_ee_m_ideal
        n6_marker_actual = axis_in_marker_frame(marker_data_6, poses_6, target_ideal_6)

        poses_5 = marker_data_5.get('captured_poses', [])
        target_ideal_5 = self.rodrigues_rotation(y_ee_m_ideal, target_ideal_6, theta_6)
        n5_marker_actual = axis_in_marker_frame(marker_data_5, poses_5, target_ideal_5)
 
        # [BYPASS] Bypassed permanently to calculate using ONLY the marker and rotation axis trajectory.
        kinematic_success = False

        if not kinematic_success:

            if marker_data_4 is not None:
                # Joint 6 angle correction for Joint 4 sweep
                theta_6_4 = marker_data_4.get('theta_6', None)
                if theta_6_4 is None:
                    q_full_4 = marker_data_4.get('captured_q_full', [])
                    if len(q_full_4) > 0:
                        if not self.robot:
                            raise RuntimeError("Robot instance is not initialized")
                        model = self.robot.model()
                        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                        q_idx = arm_idx[6]
                        theta_6_4 = np.mean([q[q_idx] for q in q_full_4])
                    else:
                        theta_6_4 = 0.0

                if hasattr(self, 'joint_offsets') and self.joint_offsets:
                    offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                    offset_val = offsets.get("wrist_roll" if self.is_v13() else "wrist_yaw2", 0.0)
                    theta_6_4 -= np.radians(offset_val)

                # --- 3-Axis SVD Alignment (Using Joint 4, 5, and 6) ---
                poses_4 = marker_data_4.get('captured_poses', [])
                target_ideal_4_base = z_ee_m_ideal if self.is_v13() else x_ee_m_ideal
                target_ideal_4 = self.rodrigues_rotation(target_ideal_4_base, target_ideal_6, theta_6_4)
                n4_marker_actual = axis_in_marker_frame(marker_data_4, poses_4, target_ideal_4)

                if not self.is_v13():
                    z_col = n6_marker_actual
                    y_col_rot = n5_marker_actual - np.dot(n5_marker_actual, z_col) * z_col
                    y_col_rot /= np.linalg.norm(y_col_rot)
                    x_col_rot = n4_marker_actual - np.dot(n4_marker_actual, z_col) * z_col
                    x_col_rot /= np.linalg.norm(x_col_rot)

                    # Apply Joint 6 angle rotations back
                    if abs(theta_6) > 1e-5:
                        y_col = self.rodrigues_rotation(y_col_rot, z_col, theta_6)
                    else:
                        y_col = y_col_rot

                    if abs(theta_6_4) > 1e-5:
                        x_col = self.rodrigues_rotation(x_col_rot, z_col, theta_6_4)
                    else:
                        x_col = x_col_rot

                    M = np.column_stack((x_col, y_col, z_col))
                else:
                    # v1.3 Spherical Wrist: J6 is X-axis, J5 is Y-axis, J4 is Z-axis
                    x_col = n6_marker_actual
                    y_col_rot = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
                    y_col_rot /= np.linalg.norm(y_col_rot)
                    z_col_rot = n4_marker_actual - np.dot(n4_marker_actual, x_col) * x_col
                    z_col_rot /= np.linalg.norm(z_col_rot)

                    if abs(theta_6) > 1e-5:
                        y_col = self.rodrigues_rotation(y_col_rot, x_col, theta_6)
                    else:
                        y_col = y_col_rot

                    if abs(theta_6_4) > 1e-5:
                        z_col = self.rodrigues_rotation(z_col_rot, x_col, theta_6_4)
                    else:
                        z_col = z_col_rot

                    M = np.column_stack((x_col, y_col, z_col))

                # Use SVD to clean up orthogonality errors and build R_m_ee
                U, S, Vt = np.linalg.svd(M)
                R_m_ee_actual = U @ Vt
                if np.linalg.det(R_m_ee_actual) < 0:
                    U[:, 2] *= -1
                    R_m_ee_actual = U @ Vt
                
                R_ee_m_actual = R_m_ee_actual.T
            else:
                if not self.is_v13():
                    # --- 2-Axis Gram-Schmidt Alignment (Joint 5 and 6) ---
                    z_col = n6_marker_actual
                    y_col_rotated = n5_marker_actual - np.dot(n5_marker_actual, z_col) * z_col
                    y_col_rotated /= np.linalg.norm(y_col_rotated)
                    
                    if abs(theta_6) > 1e-5:
                        y_col = self.rodrigues_rotation(y_col_rotated, z_col, theta_6)
                    else:
                        y_col = y_col_rotated
                        
                    x_col = np.cross(y_col, z_col)
                    
                    R_m_ee_actual = np.column_stack((x_col, y_col, z_col))
                    R_ee_m_actual = R_m_ee_actual.T
                else:
                    x_col = n6_marker_actual
                    y_col_rotated = n5_marker_actual - np.dot(n5_marker_actual, x_col) * x_col
                    y_col_rotated /= np.linalg.norm(y_col_rotated)
                    
                    if abs(theta_6) > 1e-5:
                        y_col = self.rodrigues_rotation(y_col_rotated, x_col, theta_6)
                    else:
                        y_col = y_col_rotated
                        
                    z_col = np.cross(x_col, y_col)
                    
                    R_m_ee_actual = np.column_stack((x_col, y_col, z_col))
                    R_ee_m_actual = R_m_ee_actual.T

        # 4. 오일러 각도 추출
        euler_deg = R_scipy.from_matrix(R_ee_m_actual).as_euler('ZYX', degrees=True)
        yaw_e, pitch_e, roll_e = euler_deg
        
        # v1.2: Z축 회전 방향 비틀림(Torsion) 오차 배제 - 명목 설계값 yaw으로 고정
        ver_key = "1.3" if self.is_v13() else "1.2"
        nominal_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
        if not self.is_v13():
            yaw_e = nominal_vec[5]
            if arm_side == "right" and yaw_e < 0:
                yaw_e += 360.0
            if self.bracket_roll_locked():
                # The bracket seats on the flange at its design roll; the fitted roll rides on the
                # J5-J6 orthogonality estimate and moved up to 1.5 deg between passes. On
                # 2026-09-21 a 90.43 deg roll put ~0.5 deg into both J0 offsets in Step 2, and the
                # same data with roll 90 brought J0 back to the baseline.
                if log_callback:
                    log_callback(f"[INFO] {arm_side} bracket roll held at the design value {nominal_vec[3]:.1f}° "
                                 f"(sweep fit gave {roll_e:.2f}°); pitch still fitted ({pitch_e:.2f}°).")
                roll_e = float(nominal_vec[3])
        else:
            # v1.3: In ZYX Euler angle representation with Yaw = -90 deg:
            # - pitch_e rotates around Flange X_ee (co-axial with Joint 6 Roll).
            # - roll_e rotates around Flange -Y_ee (co-axial with Joint 5 Pitch).
            pitch_e = float(nominal_vec[4])
            roll_e = float(nominal_vec[3])
            if arm_side == "right" and yaw_e < 0 and abs(yaw_e - 270.0) < 45.0:
                yaw_e += 360.0

        # 5. 평행이동 오프셋 계산 (Least-Squares Solver allowing small attachment errors)
        radius_6 = marker_data_6.get('radius_encoder', 0.0)
        radius_5 = marker_data_5.get('radius_encoder', 0.0)
        radius_4 = marker_data_4.get('radius_encoder', 0.0) if marker_data_4 is not None else 0.0
        
        x_nom = nominal_vec[0] * 1000.0
        y_nom = nominal_vec[1] * 1000.0
        z_nom = nominal_vec[2] * 1000.0
        
        opt_delta_5_rad = 0.0
        opt_delta_6_rad = 0.0
        
        z_sign = self.get_z_sign(arm_side)

        from scipy.optimize import least_squares
        has_j4 = (marker_data_4 is not None and radius_4 > 1e-3)

        if self.is_v13():
            # v1.3 Spherical Wrist: J6 is Roll (X), J5 is Pitch (Y), J4 is Yaw (Z)
            # Pivot is located at Z = +125mm in EE frame (z_sign * L_5_ee = -125mm)
            def residuals_trans(params):
                xe, ye, ze = params
                Z_prime = ze + z_sign * L_5_ee
                r6_pred = np.sqrt(ye**2 + Z_prime**2)
                r5_pred = np.sqrt(xe**2 + Z_prime**2)
                r4_pred = np.sqrt(xe**2 + ye**2)
                
                res = [
                    (r6_pred - radius_6),
                    (r5_pred - radius_5)
                ]
                if has_j4:
                    res.append(r4_pred - radius_4)
                    
                reg_weight = 1e-4
                res.append(reg_weight * (xe - x_nom))
                res.append(reg_weight * (ye - y_nom))
                res.append(reg_weight * (ze - z_nom))
                return res

            initial_guess = [x_nom, y_nom, z_nom]
            lower_bounds = [x_nom - 40.0, y_nom - 30.0, z_nom - 40.0]
            upper_bounds = [x_nom + 40.0, y_nom + 30.0, z_nom + 40.0]
            opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
            x_e, y_e, z_e = opt_res.x
        else:
            # v1.2 Non-Spherical Wrist: J6 is Yaw2 (Z), J5 is Pitch (Y), J4 is Forearm Roll (X)
            def residuals_trans(params):
                xe, ye, ze = params
                Z_prime = ze + z_sign * L_5_ee
                r6_pred = np.sqrt(xe**2 + ye**2)
                r5_pred = np.sqrt(xe**2 + Z_prime**2)
                r4_pred = np.sqrt(ye**2 + Z_prime**2)
                
                res = [
                    (r6_pred - radius_6),
                    (r5_pred - radius_5)
                ]
                if has_j4:
                    res.append(r4_pred - radius_4)
                    
                reg_weight = 1e-7
                res.append(reg_weight * (xe - x_nom))
                res.append(reg_weight * (ye - y_nom))
                res.append(reg_weight * (ze - z_nom))
                return res

            initial_guess = [x_nom, y_nom, z_nom]
            lower_bounds = [x_nom - 40.0, y_nom - 40.0, -250.0]
            upper_bounds = [x_nom + 40.0, y_nom + 40.0, 10.0]
            opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
            x_e, y_e, z_e = opt_res.x
            fit_y, fit_z = y_e, z_e
            if abs(fit_y - y_nom) > self.BRACKET_Y_WARN_MM and log_callback:
                log_callback(f"[WARN] {arm_side} bracket: the sweep radii put y at {fit_y:.2f} mm, "
                             f"{abs(fit_y - y_nom):.2f} mm from the design {y_nom:.1f} mm (both brackets "
                             f"measure 54 mm). Check the sweep plots and the marker/bracket mounting.")
            if self.bracket_y_locked():
                # y is essentially the J6 sweep radius. Up to 2026-09-22 that radius came from a
                # position-only fit, and the short arc (45 deg of a ~54 mm circle, 4 mm sagitta)
                # turned 0.1 mm of fit error into ~1.3 mm of radius: the fitted y wandered 50-54 mm.
                # The encoder-angle radii now used put y within 0.5 mm of 54 on both arms, so the
                # lock is a guard; the free y is still logged and a WARN fires above past 1 mm.
                # x/z are refitted with y held (the J4/J5 radii mix y and z), within the assembly
                # tolerance the brackets are known to hold.
                tol = self.bracket_offset_tolerance_mm()

                def residuals_fixed_y(params):
                    return residuals_trans([params[0], y_nom, params[1]])[:-3] + [
                        1e-7 * (params[0] - x_nom), 1e-7 * (params[1] - z_nom)]

                fixed = least_squares(residuals_fixed_y, [x_nom, z_nom],
                                      bounds=([x_nom - tol, z_nom - tol], [x_nom + tol, z_nom + tol]), loss='huber')
                x_e, z_e = fixed.x
                y_e = y_nom
                at_bound = abs(abs(z_e - z_nom) - tol) < 1e-3 or abs(abs(x_e - x_nom) - tol) < 1e-3
                if log_callback:
                    log_callback(f"[INFO] {arm_side} bracket y held at the design value {y_nom:.1f} mm "
                                 f"(sweep fit alone gave y {fit_y:.1f}, z {fit_z:.1f}); refitted z {z_e:.2f} mm, x {x_e:.2f} mm "
                                 f"(allowed ±{tol:.1f} mm around {z_nom:.1f} / {x_nom:.1f}).")
                    if at_bound:
                        log_callback(f"[WARN] {arm_side} bracket x/z reached the ±{tol:.1f} mm assembly limit; "
                                     f"check the bracket mounting or the sweep plots.")

        print(f"DEBUG SOLVER v1.2: arm_side={arm_side}", flush=True)
        print(f"  L_5_ee = {L_5_ee:.4f}", flush=True)
        print(f"  radius_6 = {radius_6:.4f}, radius_5 = {radius_5:.4f}, radius_4 = {radius_4:.4f}", flush=True)
        print(f"  x_nom = {x_nom:.4f}, y_nom = {y_nom:.4f}, z_nom = {z_nom:.4f}", flush=True)
        print(f"  x_e = {x_e:.4f}, y_e = {y_e:.4f}, z_e = {z_e:.4f}", flush=True)
        print(f"  Initial guess: {initial_guess}", flush=True)
        print(f"  Lower bounds: {lower_bounds}", flush=True)
        print(f"  Upper bounds: {upper_bounds}", flush=True)
        print(f"  Optimal residuals: {residuals_trans(opt_res.x)}", flush=True)

        # Circle fitting validation checks
        if not self.is_v13():
            r6_err = abs(radius_6 - np.sqrt(x_e**2 + y_e**2))
            r5_err = abs(radius_5 - np.sqrt(x_e**2 + (z_e + z_sign * L_5_ee)**2))
            r4_err = abs(radius_4 - np.sqrt((z_sign * L_5_ee + z_e)**2 + y_e**2)) if marker_data_4 is not None else 0.0
        else:
            Z_prime = z_e + z_sign * L_5_ee
            r6_err = abs(radius_6 - np.sqrt(y_e**2 + Z_prime**2))
            r5_err = abs(radius_5 - np.sqrt(x_e**2 + Z_prime**2))
            r4_err = abs(radius_4 - np.sqrt(x_e**2 + y_e**2)) if marker_data_4 is not None else 0.0

        if marker_data_4 is not None:
            print(f"[VALIDATION] {arm_side.upper()} ARM BRACKET SWEEP CIRCLE RESIDUALS:", flush=True)
            print(f"  * J6 Sweep Radius Err: {r6_err:.4f} mm", flush=True)
            print(f"  * J5 Sweep Radius Err: {r5_err:.4f} mm", flush=True)
            print(f"  * J4 Sweep Radius Err: {r4_err:.4f} mm", flush=True)
            max_err = max(r6_err, r5_err, r4_err)
            if max_err < 1.0:
                print(f"  [SUCCESS] Circle reconstruction PASSED (Max Residual: {max_err:.4f} mm < 1.0 mm)", flush=True)
            else:
                print(f"  [WARNING] Circle reconstruction shows deviation (Max Residual: {max_err:.4f} mm)", flush=True)
        else:
            print(f"[VALIDATION] {arm_side.upper()} ARM BRACKET SWEEP CIRCLE RESIDUALS (No J4 data):", flush=True)
            print(f"  * J6 Sweep Radius Err: {r6_err:.4f} mm", flush=True)
            print(f"  * J5 Sweep Radius Err: {r5_err:.4f} mm", flush=True)
            max_err = max(r6_err, r5_err)
            if max_err < 1.0:
                print(f"  [SUCCESS] Circle reconstruction PASSED (Max Residual: {max_err:.4f} mm < 1.0 mm)", flush=True)
            else:
                print(f"  [WARNING] Circle reconstruction shows deviation (Max Residual: {max_err:.4f} mm)", flush=True)

        # 6. 알고리즘 신뢰도 평가 점수
        dot_val = np.dot(n6_marker_actual, n5_marker_actual)
        ortho_err = abs(90.0 - np.degrees(np.arccos(np.clip(abs(dot_val), -1.0, 1.0))))
        
        rot_err_mat = R_ee_m_actual.T @ R_ee_m_ideal
        rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(rot_err_mat) - 1) / 2, -1.0, 1.0)))
        
        bracket_result = {
            'converged': True,
            'x_e': x_e, 'y_e': y_e, 'z_e': z_e,
            'roll_e': roll_e, 'pitch_e': pitch_e, 'yaw_e': yaw_e,
            'L_5_ee': L_5_ee, 'radius_6': radius_6, 'radius_5': radius_5, 'radius_4': radius_4,
            'ortho_err': ortho_err,
            'rmse_6': marker_data_6.get('rmse', 0.0),
            'rmse_5': marker_data_5.get('rmse', 0.0),
            'rmse_4': marker_data_4.get('rmse', 0.0) if marker_data_4 is not None else 0.0,
            'rot_err_deg': rot_err_deg, 'tilt_diff': 0.0,
            'warn_large_angle': rot_err_deg > 15.0,
            'n6_marker_actual': n6_marker_actual,
            'n5_marker_actual': n5_marker_actual,
            'y_ee_m_ideal': y_ee_m_ideal
        }
        if bracket_axis_refit is not None:
            bracket_result['bracket_axis_refit'] = bracket_axis_refit
        return bracket_result

    def generate_marker_plot(self, res_5, res_6, res_4, unified_res, arm_side, is_v13, save_path):
        """
        Generates unified marker calibration plots and saves the image to disk.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def refit_view(res):
            """Points and circle in the plane of the jointly refit circle, or None if no refit.

            The refit replaces axis_opt / c_opt / radius, but pts_2d, uc_opt and vc_opt still
            belong to the independent fit. Drawing the new radius around the old 2D centre put the
            circle millimetres away from points it actually fits to ~0.2 mm (2026-09-21 plots).
            """
            if 'axis_opt_independent' not in res or res.get('c_opt') is None or not res.get('captured_poses'):
                return None
            P = np.array([np.asarray(T)[:3, 3] * 1000.0 for T in res['captured_poses']])
            n = np.asarray(res['axis_opt'], dtype=float)
            n = n / np.linalg.norm(n)
            helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            ex = np.cross(n, helper)
            ex /= np.linalg.norm(ex)
            ey = np.cross(n, ex)
            d = P - np.asarray(res['c_opt'], dtype=float)
            h = d @ n
            pts = np.c_[d @ ex, d @ ey]
            err = np.sqrt(h ** 2 + (np.linalg.norm(pts, axis=1) - res['radius']) ** 2)
            return pts, 0.0, 0.0, float(np.sqrt(np.mean(err ** 2)))

        def plot_single_axis(ax, res, axis_num, color):
            if res is None or 'pts_2d' not in res:
                ax.set_title(f"Axis {axis_num} Sweep: No Data")
                ax.axis('off')
                return
            view = refit_view(res)
            if view is not None:
                pts_2d, uc, vc, rmse = view
                label = 'Constrained Fit'
            else:
                pts_2d, uc, vc, rmse = res['pts_2d'], res['uc_opt'], res['vc_opt'], res['rmse']
                label = 'Fitted Circle'
            ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=color, s=15, alpha=0.6, label='Captured Points')
            circle = plt.Circle((uc, vc), res['radius'], color='r', fill=False, label=label)
            ax.add_patch(circle)
            ax.plot(uc, vc, 'rx', label='Center')

            x_min, x_max = pts_2d[:, 0].min(), pts_2d[:, 0].max()
            y_min, y_max = pts_2d[:, 1].min(), pts_2d[:, 1].max()
            span = max(x_max - x_min, y_max - y_min)
            margin = max(1.0, span * 0.5)
            cx = (x_max + x_min) / 2
            cy = (y_max + y_min) / 2
            ax.set_xlim(cx - span/2 - margin, cx + span/2 + margin)
            ax.set_ylim(cy - span/2 - margin, cy + span/2 + margin)
            ax.set_aspect('equal')
            ax.grid(True)
            title = f"Axis {axis_num} Sweep (Radius: {res['radius']:.2f}mm, RMSE: {rmse:.3f}mm)"
            if view is not None and res.get('radius_encoder') is not None:
                title += f"\nencoder-angle fit (bracket uses): r {res['radius_encoder']:.2f}mm, RMSE {res['rmse']:.3f}mm"
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)

        # Plot results
        if is_v13:
            fig, axes = plt.subplots(2, 2, figsize=(16, 11))
            ax1, ax2 = axes[0, 0], axes[0, 1]
            ax3, ax4 = axes[1, 0], axes[1, 1]
            
            plot_single_axis(ax1, res_6, "6 (Wrist Roll)", 'blue')
            plot_single_axis(ax2, res_5, "5 (Wrist Pitch)", 'green')
            plot_single_axis(ax3, res_4, "4 (Wrist Yaw)", 'purple')
            
            ax4.axis('off')
            ang_45 = unified_res.get('ang_45', 90.0)
            ang_56 = unified_res.get('ang_56', 90.0)
            ang_46 = unified_res.get('ang_46', 90.0)
            ortho_status = "PASS (<0.1°)" if unified_res.get('ortho_err', 0.0) < 0.1 else "ALIGNED"
            
            summary_text = (
                f"=== UNIFIED 3-AXIS SPHERICAL WRIST SUMMARY ({arm_side.upper()} ARM) ===\n\n"
                f"1. Recommended Joint Offsets:\n"
                f"   * Joint 5 (Wrist Pitch): {unified_res.get('d5_opt_deg', 0.0):+.4f}°\n"
                f"   * Joint 6 (Wrist Roll) : {unified_res.get('d6_opt_deg', 0.0):+.4f}°\n\n"
                f"2. Marker Bracket Calibration:\n"
                f"   * Position (X, Y, Z)   : [{unified_res['x_e']:.2f}, {unified_res['y_e']:.2f}, {unified_res['z_e']:.2f}] mm\n"
                f"   * Orientation (R, P, Y): [{unified_res['roll_e']:.2f}°, {unified_res['pitch_e']:.2f}°, {unified_res['yaw_e']:.2f}°]\n\n"
                f"3. Quantitative Verification Metrics:\n"
                f"   * Orthogonality J4-J5  : {ang_45:.3f}° (Dev: {abs(ang_45-90.0):.3f}°)\n"
                f"   * Orthogonality J5-J6  : {ang_56:.3f}° (Dev: {abs(ang_56-90.0):.3f}°)\n"
                f"   * Orthogonality J4-J6  : {ang_46:.3f}° (Dev: {abs(ang_46-90.0):.3f}°)\n"
                f"   * Max Radius Residual  : {unified_res.get('max_radius_err', 0.0):.3f} mm\n"
                f"   * Alignment Status     : {ortho_status}\n"
            )
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.95, edgecolor='#ced4da'))
            
            fig.suptitle(f"RB-Y1 v1.3 Spherical Wrist Simultaneous Calibration ({arm_side.upper()} Arm)", fontsize=14, fontweight='bold')
        elif res_4 is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            plot_single_axis(ax1, res_6, 6, 'blue')
            plot_single_axis(ax2, res_5, 5, 'green')
            plot_single_axis(ax3, res_4, 4, 'purple')
            fig.suptitle(f"Unified Marker Sweep Results ({arm_side.upper()} Arm)\n"
                         f"Y-Offset: {unified_res['y_e']:.2f} mm | Z-Offset: {unified_res['z_e']:.2f} mm\n"
                         f"Roll: {unified_res['roll_e']:.2f}° | Pitch: {unified_res['pitch_e']:.2f}° | Yaw: {unified_res['yaw_e']:.2f}°\n"
                         f"Opt d5: {unified_res.get('opt_delta_5', 0.0):.3f}° | Opt d6: {unified_res.get('opt_delta_6', 0.0):.3f}° | Min Radius: {unified_res.get('min_radius', 0.0):.2f} mm", fontsize=12, fontweight='bold')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plot_single_axis(ax1, res_6, 6, 'blue')
            plot_single_axis(ax2, res_5, 5, 'green')
            fig.suptitle(f"Unified Marker Sweep Results ({arm_side.upper()} Arm)\n"
                         f"Y-Offset: {unified_res['y_e']:.2f} mm | Z-Offset: {unified_res['z_e']:.2f} mm\n"
                         f"Roll: {unified_res['roll_e']:.2f}° | Pitch: {unified_res['pitch_e']:.2f}° | Yaw: {unified_res['yaw_e']:.2f}°", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        try:
            FileStorage.ensure_dir(os.path.dirname(save_path), exist_ok=True)
            ArtifactStorage.save_figure(save_path, dpi=150)
            return True
        except Exception as e:
            logging.warning(f"[generate_marker_plot] Failed to save plot: {e}")
            return False
        finally:
            plt.close()