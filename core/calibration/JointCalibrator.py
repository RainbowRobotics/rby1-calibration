import time
import logging
import os
import numpy as np
import rby1_sdk as rby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator
class DebugLogger:
    def __init__(self, original_log_callback, file_path):
        self.original_log_callback = original_log_callback
        self.file_path = file_path
        self.buffer = []
        
    def log(self, msg):
        self.buffer.append(msg)
        msg_upper = msg.upper()
        if (
            "[SAFETY WARNING]" in msg_upper or
            "[SUCCESS]" in msg_upper or
            "[ERROR]" in msg_upper or
            "[WARN]" in msg_upper or
            "[INFO]" in msg_upper or
            "[VALIDATION SWEEP]" in msg_upper or
            "[ITERATION" in msg_upper or
            "RECOMMENDED ABSOLUTE OFFSET" in msg_upper or
            "STEP CORRECTION" in msg_upper or
            "COMMENCING" in msg_upper or
            "SWEPT" in msg_upper or
            "SWEEP COMPLETE" in msg_upper or
            "STARTING" in msg_upper
        ):
            if self.original_log_callback:
                self.original_log_callback(msg)
                
    def save(self):
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n=== NEW ITERATION ===\n")
                f.write("\n".join(self.buffer) + "\n")
        except Exception:
            pass

class JointCalibrator(BaseCalibrator):
    def __init__(self, marker_st=None, robot=None):
        super().__init__(marker_st, robot)
        self.use_angle_based_fitting = True

    def perform_joint_calibration(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=20.0, use_angle_based_fitting=None, save_debug=False, pass_idx=1, pass1_res=None):
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        config_dir = os.path.abspath(os.path.dirname(__file__))
        from core.paths import CONFIG_PATHS
        result_txt_dir = CONFIG_PATHS["txt_dir"]
        os.makedirs(result_txt_dir, exist_ok=True)
        debug_file_path = os.path.join(result_txt_dir, f"joint_calib_debug_{arm_side}_{mode}.txt")
        
        # 처음 시작(pass 1)일 때는 덮어쓰기 위해 기존 파일들 삭제
        if pass_idx == 1:
            if os.path.exists(debug_file_path):
                try: os.remove(debug_file_path)
                except: pass
                
            jcfg = self.JOINT_CONFIGS.get(mode, {})
            for key_type, j_key in [("joint_A", "sweep_joint_A"), ("joint_B", "sweep_joint_B")]:
                axis = jcfg.get(j_key)
                if axis is not None:
                    fname = os.path.join(result_txt_dir, f"sweep_points_{arm_side}_{key_type}_axis_{axis}.txt")
                    if os.path.exists(fname):
                        try: os.remove(fname)
                        except: pass

        logger = DebugLogger(log_callback, debug_file_path)
        original_log = log_callback
        log_callback = logger.log

        try:
            self.last_staged_offset = None
            self.last_diff_angle = None
            if original_log:
                log_callback("\n" + "="*60)
                log_callback("   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE")
                log_callback(f"   Target Arm: {arm_side.upper()} | Joint Target: {mode.upper()}")
                log_callback("="*60 + "\n")
                
            # save_debug는 첫 번째 sweep(원본 데이터)에서만 저장
            _sweep_count = [0]
            def run_single_sweep(offset):
                _sweep_count[0] += 1
                do_save = save_debug and (_sweep_count[0] == 1)
                return self.perform_calibration_sweep_continuous(
                    arm_side, mode, log_callback=log_callback, status_callback=status_callback,
                    current_offset_deg=offset, sweep_duration=sweep_duration,
                    use_angle_based_fitting=use_angle_based_fitting, save_debug=do_save
                )
                
            max_iterations = 6
            staged_offset = current_offset_deg
            staged_offsets_history = [staged_offset]
            final_res = None
            first_res = None
            converged = False
            
            # Sign-reversal tracking state
            prev_error = None
            prev_step_correction = 0.0
            direction_multiplier = 1.0
            dynamic_damping = 1.0
            prev_step_correction = 0.0
            
            for i in range(1, max_iterations + 1):
                # Update self.joint_offsets with staged_offset for proper FK offset subtraction in this iteration
                jcfg = self.JOINT_CONFIGS.get(mode, {})
                offset_key = jcfg.get("offset_key")
                if offset_key:
                    if arm_side in self.joint_offsets:
                        self.joint_offsets[arm_side][offset_key] = staged_offset
                    else:
                        self.joint_offsets[offset_key] = staged_offset

                if getattr(self, 'stop_requested', False):
                    if log_callback: log_callback("[INFO] Joint calibration aborted due to stop request.")
                    return None
                    
                if log_callback:
                    log_callback(f"\n[ITERATION {i}/{max_iterations}] Sweeping physically with staged offset {staged_offset:.4f}°...")
                
                # Perform physical sweep (or simulated sweep in mock mode) at the current staged offset
                res = run_single_sweep(staged_offset)
                if not res:
                    if log_callback: log_callback(f"[ERROR] Iteration {i} sweep failed. Aborting calibration.")
                    return None
                
                if i == 1:
                    first_res = res
                final_res = res
        
                angle_error = res.get('angle_between_normals', 0.0)
                sign = res.get('sign', 1.0)
                
                # wrist_roll_v13 has perpendicular axes (target 90 deg), other modes have parallel axes (target 0 deg)
                if mode == "wrist_roll_v13":
                    angle_dev = abs(angle_error - 90.0)
                    center_dist = res.get('perp_dist_after', 999.0)
                elif mode == "wrist_pitch_v13":
                    angle_dev = angle_error
                    center_dist = res.get('perp_dist_after', 999.0)
                else:
                    angle_dev = angle_error
                    center_dist = res.get('center_dist', 999.0)
                    
                r_A = res.get('r_A', 0.0)
                r_B = res.get('r_B', 0.0)
                size_error = abs(r_A - r_B)
                current_error = max(size_error, center_dist)
                
                # Print iteration summary
                if log_callback:
                    if mode in ("wrist_roll_v13", "wrist_pitch_v13"):
                        log_callback(f"  * Angle Error (Deviation)          : {angle_dev:.4f}°")
                        log_callback(f"  * Perpendicular Distance (After)   : {center_dist:.4f} mm")
                        log_callback(f"  * Perpendicular Distance (Before)  : {res.get('perp_dist_before', 999.0):.4f} mm")
                    else:
                        log_callback(f"  * Angle Error (Deviation)     : {angle_error:.4f}°")
                        if mode == "wrist_pitch_v13":
                            log_callback(f"  * Forearm Length (Center Dist): {center_dist:.4f} mm")
                            log_callback(f"  * Radii Difference (r3 - r5)  : {size_error:.4f} mm")
                        else:
                            log_callback(f"  * Circle Size Error (r_A-r_B) : {size_error:.4f} mm")
                            log_callback(f"  * Center Distance Error       : {center_dist:.4f} mm")
                            log_callback(f"  * Max Fitting Error Metric    : {current_error:.4f} mm")
                
                # Use the pre-calculated damped optimal offset correction to ensure convergence
                raw_optimal_offset = res.get('optimal_offset', 0.0)
                
                # Dynamic damping: halve the damping factor if the correction direction flips
                # This squashes noise-floor oscillations rapidly
                if i > 1 and raw_optimal_offset * prev_step_correction < 0:
                    dynamic_damping *= 0.7
                elif i == 1:
                    dynamic_damping = 1.0
                    
                step_correction = direction_multiplier * raw_optimal_offset * dynamic_damping
                
                # Calculate relative step delta for convergence check
                if mode in ("wrist_roll_v13", "wrist_yaw2"):
                    step_correction_delta = step_correction - staged_offset
                else:
                    step_correction_delta = step_correction

                # Convergence check:
                # step correction delta < 0.1° to handle bracket RPY noise
                converged_criteria = (abs(step_correction_delta) < 0.1)
                
                if converged_criteria:
                    converged = True
                    if log_callback:
                        log_callback(f"\n[SUCCESS] Calibration CONVERGED successfully:")
                        log_callback(f"  * Step Correction: {step_correction_delta:.4f}° < 0.1° (reached resolution limit)")
                        log_callback(f"  * Recommended Absolute Offset: {staged_offset:.4f}°")
                    break
                
                # Normal update: apply correction
                prev_error = angle_dev
                prev_step_correction = step_correction_delta
                if mode in ("wrist_roll_v13", "wrist_yaw2"):
                    # J6 modes return absolute recommended offset, not relative steps.
                    # Update staged_offset directly with the absolute value.
                    staged_offset = step_correction
                else:
                    staged_offset += step_correction
                
                # Safety: clamp staged_offset to the joint's configured offset range
                jcfg = self.JOINT_CONFIGS.get(mode, {})
                off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
                if staged_offset < off_min or staged_offset > off_max:
                    if log_callback:
                        log_callback(f"  [SAFETY WARNING] Staged offset {staged_offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                    staged_offset = float(np.clip(staged_offset, off_min, off_max))
                    
                staged_offsets_history.append(staged_offset)
                if log_callback:
                    log_callback(f"  * Updated Absolute Offset     : {staged_offset:.4f}°")
                    
            # Damping fallback for oscillation/noise-floor:
            if not converged and len(staged_offsets_history) >= 3:
                avg_offset = float(np.mean(staged_offsets_history[-3:]))
                if log_callback:
                    log_callback(f"\n[INFO] Joint {mode} did not meet 0.1° convergence tolerance due to measurement noise floor.")
                    log_callback(f"       Damping fallback: Averaged last 3 offsets ({', '.join(f'{v:.4f}°' for v in staged_offsets_history[-3:])}) -> {avg_offset:.4f}°")
                staged_offset = avg_offset

            # Final range safety: clamp to configured offset_range
            jcfg = self.JOINT_CONFIGS.get(mode, {})
            off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
            if staged_offset < off_min or staged_offset > off_max:
                if log_callback:
                    log_callback(f"  [SAFETY WARNING] Recommended final offset {staged_offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                staged_offset = float(np.clip(staged_offset, off_min, off_max))
        
            if getattr(self, 'stop_requested', False):
                if log_callback: log_callback("[INFO] Joint calibration aborted before final report.")
                return None
        
            # Build clean final output dict — UI only needs these fields
            final_output = {
                'mode': mode,
                'recommended_joint_offset': staged_offset,
                'optimal_offset': staged_offset,
                'converged': converged,
                'perp_dist_before': final_res.get('perp_dist_before', float('nan')) if final_res else float('nan'),
                'perp_dist_after': final_res.get('perp_dist_after', float('nan')) if final_res else float('nan'),
                'axial_offset_mm': final_res.get('axial_offset_mm', float('nan')) if final_res else float('nan'),
                'lateral_offset_mm': final_res.get('lateral_offset_mm', float('nan')) if final_res else float('nan'),
                'r_A': final_res.get('r_A', float('nan')) if final_res else float('nan'),
                'r_B': final_res.get('r_B', float('nan')) if final_res else float('nan'),
            }
        
            # Save first_res and final_res inside final_output so that the caller (FullAutoWorker) can retrieve them
            final_output['first_res'] = first_res
            final_output['final_res'] = final_res

            # Plot generation logic
            validation_res = final_res
            if validation_res and (first_res or pass1_res):
                if pass_idx == 2 and pass1_res is not None:
                    # True cross-pass BEFORE (Pass 1 start) vs AFTER (Pass 2 validation) comparison plot
                    first_res_for_plot = pass1_res.get('first_res', first_res)
                    plot_path = self.save_calibration_comparison_plot(
                        arm_side, mode, first_res_for_plot, validation_res, 
                        log_callback=log_callback, force_overwrite=True
                    )
                else:
                    # In Pass 1 or manual mode, save a comparison plot of the current pass.
                    # In Pass 1, this will be overwritten later when Pass 2 completes.
                    plot_path = self.save_calibration_comparison_plot(
                        arm_side, mode, first_res, validation_res, 
                        log_callback=log_callback, force_overwrite=True
                    )
                final_output['plot_path_combined'] = plot_path
            
            return final_output
        finally:
            logger.save()




    def save_debug_orthogonal_plot(self, arm_side, frame, dataset_A, dataset_B, dyn_model, T_mount_to_cam, optimal_offset_rad, ee_name, arm_idx, cand_joint, angle_error_deg=None, log_callback=None):
        return
        try:
            pts_a = []
            pts_b = []
            
            # Project points depending on frame
            for q_full, pose in dataset_A:
                p_cam = pose[:3, 3]
                if frame == "camera":
                    pts_a.append(p_cam * 1000.0)
                elif frame == "torso":
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    pts_a.append(p_meas_t5 * 1000.0)
                elif frame == "ee":
                    q_mod = np.array(q_full)
                    q_mod[arm_idx[cand_joint]] += optimal_offset_rad
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                    pts_a.append(p_ee * 1000.0)

            for q_full, pose in dataset_B:
                p_cam = pose[:3, 3]
                if frame == "camera":
                    pts_b.append(p_cam * 1000.0)
                elif frame == "torso":
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    pts_b.append(p_meas_t5 * 1000.0)
                elif frame == "ee":
                    q_mod = np.array(q_full)
                    q_mod[arm_idx[cand_joint]] += optimal_offset_rad
                    T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_mod, ee_name)
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q_full, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                    p_meas_t5 = T_t5_to_cam[:3, :3] @ p_cam + T_t5_to_cam[:3, 3]
                    p_ee = T_t5_to_ee[:3, :3].T @ (p_meas_t5 - T_t5_to_ee[:3, 3])
                    pts_b.append(p_ee * 1000.0)

            pts_a = np.array(pts_a)
            pts_b = np.array(pts_b)
            
            # 3D fit circles
            c_A, R_c_A, r_A, rmse_A, pts_2d_A, uc_A, vc_A = BaseCalibrator.fit_circle_3d(pts_a, robust=not self.is_mock)
            c_B, R_c_B, r_B, rmse_B, pts_2d_B, uc_B, vc_B = BaseCalibrator.fit_circle_3d(pts_b, robust=not self.is_mock)
            
            n_A = R_c_A[:, 2]
            n_B = R_c_B[:, 2]
            u_A = R_c_A[:, 0]
            v_A = R_c_A[:, 1]
            u_B = R_c_B[:, 0]
            v_B = R_c_B[:, 1]

            angle_between_normals = np.degrees(np.arccos(np.clip(abs(np.dot(n_A, n_B)), -1.0, 1.0)))
            diff_centers = c_B - c_A
            center_dist = np.linalg.norm(diff_centers - np.dot(diff_centers, n_A) * n_A)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            def generate_circle_pts(center, normal, radius, u, v, num_points=100):
                theta = np.linspace(0, 2*np.pi, num_points)
                circle_pts = []
                for t in theta:
                    p = center + radius * (np.cos(t) * u + np.sin(t) * v)
                    circle_pts.append(p)
                return np.array(circle_pts)

            circle_pts_a = generate_circle_pts(c_A, n_A, r_A, u_A, v_A)
            circle_pts_b = generate_circle_pts(c_B, n_B, r_B, u_B, v_B)
            
            # Subplot 1 (Top-Left): Sweep A Circle Fit (2D plane projection)
            theta_fit = np.linspace(0, 2*np.pi, 200)
            fit_x = uc_A + r_A * np.cos(theta_fit)
            fit_y = vc_A + r_A * np.sin(theta_fit)
            axes[0, 0].scatter(pts_2d_A[:, 0], pts_2d_A[:, 1], c='red', s=15, alpha=0.6, label='Raw Points')
            axes[0, 0].plot(fit_x, fit_y, 'r--', linewidth=2, label=f'Fit Circle (r={r_A:.1f}mm)')
            axes[0, 0].scatter([uc_A], [vc_A], c='darkred', marker='X', s=80, label='Center')
            axes[0, 0].set_xlabel('U (mm)')
            axes[0, 0].set_ylabel('V (mm)')
            axes[0, 0].set_title(f'Sweep A Local 2D Circle Fit (RMSE: {rmse_A:.4f} mm)')
            axes[0, 0].set_aspect('equal')
            axes[0, 0].grid(True)
            axes[0, 0].legend()
            
            # Subplot 2 (Top-Right): Sweep B Circle Fit (2D plane projection)
            fit_x_b = uc_B + r_B * np.cos(theta_fit)
            fit_y_b = vc_B + r_B * np.sin(theta_fit)
            axes[0, 1].scatter(pts_2d_B[:, 0], pts_2d_B[:, 1], c='blue', s=15, alpha=0.6, label='Raw Points')
            axes[0, 1].plot(fit_x_b, fit_y_b, 'b--', linewidth=2, label=f'Fit Circle (r={r_B:.1f}mm)')
            axes[0, 1].scatter([uc_B], [vc_B], c='darkblue', marker='X', s=80, label='Center')
            axes[0, 1].set_xlabel('U (mm)')
            axes[0, 1].set_ylabel('V (mm)')
            axes[0, 1].set_title(f'Sweep B Local 2D Circle Fit (RMSE: {rmse_B:.4f} mm)')
            axes[0, 1].set_aspect('equal')
            axes[0, 1].grid(True)
            axes[0, 1].legend()

            # Subplot 3 (Bottom-Left): Comparison Top View (X-Y Projection)
            axes[1, 0].scatter(pts_a[:, 0], pts_a[:, 1], c='red', s=15, alpha=0.5, label='Sweep A Raw')
            axes[1, 0].scatter(pts_b[:, 0], pts_b[:, 1], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
            axes[1, 0].plot(circle_pts_a[:, 0], circle_pts_a[:, 1], 'r-', linewidth=1.5, label='Sweep A Fit')
            axes[1, 0].plot(circle_pts_b[:, 0], circle_pts_b[:, 1], 'b-', linewidth=1.5, label='Sweep B Fit')
            axes[1, 0].scatter([c_A[0]], [c_A[1]], c='darkred', marker='X', s=100, label='Center A')
            axes[1, 0].scatter([c_B[0]], [c_B[1]], c='darkblue', marker='X', s=100, label='Center B')
            axes[1, 0].plot([c_A[0], c_B[0]], [c_A[1], c_B[1]], color='purple', linestyle=':', linewidth=2, label='Center Shift')
            axes[1, 0].set_xlabel('X (mm)')
            axes[1, 0].set_ylabel('Y (mm)')
            axes[1, 0].set_title('Top View Comparison (X-Y Projection)')
            axes[1, 0].set_aspect('equal')
            axes[1, 0].grid(True)
            axes[1, 0].legend()

            # Subplot 4 (Bottom-Right): Comparison Side View (Y-Z Projection)
            axes[1, 1].scatter(pts_a[:, 1], pts_a[:, 2], c='red', s=15, alpha=0.5, label='Sweep A Raw')
            axes[1, 1].scatter(pts_b[:, 1], pts_b[:, 2], c='blue', s=15, alpha=0.5, label='Sweep B Raw')
            axes[1, 1].plot(circle_pts_a[:, 1], circle_pts_a[:, 2], 'r-', linewidth=1.5, label='Sweep A Fit')
            axes[1, 1].plot(circle_pts_b[:, 1], circle_pts_b[:, 2], 'b-', linewidth=1.5, label='Sweep B Fit')
            axes[1, 1].scatter([c_A[1]], [c_A[2]], c='darkred', marker='X', s=100, label='Center A')
            axes[1, 1].scatter([c_B[1]], [c_B[2]], c='darkblue', marker='X', s=100, label='Center B')
            
            # Normal Vectors Projection
            scale = min(r_A, r_B) * 0.4
            axes[1, 1].arrow(c_A[1], c_A[2], n_A[1]*scale, n_A[2]*scale, color='darkred', head_width=2, width=0.5, label='Normal A')
            axes[1, 1].arrow(c_B[1], c_B[2], n_B[1]*scale, n_B[2]*scale, color='darkblue', head_width=2, width=0.5, label='Normal B')
            axes[1, 1].set_xlabel('Y (mm)')
            axes[1, 1].set_ylabel('Z (mm)')
            axes[1, 1].set_title('Side View Comparison (Y-Z Projection)')
            axes[1, 1].set_aspect('equal')
            axes[1, 1].grid(True)
            axes[1, 1].legend()

            display_angle = angle_error_deg if angle_error_deg is not None else angle_between_normals
            status_text = "PASS" if (display_angle < 0.1 and center_dist < 0.1) else "WARNING"
            fig.suptitle(
                f"Orthogonal Multi-View Analysis ({arm_side.upper()} Arm, {frame.upper()} Frame)\n"
                f"Status: {status_text} | Axis Angle Error: {display_angle:.4f}° (Target < 0.1°)\n"
                f"Axis Center Distance: {center_dist:.4f} mm (Target < 0.1 mm)",
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            
            from core.paths import CONFIG_PATHS
            result_dir = CONFIG_PATHS["plot_dir"]
            os.makedirs(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"debug_orthogonal_circles_{arm_side}_{frame}.png"))
            plt.savefig(plot_save_path, dpi=150)
            plt.close()
            if log_callback:
                log_callback(f"[SUCCESS] Orthogonal debug plot saved to: {plot_save_path}")
                log_callback(f"  * Alignment check: {status_text} (Angle error = {display_angle:.4f}°, Center distance = {center_dist:.4f} mm)")
        except Exception as e:
            if log_callback:
                log_callback(f"[WARN] Failed to save orthogonal debug plot for {frame}: {e}")

    def save_calibration_comparison_plot(self, arm_side, mode, first_res, final_res, log_callback=None, force_overwrite=False):
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            def plot_column(res, col_idx, stage_name):
                # Read from internal _plot_data bundle
                pd = res.get('_plot_data', {})
                pts_a      = pd.get('pts_a_cam')
                pts_b      = pd.get('pts_b_cam')
                c_A        = pd.get('c_A')
                c_B        = pd.get('c_B')
                n_A        = pd.get('n_A')
                n_B        = pd.get('n_B')
                r_A        = pd.get('r_A', res.get('r_A', 1.0))
                r_B        = pd.get('r_B', res.get('r_B', 1.0))
                angle_error  = pd.get('angle_between_normals', res.get('angle_between_normals', 0.0))
                center_dist  = pd.get('center_dist', res.get('center_dist', 0.0))

                if pts_a is None or c_A is None or n_A is None:
                    # No plot data available for this sweep — leave panel blank
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

            plot_column(first_res, 0, "BEFORE")
            plot_column(final_res, 1, "AFTER")

            before_dist_str = ""
            after_dist_str = ""
            if mode == "wrist_pitch_v13":
                first_pd = first_res.get('_plot_data', {})
                final_pd = final_res.get('_plot_data', {})
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

            first_pd = first_res.get('_plot_data', first_res)
            final_pd = final_res.get('_plot_data', final_res)
            fig.suptitle(
                f"Joint Calibration: {arm_side.upper()} Arm - {mode.upper()}\n"
                f"Before: Angle Dev = {first_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {first_pd.get('center_dist', 0.0):.2f} mm{before_dist_str}\n"
                f"After : Angle Dev = {final_pd.get('angle_between_normals', 0.0):.3f}°, Center Dist = {final_pd.get('center_dist', 0.0):.2f} mm{after_dist_str}",
                fontsize=16, fontweight='bold'
            )
            plt.tight_layout()

            camera_ws_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            result_dir = os.path.join(camera_ws_dir, "result", "result_img")
            os.makedirs(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"circle_fit_{arm_side}_{mode}_joint_calib.png"))
            if not force_overwrite and os.path.exists(plot_save_path):
                plt.close()
                if log_callback:
                    log_callback(f"[INFO] Comparison plot already exists at: {plot_save_path}, skipping overwrite.")
            else:
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

    def perform_calibration_sweep_continuous(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=20.0, use_angle_based_fitting=None, save_debug=False):
        if getattr(self, 'stop_requested', False):
            return None

        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   STARTING {mode.upper()} CONTINUOUS OFFSET CALIBRATION SWEEP")
            if current_offset_deg != 0.0:
                log_callback(f"   [Baseline Shift (Current Applied Offset): {current_offset_deg:.4f}°]")
            log_callback("="*50)

        is_camera_mock = (self.marker_st is None or type(self.marker_st).__name__ == "SimulatedMarkerTransform")

        if not is_camera_mock:
            # Pre-check marker visibility
            initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not initial_check:
                if log_callback: log_callback("[ERROR] Marker is not visible.")
                if status_callback: status_callback(False)
                return None
            if status_callback: status_callback(True)
        else:
            if status_callback: status_callback(True)

        if getattr(self, 'stop_requested', False):
            return None

        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        initial_joint_pos = list(state.position[arm_idx])

        # Define joint parameters from JOINT_CONFIGS
        jcfg = self.JOINT_CONFIGS[mode]
        cand_joint = jcfg["cand_joint"]
        sweep_joint_A = jcfg["sweep_joint_A"]
        sweep_joint_B = jcfg["sweep_joint_B"]

        dyn_model = self.robot.get_dynamics()
        ee_name = f"ee_{arm_side}"

        # Arm cand baseline pose (shifted by current offset)
        ver_key = "v1.3" if self.is_v13() else "v1.2"
        ready_pose_nom = self.get_ready_pose(ver_key, "joint", mode, arm_side)
        q_cand = list(initial_joint_pos)
        q_cand[cand_joint] = ready_pose_nom[cand_joint] + np.radians(current_offset_deg)

        # Determine sweep ranges from JOINT_CONFIGS
        range_A = jcfg.get("sweep_range_A", 20.0)
        range_B = jcfg.get("sweep_range_B", 20.0)

        # 1. PHYSICAL SWEEP JOINT A
        logging.info(f"\n--- Commencing Continuous Sweep on Joint A (Index {sweep_joint_A}, duration={sweep_duration}s) ---")
        dataset_A = self.perform_single_joint_sweep(
            arm_side, sweep_joint_A, q_cand, -range_A, range_A, sweep_duration,
            q_head=None, label="Joint A", log_callback=log_callback,
            current_offset_deg=current_offset_deg, cand_joint=cand_joint
        )
        if dataset_A is None:
            return None

        if getattr(self, 'stop_requested', False):
            return None
            
        if self.robot:
            time.sleep(1.0)
        else:
            time.sleep(0.01)

        # 2. PHYSICAL SWEEP JOINT B
        logging.info(f"\n--- Commencing Continuous Sweep on Joint B (Index {sweep_joint_B}, duration={sweep_duration}s) ---")
        dataset_B = self.perform_single_joint_sweep(
            arm_side, sweep_joint_B, q_cand, -range_B, range_B, sweep_duration,
            q_head=None, label="Joint B", log_callback=log_callback,
            current_offset_deg=current_offset_deg, cand_joint=cand_joint
        )
        if dataset_B is None:
            return None

        if getattr(self, 'stop_requested', False):
            return None

        # Return arm to original ready pose (head=None)
        logging.info("[INFO] Sweep finished. Returning arm to initial pose...")
        if arm_side == "left":
            ok = self.movej(self.robot, left_arm=initial_joint_pos, head=None, minimum_time=2.5, apply_offsets=False)
        else:
            ok = self.movej(self.robot, right_arm=initial_joint_pos, head=None, minimum_time=2.5, apply_offsets=False)

        if not ok or getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[ERROR] Failed to return arm to initial pose or stop was requested.")
            return None

        # Save FULL captured continuous sweep points to debug txt files before downsampling
        if save_debug:
            self.save_debug_points(
                arm_side, sweep_joint_A, dataset_A, initial_joint_pos, ee_name, dyn_model, None, "joint_A", log_callback
            )
            self.save_debug_points(
                arm_side, sweep_joint_B, dataset_B, initial_joint_pos, ee_name, dyn_model, None, "joint_B", log_callback
            )
        
        # Keep up to 200 points for speed and accuracy
        raw_len_A = len(dataset_A)
        raw_len_B = len(dataset_B)
        
        max_pts = 200
        if len(dataset_A) > max_pts:
            indices_A = np.round(np.linspace(0, len(dataset_A) - 1, max_pts)).astype(int)
            dataset_A = [dataset_A[idx] for idx in indices_A]
        if len(dataset_B) > max_pts:
            indices_B = np.round(np.linspace(0, len(dataset_B) - 1, max_pts)).astype(int)
            dataset_B = [dataset_B[idx] for idx in indices_B]
            
        logging.info(f"Swept {raw_len_A} dense raw coordinate frames during Joint A motion... downsampled to {len(dataset_A)} for optimization.")
        logging.info(f"Swept {raw_len_B} dense raw coordinate frames during Joint B motion... downsampled to {len(dataset_B)} for optimization.")

        return self.compute_calibration_results(
            arm_side=arm_side,
            mode=mode,
            dataset_A=dataset_A,
            dataset_B=dataset_B,
            initial_joint_pos=initial_joint_pos,
            current_offset_deg=current_offset_deg,
            use_angle_based_fitting=use_angle_based_fitting,
            save_debug=save_debug,
            log_callback=log_callback,
            cand_joint=cand_joint,
            sweep_joint_A=sweep_joint_A,
            sweep_joint_B=sweep_joint_B
        )

    def compute_calibration_results(self, arm_side, mode, dataset_A, dataset_B, initial_joint_pos, current_offset_deg=0.0, use_angle_based_fitting=None, save_debug=False, log_callback=None, cand_joint=None, sweep_joint_A=None, sweep_joint_B=None):
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        # Define nominal axes in parent link frame for each mode
        if mode == "wrist_roll_v13":
            a_cand_local = np.array([1.0, 0.0, 0.0])
            a_A_local = np.array([1.0, 0.0, 0.0])
            a_B_local = np.array([0.0, 1.0, 0.0])
        elif mode == "wrist_pitch_v13":
            a_cand_local = np.array([0.0, 1.0, 0.0])
            a_A_local = np.array([0.0, 1.0, 0.0])
            a_B_local = np.array([0.0, 1.0, 0.0])
        elif mode in ("wrist_pitch", "elbow"):
            a_cand_local = np.array([0.0, 1.0, 0.0])
            a_A_local = np.array([0.0, 0.0, 1.0])
            a_B_local = np.array([0.0, 0.0, 1.0])
        elif mode == "wrist_yaw2":
            a_cand_local = np.array([0.0, 0.0, 1.0])
            a_A_local = np.array([0.0, 0.0, 1.0])
            a_B_local = np.array([0.0, 1.0, 0.0])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Get joint index configs
        jcfg = self.JOINT_CONFIGS[mode]
        cand_joint = jcfg["cand_joint"]
        sweep_joint_A = jcfg["sweep_joint_A"]
        sweep_joint_B = jcfg["sweep_joint_B"]

        if not self.robot:
            raise RuntimeError("Robot instance is not initialized")
            
        state = self.robot.get_state()
        model = self.robot.model()
        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
        dyn_model = self.robot.get_dynamics()
        ee_name = f"ee_{arm_side}"

        # Compute dynamic nominal axes using forward kinematics (FK) at the ready pose
        # Determine nominal ready pose for the arm (zero offsets)
        ver_key = "v1.3" if self.is_v13() else "v1.2"
        ready_pose_nom = self.get_ready_pose(ver_key, "joint", mode, arm_side)
        
        q_ready_full = np.array(state.position)
        for idx, val in zip(arm_idx, ready_pose_nom):
            q_ready_full[idx] = val

        def get_link_name(j_idx):
            return f"link_{arm_side}_arm_{j_idx}"

        T_cand = self.compute_fk(self.robot, dyn_model, q_ready_full, get_link_name(cand_joint - 1))
        T_A = self.compute_fk(self.robot, dyn_model, q_ready_full, get_link_name(sweep_joint_A - 1))
        T_B = self.compute_fk(self.robot, dyn_model, q_ready_full, get_link_name(sweep_joint_B - 1))

        a_cand_t5 = T_cand[:3, :3] @ a_cand_local
        a_A_t5 = T_A[:3, :3] @ a_A_local
        a_B_t5 = T_B[:3, :3] @ a_B_local
        
        logging.debug(f"[INFO] Dynamically calculated nominal axes from FK (Arm: {arm_side}, Mode: {mode}):")
        logging.debug(f"       a_cand_t5 = {a_cand_t5.tolist()}")
        logging.debug(f"       a_A_t5    = {a_A_t5.tolist()}")
        logging.debug(f"       a_B_t5    = {a_B_t5.tolist()}")

        # Use nominal fixed camera rotation relative to torso (ZYX [-90, 0, -90])
        # to avoid using uncalibrated mount_to_cam values or head kinematics.
        R_rob_to_cam = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()

        # Define nominal axes in the camera frame using transpose of R_rob_to_cam (since R_rob_to_cam is R_cam_to_torso)
        a_cand_cam = R_rob_to_cam.T @ a_cand_t5
        a_A_cam = R_rob_to_cam.T @ a_A_t5
        a_B_cam_nom = R_rob_to_cam.T @ a_B_t5

        a_cand_cam /= np.linalg.norm(a_cand_cam)
        a_A_cam /= np.linalg.norm(a_A_cam)
        a_B_cam_nom /= np.linalg.norm(a_B_cam_nom)

        # 2. Extract poses and angles in the camera frame
        poses_A = [pose for _, pose in dataset_A]
        angles_A = [np.degrees(q_full[arm_idx[sweep_joint_A]] - initial_joint_pos[sweep_joint_A]) for q_full, _ in dataset_A]
        poses_B = [pose for _, pose in dataset_B]
        angles_B = [np.degrees(q_full[arm_idx[sweep_joint_B]] - initial_joint_pos[sweep_joint_B]) for q_full, _ in dataset_B]

        # 3. Fit Sweep A and B axes in the camera frame
        robust_fit = not self.is_mock
        res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_A, angles_A, axis_prior=a_A_cam, robust=robust_fit)
        res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_B, angles_B, axis_prior=a_B_cam_nom, robust=robust_fit)

        n_A = res_A['axis_opt']
        n_B = res_B['axis_opt']
        if np.dot(n_A, n_B) < 0:
            n_B = -n_B

        r_A = res_A['radius']
        r_B = res_B['radius']
        rmse_A = res_A['rmse']
        rmse_B = res_B['rmse']
        c_A_c = res_A['c_opt']
        c_B_c = res_B['c_opt']

        pts_a_cam = np.array([pose[:3, 3] * 1000.0 for _, pose in dataset_A])
        pts_b_cam = np.array([pose[:3, 3] * 1000.0 for _, pose in dataset_B])

        # Compute center distance in camera frame
        diff_centers = c_B_c - c_A_c
        center_dist = np.linalg.norm(diff_centers - np.dot(diff_centers, n_A) * n_A)
        angle_between_normals = np.degrees(np.arccos(np.clip(np.dot(n_A, n_B), -1.0, 1.0)))

        # Enforce that normal vectors point in the direction of the physical kinematic axes
        n_A = n_A if np.dot(n_A, a_A_cam) > 0 else -n_A
        n_B = n_B if np.dot(n_B, a_B_cam_nom) > 0 else -n_B

        # Project nominal and actual axes onto the plane perpendicular to the candidate joint axis
        if mode == "wrist_pitch_v13":
            try:
                # Phase-based fitting for parallel axes (J5 vs J3)
                mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                T_mount_to_cam = self.make_transform(mount_to_cam)
                
                q_first, T_cam_to_marker_first = dataset_A[0]
                T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_first, "link_head_2", "link_torso_5")
                T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                T_torso_to_cam = np.linalg.inv(T_t5_to_cam)
                
                ver_key = "v1.3" if self.is_v13() else "v1.2"
                key = f"Tf_to_marker_{arm_side}"
                if self.camera_config and key in self.camera_config:
                    bracket_vec = self.camera_config[key]
                else:
                    bracket_vec = self.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
                T_ee_to_marker = self.make_transform(bracket_vec)
                
                arm_idx = self.robot.model().left_arm_idx if arm_side == "left" else self.robot.model().right_arm_idx
                
                angles = []
                for q_full, T_cam_to_marker in dataset_A:
                    q_nom = np.array(q_full)
                    if hasattr(self, 'joint_offsets') and self.joint_offsets:
                        offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                        q_nom[arm_idx[3]] -= np.radians(offsets.get("elbow", 0.0))
                        q_nom[arm_idx[5]] -= np.radians(offsets.get("wrist_pitch", 0.0))
                        q_nom[arm_idx[6]] -= np.radians(offsets.get("wrist_roll", 0.0))
                    
                    T_torso_to_ee_nom = self.compute_fk(self.robot, dyn_model, q_nom, ee_name, "link_torso_5")
                    T_cam_to_marker_nom = T_torso_to_cam @ T_torso_to_ee_nom @ T_ee_to_marker
                    p_nom = T_cam_to_marker_nom[:3, 3]
                    p_meas = T_cam_to_marker[:3, 3]
                    
                    v_nom = p_nom - c_A_c / 1000.0
                    v_nom = v_nom - np.dot(v_nom, n_A) * n_A
                    v_nom /= np.linalg.norm(v_nom)
                    
                    v_meas = p_meas - c_A_c / 1000.0
                    v_meas = v_meas - np.dot(v_meas, n_A) * n_A
                    v_meas /= np.linalg.norm(v_meas)
                    
                    ang = np.arctan2(np.dot(np.cross(v_nom, v_meas), n_A), np.dot(v_nom, v_meas))
                    angles.append(ang)
                    
                # We return negative of the mean to represent the compensation offset
                optimal_offset_deg = float(np.degrees(np.mean(angles)))
            except Exception as e:
                import traceback
                if log_callback:
                    log_callback(f"[WARN] Phase-based fitting failed: {e}\n{traceback.format_exc()}. Using 0.0.")
                optimal_offset_deg = 0.0
            
            diff_angle = np.radians(optimal_offset_deg)
        elif mode in ("wrist_roll_v13", "wrist_yaw2"):
            try:
                from core.calibration.MarkerCalibrator import MarkerCalibrator
                mc = MarkerCalibrator(self.marker_st, self.robot)
                mc.camera_config = self.camera_config
                mc.robot_version = self.robot_version
                mc.joint_offsets = self.joint_offsets
                
                res_5 = {
                    'captured_q_full': [q_full for q_full, _ in dataset_B],
                    'captured_poses': [pose for _, pose in dataset_B],
                    'axis_opt': n_B,
                    'radius': r_B,
                }
                res_6 = {
                    'captured_q_full': [q_full for q_full, _ in dataset_A],
                    'captured_poses': [pose for _, pose in dataset_A],
                    'axis_opt': n_A,
                    'radius': r_A,
                }
                
                unified_res = mc.compute_unified_bracket_calibration(
                    res_5, res_6, arm_side, marker_data_4=None,
                    calib_roll_or_yaw_deg=None, calib_pitch_deg=0.0,
                    lock_bracket=True
                )
                if mode in ("wrist_roll_v13", "wrist_yaw2"):
                    # Coordinate-free vector projection method (ultimate solution)
                    # n6_marker_actual is J6 axis (normal to the plane of J6 rotation)
                    # n5_marker_actual is J5 axis (vector that rotates around J6 axis)
                    # y_ee_m_ideal is the nominal J5 axis in the marker frame
                    z_axis = unified_res['n6_marker_actual']
                    n5_act = unified_res['n5_marker_actual']
                    ref_y = unified_res['y_ee_m_ideal']
                    
                    # Project J5 axis onto the plane perpendicular to J6 axis
                    n5_proj = n5_act - np.dot(n5_act, z_axis) * z_axis
                    n5_proj /= np.linalg.norm(n5_proj)
                    
                    ref_x = np.cross(z_axis, ref_y)
                    ref_x /= np.linalg.norm(ref_x)
                    
                    # Calculate angle around z_axis from ref_y to n5_proj
                    diff_angle = np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y))
                    raw_diff_deg = np.degrees(diff_angle)
                    
                    # Compensate for the initial ready pose angle of J7 (index 6)
                    # We must take the J7 angle from dataset_B (the J5 sweep), because during dataset_A (the J6 sweep), J7 is moving.
                    q_full_B_first = dataset_B[0][0]
                    j7_ready_pose_deg = np.degrees(q_full_B_first[arm_idx[6]])
                    
                    # Due to kinematics, the apparent marker rotation is the negative of the actual J7 rotation:
                    # raw_diff_deg = - (j7_ready_pose_deg + physical_offset)
                    # physical_offset = - raw_diff_deg - j7_ready_pose_deg
                    # We must return the compensation offset (correction), which is the negative of the physical offset:
                    optimal_offset_deg = raw_diff_deg + j7_ready_pose_deg
                    
                    if log_callback:
                        log_callback(f"[INFO] {mode}: J7 ready pose={j7_ready_pose_deg:.2f}°, raw_diff={raw_diff_deg:.2f}°, optimal_offset={optimal_offset_deg:.2f}°")
            except Exception as e:
                import traceback
                if log_callback:
                    log_callback(f"[WARN] MarkerCalibrator fallback failed: {e}\n{traceback.format_exc()}. Using 0.0.")
                optimal_offset_deg = 0.0
                diff_angle = 0.0
        else:
            # Project axes onto the candidate joint's rotation plane to absorb physical DH twist errors
            # This ensures smooth zero-crossing even if the sweep axes are not perfectly parallel.
            a_A_proj = a_A_cam - np.dot(a_A_cam, a_cand_cam) * a_cand_cam
            a_B_proj = a_B_cam_nom - np.dot(a_B_cam_nom, a_cand_cam) * a_cand_cam
            if np.linalg.norm(a_A_proj) > 1e-6: a_A_proj /= np.linalg.norm(a_A_proj)
            if np.linalg.norm(a_B_proj) > 1e-6: a_B_proj /= np.linalg.norm(a_B_proj)
            
            n_A_proj = n_A - np.dot(n_A, a_cand_cam) * a_cand_cam
            n_B_proj = n_B - np.dot(n_B, a_cand_cam) * a_cand_cam
            if np.linalg.norm(n_A_proj) > 1e-6: n_A_proj /= np.linalg.norm(n_A_proj)
            if np.linalg.norm(n_B_proj) > 1e-6: n_B_proj /= np.linalg.norm(n_B_proj)
            
            nominal_angle = np.arctan2(np.dot(np.cross(a_A_proj, a_B_proj), a_cand_cam), np.dot(a_A_proj, a_B_proj))
            actual_angle = np.arctan2(np.dot(np.cross(n_A_proj, n_B_proj), a_cand_cam), np.dot(n_A_proj, n_B_proj))
            diff_angle = actual_angle - nominal_angle
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi


        # Match the physical motor driver rotations (negative feedback loop)
        sign = 1.0 if diff_angle > 0.0 else -1.0
        size_error = abs(r_A - r_B)
        if mode not in ("wrist_roll_v13", "wrist_pitch_v13", "wrist_yaw2") and (center_dist > 100.0 or size_error > 100.0):
            if log_callback:
                log_callback("[ERROR] Circle fitting failed or error is too large. Aborting step adjustment.")
            optimal_offset_deg = 0.0
        else:
            if mode in ("elbow", "wrist_pitch"):
                # Robust Center-Distance Method for parallel joints
                # The normal vector of a small arc is highly sensitive to vibrations.
                # However, the distance between the rotation centers is extremely robust and proportional to the angle error.
                mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                T_mount_to_cam = self.make_transform(mount_to_cam)
                q_init = dataset_A[0][0]
                T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_init, "link_head_2", "link_torso_5")
                T_torso_to_cam = np.linalg.inv(T_t5_to_head @ T_mount_to_cam)
                
                arm_side_str = "left" if arm_side == "left" else "right"
                
                # Get the true axis of the candidate joint using the exact same T_torso_to_cam frame
                T_cand_parent = self.compute_fk(self.robot, dyn_model, q_init, f"link_{arm_side_str}_arm_{cand_joint - 1}")
                a_cand_t5 = T_cand_parent[:3, :3] @ a_cand_local
                true_a_cand_cam = T_torso_to_cam[:3, :3] @ a_cand_t5
                if np.linalg.norm(true_a_cand_cam) > 1e-6:
                    true_a_cand_cam /= np.linalg.norm(true_a_cand_cam)
                
                # Get the true position of the pivot (child link, e.g. link_3 for elbow)
                T_cand_child = self.compute_fk(self.robot, dyn_model, q_init, f"link_{arm_side_str}_arm_{cand_joint}")
                p_cand_cam = (T_torso_to_cam @ T_cand_child)[:3, 3] * 1000.0  # mm
                
                vec_elbow_to_cA = c_A_c - p_cand_cam
                
                # Cross direction gives the expected displacement direction for a positive physical offset
                dir_vec = np.cross(true_a_cand_cam, vec_elbow_to_cA)
                L2 = np.sum(dir_vec**2)
                
                if L2 > 1.0:
                    proj = np.dot(diff_centers, dir_vec)
                    sin_theta = np.clip(proj / L2, -1.0, 1.0)
                    # We return the negative of the physical error as the compensation offset
                    optimal_offset_deg = -float(np.degrees(np.arcsin(sin_theta)))
                else:
                    optimal_offset_deg = 0.0
                    
            elif mode not in ("wrist_roll_v13", "wrist_yaw2"):
                optimal_offset_deg = -np.degrees(diff_angle)

        if log_callback:
            log_callback("\n" + "="*50)
            log_callback(f"   SWEEP ANALYSIS & RESULTS ({mode.upper()})")
            log_callback("="*50)
            log_callback(f"  * Camera Circle Normals Angle (Reference): {angle_between_normals:.4f} deg")
            log_callback(f"  * Circle Size Error (abs: r_A - r_B)     : {size_error:.4f} mm")
            log_callback(f"  * Estimated Circle Center Distance       : {center_dist:.4f} mm")
            log_callback(f"  * Calculated Offset Correction           : {optimal_offset_deg:.6f} deg")
            log_callback("="*50)

        if save_debug and dyn_model:
            try:
                mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                T_mount_to_cam = self.make_transform(mount_to_cam)
                self.save_debug_orthogonal_plot(
                    arm_side, "camera", dataset_A, dataset_B, dyn_model, T_mount_to_cam, 
                    np.radians(optimal_offset_deg), ee_name, arm_idx, cand_joint, 
                    angle_error_deg=angle_between_normals, log_callback=log_callback
                )
            except Exception as e:
                if log_callback:
                    log_callback(f"[WARN] Failed to save debug orthogonal plot: {e}")

        return {
            'mode': mode,
            'optimal_offset': optimal_offset_deg,
            'recommended_joint_offset': optimal_offset_deg,
            'converged': False,
            '_dataset_A': dataset_A,
            '_dataset_B': dataset_B,
            '_initial_joint_pos': initial_joint_pos,
            'angle_between_normals': angle_between_normals,
            'sign': sign,
            'center_dist': center_dist,
            'r_A': r_A,
            'r_B': r_B,
            '_plot_data': {
                'pts_a_cam': pts_a_cam,
                'pts_b_cam': pts_b_cam,
                'c_A': c_A_c,
                'c_B': c_B_c,
                'n_A': n_A,
                'n_B': n_B,
                'r_A': r_A,
                'r_B': r_B,
                'angle_between_normals': angle_between_normals,
                'center_dist': center_dist,
            },
        }
