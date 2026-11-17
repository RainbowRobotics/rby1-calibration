from core.storage import FileStorage, ArtifactStorage
import time
import logging
import os
import numpy as np
# import rby1_sdk as rby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from scipy.optimize import least_squares, minimize_scalar
# from scipy.spatial.transform import Rotation as R_scipy
from .CalibratorBase import BaseCalibrator

# Step 1 joint-sweep timing (2026-09-15 time reduction): one direction per sweep; all captured
# frames are fitted (no downsampling) to offset the lower sample count of the shorter sweep.
JOINT_SWEEP_DURATION_S = 8.0
# wrist_yaw2 (J6) pairs marker orientation with encoder angles, so a faster one-direction sweep
# biases it (real test 2026-09-15, right arm, 3 reps: 12 s mean -2.07 std 0.11 vs 8 s mean -2.71
# std 0.32). Elbow was unaffected (12 s -1.982/0.024 vs 8 s -1.979/0.025).
# v1.3's J6 (wrist_roll_v13) is estimated the same way, so it gets the same duration (2026-10-15).
# The v1.3 J5 stage sweeps J6 too: at 8 s its estimate scattered 0.24 deg on the first v1.3 robot run
# (2026-10-23) while the J6 stage at 12 s held its axis angle to 0.015 deg.
JOINT_SWEEP_DURATION_BY_MODE_S = {"wrist_yaw2": 12.0, "wrist_roll_v13": 12.0, "wrist_pitch_v13": 12.0}
# How the v1.2 J6 (wrist_yaw2) offset is estimated.
#   "position"   : marker positions only -- the angle, about the J6 axis, between the J5 axis and
#                  the marker's radial offset, compared with the same angle from the URDF.
#   "orientation": the J5 axis mapped into the marker frame through the measured marker
#                  orientation (the method up to 2026-09-22).
# 2026-09-22: a stationary marker seen at different image positions (Step 2 head-only poses) kept
# its position to 0.2 mm but its measured orientation drifted 0.2-0.36 deg with the image x
# position (corr ~0.96). Fed into the orientation method in a synthetic sweep, 0.25 deg per 100 px
# moved the right J6 estimate by 0.6 deg (the size of the bias seen on the robot); the position
# method does not use the orientation. Its gauge is the bracket's lateral x (1 mm ~ 1 deg), which
# is the same bracket model Step 2 uses, whereas the orientation method's gauge is the bracket yaw.
# Kept on "orientation": on the real 11:22 left-arm sweep, taken with J6 verified by eye, the
# orientation method read -0.82 deg and the position method +1.17 deg -- they miss in opposite
# directions, i.e. the bracket mounting (yaw vs lateral x, the J6 gauge) dominates either way.
WRIST_YAW2_METHOD = "orientation"
# Both methods are always computed and reported (2026-09-29); a larger disagreement is logged as a
# warning. They miss for different reasons (marker-orientation bias vs bracket lateral mounting).
WRIST_YAW2_METHOD_DISAGREE_WARN_DEG = 0.5


def choose_wrist_yaw2_offset(orientation_deg, position_deg, method):
    """The J6 offset to use and the position-minus-orientation disagreement (None when the position
    estimate is unavailable). "position" falls back to the orientation estimate without it."""
    disagreement = None if position_deg is None else float(position_deg) - float(orientation_deg)
    if method == "position" and position_deg is not None:
        return float(position_deg), disagreement
    return float(orientation_deg), disagreement
# Which sweeps the v1.2 J6 (wrist_yaw2) offset is computed from in Step 1.
#   "bracket_sweeps": the marker-bracket J6 (+-30 deg) and J5 (90 -> 50 deg) sweeps Step 1 runs
#                     at the marker pose just before (wrist_yaw2_from_bracket_sweeps).
#   "joint_sweeps"  : a dedicated, iterated J6/J5 sweep pair at the wrist_yaw2 joint pose
#                     (perform_joint_calibration; the method up to 2026-09-22).
# Both go through the same compute_calibration_results. On 2026-09-22, at the J6 zero the user had
# checked by eye, the bracket sweeps read -0.07 deg (right) / +0.30 deg (left) while the dedicated
# sweeps kept reading about -0.9 deg on both arms. The bracket sweeps also save the J6 sweep pairs
# (2+ iterations x 2 sweeps x 12 s per arm and pass).
WRIST_YAW2_SOURCE = "bracket_sweeps"
MAX_FIT_POINTS = 1000
# Per-sweep target-estimate scatter measured on the real robot (deg). Used as a floor on the
# sample std so a mean of only 2 agreeing sweeps cannot pass on luck.
# elbow: 0.025 deg measured with +-20 deg sweeps fitting all frames (2026-09-15); floor 0.05, raised to 0.12 on
# 2026-10-27: three runs on one v1.2 reset scattered 0.11-0.12 deg per sweep and their means 0.3 deg apart.
# v1.3 (2026-10-15; before, both fell back to 3 samples and no floor): wrist_roll_v13 is the wrist_yaw2
# method, so it takes its floor; wrist_pitch_v13 measured 0.06-0.13 deg per sweep pair in the simulator.
JOINT_ESTIMATE_NOISE_DEG = {"wrist_pitch": 0.03, "wrist_yaw2": 0.06, "elbow": 0.12,
                            "wrist_roll_v13": 0.06, "wrist_pitch_v13": 0.10}
MIN_MEAN_SAMPLES_BY_MODE = {"wrist_pitch": 2, "wrist_yaw2": 2, "elbow": 2, "wrist_roll_v13": 2, "wrist_pitch_v13": 2}


class DebugLogger:
    def __init__(self, original_log_callback, file_path):
        self.original_log_callback = original_log_callback
        self.file_path = file_path
        self.buffer = []
        
    # GUI gets only the per-iteration headline (number, offset, pass/fail vs
    # tolerance) plus warnings/errors/final result; the full bullet-by-bullet
    # breakdown (circle radii, center distance, etc. -- still useful for
    # offline debugging) is always buffered below and written to
    # joint_calib_debug_{arm}_{mode}.txt regardless of what reaches the GUI.
    def log(self, msg):
        self.buffer.append(msg)
        msg_upper = msg.upper()
        if (
            "[ALERT]" in msg_upper or
            "[SAFETY WARNING]" in msg_upper or
            "[SUCCESS]" in msg_upper or
            "[ERROR]" in msg_upper or
            "[WARN]" in msg_upper or
            "[INFO]" in msg_upper or
            "[VALIDATION SWEEP]" in msg_upper or
            "[ITERATION" in msg_upper or
            "RECOMMENDED ABSOLUTE OFFSET" in msg_upper
        ):
            if self.original_log_callback:
                self.original_log_callback(msg)
                
    def save(self):
        try:
            with FileStorage.open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n=== NEW ITERATION ===\n")
                f.write("\n".join(self.buffer) + "\n")
        except Exception:
            pass

class JointCalibrator(BaseCalibrator):
    def __init__(self, marker_st=None, robot=None):
        super().__init__(marker_st, robot)
        self.use_angle_based_fitting = True

    @staticmethod
    def sweep_points_tag(mode, key_type):
        """Name part of a joint calibration's sweep_points file. The mode is in it because two modes
        of one arm can sweep the same axis (v1.3: J6 in wrist_roll_v13 and wrist_pitch_v13, J4 in
        wrist_pitch_v13 and elbow); with the axis alone, the later mode deleted the earlier one's file."""
        return f"{mode}_{key_type}"

    def perform_joint_calibration(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=None, use_angle_based_fitting=None, save_debug=False, pass_idx=1, pass1_res=None):
        if sweep_duration is None:
            sweep_duration = JOINT_SWEEP_DURATION_BY_MODE_S.get(mode, JOINT_SWEEP_DURATION_S)
        if use_angle_based_fitting is None:
            use_angle_based_fitting = getattr(self, 'use_angle_based_fitting', True)

        config_dir = os.path.abspath(os.path.dirname(__file__))
        from core.storage import CONFIG_PATHS
        result_txt_dir = CONFIG_PATHS["txt_dir"]
        FileStorage.ensure_dir(result_txt_dir, exist_ok=True)
        debug_file_path = os.path.join(result_txt_dir, f"joint_calib_debug_{arm_side}_{mode}.txt")
        
        # 처음 시작(pass 1)일 때는 덮어쓰기 위해 기존 파일들 삭제
        if pass_idx == 1:
            if os.path.exists(debug_file_path):
                try: FileStorage.remove(debug_file_path)
                except: pass
                
            jcfg = self.JOINT_CONFIGS.get(mode, {})
            for key_type, j_key in [("joint_A", "sweep_joint_A"), ("joint_B", "sweep_joint_B")]:
                axis = jcfg.get(j_key)
                if axis is not None:
                    fname = os.path.join(result_txt_dir, f"sweep_points_{arm_side}_{self.sweep_points_tag(mode, key_type)}_axis_{axis}.txt")
                    if os.path.exists(fname):
                        try: FileStorage.remove(fname)
                        except: pass

        logger = DebugLogger(log_callback, debug_file_path)
        original_log = log_callback
        log_callback = logger.log

        try:
            self.current_calib_mode = mode
            self.last_staged_offset = None
            self.last_diff_angle = None
            if original_log:
                log_callback("\n" + "="*60)
                log_callback("   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE")
                log_callback(f"   Target Arm: {arm_side.upper()} | Joint Target: {mode.upper()}")
                log_callback("="*60 + "\n")
            exposure = self.camera_exposure_record()
            if exposure is not None:
                log_callback(f"[INFO] Camera exposure for these sweeps: {'auto' if exposure['auto'] else 'manual'}, "
                             f"setpoint {exposure['setpoint']:.0f} (x {exposure['setpoint_unit_us']:g} us), "
                             f"last frame {exposure['last_frame_us']:.0f} us")

            max_readjust_retries = 2
            readjust_retry_count = 0
            max_iterations = 8
            # Mean-based convergence: estimates measured within NEAR_TARGET_DEG of the staged
            # offset (past the large-step transient) are averaged; converge once the standard
            # error of that mean is within tolerance.
            MIN_MEAN_SAMPLES = MIN_MEAN_SAMPLES_BY_MODE.get(mode, 3)
            ESTIMATE_NOISE_DEG = JOINT_ESTIMATE_NOISE_DEG.get(mode, 0.0)
            NEAR_TARGET_DEG = 0.3
            CONVERGENCE_TOL_DEG = 0.06

            while True:
                staged_offset = current_offset_deg
                staged_offsets_history = [staged_offset]
                # Per-iteration (staged offset, measured absolute target) pairs for the
                # noise-floor stability convergence check.
                target_estimates = []
                convergence_basis = None
                final_res = None
                first_res = None
                converged = False
                
                # Sign-reversal tracking state
                prev_error = None
                prev_step_correction = 0.0
                direction_multiplier = 1.0
                dynamic_damping = 1.0
                prev_raw_optimal_offset = None
                # The actual (post-clamp) staged_offset change applied last iteration, used
                # to adjust the runaway-step check below for relative/delta modes.
                prev_applied_step_deg = 0.0

                # Refresh starting pose from robot state in case user taught a new pose
                first_starting_pose = None
                if self.robot:
                    try:
                        state = self.robot.get_state()
                        model = self.robot.model()
                        arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                        first_starting_pose = list(state.position[arm_idx])
                    except Exception:
                        pass

                # The debug plot is saved for the first sweep only (the uncorrected state). The raw
                # sweep points are saved for every sweep: on 2026-10-06 the elbow estimates scattered
                # 0.15 deg between sweeps and only the first sweep was on disk to look at.
                _sweep_count = [0]
                def run_single_sweep(offset):
                    _sweep_count[0] += 1
                    do_save = save_debug and (_sweep_count[0] == 1)
                    points_note = (f"pass {pass_idx}, sweep {_sweep_count[0]}, staged offset {offset:.4f} deg"
                                   if save_debug else None)
                    return self.perform_calibration_sweep_continuous(
                        arm_side, mode, log_callback=log_callback, status_callback=status_callback,
                        current_offset_deg=offset, sweep_duration=sweep_duration,
                        use_angle_based_fitting=use_angle_based_fitting, save_debug=do_save,
                        debug_points_note=points_note,
                        first_starting_pose=first_starting_pose,
                        # the previous sweep of this sequence already saw the marker
                        skip_visibility_check=(_sweep_count[0] > 1)
                    )

                restart_sequence = False
                
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
                    
                    # Perform continuous sweep at the current staged offset
                    res = run_single_sweep(staged_offset)
                    if not res:
                        if readjust_retry_count < max_readjust_retries and hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                            readjust_retry_count += 1
                            if log_callback:
                                log_callback(f"\n[WARNING] Iteration {i} sweep failed (marker lost or insufficient points).")
                                log_callback(f"[INFO] Prompting user for posture readjustment (Attempt {readjust_retry_count}/{max_readjust_retries})...")
                            # Reset staged offset in memory before moving
                            if offset_key:
                                if arm_side in self.joint_offsets:
                                    self.joint_offsets[arm_side][offset_key] = current_offset_deg
                                else:
                                    self.joint_offsets[offset_key] = current_offset_deg
                            self.perform_move_to_ready_pose(arm_side, mode=mode, log_callback=log_callback)
                            resolved = self.marker_problem_callback(arm_side, mode=mode)
                            if resolved:
                                if log_callback:
                                    log_callback(f"[INFO] Posture readjusted. Resetting staged offset to initial ({current_offset_deg:.4f}°) and restarting calibration from Iteration 1...")
                                self.perform_move_to_ready_pose(arm_side, mode=mode, log_callback=log_callback)
                                time.sleep(1.0)
                                restart_sequence = True
                                break
                            else:
                                if log_callback: log_callback("[ERROR] Posture readjustment cancelled by user. Aborting calibration.")
                                return None
                        else:
                            if log_callback: log_callback(f"[ERROR] Iteration {i} sweep failed (consecutive retry limit reached). Aborting calibration.")
                            return None
                    
                    if i == 1:
                        first_res = res
                    final_res = res
            
                    angle_error = res.get('angle_between_normals', 0.0)
                    sign = res.get('sign', 1.0)
                    
                    # Modes with perpendicular axes (target 90 deg): wrist_yaw2 (v1.2), wrist_roll_v13, wrist_pitch_v13
                    if mode in ("wrist_yaw2", "wrist_roll_v13", "wrist_pitch_v13"):
                        angle_dev = abs(angle_error - 90.0)
                        center_dist = res.get('perp_dist_after', res.get('center_dist', 999.0))
                    else: # Modes with parallel axes (target 0 deg): wrist_pitch (v1.2), elbow
                        angle_dev = angle_error
                        center_dist = res.get('center_dist', 999.0)
                        
                    r_A = res.get('r_A', 0.0)
                    r_B = res.get('r_B', 0.0)
                    size_error = abs(r_A - r_B)
                    current_error = max(size_error, center_dist)
                    
                    # Print iteration summary
                    if log_callback:
                        if mode in ("wrist_yaw2", "wrist_roll_v13", "wrist_pitch_v13"):
                            log_callback(f"  * Angle Error (from 90.0°)         : {angle_dev:.4f}° (Raw: {angle_error:.4f}°)")
                            log_callback(f"  * Circle Radii (r_A / r_B)         : {r_A:.1f} mm / {r_B:.1f} mm")
                            log_callback(f"  * Center Distance                  : {center_dist:.4f} mm")
                        else:
                            log_callback(f"  * Angle Error (Deviation)          : {angle_error:.4f}°")
                            log_callback(f"  * Circle Size Error (r_A-r_B)      : {size_error:.4f} mm")
                            log_callback(f"  * Center Distance Error            : {center_dist:.4f} mm")
                            log_callback(f"  * Max Fitting Error Metric         : {current_error:.4f} mm")
                    
                    raw_optimal_offset = res.get('optimal_offset', 0.0)

                    # Runtime Anomaly Detection during sweep iterations:
                    # 1. center_dist > 40.0 mm (ONLY for parallel concentric modes: elbow, wrist_pitch)
                    # 2. Unexplained jump > 3.5 deg between consecutive raw measurements. Skipped
                    #    on iteration 1: with no prior measurement there is nothing for a "jump"
                    #    to be relative to, and a genuinely large true joint offset (e.g. an
                    #    uncalibrated wrist that is several degrees off) is a normal, correct
                    #    first reading -- not an anomaly.
                    #    wrist_roll_v13/wrist_yaw2 report an ABSOLUTE target offset, which should
                    #    stay stable across iterations regardless of how far staged_offset just
                    #    moved -- compare the raw values directly.
                    #    Every other mode (elbow, wrist_pitch, wrist_pitch_v13) reports a RELATIVE
                    #    remaining-correction, which is *expected* to shrink by roughly the
                    #    previous iteration's applied step each time -- compare only the part of
                    #    the jump that isn't explained by that expected shrinkage. Without this
                    #    adjustment, raising MAX_STEP_CLAMP_DEG (2026-09-14, 1.5->2.5) made a
                    #    normal convergence step (~2.5 deg) plus ordinary measurement noise
                    #    (~1 deg) exceed the flat 3.5 deg threshold and falsely abort calibration
                    #    (observed live on v1.3 wrist_pitch_v13, 2026-09-15).
                    is_anomalous = False
                    anomaly_reasons = []
                    if mode in ("elbow", "wrist_pitch") and center_dist > 40.0:
                        is_anomalous = True
                        anomaly_reasons.append(f"Center distance error {center_dist:.2f} mm > 40.0 mm")
                    if prev_raw_optimal_offset is not None:
                        raw_jump = raw_optimal_offset - prev_raw_optimal_offset
                        if mode in ("wrist_roll_v13", "wrist_yaw2"):
                            unexplained_jump = raw_jump
                        else:
                            unexplained_jump = raw_jump + prev_applied_step_deg
                        if abs(unexplained_jump) > 3.5:
                            is_anomalous = True
                            anomaly_reasons.append(
                                f"Optimal offset jumped {raw_jump:+.2f}° "
                                f"between iterations ({prev_raw_optimal_offset:.2f}° -> {raw_optimal_offset:.2f}°), "
                                f"{unexplained_jump:+.2f}° unexplained by the last applied step "
                                f"({prev_applied_step_deg:+.2f}°) > 3.5° (runaway step)"
                            )
                    prev_raw_optimal_offset = raw_optimal_offset

                    if is_anomalous:
                        if log_callback:
                            log_callback(f"\n[ALERT] Runtime measurement anomaly detected in Iteration {i}:")
                            for r in anomaly_reasons:
                                log_callback(f"  - {r}")
                        if readjust_retry_count < max_readjust_retries and hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                            readjust_retry_count += 1
                            if log_callback:
                                log_callback(f"[INFO] Moving arm to ready pose and prompting user for posture readjustment (Attempt {readjust_retry_count}/{max_readjust_retries})...")
                            # Reset staged offset in memory before moving
                            if offset_key:
                                if arm_side in self.joint_offsets:
                                    self.joint_offsets[arm_side][offset_key] = current_offset_deg
                                else:
                                    self.joint_offsets[offset_key] = current_offset_deg
                            self.perform_move_to_ready_pose(arm_side, mode=mode, log_callback=log_callback)
                            resolved = self.marker_problem_callback(arm_side, mode=mode)
                            if resolved:
                                if log_callback:
                                    log_callback(f"[INFO] Posture readjusted. Resetting staged offset to initial ({current_offset_deg:.4f}°) and restarting calibration from Iteration 1...")
                                self.perform_move_to_ready_pose(arm_side, mode=mode, log_callback=log_callback)
                                time.sleep(1.0)
                                restart_sequence = True
                                break
                            else:
                                if log_callback: log_callback("[ERROR] Posture readjustment cancelled by user. Aborting calibration.")
                                return None
                        else:
                            if log_callback:
                                log_callback(f"[ERROR] Consecutive anomaly retry limit reached ({readjust_retry_count}/{max_readjust_retries}). Aborting calibration.")
                            return None
                    
                    # Calculate raw measurement residual first:
                    # J6 modes (wrist_roll_v13, wrist_yaw2) return absolute recommended target offset.
                    # Other modes return relative correction offset.
                    if mode in ("wrist_roll_v13", "wrist_yaw2"):
                        raw_residual = direction_multiplier * raw_optimal_offset - staged_offset
                    else:
                        raw_residual = direction_multiplier * raw_optimal_offset

                    # Dynamic damping: halve the damping factor if the correction direction flips
                    # This squashes noise-floor oscillations rapidly
                    if i > 1 and raw_residual * prev_step_correction < 0:
                        dynamic_damping *= 0.8
                    elif i == 1:
                        dynamic_damping = 1.0

                    # Apply damping to the residual step delta, NEVER directly to the absolute target itself
                    step_correction_delta = raw_residual * dynamic_damping

                    # Convergence check:
                    # Based on raw measurement residual (< 0.06°) to ensure genuine physical convergence
                    # rather than premature termination from small damped steps.
                    converged_criteria = (abs(raw_residual) < 0.06)
                    target_estimates.append((staged_offset, staged_offset + raw_residual))

                    if log_callback:
                        tol_status = "OK (within 0.06°)" if converged_criteria else "retry (exceeds 0.06° tolerance)"
                        log_callback(
                            f"[ITERATION {i}/{max_iterations}] offset={staged_offset:.4f}°, "
                            f"residual={raw_residual:+.4f}° -> {tol_status}"
                        )

                    # Mean-based convergence: a single sweep's estimate scatters by ~0.05° (J6)
                    # to ~0.15° (J3), so a lone residual landing inside the tolerance is luck,
                    # not precision. Average the target estimates measured near the staged
                    # offset and converge when the standard error of that mean is within
                    # tolerance.
                    # A lone in-tolerance residual is not treated as convergence; only the mean is.
                    converged_criteria = False
                    near = [e for s, e in target_estimates if abs(e - s) <= NEAR_TARGET_DEG]
                    if len(near) >= MIN_MEAN_SAMPLES:
                        est_n = np.array(near)
                        est_std = max(float(np.std(est_n, ddof=1)), ESTIMATE_NOISE_DEG)
                        est_se = est_std / np.sqrt(len(est_n))
                        mean_ok = est_se <= CONVERGENCE_TOL_DEG
                        if log_callback:
                            log_callback(
                                f"  * Mean of {len(est_n)} near-target estimates: {float(np.mean(est_n)):.4f}° "
                                f"(std={est_std:.4f}°, std err={est_se:.4f}°) -> "
                                f"{'OK (within 0.06°)' if mean_ok else 'need more samples'}"
                            )
                        if mean_ok:
                            converged = True
                            convergence_basis = "mean_std_err"
                            staged_offset = float(np.mean(est_n))
                            if log_callback:
                                log_callback(f"\n[SUCCESS] Calibration CONVERGED (mean of repeated sweeps):")
                                log_callback(f"  * Near-target estimates: {', '.join(f'{v:.4f}°' for v in est_n)}")
                                log_callback(f"  * Std err {est_se:.4f}° <= {CONVERGENCE_TOL_DEG}°")
                                log_callback(f"  * Recommended Absolute Offset (mean): {staged_offset:.4f}°")
                            break

                    if converged_criteria:
                        converged = True
                        convergence_basis = "residual"
                        if mode in ("wrist_roll_v13", "wrist_yaw2"):
                            # On convergence for absolute modes, record the exact target offset
                            staged_offset = direction_multiplier * raw_optimal_offset
                        if log_callback:
                            log_callback(f"\n[SUCCESS] Calibration CONVERGED successfully:")
                            log_callback(f"  * Residual Error: {raw_residual:.4f}° < 0.06° (reached resolution limit)")
                            log_callback(f"  * Recommended Absolute Offset: {staged_offset:.4f}°")
                        break
                    
                    # Normal update: apply correction with per-iteration step clamping.
                    # 2026-09-14: raised from 1.5deg to 2.5deg to cut iteration count for
                    # larger real offsets (e.g. right after a motor/joint swap) -- a 5.4deg
                    # offset now takes ~3 iterations instead of ~5. Kept well under the
                    # runaway-step anomaly threshold (3.5deg, on the raw per-iteration
                    # measurement -- see MAX_STEP_CLAMP_DEG usage below and the anomaly
                    # check a few lines above) so a genuine overshoot/marker-lost event is
                    # still caught before being acted on.
                    # 2026-09-15: 2.5 -> 5.0 deg so post-Home-Offset-Reset offsets (3-5 deg) are
                    # reached in one step.
                    MAX_STEP_CLAMP_DEG = 5.0
                    prev_error = angle_dev
                    prev_step_correction = step_correction_delta
                    staged_offset_before_step = staged_offset
                    clamped_step = float(np.clip(step_correction_delta, -MAX_STEP_CLAMP_DEG, MAX_STEP_CLAMP_DEG))
                    staged_offset = staged_offset + clamped_step

                    # Safety: clamp staged_offset to the joint's configured offset range
                    jcfg = self.JOINT_CONFIGS.get(mode, {})
                    off_min, off_max = jcfg.get('offset_range', (-10.0, 10.0))
                    if staged_offset < off_min or staged_offset > off_max:
                        if log_callback:
                            log_callback(f"  [SAFETY WARNING] Staged offset {staged_offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                        staged_offset = float(np.clip(staged_offset, off_min, off_max))
                    # Record the *actual* change (post safety-clamp too) for next iteration's
                    # runaway-step check above.
                    prev_applied_step_deg = staged_offset - staged_offset_before_step
                        
                    staged_offsets_history.append(staged_offset)
                    if log_callback:
                        log_callback(f"  * Updated Absolute Offset     : {staged_offset:.4f}°")
                        
                if restart_sequence:
                    continue
                else:
                    break

            # Damping fallback for oscillation/noise-floor:
            if not converged and len(staged_offsets_history) >= 3:
                avg_offset = float(np.mean(staged_offsets_history[-3:]))
                if log_callback:
                    log_callback(f"\n[INFO] Joint {mode} did not meet 0.06° convergence tolerance due to measurement noise floor.")
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
                'convergence_basis': convergence_basis,
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
            self.current_calib_mode = None
            logger.save()

    def wrist_yaw2_from_bracket_sweeps(self, arm_side, res_6, res_5, log_callback=None, pass_idx=1):
        """v1.2 J6 (wrist_yaw2) offset from the marker-bracket J6 and J5 sweeps (WRIST_YAW2_SOURCE).

        The bracket's axis-6 and axis-5 sweeps are the J6/J5 sweep pair compute_calibration_results
        needs, taken at the marker pose: the J5 axis's angle about the J6 axis, against the design
        bracket, is the J6 offset. That estimate is absolute (it does not depend on the staged J6
        offset), so one pair gives the answer and nothing moves the robot. res_6 / res_5 are
        MarkerCalibrator.perform_calibration_sweep results (captured_q_full, captured_poses).
        Returns a dict shaped like perform_joint_calibration's result, or None without enough data.
        """
        mode = "wrist_yaw2"
        from core.storage import CONFIG_PATHS
        result_txt_dir = CONFIG_PATHS["txt_dir"]
        FileStorage.ensure_dir(result_txt_dir, exist_ok=True)
        debug_file_path = os.path.join(result_txt_dir, f"joint_calib_debug_{arm_side}_{mode}.txt")
        if pass_idx == 1 and os.path.exists(debug_file_path):
            try: FileStorage.remove(debug_file_path)
            except Exception: pass
        logger = DebugLogger(log_callback, debug_file_path)
        log = logger.log
        try:
            if not self.robot:
                raise RuntimeError("Robot instance is not initialized or connected.")
            model = self.robot.model()
            arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
            dataset_A = list(zip(res_6.get('captured_q_full', []), res_6.get('captured_poses', [])))
            dataset_B = list(zip(res_5.get('captured_q_full', []), res_5.get('captured_poses', [])))
            if len(dataset_A) < 20 or len(dataset_B) < 20:
                log(f"[ERROR] {arm_side} J6 from the bracket sweeps: too few frames "
                    f"(J6 sweep {len(dataset_A)}, J5 sweep {len(dataset_B)}; need 20 each).")
                return None
            if len(dataset_A) > MAX_FIT_POINTS:
                dataset_A = [dataset_A[i] for i in np.round(np.linspace(0, len(dataset_A) - 1, MAX_FIT_POINTS)).astype(int)]
            if len(dataset_B) > MAX_FIT_POINTS:
                dataset_B = [dataset_B[i] for i in np.round(np.linspace(0, len(dataset_B) - 1, MAX_FIT_POINTS)).astype(int)]
            initial_joint_pos = list(np.asarray(dataset_A[0][0])[arm_idx])

            res = self.compute_calibration_results(arm_side, mode, dataset_A, dataset_B, initial_joint_pos,
                                                   log_callback=log)
            offset = float(res['optimal_offset'])
            converged = not res.get('estimate_failed', False)
            if not converged:
                log(f"[ERROR] {arm_side} J6 could not be computed from the bracket sweeps (see the warning above).")
            off_min, off_max = self.JOINT_CONFIGS[mode].get('offset_range', (-10.0, 10.0))
            if not off_min <= offset <= off_max:
                log(f"  [SAFETY WARNING] J6 offset {offset:.4f}° exceeds safe bounds [{off_min}°, {off_max}°]. Clamping.")
                offset = float(np.clip(offset, off_min, off_max))
            both = ""
            if res.get('wrist_yaw2_orientation_deg') is not None:
                pos = res.get('wrist_yaw2_position_deg')
                both = (f"; orientation {res['wrist_yaw2_orientation_deg']:+.4f}°, position "
                        f"{'n/a' if pos is None else f'{pos:+.4f}°'}, method {WRIST_YAW2_METHOD}")
            log(f"[INFO] {arm_side} J6 (wrist_yaw2) from the bracket sweeps: {offset:+.4f}° "
                f"(J6 sweep {len(dataset_A)} frames, r {res['r_A']:.1f} mm; J5 sweep {len(dataset_B)} frames, "
                f"r {res['r_B']:.1f} mm{both})")
            return {
                'mode': mode,
                'recommended_joint_offset': offset,
                'optimal_offset': offset,
                'converged': converged,
                'convergence_basis': 'bracket_sweeps',
                'wrist_yaw2_method': WRIST_YAW2_METHOD,
                'wrist_yaw2_orientation_deg': res.get('wrist_yaw2_orientation_deg'),
                'wrist_yaw2_position_deg': res.get('wrist_yaw2_position_deg'),
                'perp_dist_before': float('nan'),
                'perp_dist_after': float('nan'),
                'axial_offset_mm': float('nan'),
                'lateral_offset_mm': float('nan'),
                'r_A': res['r_A'],
                'r_B': res['r_B'],
                'first_res': res,
                'final_res': res,
            }
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
            c_A, R_c_A, r_A, rmse_A, pts_2d_A, uc_A, vc_A = BaseCalibrator.fit_circle_3d(pts_a, robust=True)
            c_B, R_c_B, r_B, rmse_B, pts_2d_B, uc_B, vc_B = BaseCalibrator.fit_circle_3d(pts_b, robust=True)
            
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
            
            from core.storage import CONFIG_PATHS
            result_dir = CONFIG_PATHS["plot_dir"]
            FileStorage.ensure_dir(result_dir, exist_ok=True)
            plot_save_path = os.path.abspath(os.path.join(result_dir, f"debug_orthogonal_circles_{arm_side}_{frame}.png"))
            ArtifactStorage.save_figure(plot_save_path, dpi=150)
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
                # Read from internal _plot_data bundle
                pd = res.get('_plot_data', res)
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
            if not force_overwrite and os.path.exists(plot_save_path):
                plt.close()
                if log_callback:
                    log_callback(f"[INFO] Comparison plot already exists at: {plot_save_path}, skipping overwrite.")
            else:
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

    def perform_calibration_sweep_continuous(self, arm_side, mode, log_callback=None, status_callback=None, current_offset_deg=0.0, sweep_duration=None, use_angle_based_fitting=None, save_debug=False, first_starting_pose=None, skip_visibility_check=False, debug_points_note=None):
        """debug_points_note: when given, the raw sweep points are saved with this line even if
        save_debug (plot and points) is off for this sweep."""
        if sweep_duration is None:
            sweep_duration = JOINT_SWEEP_DURATION_BY_MODE_S.get(mode, JOINT_SWEEP_DURATION_S)
        self.current_calib_mode = mode
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

        if skip_visibility_check:
            if status_callback: status_callback(True)
        elif not getattr(self.marker_st, 'sim', False):
            # Pre-check marker visibility after settling delay
            time.sleep(1.0)
            initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
            if not initial_check:
                if log_callback: log_callback("[ERROR] Marker is not visible.")
                if hasattr(self, 'marker_problem_callback') and self.marker_problem_callback:
                    if log_callback: log_callback("[INFO] Prompting user for manual teaching due to marker visibility error...")
                    # Back to the ready pose before asking: the arm may still be where the
                    # previous sweep ended, not where this sweep starts from.
                    self.perform_move_to_ready_pose(arm_side, mode=mode, log_callback=log_callback)
                    resolved = self.marker_problem_callback(arm_side, mode=mode)
                    if resolved:
                        self.perform_move_to_ready_pose(arm_side, mode=mode, log_callback=log_callback)
                        time.sleep(1.0)
                        initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
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
        if first_starting_pose is not None:
            initial_joint_pos = list(first_starting_pose)
        else:
            initial_joint_pos = list(np.array(state.position)[arm_idx])

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
        norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
        ref_pose = None
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            arm_dict = self.user_taught_ready_poses.get(arm_side)
            if isinstance(arm_dict, dict):
                ref_pose = arm_dict.get(norm_mode)
        q_cand = list(ref_pose) if ref_pose is not None else list(initial_joint_pos)
        q_cand[cand_joint] = ready_pose_nom[cand_joint] + np.radians(current_offset_deg)

        # Determine sweep ranges from JOINT_CONFIGS
        range_A = jcfg.get("sweep_range_A", 20.0)
        range_B = jcfg.get("sweep_range_B", 20.0)

        # 1. PHYSICAL SWEEP JOINT A
        logging.info(f"\n--- Commencing Continuous Sweep on Joint A (Index {sweep_joint_A}, duration={sweep_duration}s) ---")
        dataset_A = self.perform_single_joint_sweep(
            arm_side, sweep_joint_A, q_cand, -range_A, range_A, sweep_duration,
            q_head=None, label="Joint A", log_callback=log_callback,
            current_offset_deg=current_offset_deg, cand_joint=cand_joint, mode=mode
        )
        if dataset_A is None:
            return None

        if getattr(self, 'stop_requested', False):
            return None
            
        if self.robot:
            time.sleep(0.5)
        else:
            time.sleep(0.01)

        # 2. PHYSICAL SWEEP JOINT B
        norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            arm_dict = self.user_taught_ready_poses.get(arm_side)
            if isinstance(arm_dict, dict):
                ref_pose = arm_dict.get(norm_mode)
                if ref_pose is not None:
                    q_cand = list(ref_pose)
                    q_cand[cand_joint] = ready_pose_nom[cand_joint] + np.radians(current_offset_deg)

        logging.info(f"\n--- Commencing Continuous Sweep on Joint B (Index {sweep_joint_B}, duration={sweep_duration}s) ---")
        dataset_B = self.perform_single_joint_sweep(
            arm_side, sweep_joint_B, q_cand, -range_B, range_B, sweep_duration,
            q_head=None, label="Joint B", log_callback=log_callback,
            current_offset_deg=current_offset_deg, cand_joint=cand_joint, mode=mode
        )
        if dataset_B is None:
            return None
        if dataset_A:
            initial_joint_pos = list(dataset_A[0][0][arm_idx])

        if getattr(self, 'stop_requested', False):
            if log_callback: log_callback("[INFO] Stop requested during sweep.")
            return None

        # Save FULL captured continuous sweep points to debug txt files before downsampling
        if save_debug or debug_points_note is not None:
            self.save_debug_points(
                arm_side, sweep_joint_A, dataset_A, initial_joint_pos, ee_name, dyn_model, None, "joint_A", log_callback,
                note=debug_points_note, file_tag=self.sweep_points_tag(mode, "joint_A")
            )
            self.save_debug_points(
                arm_side, sweep_joint_B, dataset_B, initial_joint_pos, ee_name, dyn_model, None, "joint_B", log_callback,
                note=debug_points_note, file_tag=self.sweep_points_tag(mode, "joint_B")
            )
        
        # Fit all captured frames (cap only as a runtime guard)
        raw_len_A = len(dataset_A)
        raw_len_B = len(dataset_B)

        max_pts = MAX_FIT_POINTS
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

    def wrist_yaw2_offset_from_positions(self, arm_side, dataset_A, n6, n5, c6, arm_idx, dyn_model,
                                         log_callback=None, edge_deg=(0.5, 1.0, 2.0), min_frames=8):
        """v1.2 J6 offset from marker positions only (see WRIST_YAW2_METHOD).

        During the J6 sweep the marker circles the J6 axis (n6, through c6); the J5 axis n5 is
        fixed. For a frame, the angle about n6 from the J5 axis to the marker's radial offset is
        compared with the same angle from the URDF at that frame's encoder angles and the current
        bracket; the difference is how far J6 physically sits from its encoder reading. Only frames
        near the two ends of the sweep are used, where the joint is slow, so the camera latency
        (~65 ms) does not skew the encoder pairing: in a synthetic smooth-profile sweep with 65 ms
        of latency the bias was +0.08 deg using the last 2 deg of each end and +0.02 deg using
        0.5 deg. The window widens (edge_deg) until min_frames frames are in it.
        n6/n5 must already point like the URDF axes.
        Returns the offset in degrees (same convention as the other methods), or None.
        """
        if not dataset_A or n6 is None or n5 is None or c6 is None:
            return None
        n6 = np.asarray(n6, dtype=float) / np.linalg.norm(n6)
        n5 = np.asarray(n5, dtype=float) / np.linalg.norm(n5)

        def angle_about(axis, ref, vec):
            vec = vec - np.dot(vec, axis) * axis
            ref = ref - np.dot(ref, axis) * axis
            if np.linalg.norm(vec) < 1e-9 or np.linalg.norm(ref) < 1e-9:
                return None
            vec, ref = vec / np.linalg.norm(vec), ref / np.linalg.norm(ref)
            return float(np.arctan2(np.dot(axis, np.cross(ref, vec)), np.dot(ref, vec)))

        j6 = np.array([q[arm_idx[6]] for q, _ in dataset_A], dtype=float)
        near_ends = np.ones(len(dataset_A), dtype=bool)
        for edge in edge_deg:
            window = (j6 <= j6.min() + np.radians(edge)) | (j6 >= j6.max() - np.radians(edge))
            if window.sum() >= min_frames:
                near_ends = window
                break

        tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}")
        if tf_vec is None:
            tf_vec = self.camera_config.get(f"Tf_to_marker_{arm_side}_v12")
        if tf_vec is None:
            return None
        T_ee_m = self.make_transform(tf_vec)
        diffs = []
        for (q_full, pose), use in zip(dataset_A, near_ends):
            if not use:
                continue
            psi_meas = angle_about(n6, n5, np.asarray(pose)[:3, 3] * 1000.0 - c6)
            T5 = self.compute_fk(self.robot, dyn_model, q_full, f"link_{arm_side}_arm_5")
            T4 = self.compute_fk(self.robot, dyn_model, q_full, f"link_{arm_side}_arm_4")
            T6 = self.compute_fk(self.robot, dyn_model, q_full, f"link_{arm_side}_arm_6")
            T_ee = self.compute_fk(self.robot, dyn_model, q_full, f"ee_{arm_side}")
            a6 = T5[:3, :3] @ np.array([0.0, 0.0, 1.0])
            a5 = T4[:3, :3] @ np.array([0.0, 1.0, 0.0])
            p_marker = (T_ee @ T_ee_m)[:3, 3] * 1000.0
            psi_nom = angle_about(a6 / np.linalg.norm(a6), a5, p_marker - T6[:3, 3] * 1000.0)
            if psi_meas is None or psi_nom is None:
                continue
            diffs.append((psi_meas - psi_nom + np.pi) % (2 * np.pi) - np.pi)
        if len(diffs) < 3:
            return None
        diffs = np.degrees(np.array(diffs))
        offset = -float(np.median(diffs))
        if log_callback:
            log_callback(f"  * Position-based J6: {len(diffs)} frames near the sweep ends, "
                         f"spread {np.std(diffs):.3f}° -> offset {offset:+.4f}°")
        return offset

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
            a_A_local = np.array([1.0, 0.0, 0.0])
            a_B_local = np.array([0.0, 0.0, 1.0])
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
            raise RuntimeError("Robot instance is not initialized or connected.")
            
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

        # Calculate accurate transformation from torso to camera frame
        if self.is_head_active():
            mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
            T_mount_to_cam = self.make_transform(mount_to_cam)
            q_init = dataset_A[0][0]
            T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_init, "link_head_2", "link_torso_5")
            T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
        else:
            head_base_to_cam = self.camera_config.get("head_base_to_cam", [0.098, 0.009, 0.012, -90.0, 0.0, -90.0])
            T_head_base_to_cam = self.make_transform(head_base_to_cam)
            q_init = dataset_A[0][0]
            try:
                T_t5_to_head_0 = self.compute_fk(self.robot, dyn_model, q_init, "link_head_0", "link_torso_5")
            except Exception:
                T_t5_to_head_0 = np.eye(4)
            T_t5_to_cam = T_t5_to_head_0 @ T_head_base_to_cam

        T_torso_to_cam = np.linalg.inv(T_t5_to_cam)
        R_torso_to_cam = T_torso_to_cam[:3, :3]

        # Define nominal axes in the camera frame
        a_cand_cam = R_torso_to_cam @ a_cand_t5
        a_A_cam = R_torso_to_cam @ a_A_t5
        a_B_cam_nom = R_torso_to_cam @ a_B_t5

        a_cand_cam /= np.linalg.norm(a_cand_cam)
        a_A_cam /= np.linalg.norm(a_A_cam)
        a_B_cam_nom /= np.linalg.norm(a_B_cam_nom)

        # 2. Extract poses and angles in the camera frame
        poses_A = [pose for _, pose in dataset_A]
        angles_A = [np.degrees(q_full[arm_idx[sweep_joint_A]] - initial_joint_pos[sweep_joint_A]) for q_full, _ in dataset_A]
        poses_B = [pose for _, pose in dataset_B]
        angles_B = [np.degrees(q_full[arm_idx[sweep_joint_B]] - initial_joint_pos[sweep_joint_B]) for q_full, _ in dataset_B]

        # 3. Fit Sweep A and B axes in the camera frame
        robust_fit = True
        res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_A, angles_A, axis_prior=a_A_cam, robust=robust_fit)
        res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_B, angles_B, axis_prior=a_B_cam_nom, robust=robust_fit)

        n_A = res_A['axis_opt']
        n_B = res_B['axis_opt']
        if mode not in ("wrist_roll_v13", "wrist_pitch_v13") and np.dot(n_A, n_B) < 0:
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
        estimate_failed = False
        wrist_yaw2_orientation_deg = wrist_yaw2_position_deg = None
        if mode in ("wrist_roll_v13", "wrist_yaw2"):
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
                    
                    # Compensate for the initial nominal ready pose angle of J7 (index 6)
                    # We take J7 angle from dataset_B and subtract any current staged offset to get the true nominal initial pose
                    q_full_B_first = dataset_B[0][0]
                    j7_current_pos_deg = np.degrees(q_full_B_first[arm_idx[6]])
                    
                    staged_j7_offset_deg = 0.0
                    if hasattr(self, 'joint_offsets') and self.joint_offsets:
                        offsets = self.joint_offsets[arm_side] if arm_side in self.joint_offsets else self.joint_offsets
                        staged_j7_offset_deg = offsets.get("wrist_roll" if self.is_v13() else "wrist_yaw2", 0.0)
                    
                    j7_ready_pose_deg = j7_current_pos_deg - staged_j7_offset_deg
                    # 2026-09-15: 0.8 -> 1.0. Real runs show raw gain ~1.0 and the outer loop
                    # now averages repeated estimates, so under-stepping only cost iterations.
                    damping_factor = 1.0
                    optimal_offset_deg = (raw_diff_deg * damping_factor) + j7_current_pos_deg
                    
                    if log_callback:
                        log_callback(f"[INFO] {mode}: J7 nominal ready pose={j7_ready_pose_deg:.2f}°, raw_diff={raw_diff_deg:.2f}°, optimal_offset={optimal_offset_deg:.2f}°")
            except Exception as e:
                import traceback
                if log_callback:
                    log_callback(f"[WARN] MarkerCalibrator fallback failed: {e}\n{traceback.format_exc()}. Using 0.0.")
                optimal_offset_deg = 0.0
                diff_angle = 0.0
                estimate_failed = True
            if mode == "wrist_yaw2" and not estimate_failed:
                # Both J6 estimates are always computed and reported; WRIST_YAW2_METHOD picks the one
                # used (2026-09-29: on the D405 the orientation method missed the baseline by
                # +0.43 / -0.85 deg while the position method stayed within 0.1-0.2 deg).
                wrist_yaw2_orientation_deg = float(optimal_offset_deg)
                wrist_yaw2_position_deg = self.wrist_yaw2_offset_from_positions(
                    arm_side, dataset_A, n_A, n_B, c_A_c, arm_idx, dyn_model, log_callback=log_callback)
                optimal_offset_deg, j6_disagreement = choose_wrist_yaw2_offset(
                    wrist_yaw2_orientation_deg, wrist_yaw2_position_deg, WRIST_YAW2_METHOD)
                if log_callback:
                    if wrist_yaw2_position_deg is None:
                        log_callback(f"[WARN] wrist_yaw2: position-based J6 could not be computed; "
                                     f"using the orientation method ({wrist_yaw2_orientation_deg:+.3f}°).")
                    else:
                        log_callback(f"[INFO] wrist_yaw2 J6 estimates: orientation {wrist_yaw2_orientation_deg:+.3f}°, "
                                     f"position {wrist_yaw2_position_deg:+.3f}° (difference {j6_disagreement:+.3f}°); "
                                     f"using the {WRIST_YAW2_METHOD} method.")
                        if abs(j6_disagreement) > WRIST_YAW2_METHOD_DISAGREE_WARN_DEG:
                            log_callback(f"[WARN] wrist_yaw2: the two J6 estimates differ by {abs(j6_disagreement):.2f}° "
                                         f"(> {WRIST_YAW2_METHOD_DISAGREE_WARN_DEG}°). The orientation method follows the "
                                         f"marker-orientation measurement, the position method the bracket's lateral "
                                         f"mounting; check both before trusting J6.")
        else:
            # wrist_pitch_v13 (J5) falls through to here too, same as v1.2's wrist_pitch and
            # elbow -- only the JOINT_CONFIGS axis assignment (cand_joint/sweep_joint_A/B) and
            # nominal local axis vectors differ per mode/version, not the computation method.
            # Unlike elbow/wrist_pitch (whose A/B sweep axes are nominally concentric, so their
            # own center-distance method below overrides this with a vibration-robust
            # measurement), wrist_pitch_v13's A/B axes (joints 6 and 4) are nominally
            # *orthogonal*, not concentric -- the center-distance method's parallel-axis
            # assumption doesn't hold, so this signed projected-angle deviation IS the final
            # optimal_offset_deg for wrist_pitch_v13 (see the final `elif` below).
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
                if self.is_head_active():
                    mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
                    T_mount_to_cam = self.make_transform(mount_to_cam)
                    q_init = dataset_A[0][0]
                    T_t5_to_head = self.compute_fk(self.robot, dyn_model, q_init, "link_head_2", "link_torso_5")
                    T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
                else:
                    head_base_to_cam = self.camera_config.get("head_base_to_cam", [0.098, 0.009, 0.012, -90.0, 0.0, -90.0])
                    T_head_base_to_cam = self.make_transform(head_base_to_cam)
                    q_init = dataset_A[0][0]
                    try:
                        T_t5_to_head_0 = self.compute_fk(self.robot, dyn_model, q_init, "link_head_0", "link_torso_5")
                    except Exception:
                        T_t5_to_head_0 = np.eye(4)
                    T_t5_to_cam = T_t5_to_head_0 @ T_head_base_to_cam

                T_torso_to_cam = np.linalg.inv(T_t5_to_cam)
                
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
                
                # Project centers to candidate plane
                c_A_proj = c_A_c - np.dot(c_A_c - p_cand_cam, true_a_cand_cam) * true_a_cand_cam
                c_B_proj = c_B_c - np.dot(c_B_c - p_cand_cam, true_a_cand_cam) * true_a_cand_cam
                
                v_A = c_A_proj - p_cand_cam
                v_B = c_B_proj - p_cand_cam
                L1 = np.linalg.norm(v_A) # Length of link between cand joint and sweep joint A
                L2 = np.linalg.norm(v_B) # Length of link between cand joint and sweep joint B
                
                if L1 > 1e-3 and L2 > 1e-3:
                    # Direction perpendicular to v_A in the rotation plane
                    dir_vec = np.cross(true_a_cand_cam, v_A / L1)
                    # Project center displacement onto this perpendicular direction
                    diff_centers = c_B_proj - c_A_proj
                    proj = np.dot(diff_centers, dir_vec)
                    sin_theta = np.clip(proj / L2, -1.0, 1.0)
                    # We return the negative of the physical error as the compensation offset
                    optimal_offset_deg = -float(np.degrees(np.arcsin(sin_theta)))
                else:
                    optimal_offset_deg = 0.0
                    
            elif mode not in ("wrist_roll_v13", "wrist_yaw2"):
                # Empirically verified sign against the live v1.3 simulator (2026-09-15):
                # the negated form drove staged_offset in the wrong direction, growing the
                # residual monotonically each iteration instead of shrinking it.
                optimal_offset_deg = np.degrees(diff_angle)

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
            # True when the J6 estimate fell back to 0.0 (the value is not a measurement).
            'estimate_failed': estimate_failed,
            # v1.2 J6 only: both estimates, whichever WRIST_YAW2_METHOD used (None elsewhere).
            'wrist_yaw2_orientation_deg': wrist_yaw2_orientation_deg,
            'wrist_yaw2_position_deg': wrist_yaw2_position_deg,
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
