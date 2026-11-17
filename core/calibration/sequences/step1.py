import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from core.storage import CONFIG_PATHS
from core.calibration.JointCalibrator import WRIST_YAW2_SOURCE
from .result import require_converged_joint


def _taught_marker_pose(marker_calibrator, arm_side):
    """The marker posture the operator taught for this arm during this run, if any."""
    taught = getattr(marker_calibrator, "user_taught_ready_poses", None)
    if isinstance(taught, dict):
        arm = taught.get(arm_side, {})
        if isinstance(arm, dict) and arm.get("marker") is not None:
            return list(arm["marker"])
    return None


def run_bracket_sweeps(marker_calibrator, arm_side, first_starting_pose, *, log, emit_status,
                       save_debug, pass_idx, is_stopped, labels=None, max_rounds=3):
    """Run the axis 4, 6 and 5 marker sweeps so that all three share one arm posture.

    The bracket is solved from the three circles together, on the assumption that they were
    swept from the same posture (the wrist axes then meet at one point). When the marker leaves
    the view part way through, the operator re-teaches the posture and only the sweep in progress
    and the ones after it used the new posture: on 2026-09-21 axis 4 stayed at the old posture,
    its axis ended up 29 mm from the other two, and the constrained refit bent the circles until
    the right bracket read y = -40.5 mm instead of about -54. Any sweep done at a posture that
    differs from the final one is therefore run again at the final one.

    Returns {4: res, 6: res, 5: res}, or None when a stop was requested.
    """
    labels = labels or {}
    order = (4, 6, 5)
    results, pose_used = {}, {}
    for _ in range(max_rounds):
        for axis in order:
            if axis in results:
                continue
            start = _taught_marker_pose(marker_calibrator, arm_side) or first_starting_pose
            log(f"[FULL AUTO] Sweeping Axis {axis}{labels.get(axis, '')}...")
            res = marker_calibrator.perform_calibration_sweep(
                arm_side, axis, log_callback=log, status_callback=emit_status,
                save_debug=save_debug, initial_joint_pos=start, pass_idx=pass_idx
            )
            if not res:
                raise RuntimeError(f"Axis {axis} marker sweep failed on {arm_side} arm")
            res['axis_mode'] = axis
            res['axis'] = res['axis_opt']
            results[axis] = res
            # A re-teach inside the sweep restarts that sweep at the taught posture.
            pose_used[axis] = _taught_marker_pose(marker_calibrator, arm_side) or start
            if is_stopped():
                return None
        final = np.asarray(pose_used[order[-1]], dtype=float)
        stale = [axis for axis in order
                 if not np.allclose(np.asarray(pose_used[axis], dtype=float), final, atol=np.radians(0.1))]
        if not stale:
            return results
        log(f"[FULL AUTO] The marker posture was re-taught during the bracket sweeps. Re-running "
            f"Axis {', '.join(str(a) for a in stale)} at the new posture so all three sweeps share it.")
        for axis in stale:
            del results[axis]
    raise RuntimeError(f"The marker posture kept changing during the bracket sweeps on the "
                       f"{arm_side} arm; the three sweeps never shared one posture.")


def _execute_step1_sequence(
    joint_calibrator,
    marker_calibrator,
    joint_offsets_store: dict,
    stop_event=None,
    log_callback=None,
    status_callback=None,
    bracket_finished_callback=None,
    joint_finished_callback=None,
    save_debug: bool = False,
    context=None,
    bracket_accept_callback=None,
    symmetry_callback=None,
) -> None:
    """
    Executes the full automated multi-pass calibration sequence for dual arms.
    Supports both robot model v1.2 (sequential J5 -> Marker -> J6 -> J3) and
    v1.3 (unified 3-axis spherical wrist -> Marker -> J3).

    :param joint_calibrator: JointCalibrator instance
    :param marker_calibrator: MarkerCalibrator instance
    :param joint_offsets_store: In-memory store for joint offsets dict
    :param stop_event: threading.Event to signal cancellation
    :param log_callback: Callable[[str], None] for logging messages
    :param status_callback: Callable[[bool], None] for status updates
    :param bracket_finished_callback: Callable[[dict], None] for bracket result data
    :param joint_finished_callback: Callable[[dict], None] for joint result data
    :param save_debug: Whether to save debug trajectory data
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    def emit_status(val: bool):
        if status_callback:
            status_callback(val)

    def emit_bracket(data: dict):
        if context:
            context.complete("bracket_" + str(len(context.result.completed)), data)
        if bracket_finished_callback:
            bracket_finished_callback(data)

    def emit_joint(data: dict):
        if context:
            context.complete("joint_" + str(len(context.result.completed)), data)
        if joint_finished_callback:
            joint_finished_callback(data)

    def _wait(seconds):
        if stop_event is not None:
            stop_event.wait(seconds)
        else:
            time.sleep(seconds)
        if context:
            context.check_cancelled()

    def is_stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    if context:
        context.check_cancelled()
    log("Starting Step 1 sequential calibration...")
    if joint_calibrator.robot is None:
        raise RuntimeError("Robot is not connected.")

    version_num = marker_calibrator.get_robot_version()
    is_v13 = (version_num == "1.3")

    for arm_side in ["right", "left"]:
        pass1_joint_results = {"wrist_pitch": None, "wrist_roll": None, "elbow": None}
        # Backup of parameters before Pass 1 for early exit / change checking
        prev_j6 = joint_offsets_store[arm_side]["joint6"]
        prev_j5 = joint_offsets_store[arm_side]["joint5"]
        prev_j3 = joint_offsets_store[arm_side]["joint3"]

        # Nominal bracket baseline to calculate bracket parameter changes
        ver_key = "1.3" if is_v13 else "1.2"
        nominal_vec = joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
        tf_vec_init = marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
        if tf_vec_init is not None and len(tf_vec_init) == 6:
            prev_bracket_pos = np.array(tf_vec_init[:3]) * 1000.0
            prev_bracket_rot = np.array(tf_vec_init[3:])
        else:
            prev_bracket_pos = np.array(nominal_vec[:3]) * 1000.0
            prev_bracket_rot = np.array(nominal_vec[3:])

        res_4 = None
        res_5 = None
        res_6 = None

        # Pass 1 calibrates, pass 2 re-sweeps only what did not converge and verifies the
        # bracket. A third pass had no extra logic: it re-swept already-converged joints and
        # overwrote the pass1-vs-pass2 comparison plots.
        MAX_PASSES = 2
        for pass_idx in range(1, MAX_PASSES + 1):
            log("\n" + "=" * 50)
            log(f"   STARTING PASS {pass_idx}/{MAX_PASSES} FOR {arm_side.upper()} ARM")
            log("=" * 50 + "\n")
            log(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")

            for calibrator in [joint_calibrator, marker_calibrator]:
                if arm_side not in calibrator.joint_offsets:
                    calibrator.joint_offsets[arm_side] = {}
                calibrator.joint_offsets[arm_side]["wrist_pitch"] = joint_offsets_store[arm_side]["joint5"]
                if is_v13:
                    calibrator.joint_offsets[arm_side]["wrist_roll"] = joint_offsets_store[arm_side]["joint6"]
                    calibrator.joint_offsets[arm_side]["wrist_yaw2"] = 0.0
                else:
                    calibrator.joint_offsets[arm_side]["wrist_roll"] = 0.0
                    calibrator.joint_offsets[arm_side]["wrist_yaw2"] = joint_offsets_store[arm_side]["joint6"]
                calibrator.joint_offsets[arm_side]["elbow"] = joint_offsets_store[arm_side]["joint3"]

            # --- Step 1: Sequential Calibration Execution ---
            if is_v13:
                # === v1.3 CONCURRENT-AXIS WRIST SEQUENCE ===
                # v1.3's wrist is spherical: J4/J5/J6 rotation axes are concurrent, so the
                # marker bracket transform is derivable directly from one 3-axis sweep
                # (independent of the J5/J6 offsets) and must be computed FIRST -- unlike
                # v1.2, where the bracket fit needs J5 already known. Once the bracket is
                # fixed, J6 then J5 are each calibrated with the SAME iterative
                # sweep-measure-correct-repeat-until-converged loop used everywhere else
                # (JointCalibrator.perform_joint_calibration, JOINT_CONFIGS-driven:
                # wrist_roll_v13, wrist_pitch_v13). Only the swept/reference axis pair
                # differs from v1.2 -- the computation method itself must not.
                # 1. 3-Axis Continuous Sweeps (Axis 4: Yaw, Axis 6: Roll, Axis 5: Pitch) --
                # collected once per pass purely to derive the bracket transform below.
                joint_calibrator.current_calib_mode = None
                marker_calibrator.current_calib_mode = "marker"
                log(f"[FULL AUTO] Starting 3-Axis Bracket Sweeps for {arm_side} arm (Pass {pass_idx})...")
                log(f"[FULL AUTO] Moving {arm_side} arm to marker ready pose...")
                if not marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=log):
                    raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                if is_stopped():
                    return

                state = joint_calibrator.robot.get_state()
                model = joint_calibrator.robot.model()
                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                first_starting_pose = list(state.position[arm_idx])
                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                sweeps = run_bracket_sweeps(
                    marker_calibrator, arm_side, first_starting_pose, log=log, emit_status=emit_status,
                    save_debug=save_debug, pass_idx=pass_idx, is_stopped=is_stopped, labels={4: ' (Wrist Yaw)', 6: ' (Wrist Roll)', 5: ' (Wrist Pitch)'}
                )
                if sweeps is None:
                    return
                res_4, res_6, res_5 = sweeps[4], sweeps[6], sweeps[5]

                # 2. Compute Marker Bracket FIRST from the 3-axis sweeps (concurrent-axis
                # wrist geometry; does not require J5/J6 offsets as input).
                log(f"\n[FULL AUTO] Computing Marker Bracket Transform for {arm_side} arm...")
                bracket_res = marker_calibrator.compute_marker_bracket_from_orthogonal_sweeps(
                    res_4, res_5, res_6, arm_side
                )
                log(f"[INFO] {arm_side} bracket position x {bracket_res['x_e']:.2f}, y {bracket_res['y_e']:.2f}, "
                    f"z {bracket_res['z_e']:.2f} mm ({bracket_res['position_source']}).")
                if 'yaw_measured_e' in bracket_res:
                    log(f"[INFO] {arm_side} bracket rotation held at the design (yaw {bracket_res['yaw_e']:.2f} deg); "
                        f"the sweep axes alone give yaw {bracket_res['yaw_measured_e']:.2f} deg.")
                if bracket_res['position_at_limit']:
                    log(f"[WARN] {arm_side} bracket x/z reached the +-{marker_calibrator.bracket_offset_tolerance_mm():g} mm "
                        f"limit around the design value; check the bracket mounting.")
                bracket_res['res_5'] = res_5
                bracket_res['res_6'] = res_6
                bracket_res['res_4'] = res_4
                bracket_res['arm_side'] = arm_side
                bracket_res['pass_idx'] = pass_idx

                plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                plot_saved = marker_calibrator.generate_marker_plot(res_5, res_6, res_4, bracket_res, arm_side, is_v13, plot_path)
                if plot_saved:
                    bracket_res['plot_path_combined'] = plot_path

                x_m, y_m, z_m = bracket_res['x_e'] / 1000.0, bracket_res['y_e'] / 1000.0, bracket_res['z_e'] / 1000.0
                new_vals = [x_m, y_m, z_m, bracket_res['roll_e'], bracket_res['pitch_e'], bracket_res['yaw_e']]
                key = f"Tf_to_marker_{arm_side}"
                if bracket_accept_callback is not None:
                    bracket_accept_callback(arm_side, new_vals)
                else:
                    marker_calibrator.camera_config[key] = new_vals
                    joint_calibrator.camera_config[key] = new_vals

                emit_bracket(bracket_res)
                _wait(0.5)
                if is_stopped():
                    return

                # 3. Calibrate J6 (Wrist Roll) under the now-known bracket -- iterative
                # loop, mirrors v1.2's wrist_yaw2 stage exactly (same
                # perform_joint_calibration / require_converged_joint / pass1 caching),
                # just using the wrist_roll_v13 JOINT_CONFIGS entry (sweep axis 6 vs
                # reference axis 5, instead of v1.2's axis 6 vs axis 5 pairing for yaw2).
                pass1_res_roll = pass1_joint_results.get("wrist_roll")
                if pass_idx == 2 and pass1_res_roll and pass1_res_roll.get("converged", False):
                    log(f"[FULL AUTO] J6 (Wrist Roll) converged in Pass 1 ({pass1_res_roll['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_roll = pass1_res_roll["recommended_joint_offset"]
                    joint_offsets_store[arm_side]["joint6"] = opt_roll
                    joint_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                    marker_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                    emit_joint(pass1_res_roll)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "wrist_roll_v13"
                    log(f"\n[FULL AUTO] Calibrating J6 (Wrist Roll) under locked bracket...")
                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_roll_v13", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_roll_v13 on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_roll = joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_roll_v13",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint6"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_roll
                    )
                    require_converged_joint(joint_res_roll, context, f"{arm_side}_wrist_roll_v13")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_roll"] = joint_res_roll
                    joint_res_roll['arm_side'] = arm_side
                    joint_res_roll['mode'] = "wrist_roll_v13"
                    joint_res_roll['pass_idx'] = pass_idx

                    opt_roll = joint_res_roll["recommended_joint_offset"]
                    log(f"[FULL AUTO] Staging J6 offset: {opt_roll:.4f}°")
                    joint_offsets_store[arm_side]["joint6"] = opt_roll
                    joint_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                    marker_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_roll_v13", pass1_res_roll if pass1_res_roll else joint_res_roll, joint_res_roll,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_roll['plot_path_combined'] = plot_path

                    emit_joint(joint_res_roll)
                    _wait(0.5)
                    if is_stopped():
                        return

                # 4. Calibrate J5 (Wrist Pitch) now that J6 and the bracket are known --
                # iterative loop, mirrors v1.2's wrist_pitch stage exactly, using the
                # wrist_pitch_v13 JOINT_CONFIGS entry (sweep axis 6 vs reference axis 4).
                pass1_res_pitch = pass1_joint_results.get("wrist_pitch")
                if pass_idx == 2 and pass1_res_pitch and pass1_res_pitch.get("converged", False):
                    log(f"[FULL AUTO] J5 (Wrist Pitch) converged in Pass 1 ({pass1_res_pitch['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_pitch = pass1_res_pitch["recommended_joint_offset"]
                    joint_offsets_store[arm_side]["joint5"] = opt_pitch
                    joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    emit_joint(pass1_res_pitch)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "wrist_pitch_v13"
                    log(f"[FULL AUTO] Calibrating J5 (Wrist Pitch) under locked bracket + J6...")
                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch_v13", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_pitch_v13 on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_pitch = joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_pitch_v13",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint5"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_pitch
                    )
                    require_converged_joint(joint_res_pitch, context, f"{arm_side}_wrist_pitch_v13")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_pitch"] = joint_res_pitch
                    joint_res_pitch['arm_side'] = arm_side
                    joint_res_pitch['mode'] = "wrist_pitch_v13"
                    joint_res_pitch['pass_idx'] = pass_idx

                    opt_pitch = joint_res_pitch["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    joint_offsets_store[arm_side]["joint5"] = opt_pitch

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_pitch_v13", pass1_res_pitch if pass1_res_pitch else joint_res_pitch, joint_res_pitch,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_pitch['plot_path_combined'] = plot_path

                    emit_joint(joint_res_pitch)
                    _wait(0.5)
                    if is_stopped():
                        return

                # 5. Calibrate J3 Elbow (unchanged)
                pass1_res_elbow = pass1_joint_results.get("elbow")
                if pass_idx == 2 and pass1_res_elbow and pass1_res_elbow.get("converged", False):
                    log(f"[FULL AUTO] J3 (Elbow) converged in Pass 1 ({pass1_res_elbow['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_elbow = pass1_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow
                    emit_joint(pass1_res_elbow)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "elbow"
                    log("[FULL AUTO] Sweeping Elbow (Joint 3)...")
                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_elbow = joint_calibrator.perform_joint_calibration(
                        arm_side, "elbow",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint3"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_elbow
                    )
                    require_converged_joint(joint_res_elbow, context, f"{arm_side}_elbow")
                    if pass_idx == 1:
                        pass1_joint_results["elbow"] = joint_res_elbow
                    joint_res_elbow['arm_side'] = arm_side
                    joint_res_elbow['mode'] = "elbow"
                    joint_res_elbow['pass_idx'] = pass_idx

                    opt_elbow = joint_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "elbow", pass1_res_elbow if pass1_res_elbow else joint_res_elbow, joint_res_elbow,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_elbow['plot_path_combined'] = plot_path

                    emit_joint(joint_res_elbow)
                    _wait(0.5)

            else:
                # === v1.2 CALIBRATION SEQUENCE ===
                # 1. Calibrate J5 (Wrist Pitch) FIRST
                pass1_res_pitch = pass1_joint_results.get("wrist_pitch")
                if pass_idx == 2 and pass1_res_pitch and pass1_res_pitch.get("converged", False):
                    log(f"[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 ({pass1_res_pitch['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_pitch = pass1_res_pitch["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    joint_offsets_store[arm_side]["joint5"] = opt_pitch
                    emit_joint(pass1_res_pitch)
                else:
                    log(f"[FULL AUTO 1/3] Calibrating J5 (Wrist Pitch) first on v1.2 {arm_side} arm...")
                    for calibrator in [joint_calibrator, marker_calibrator]:
                        calibrator.joint_offsets[arm_side]["wrist_pitch"] = joint_offsets_store[arm_side]["joint5"]
                        calibrator.joint_offsets[arm_side]["wrist_roll"] = 0.0
                        calibrator.joint_offsets[arm_side]["wrist_yaw2"] = joint_offsets_store[arm_side]["joint6"]
                        calibrator.joint_offsets[arm_side]["elbow"] = joint_offsets_store[arm_side]["joint3"]

                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_pitch on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_pitch = joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_pitch",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint5"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_pitch
                    )
                    require_converged_joint(joint_res_pitch, context, f"{arm_side}_wrist_pitch")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_pitch"] = joint_res_pitch
                    joint_res_pitch['arm_side'] = arm_side
                    joint_res_pitch['mode'] = "wrist_pitch"
                    joint_res_pitch['pass_idx'] = pass_idx

                    opt_pitch = joint_res_pitch["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    joint_offsets_store[arm_side]["joint5"] = opt_pitch

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_pitch", pass1_res_pitch if pass1_res_pitch else joint_res_pitch, joint_res_pitch,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_pitch['plot_path_combined'] = plot_path

                    emit_joint(joint_res_pitch)
                    _wait(0.5)
                if is_stopped():
                    return

                # 2. Marker Bracket Sweeps (Axis 4, 6, 5) with calibrated J5
                joint_calibrator.current_calib_mode = None
                marker_calibrator.current_calib_mode = "marker"
                log(f"[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 {arm_side} arm (Pass {pass_idx}/2)...")
                log(f"[FULL AUTO] Moving {arm_side} arm to ready pose...")
                if not marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=log):
                    raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                if is_stopped():
                    return

                state = joint_calibrator.robot.get_state()
                model = joint_calibrator.robot.model()
                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                first_starting_pose = list(state.position[arm_idx])
                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                sweeps = run_bracket_sweeps(
                    marker_calibrator, arm_side, first_starting_pose, log=log, emit_status=emit_status,
                    save_debug=save_debug, pass_idx=pass_idx, is_stopped=is_stopped, labels=None
                )
                if sweeps is None:
                    return
                res_4, res_6, res_5 = sweeps[4], sweeps[6], sweeps[5]

                # 3. Compute Marker Bracket (1-time lock)
                log("\n[FULL AUTO] Computing unified marker bracket calibration for v1.2...")
                staged_pitch = joint_offsets_store[arm_side]["joint5"]
                staged_yaw2 = joint_offsets_store[arm_side]["joint6"]
                unified_res = marker_calibrator.compute_unified_bracket_calibration(
                    res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_or_yaw_deg=staged_yaw2, calib_pitch_deg=staged_pitch, log_callback=log
                )

                unified_res['res_5'] = res_5
                unified_res['res_6'] = res_6
                if res_4 is not None:
                    unified_res['res_4'] = res_4
                unified_res['arm_side'] = arm_side
                unified_res['pass_idx'] = pass_idx

                plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                plot_saved = marker_calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, arm_side, is_v13, plot_path)
                if plot_saved:
                    unified_res['plot_path_combined'] = plot_path

                x_m, y_m, z_m = unified_res['x_e'] / 1000.0, unified_res['y_e'] / 1000.0, unified_res['z_e'] / 1000.0
                new_vals = [x_m, y_m, z_m, unified_res['roll_e'], unified_res['pitch_e'], unified_res['yaw_e']]
                key = f"Tf_to_marker_{arm_side}"
                if bracket_accept_callback is not None:
                    bracket_accept_callback(arm_side, new_vals)
                else:
                    marker_calibrator.camera_config[key] = new_vals
                    joint_calibrator.camera_config[key] = new_vals

                emit_bracket(unified_res)
                _wait(0.5)
                if is_stopped():
                    return

                # 4. Calibrate J6 (Wrist Yaw 2) & Iteration 2 Verification
                pass1_res_yaw2 = pass1_joint_results.get("wrist_yaw2")
                if WRIST_YAW2_SOURCE == "bracket_sweeps":
                    # From the axis-6 / axis-5 bracket sweeps just taken (see WRIST_YAW2_SOURCE):
                    # no robot motion, so it is recomputed every pass and the pass evaluation
                    # below compares the two passes' values.
                    log(f"\n[FULL AUTO] Computing J6 (Wrist Yaw 2) from the bracket J6/J5 sweeps...")
                    joint_res_roll = joint_calibrator.wrist_yaw2_from_bracket_sweeps(
                        arm_side, res_6, res_5, log_callback=log, pass_idx=pass_idx
                    )
                    require_converged_joint(joint_res_roll, context, f"{arm_side}_wrist_yaw2")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_yaw2"] = joint_res_roll

                    opt_roll = joint_res_roll["recommended_joint_offset"]
                    log(f"[FULL AUTO] Staging J6 offset: {opt_roll:.4f}°")
                    joint_offsets_store[arm_side]["joint6"] = opt_roll
                    joint_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    marker_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_yaw2", pass1_res_yaw2 if pass1_res_yaw2 else joint_res_roll, joint_res_roll,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_roll['plot_path_combined'] = plot_path

                    joint_res_roll['arm_side'] = arm_side
                    joint_res_roll['mode'] = "wrist_yaw2"
                    joint_res_roll['pass_idx'] = pass_idx
                    emit_joint(joint_res_roll)
                    _wait(0.5)
                    if is_stopped():
                        return
                elif pass_idx == 2 and pass1_res_yaw2 and pass1_res_yaw2.get("converged", False):
                    log(f"[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 ({pass1_res_yaw2['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_roll = pass1_res_yaw2["recommended_joint_offset"]
                    joint_offsets_store[arm_side]["joint6"] = opt_roll
                    joint_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    marker_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    emit_joint(pass1_res_yaw2)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "wrist_yaw2"
                    log(f"\n[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...")
                    # Return to the ready pose first, as every other stage does. This one did not,
                    # and perform_joint_calibration centres both of its sweeps on wherever the arm
                    # happens to be, resetting only J6. The axis-5 marker sweep immediately before
                    # ends 40 deg away (MARKER_CONFIGS axis_5 runs 0 -> -40 deg), so the J5 arc was
                    # swept 40 deg low and walked the marker off the bottom of the frame: on
                    # 2026-09-21 it captured 22 of 30 deg on the right arm and 26 on the left, and
                    # a J5 arc cut short that unevenly biases the circle centre the J6 offset is
                    # solved from.
                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_yaw2", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_yaw2 on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_roll = joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_yaw2",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint6"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_yaw2
                    )
                    require_converged_joint(joint_res_roll, context, f"{arm_side}_wrist_yaw2")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_yaw2"] = joint_res_roll

                    opt_roll = joint_res_roll["recommended_joint_offset"]
                    log(f"[FULL AUTO] Staging J6 offset: {opt_roll:.4f}°")
                    joint_offsets_store[arm_side]["joint6"] = opt_roll
                    joint_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    marker_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_yaw2", pass1_res_yaw2 if pass1_res_yaw2 else joint_res_roll, joint_res_roll,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_roll['plot_path_combined'] = plot_path

                    joint_res_roll['arm_side'] = arm_side
                    joint_res_roll['mode'] = "wrist_yaw2"
                    joint_res_roll['pass_idx'] = pass_idx
                    emit_joint(joint_res_roll)
                    _wait(0.5)
                    if is_stopped():
                        return

                # 5. Calibrate J3 Elbow
                pass1_res_elbow = pass1_joint_results.get("elbow")
                if pass_idx == 2 and pass1_res_elbow and pass1_res_elbow.get("converged", False):
                    log(f"[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 ({pass1_res_elbow['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_elbow = pass1_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow
                    emit_joint(pass1_res_elbow)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "elbow"
                    log("[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...")
                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_elbow = joint_calibrator.perform_joint_calibration(
                        arm_side, "elbow",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint3"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_elbow
                    )
                    require_converged_joint(joint_res_elbow, context, f"{arm_side}_elbow")
                    if pass_idx == 1:
                        pass1_joint_results["elbow"] = joint_res_elbow
                    joint_res_elbow['arm_side'] = arm_side
                    joint_res_elbow['mode'] = "elbow"
                    joint_res_elbow['pass_idx'] = pass_idx

                    opt_elbow = joint_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "elbow", pass1_res_elbow if pass1_res_elbow else joint_res_elbow, joint_res_elbow,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_elbow['plot_path_combined'] = plot_path

                    emit_joint(joint_res_elbow)
                    _wait(0.5)

            # Pass Evaluation & Convergence Check
            j6_change = abs(joint_offsets_store[arm_side]["joint6"] - prev_j6)
            j5_change = abs(joint_offsets_store[arm_side]["joint5"] - prev_j5)
            j3_change = abs(joint_offsets_store[arm_side]["joint3"] - prev_j3)

            tf_vec_now = marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
            if tf_vec_now is not None and len(tf_vec_now) == 6:
                now_bracket_pos = np.array(tf_vec_now[:3]) * 1000.0
                now_bracket_rot = np.array(tf_vec_now[3:])
            else:
                now_bracket_pos = prev_bracket_pos
                now_bracket_rot = prev_bracket_rot

            if is_v13 and pass_idx == 1 and tf_vec_now is not None:
                # Pass 1 solved the bracket position before J6 was known (x 0.46 mm short at a 3.6 deg J6 offset,
                # 2026-10-16); re-solve it with the J6 offset staged since, so pass 2 is compared with like.
                solved = marker_calibrator.v13_bracket_position_from_radii(
                    res_5, res_6, arm_side, tuple(np.array(nominal_vec[:3]) * 1000.0))
                if solved is not None:
                    log(f"[INFO] {arm_side} pass 1 bracket position with the J6 offset found in pass 1: "
                        f"x {solved[0][0]:.2f}, z {solved[0][2]:.2f} mm (solved before J6: "
                        f"x {now_bracket_pos[0]:.2f}, z {now_bracket_pos[2]:.2f} mm).")
                    now_bracket_pos = np.array(solved[0])

            pos_change = np.linalg.norm(now_bracket_pos - prev_bracket_pos)

            # Compute rotation change in degrees
            R_prev = R_scipy.from_euler('ZYX', [prev_bracket_rot[2], prev_bracket_rot[1], prev_bracket_rot[0]], degrees=True)
            R_now = R_scipy.from_euler('ZYX', [now_bracket_rot[2], now_bracket_rot[1], now_bracket_rot[0]], degrees=True)
            rot_change = np.rad2deg(np.linalg.norm((R_now * R_prev.inv()).as_rotvec()))

            log(f"\n[PASS {pass_idx} EVALUATION] Staged parameter changes for {arm_side.upper()} Arm:")
            log(f"  * Joint 6 Change      : {j6_change:.4f}°")
            log(f"  * Joint 5 Change      : {j5_change:.4f}°")
            log(f"  * Joint 3 Change      : {j3_change:.4f}°")
            log(f"  * Bracket Pos Change  : {pos_change:.4f} mm")
            log(f"  * Bracket Rot Change  : {rot_change:.4f}°")

            # Convergence Criteria on Pass >= 2: joints < 0.10°, bracket pos < 0.5 mm, bracket rot < 0.15°
            if pass_idx >= 2:
                if j6_change < 0.10 and j5_change < 0.10 and j3_change < 0.10 and pos_change < 0.5 and rot_change < 0.15:
                    log(f"[PASS {pass_idx} EVALUATION] All parameters converged physically (Step changes < 0.10°).")
                    log(f"[PASS {pass_idx} EVALUATION] Calibration completed in Pass {pass_idx}!")
                    break
                else:
                    log(f"[PASS {pass_idx} EVALUATION] [WARN] Pass-to-pass changes exceed tolerance "
                        f"(J6: {j6_change:.3f}°, J5: {j5_change:.3f}°, J3: {j3_change:.3f}°, "
                        f"bracket pos: {pos_change:.3f} mm, bracket rot: {rot_change:.3f}°; "
                        f"limits 0.10° / 0.5 mm / 0.15°). Keeping Pass {pass_idx} values.")

            # Update prev values for next pass check
            prev_j6 = joint_offsets_store[arm_side]["joint6"]
            prev_j5 = joint_offsets_store[arm_side]["joint5"]
            prev_j3 = joint_offsets_store[arm_side]["joint3"]
            prev_bracket_pos = now_bracket_pos
            prev_bracket_rot = now_bracket_rot

        log(f"[INFO] {arm_side.upper()} arm sequential calibration completed successfully.")
        if is_stopped():
            return
        _wait(1.0)

    if symmetry_callback is not None:
        try:
            symmetry_callback()
        except Exception as error:
            log(f"[WARN] Bracket symmetry step failed: {error}")

    log("\n" + "=" * 50)
    log("   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!")
    log("=" * 50 + "\n")

    # Print Final Calibrated Results Report in the same style as simulated ground truth
    log("[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):")
    for arm in ["right", "left"]:
        j_store = joint_offsets_store.get(arm, {})
        j6_cal = j_store.get("joint6", 0.0)
        j5_cal = j_store.get("joint5", 0.0)
        j3_cal = j_store.get("joint3", 0.0)

        tf_vec = marker_calibrator.camera_config.get(f"Tf_to_marker_{arm}")
        if tf_vec is not None and len(tf_vec) == 6:
            x_cal = tf_vec[0] * 1000.0
            y_cal = tf_vec[1] * 1000.0
            z_cal = tf_vec[2] * 1000.0
            r_cal = tf_vec[3]
            p_cal = tf_vec[4]
            y_cal_deg = tf_vec[5]
        else:
            ver_key = "1.3" if is_v13 else "1.2"
            nominal_vec = joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
            x_cal = nominal_vec[0] * 1000.0
            y_cal = nominal_vec[1] * 1000.0
            z_cal = nominal_vec[2] * 1000.0
            r_cal = nominal_vec[3]
            p_cal = nominal_vec[4]
            y_cal_deg = nominal_vec[5]

        ver_key = "1.3" if is_v13 else "1.2"
        nominal_vec = joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
        x_nom = nominal_vec[0] * 1000.0
        y_nom = nominal_vec[1] * 1000.0
        z_nom = nominal_vec[2] * 1000.0
        r_nom = nominal_vec[3]
        p_nom = nominal_vec[4]
        y_nom_deg = nominal_vec[5]

        dx = x_cal - x_nom
        dy = y_cal - y_nom
        dz = z_cal - z_nom

        R_ideal = R_scipy.from_euler('ZYX', [y_nom_deg, p_nom, r_nom], degrees=True)
        R_actual = R_scipy.from_euler('ZYX', [y_cal_deg, p_cal, r_cal], degrees=True)
        R_offset = R_actual * R_ideal.inv()
        yaw_off, pitch_off, roll_off = R_offset.as_euler('ZYX', degrees=True)

        roll_off = (roll_off + 180) % 360 - 180
        pitch_off = (pitch_off + 180) % 360 - 180
        yaw_off = (yaw_off + 180) % 360 - 180

        log(f"  --- {arm.upper()} ARM ---")
        log(f"  * Bracket Pos: X: {dx:+.1f}, Y: {dy:+.1f}, Z: {dz:+.1f} mm")
        log(f"  * Bracket Rot: R: {roll_off:+.2f}, P: {pitch_off:+.2f}, Y: {yaw_off:+.2f} deg")
        log(f"  * Joint Offsets: Joint 6: {j6_cal:+.2f}°, Joint 5: {j5_cal:+.2f}°, Joint 3: {j3_cal:+.2f}°")
    log("==================================================\n")


def execute_step1_sequence(joint_calibrator, marker_calibrator, joint_offsets_store,
                           stop_event=None, log_callback=None, status_callback=None,
                           bracket_finished_callback=None, joint_finished_callback=None,
                           save_debug=False, context=None,
                           bracket_accept_callback=None, symmetry_callback=None):
    from copy import deepcopy
    from .result import SequenceContext, SequenceCancelled
    context = context or SequenceContext("step1", stop_event)
    try:
        context.check_cancelled()
        _execute_step1_sequence(
            joint_calibrator, marker_calibrator, joint_offsets_store,
            context.stop_event, log_callback, status_callback,
            bracket_finished_callback, joint_finished_callback, save_debug, context,
            bracket_accept_callback, symmetry_callback,
        )
        context.check_cancelled()
        context.complete("joint_offsets", joint_offsets_store)
        context.complete("camera_config", marker_calibrator.camera_config)
        context.result.finish("completed")
    except SequenceCancelled as error:
        context.result.partial.update(error.partial)
        context.result.finish("cancelled", str(error))
    except Exception as error:
        context.result.finish("cancelled" if context.stop_event.is_set() else "failed", str(error))
    finally:
        context.result.partial["joint_offsets"] = deepcopy(joint_offsets_store)
        context.result.partial["camera_config"] = deepcopy(marker_calibrator.camera_config)
        for name, calibrator in (("joint", joint_calibrator), ("marker", marker_calibrator)):
            data = getattr(calibrator, "partial_data", None)
            if isinstance(data, dict):
                context.result.partial[name] = deepcopy(data)
    return context.result.snapshot()
