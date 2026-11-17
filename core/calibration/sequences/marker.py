import os
from core.storage import CONFIG_PATHS


def run_marker(core, context, arm_side="right", use_head_tracking=False,
               tolerance=0.5, save_debug=False):
    c = core.marker_calibrator
    context.check_cancelled()
    if not c.perform_move_to_ready_pose(arm_side, mode="marker", log_callback=core.log_msg):
        context.check_cancelled()
        raise RuntimeError("Marker ready pose failed")
    model = core.robot.model()
    indices = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
    initial = list(core.robot.get_state().position[indices])
    taught = c.user_taught_ready_poses.get(arm_side, {}).get("marker")
    if taught is not None:
        initial = list(taught)
    sweeps = {}
    for axis in (4, 6, 5):
        context.check_cancelled()
        result = c.perform_calibration_sweep(
            arm_side, axis, log_callback=core.log_msg, status_callback=core.emit_detection,
            use_head_tracking=use_head_tracking, save_debug=save_debug,
            initial_joint_pos=initial,
        )
        context.checkpoint(f"axis_{axis}", result)
        context.check_cancelled()
        if not result:
            raise RuntimeError(f"Marker axis {axis} sweep failed")
        result.update(axis_mode=axis, axis=result["axis_opt"])
        sweeps[axis] = result
        context.checkpoint(f"axis_{axis}", result)
    result = c.compute_unified_bracket_calibration(
        sweeps[5], sweeps[6], arm_side, tolerance=tolerance,
        marker_data_4=sweeps[4], calib_roll_deg=0.0, calib_pitch_deg=0.0, log_callback=core.log_msg,
    )
    context.checkpoint("marker_fit", result)
    context.check_cancelled()
    if not result:
        raise RuntimeError("Marker fit failed")
    result.update(res_4=sweeps[4], res_5=sweeps[5], res_6=sweeps[6])
    path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
    if c.generate_marker_plot(sweeps[5], sweeps[6], sweeps[4], result, arm_side, c.is_v13(), path):
        result["plot_path_combined"] = path
    context.check_cancelled()
    core.accept_marker_result(arm_side, result)
    # Bracket-only runs calibrate one arm at a time; symmetrize once the second arm lands.
    symmetry = core.apply_bracket_symmetry(core.log_msg)
    if symmetry is not None:
        result["bracket_symmetry"] = symmetry
        context.checkpoint("bracket_symmetry", symmetry)
    context.complete("marker", result)
    return result
