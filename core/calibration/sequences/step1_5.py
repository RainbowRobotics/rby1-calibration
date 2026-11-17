"""Head/camera step shared by manual and full execution."""


def run_step1_5(core, context, prepare=True, **options):
    context.check_cancelled()
    calibrator = core.head_camera_calibrator
    head_indices = getattr(core.model, "head_idx", None)
    if not core.include_head_motion or (core.model is not None and (head_indices is None or len(head_indices) == 0)):
        result = {"success": True, "skipped": True, "reason": "Head motion disabled or no head hardware"}
        context.complete("head_camera", result)
        return result
    if core.robot is None or core.observer is None:
        raise RuntimeError("Robot and camera/marker source must be connected before head sweep")
    if options.get("num_steps", 11) < 5:
        raise ValueError("Head sweep requires at least 5 steps")
    if core.include_head_motion and prepare:
        if not calibrator.perform_move_to_ready_pose(
            arm_side="both", log_callback=core.log_msg, stop_event=context.stop_event
        ):
            context.check_cancelled()
            raise RuntimeError("Head/camera ready pose failed")
    context.check_cancelled()
    # A fresh Step 1.5 re-identifies the head offsets from zero: never let a stale value (e.g. one
    # loaded from setting.yaml) survive into Step 2 if this sweep fails or is cancelled.
    core.joint_offsets_store["head"] = {"pan": 0.0, "tilt": 0.0}
    calibrator.calibrated_results = None
    # The sweep is told when the robot is already at the ready pose so a marker missing at the
    # start is prompted for there, without driving to the same pose a second time.
    options.setdefault("at_ready_pose", bool(core.include_head_motion and prepare))
    result = calibrator.perform_head_sweep(
        log_callback=core.log_msg, stop_event=context.stop_event, **options
    )
    context.checkpoint("head_camera", result)
    context.check_cancelled()
    if not result or not result.get("success"):
        raise RuntimeError("Head/camera calibration did not produce a valid result")
    context.complete("head_camera", result)
    # Stage the result in memory for Step 2. File and motor application are explicit.
    core.accept_head_result(result)
    return result
