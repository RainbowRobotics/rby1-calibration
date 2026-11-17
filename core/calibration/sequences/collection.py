from core.robot.motion import build_incremental_motion_plan, execute_auto_motion_step, move_to_auto_ready_pose, verify_and_align_head_at_ready_pose
from ..data import (MarkersTooCloseError, capture_one_sample, get_both_arm_config, get_head_config,
                    load_min_marker_x_gap_m, load_step2_plan_options, load_step2_wrist_diversity, marker_x_gap_m)

# How often the gap is re-measured and sent to the UI while the operator widens the arms.
SPACING_MONITOR_PERIOD_S = 0.2


def current_marker_x_gap(core, sampling_time=0):
    """Camera-x gap between the two markers now, or None while either marker is out of view."""
    transforms = {}
    for side in ("right", "left"):
        result = core.observer.get_marker_transform(sampling_time=sampling_time, side=side)
        transforms[side] = result[0] if isinstance(result, list) and result else None
    if transforms["right"] is None or transforms["left"] is None:
        return None
    return marker_x_gap_m(transforms["right"], transforms["left"])


def ensure_marker_spacing(core, context):
    """At the Step 2 init pose, after both markers are verified visible: make sure they are at
    least step2.min_marker_x_gap_m apart along the camera x axis before any Step 2 motion.

    Arms whose markers start closer than that can collide while the plan moves them. When they
    are too close the operator is asked to widen them, with the gap monitored live, and confirms
    once it is wide enough. That confirmation is final: Step 2 then runs as usual, with no
    further spacing checks."""
    min_gap, from_config = load_min_marker_x_gap_m()
    if not from_config:
        core.log_msg(f"[WARN] step2.min_marker_x_gap_m is missing from setting.yaml; "
                     f"using the default {min_gap * 100:.1f} cm.")
    gap = current_marker_x_gap(core, sampling_time=0 if core.observer.sim else 1)
    if gap is None:
        raise RuntimeError(f"Both markers must be in view to check their spacing "
                           f"(min {min_gap * 100:.1f} cm) before Step 2 motion.")
    core.log_msg(f"[Step2] Marker spacing (camera x): {gap * 100:.1f} cm (minimum {min_gap * 100:.1f} cm)")
    context.checkpoint("marker_x_gap", {"gap_m": gap, "min_gap_m": min_gap})
    if gap >= min_gap:
        return gap
    prompt = getattr(core, "prompt_marker_spacing", None)
    if prompt is None:
        raise MarkersTooCloseError(gap, min_gap)
    core.log_msg(f"[WARN] Markers are {gap * 100:.1f} cm apart, below {min_gap * 100:.1f} cm. "
                 f"Asking the operator to widen the arms.")
    # The UI opens the monitoring dialog and returns at once; the gap is measured here, in the
    # core, and sent to it until the operator confirms or cancels.
    session = prompt(gap, min_gap)
    try:
        while not session["done"].wait(SPACING_MONITOR_PERIOD_S):
            context.check_cancelled()
            core.emit("marker_gap", {"gap_m": current_marker_x_gap(core), "min_gap_m": min_gap})
    finally:
        # Lets the dialog close itself when monitoring stopped without the operator (stop, error).
        session["closed"] = True
    context.check_cancelled()
    if not session.get("accepted"):
        raise RuntimeError("Marker spacing adjustment cancelled by user")
    confirmed = session.get("gap_m")
    core.log_msg(f"[Step2] Marker spacing widened by the operator"
                 + (f" to {confirmed * 100:.1f} cm" if confirmed is not None else "")
                 + "; continuing Step 2 without further spacing checks.")
    context.checkpoint("marker_x_gap", {"gap_m": confirmed, "min_gap_m": min_gap, "widened": True})
    return confirmed


def run_collection(core, context, prepare=False, plan=None, start_index=0, max_samples=None,
                   skip_joint_pose=False):
    context.check_cancelled()
    if core.robot is None:
        raise RuntimeError("Robot is not connected")
    if start_index < 0 or (max_samples is not None and max_samples < 1):
        raise ValueError("start_index must be nonnegative and max_samples must be positive")
    if prepare:
        move_to_auto_ready_pose(core.robot, ["right", "left"], include_head_motion=core.include_head_motion,
                                robot_version=core.get_robot_version(), skip_joint_pose=skip_joint_pose)
        context.check_cancelled()
        verify_and_align_head_at_ready_pose(
            core.robot, core.observer, core.model, ["right", "left"], 10,
            include_head_motion=core.include_head_motion,
            prompt_teaching_cb=core.prompt_teaching, log_cb=core.log_msg,
        )
        context.check_cancelled()
        ensure_marker_spacing(core, context)
    context.check_cancelled()
    plan_options = load_step2_plan_options()
    if plan is None:
        wrist_diversity = []
        if core.include_head_motion:
            wrist_diversity, enabled = load_step2_wrist_diversity(core.get_robot_version())
            if enabled:
                core.log_msg(f"[INFO] Step 2 wrist-diversity poses: {len(wrist_diversity)} "
                             f"(step2.wrist_diversity_poses on, v{core.get_robot_version()}).")
        plan = build_incremental_motion_plan(
            core.robot, core.robot.get_dynamics(), core.auto_config, ["right", "left"],
            include_head_motion=core.include_head_motion, wrist_diversity=wrist_diversity,
            skip_half_steps=plan_options["skip_half_steps"],
        )
        if plan_options["skip_half_steps"]:
            core.log_msg("[INFO] Step 2 half-size steps skipped (step2.skip_half_steps on).")
    sample_time = 0 if core.observer.sim else plan_options["sample_time_s"]
    core.log_msg(f"[INFO] Step 2 capture averages {sample_time:g} s per pose (step2.sample_time_s).")
    # 2026-10-15: a simulator run ended with "No valid calibration samples collected" and nothing in
    # the log said whether the plan was empty or every capture failed.
    core.log_msg(f"[INFO] Step 2 plan: {len(plan)} poses (starting at {start_index + 1}).")
    if len(plan) <= start_index:
        raise RuntimeError(f"Step 2 pose plan is empty ({len(plan)} poses, start {start_index + 1}). The plan is "
                           f"built only while the right hand starts within max_x = {core.auto_config.max_x} m.")
    arm = get_both_arm_config(core.model, version=core.get_robot_version())
    head = get_head_config(core.model)["head_idx"] if core.include_head_motion else None
    samples = []
    context.result.partial["samples"] = samples
    failures = 0
    missed = 0
    for index, motion in enumerate(plan[start_index:], start=start_index):
        context.check_cancelled()
        # Optional steps (wrist-diversity poses) are skipped, not counted as failures, when a joint
        # target is near its limit or a marker is out of view.
        optional = isinstance(motion, dict) and bool(motion.get("optional"))
        moved = execute_auto_motion_step(core.robot, core.auto_config, motion, ["right", "left"],
                                         include_head_motion=core.include_head_motion)
        context.check_cancelled()
        if moved is None and optional:
            core.log_msg(f"[INFO] Step 2 pose {index + 1} skipped (joint limit): {motion.get('desc', '')}")
            transforms = None
        else:
            q_arm, q_head, transforms = capture_one_sample(
                core.robot, arm["arm_idx"], core.observer,
                sampling_time=sample_time, side="all", head_idx=head,
            )
        # Finish recording an acquired sample before acknowledging cancellation.
        if transforms is not None:
            # home_epoch: the robot zero this sample was taken at (CalibrationCore.home_epoch).
            samples.append({"q_arm": q_arm, "q_head": q_head, "marker": transforms,
                            "motion_index": index, "frame": core.observer.last_frame.copy(),
                            "home_epoch": getattr(core, "home_epoch", None),
                            "baseline": isinstance(motion, dict) and motion.get("type") == "restore_baseline"})
            failures = 0
        elif optional:
            if moved is not None:
                core.log_msg(f"[INFO] Step 2 pose {index + 1} skipped (marker not in view): {motion.get('desc', '')}")
        else:
            failures += 1
            missed += 1
        context.checkpoint("next_motion_index", index + 1)
        # Tell the UI after every pose. Without this the sample counter sat at 0 for the whole
        # run and only jumped to 64 at the end, so there was no way to watch it progress.
        core.emit("progress", {"stage": "collect", "pose_index": index + 1,
                               "pose_total": len(plan), "samples": len(samples),
                               "failures": failures})
        context.check_cancelled()
        if failures >= 3:
            raise RuntimeError("Marker not detected at three consecutive poses")
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError(f"No valid calibration samples collected ({len(plan) - start_index} poses in the plan, "
                           f"markers missed at {missed})")
    context.complete("samples", samples)
    return samples
