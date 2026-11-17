"""Post-Step-2 check.

Returns to the Step 2 ready pose -- a left/right symmetric Cartesian pose -- and reports two
independent numbers:

1. model vs camera: how well the freshly calibrated model predicts the marker poses the camera
   actually sees. This is the calibration's own consistency.
2. left/right symmetry: where the two calibrated end effectors really are at a pose that is
   commanded symmetrically. This is the mismatch the operator sees by eye, and it is what the
   marker bracket errors show up in.

Both are also computed with the offsets zeroed so the log shows what the calibration changed.
The ready pose is commanded with a tight Cartesian tolerance: at the default 5 mm stopping
tolerance the two arms can settle several mm apart, which would swamp the number we want.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from core.robot.motion import move_to_auto_ready_pose
from ..data import capture_one_sample, get_both_arm_config, get_head_config
from ..CalibratorBase import BaseCalibrator

# Reflection across the robot's sagittal plane (y -> -y).
MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])

# At a mirrored pose the left arm's encoders read q_left[j] == MIRROR_JOINT_SIGN[j] * q_right[j],
# so a physically symmetric set of home offsets has to satisfy the same relation. Whatever
# breaks it is what tilts the robot at a pose the operator commanded symmetrically.
MIRROR_JOINT_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

VERIFY_POS_TOLERANCE_M = 0.0005
VERIFY_ORI_TOLERANCE_RAD = 0.005


def _fk(dyn, model, q_full, ee, base="link_torso_5"):
    state = dyn.make_state([base, ee], model.robot_joint_names)
    state.set_q(q_full)
    dyn.compute_forward_kinematics(state)
    return dyn.compute_transformation(state, 0, 1)


def _rot_deg(R):
    return float(np.linalg.norm(R_scipy.from_matrix(R).as_rotvec(degrees=True)))


def _arm_offsets_rad(optimization, side):
    key = f"{side}_arm_joint_offset_deg"
    values = (optimization or {}).get(key)
    if values is None:
        return np.zeros(7)
    return np.radians(np.asarray(values, dtype=float))


def _head_offsets_rad(optimization):
    values = (optimization or {}).get("head_joint_offset_deg")
    if values is None:
        return np.zeros(2)
    return np.radians(np.asarray(values, dtype=float))


def _evaluate(core, cfg, head_idx, q_arm, q_head, markers, camera, apply_offsets, optimization):
    """Model-vs-camera residuals and the true left/right symmetry at the captured pose."""
    model = core.model
    dyn = core.robot.get_dynamics()
    camera_link, camera_vec = camera
    T_mount_cam = BaseCalibrator.make_transform(camera_vec)

    q_full = np.zeros(len(model.robot_joint_names))
    q_full[cfg["arm_idx"]] = q_arm
    if head_idx is not None and q_head is not None:
        for i, idx in enumerate(list(head_idx)):
            if i < len(q_head):
                q_full[idx] = q_head[i]
    if apply_offsets:
        arm_idx = list(cfg["arm_idx"])
        q_full[arm_idx[:7]] += _arm_offsets_rad(optimization, "right")
        q_full[arm_idx[7:]] += _arm_offsets_rad(optimization, "left")
        if head_idx is not None and q_head is not None:
            head_off = _head_offsets_rad(optimization)
            for i, idx in enumerate(list(head_idx)):
                if i < len(head_off):
                    q_full[idx] += head_off[i]

    T_t5_cam = _fk(dyn, model, q_full, camera_link) @ T_mount_cam

    per_arm = {}
    T_ee = {}
    T_ee_seen = {}
    for i, side in enumerate(("right", "left")):
        bracket = core.get_marker_bracket(side)
        if bracket is None:
            bracket = cfg["ee_to_marker_nom"][side]
        T_bracket = BaseCalibrator.make_transform(list(bracket))
        T_ee[side] = _fk(dyn, model, q_full, cfg["ee_links"][side])
        if markers is None:
            continue
        T_model = np.linalg.inv(T_t5_cam) @ T_ee[side] @ T_bracket
        delta = np.linalg.inv(T_model) @ markers[i]
        per_arm[side] = {
            "pos_err_mm": float(np.linalg.norm(delta[:3, 3]) * 1000.0),
            "rot_err_deg": _rot_deg(delta[:3, :3]),
        }
        # Where the camera says that flange is, by undoing the calibrated bracket. Comparing
        # flanges rather than markers keeps the two sides in the same frame convention -- the
        # marker frames themselves sit 180 deg apart, which would swamp any rotation metric.
        T_ee_seen[side] = T_t5_cam @ markers[i] @ np.linalg.inv(T_bracket)

    # Mirror the left flange into the right arm's half of the workspace and compare.
    symmetry = _mirror_pair(T_ee["right"], T_ee["left"])
    # The same comparison on what the camera actually saw. Once the arms stand at a genuinely
    # symmetric pose, whatever is left here is bracket mismatch plus camera error.
    measured = _mirror_pair(T_ee_seen["right"], T_ee_seen["left"]) if markers is not None else None

    return {"per_arm": per_arm, "symmetry": symmetry, "measured_symmetry": measured}


def _mirror_pair(T_right, T_left):
    T_left_mirrored = MIRROR @ T_left @ MIRROR
    d_pos_mm = (T_left_mirrored[:3, 3] - T_right[:3, 3]) * 1000.0
    return {
        "dx_mm": float(d_pos_mm[0]),
        "dy_mm": float(d_pos_mm[1]),
        "dz_mm": float(d_pos_mm[2]),
        "dist_mm": float(np.linalg.norm(d_pos_mm)),
        "drot_deg": _rot_deg(T_right[:3, :3].T @ T_left_mirrored[:3, :3]),
        "separation_mm": float((T_left[1, 3] - T_right[1, 3]) * 1000.0),
    }


def run_post_step2_verification(core, context, optimization=None, minimum_time=5.0,
                                pos_tolerance_m=VERIFY_POS_TOLERANCE_M,
                                ori_tolerance_rad=VERIFY_ORI_TOLERANCE_RAD, reference=None):
    """reference: (q_arm, q_head) of the last Step 2 sample, taken back at the ready pose. Given, the robot does not
    move to the ready pose again (user, 2026-10-27): the right arm's joints mirrored onto the left stand in for
    it, and the ready-pose model-vs-camera comparison is left out."""
    log = core.log_msg
    if core.robot is None or core.model is None:
        raise RuntimeError("Robot is not connected")

    log("")
    log("=" * 58)
    log("   POST-STEP2 VERIFICATION AT THE READY POSE")
    log("=" * 58)

    cfg = get_both_arm_config(core.model, version=core.get_robot_version())
    head_idx = get_head_config(core.model)["head_idx"] if core.include_head_motion else None
    if reference is None:
        context.check_cancelled()
        move_to_auto_ready_pose(core.robot, ["right", "left"], minimum_time=minimum_time,
                                include_head_motion=core.include_head_motion,
                                robot_version=core.get_robot_version(),
                                pos_tolerance_m=pos_tolerance_m,
                                ori_tolerance_rad=ori_tolerance_rad)
        context.check_cancelled()
        q_arm, q_head, markers = capture_one_sample(
            core.robot, cfg["arm_idx"], core.observer,
            sampling_time=0 if core.observer.sim else 1, side="all", head_idx=head_idx,
        )
        if markers is None:
            log("[WARN] Verification skipped: both markers were not visible at the ready pose.")
            return None
    else:
        q_ref = np.asarray(reference[0], dtype=float)
        q_arm = np.concatenate([q_ref[:7], MIRROR_JOINT_SIGN * q_ref[:7]])
        q_head = None if reference[1] is None or head_idx is None else np.asarray(reference[1], dtype=float)
        markers = None

    # Without a head the camera sits on the fixed head base, not on the pan/tilt link, and it is
    # described by head_base_to_cam. Step 2 already branches this way; reading the head-mounted
    # transform here would have reported numbers computed from the wrong frame.
    if core.include_head_motion and head_idx is not None:
        camera = ("link_head_2", (optimization or {}).get("mount_to_cam_new")
                  or core.marker_calibrator.camera_config.get("mount_to_cam", cfg.get("mount_to_cam_nom")))
    else:
        camera = ("link_head_0", (optimization or {}).get("head_base_to_cam_new")
                  or core.marker_calibrator.camera_config.get("head_base_to_cam",
                                                              cfg.get("head_base_to_cam_nom")))
    if camera[1] is None:
        log("[WARN] Verification skipped: no camera transform available.")
        return None
    log(f" camera frame: {camera[0]} + {[round(float(v), 5) for v in camera[1]]}")

    calibrated = _evaluate(core, cfg, head_idx, q_arm, q_head, markers, camera, True, optimization)
    baseline = _evaluate(core, cfg, head_idx, q_arm, q_head, markers, camera, False, optimization)

    if markers is not None:
        log(" model vs camera (marker pose the model predicts vs the one measured):")
        for side in ("right", "left"):
            before = baseline["per_arm"][side]
            after = calibrated["per_arm"][side]
            log(f"   {side:5s}: uncalibrated {before['pos_err_mm']:6.2f} mm / {before['rot_err_deg']:5.2f} deg"
                f"   ->  calibrated {after['pos_err_mm']:6.2f} mm / {after['rot_err_deg']:5.2f} deg")
    else:
        log(" (no second move to the ready pose: its pose is the last Step 2 sample, right arm mirrored onto the"
            " left; the markers are measured at the offset-corrected pose below)")

    sym = calibrated["symmetry"]
    log(" where the two flanges really are, at a pose commanded symmetrically:")
    log(f"   front-back (x): {sym['dx_mm']:+6.2f} mm")
    log(f"   separation (y): {sym['separation_mm']:+7.2f} mm  (mirror residual {sym['dy_mm']:+.2f} mm)")
    log(f"   up-down    (z): {sym['dz_mm']:+6.2f} mm")
    log(f"   total offset  : {sym['dist_mm']:6.2f} mm")
    log(f"   (the encoders themselves read symmetric to {baseline['symmetry']['dist_mm']:.2f} mm, "
        f"so everything above comes from the calibrated offsets)")

    joint_asym = _joint_mirror_asymmetry(optimization, q_arm, log)

    corrected = _move_to_offset_corrected_pose(core, context, cfg, head_idx, q_arm, q_head,
                                               camera, optimization, minimum_time, log)
    log("=" * 58)
    log("")

    result = {"calibrated": calibrated, "baseline": baseline,
              "joint_mirror_asymmetry_deg": joint_asym,
              "corrected_pose": corrected,
              "q_arm": np.asarray(q_arm).tolist(),
              "q_head": np.asarray(q_head).tolist() if q_head is not None else None}
    context.checkpoint("verification", result)
    return result


def _link_separations(core, cfg, head_idx, q_arm, q_head):
    """Left-minus-right y distance of every arm link, used as the clearance guard."""
    dyn = core.robot.get_dynamics()
    q_full = np.zeros(len(core.model.robot_joint_names))
    q_full[cfg["arm_idx"]] = q_arm
    if head_idx is not None and q_head is not None:
        for i, idx in enumerate(list(head_idx)):
            if i < len(q_head):
                q_full[idx] = q_head[i]
    out = []
    for k in range(7):
        p_r = _fk(dyn, core.model, q_full, f"link_right_arm_{k}")[:3, 3]
        p_l = _fk(dyn, core.model, q_full, f"link_left_arm_{k}")[:3, 3]
        out.append(float((p_l[1] - p_r[1]) * 1000.0))
    p_r = _fk(dyn, core.model, q_full, cfg["ee_links"]["right"])[:3, 3]
    p_l = _fk(dyn, core.model, q_full, cfg["ee_links"]["left"])[:3, 3]
    out.append(float((p_l[1] - p_r[1]) * 1000.0))
    return np.array(out)


def _move_to_offset_corrected_pose(core, context, cfg, head_idx, q_arm, q_head,
                                   camera, optimization, minimum_time, log):
    """Command the ready pose again, this time corrected by the offsets Step 2 just produced.

    The Cartesian move lands where the *encoders* read symmetric; subtracting the offsets from
    that joint command makes the arms stand where they are *physically* symmetric, which is the
    pose an operator judges by eye. Refuses to move if it would bring the arms closer together.
    """
    off_right = _arm_offsets_rad(optimization, "right")
    off_left = _arm_offsets_rad(optimization, "left")
    if not np.any(off_right) and not np.any(off_left):
        return None

    q_arm = np.asarray(q_arm, dtype=float)
    q_cmd = q_arm.copy()
    q_cmd[:7] -= off_right
    q_cmd[7:] -= off_left
    q_head_cmd = None
    if head_idx is not None and q_head is not None:
        q_head_cmd = np.asarray(q_head, dtype=float) - _head_offsets_rad(optimization)[:len(q_head)]

    # Measured from where the robot stands now (the last Step 2 pose when the ready-pose move is skipped).
    state = np.asarray(core.robot.get_state().position, dtype=float)
    q_head_now = None if head_idx is None else state[list(head_idx)]
    sep_now = _link_separations(core, cfg, head_idx, state[list(cfg["arm_idx"])], q_head_now)
    sep_new = _link_separations(core, cfg, head_idx, q_cmd, q_head_cmd)
    closing = sep_new - sep_now
    if np.min(sep_new) < 50.0 or np.min(closing) < -5.0:
        log(f"[WARN] Offset-corrected pose skipped: it would bring the arms closer together "
            f"(worst link loses {-np.min(closing):.1f} mm, minimum separation {np.min(sep_new):.1f} mm).")
        return None

    log("")
    log(" moving to the offset-corrected ready pose (arms separate by "
        f"{np.min(closing):+.1f}..{np.max(closing):+.1f} mm; "
        f"largest joint change {np.degrees(np.abs(q_cmd - q_arm)).max():.2f} deg)")
    context.check_cancelled()
    ok = core.marker_calibrator.movej(
        core.robot, torso=None,
        right_arm=list(q_cmd[:7]), left_arm=list(q_cmd[7:]),
        head=list(q_head_cmd) if q_head_cmd is not None else None,
        minimum_time=minimum_time, apply_offsets=False,
    )
    context.check_cancelled()
    if not ok:
        log("[WARN] Offset-corrected pose move failed; leaving the robot where it is.")
        return None

    q_arm2, q_head2, markers2 = capture_one_sample(
        core.robot, cfg["arm_idx"], core.observer,
        sampling_time=0 if core.observer.sim else 1, side="all", head_idx=head_idx,
    )
    if markers2 is None:
        log("[WARN] Markers not visible at the offset-corrected pose.")
        return None

    value = _evaluate(core, cfg, head_idx, q_arm2, q_head2, markers2, camera, True, optimization)
    fk_sym, meas = value["symmetry"], value["measured_symmetry"]
    log(" at the offset-corrected pose (this is what the robot physically looks like now):")
    log(f"   flanges, from the model : x {fk_sym['dx_mm']:+6.2f}, y {fk_sym['dy_mm']:+6.2f}, "
        f"z {fk_sym['dz_mm']:+6.2f} mm  -> {fk_sym['dist_mm']:5.2f} mm")
    log(f"   markers, as measured    : x {meas['dx_mm']:+6.2f}, y {meas['dy_mm']:+6.2f}, "
        f"z {meas['dz_mm']:+6.2f} mm  -> {meas['dist_mm']:5.2f} mm, {meas['drot_deg']:5.2f} deg")
    log(f"   marker separation       : {meas['separation_mm']:+8.2f} mm")
    log("   (the model line is ~0 by construction; the measured line is what is left --"
        " bracket mismatch plus camera error)")
    value["q_arm"] = np.asarray(q_arm2).tolist()
    value["q_head"] = np.asarray(q_head2).tolist() if q_head2 is not None else None
    return value


def _joint_mirror_asymmetry(optimization, q_arm, log):
    """Which joint's offset breaks left/right symmetry, and by how much."""
    right = (optimization or {}).get("right_arm_joint_offset_deg")
    left = (optimization or {}).get("left_arm_joint_offset_deg")
    if right is None or left is None:
        return None
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    if right.shape != (7,) or left.shape != (7,):
        return None

    q = np.asarray(q_arm, dtype=float)
    if q.shape == (14,):
        pose_residual = np.degrees(np.abs(q[7:] - MIRROR_JOINT_SIGN * q[:7])).max()
        if pose_residual > 1.0:
            log(f" [skip] joint symmetry table: the achieved pose is not mirrored "
                f"(worst joint off by {pose_residual:.2f} deg).")
            return None

    asym = left - MIRROR_JOINT_SIGN * right
    log(" home offset symmetry (left minus the mirror of right; 0 means perfectly symmetric):")
    log("   joint :    J0     J1     J2     J3     J4     J5     J6")
    log("   right : " + " ".join(f"{v:+6.3f}" for v in right))
    log("   left  : " + " ".join(f"{v:+6.3f}" for v in left))
    log("   asym  : " + " ".join(f"{v:+6.3f}" for v in asym) + "  deg")
    worst = int(np.argmax(np.abs(asym)))
    log(f"   worst : J{worst} at {asym[worst]:+.3f} deg")
    return [float(v) for v in asym]
