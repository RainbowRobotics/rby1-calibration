from core.storage import ResultStorage
from core.storage import FileStorage
from core.storage import StoragePaths
from core.storage import CONFIG_PATHS, ConfigStorage
import os
import json
import datetime
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from ..data import (format_mount_constraint, get_arm_config, get_both_arm_config, get_head_config,
                    load_camera_mount_constraint, load_step2_head_tilt_anchor_weight, load_step2_min_rot_noise_deg,
                    mount_range_ends, validate_dataset)
from ..calibration_optimizer import HEAD_ANCHOR_WEIGHT_TILT, QPCalibrationOptimizer
from ..observability import format_weak_directions, step2_weak_directions
from ..CalibratorBase import BaseCalibrator
D2R = np.pi / 180.0

def optimize_step2(
    self,
    active_arms,
    optimize_head,
    optimize_camera,
    q_arm_list,
    q_head_list,
    T_meas_list,
    result_path,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    solver_type="QP Solver",
    use_sag=False,
):
    if self.model is None:
        raise RuntimeError("Robot is not connected.")

    if len(q_arm_list.shape) == 2 and q_arm_list.shape[1] == 7 and len(active_arms) == 2:
        self.log_msg("[WARN] q_arm_list has 7 joints but active_arms has 2 arms. Falling back active_arms to single arm ['right'].")
        active_arms = ["right"]

    if len(active_arms) == 1:
        cfg = get_arm_config(self.model, active_arms[0], version=self.get_robot_version())
        ee_links = {active_arms[0]: cfg["ee_link"]}
        ee_to_marker_nom = {active_arms[0]: cfg["ee_to_marker_nom"]}
    else:
        cfg = get_both_arm_config(self.model, version=self.get_robot_version())
        ee_links = cfg["ee_links"]
        ee_to_marker_nom = cfg["ee_to_marker_nom"]

    # Override ee_to_marker_nom with actual calibrated values from memory
    for side in active_arms:
        key = f"Tf_to_marker_{side}"
        if key in self.marker_calibrator.camera_config:
            ee_to_marker_nom[side] = self.marker_calibrator.camera_config[key]
            self.log_msg(f"[INFO] Using calibrated marker bracket values for {side}: {ee_to_marker_nom[side]}")

    # Floor on the estimated orientation noise (see load_step2_min_rot_noise_deg).
    min_rot_noise_deg, from_config = load_step2_min_rot_noise_deg()
    if not from_config:
        self.log_msg(f"[WARN] step2.min_rot_noise_deg is missing from setting.yaml; using the default {min_rot_noise_deg} deg.")
    self.log_msg(f"[INFO] Step 2 marker-orientation noise floor: {min_rot_noise_deg} deg")
    min_rot_noise_rad = min_rot_noise_deg * D2R
    head_tilt_anchor_weight = load_step2_head_tilt_anchor_weight()
    if head_tilt_anchor_weight is None:
        head_tilt_anchor_weight = HEAD_ANCHOR_WEIGHT_TILT
        self.log_msg(f"[INFO] Step 2 head tilt anchor weight: {head_tilt_anchor_weight:g} (default)")
    else:
        self.log_msg(f"[INFO] Step 2 head tilt anchor weight: {head_tilt_anchor_weight:g} "
                     f"(setting.yaml; default {HEAD_ANCHOR_WEIGHT_TILT:g})")

    head_cfg = get_head_config(self.model)
    use_head_kinematics = (
        getattr(self, 'include_head_motion', True) and
        (q_head_list is not None) and
        (head_cfg.get("head_idx") is not None) and
        len(head_cfg.get("head_idx", [])) >= 2
    )
    head_idx = head_cfg["head_idx"] if use_head_kinematics else None
    optimize_head = optimize_head and use_head_kinematics
    validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)

    # Determine initial head offsets if previously calibrated
    q_head_offset_init = None
    head_stored = getattr(self, 'joint_offsets_store', {}).get("head", {})
    if head_stored and head_idx and len(head_idx) >= 2:
        pan_val = float(head_stored.get("pan", 0.0))
        tilt_val = float(head_stored.get("tilt", 0.0))
        if abs(pan_val) > 1e-4 or abs(tilt_val) > 1e-4:
            q_head_offset_init = np.radians([pan_val, tilt_val])
            self.log_msg(f"[INFO] Using locked/calibrated head offsets for Step 2 optimization: {head_stored}")

    has_step1_offsets = bool(self.joint_offsets_store.get("right") or self.joint_offsets_store.get("left"))
    apply_limits = getattr(self, "apply_joint_offset_flag", False) or has_step1_offsets
    joint_offsets = None
    if apply_limits:
        joint_offsets = {}
        for side in active_arms:
            side_dict = self.joint_offsets_store.get(side, {})
            joint_offsets[side] = {
                "joint3": side_dict.get("joint3", 0.0),
                "joint5": side_dict.get("joint5", 0.0),
                "joint6": side_dict.get("joint6", 0.0),
            }
        if q_head_offset_init is not None:
            # Step 1.5 jointly identified Head Pan/Tilt via raw-point
            # least-squares (not a gauge-fixing convention) -- anchor them
            # in Step 2 the same way J3/J5/J6 are anchored to Step 1.
            joint_offsets["head"] = {
                "pan": float(head_stored.get("pan", 0.0)),
                "tilt": float(head_stored.get("tilt", 0.0)),
            }
        self.log_msg(f"[INFO] Applying joint offset bounds: {joint_offsets}")

    # Check if Step 1.5 camera results are present
    mount_cam_from_step1_5 = None
    if hasattr(self, 'head_camera_calibrator') and self.head_camera_calibrator is not None:
        res15 = getattr(self.head_camera_calibrator, 'calibrated_results', None)
        if res15 and not res15.get("skipped", False) and "calibrated_mount_to_cam" in res15:
            mount_cam_from_step1_5 = res15["calibrated_mount_to_cam"]

    def with_nominal_head_axis_rotation(mount_to_cam):
        """Reset the camera-mount rotation about the camera x (head tilt) and y (head pan) axes to
        the CAD nominal. The optimizer keeps those components fixed at the baseline (the head joints
        absorb pointing error), so a baseline carried over from an older result must not smuggle
        them back in. Rotation about the optical (z) axis is kept."""
        if not use_head_kinematics or mount_to_cam is None:
            return mount_to_cam
        nominal = self.marker_calibrator.camera_config.get("mount_to_cam_nominal")
        if not nominal or len(nominal) < 6:
            return mount_to_cam
        T_nom = BaseCalibrator.make_transform(nominal)
        T_cur = BaseCalibrator.make_transform(mount_to_cam)
        rel = R_scipy.from_matrix(T_nom[:3, :3].T @ T_cur[:3, :3]).as_rotvec()
        removed_deg = np.degrees(rel[:2]).copy()
        if np.all(np.abs(removed_deg) < 1e-6):
            return mount_to_cam
        rel[:2] = 0.0
        R_new = T_nom[:3, :3] @ R_scipy.from_rotvec(rel).as_matrix()
        yaw, pitch, roll = R_scipy.from_matrix(R_new).as_euler("ZYX", degrees=True)
        adjusted = [float(mount_to_cam[0]), float(mount_to_cam[1]), float(mount_to_cam[2]), float(roll), float(pitch), float(yaw)]
        self.log_msg(f"[INFO] Camera mount rotation about the head tilt/pan axes reset to nominal "
                     f"(removed x {removed_deg[0]:+.3f}°, y {removed_deg[1]:+.3f}°; head joints absorb it): {adjusted}")
        return adjusted

    if len(active_arms) == 2 and q_arm_list.shape[1] >= 14:
        self.log_msg("\n[INFO] === UNIFIED DUAL-ARM JOINT-CAMERA CALIBRATION WORKFLOW (2-PASS) ===")
        cfg_both = get_both_arm_config(self.model, version=self.get_robot_version())
        mount_cam_init = with_nominal_head_axis_rotation(mount_cam_from_step1_5 or self.marker_calibrator.camera_config.get("mount_to_cam", cfg_both["mount_to_cam_nom"]))
        if mount_cam_from_step1_5:
            self.log_msg(f"[INFO] Using Step 1.5 calibrated mount_to_cam as fixed baseline: {mount_cam_init}")

        # camera_pos_bound_m=0.025 (was 0.010): widened to comfortably cover a
        # freshly-mounted/swapped camera whose position was only measured by
        # hand (~1-2cm accuracy), not to CAD tolerance. Camera POSITION errors
        # are well-conditioned in Step 2's 64-pose data (unlike camera
        # ROTATION, which stays coupled with Head Tilt/Pan -- see
        # HEAD_ANCHOR_WEIGHT/camera_rot_bound_rad, left untouched here), so
        # widening this specific bound does not reopen the cascade failure
        # mode found when also stress-testing with unrealistically large (up
        # to 5 deg) head-offset/noise ranges in the same sweep.
        camera_pos_bound_m = 0.025

        # Design-accurate camera mount (camera_info.yaml mount_to_cam_position_range_m): the camera
        # starts at its design position and may only move inside the range, its angles stay fixed, and
        # J6 is solved here instead of being held at Step 1's value. Step 1.5 still gives the head
        # offsets; its camera position is not carried over.
        mount_constraint = load_camera_mount_constraint() if (use_head_kinematics and optimize_camera) else None
        # v1.3's J6 is the wrist roll, measured directly in Step 1 (0.015 deg on the first robot). Freed here it
        # took a bracket rotation error instead: left J6 +3.7 deg and J1 +3.1 deg past its -1 deg limit (2026-10-23).
        is_v13 = self.get_robot_version() == "1.3"
        solve_j6 = mount_constraint is not None and not is_v13
        # v1.3's J5 is solved here instead: Step 1's J5 rests on the J6 circle normal, 5-7 deg off on the real robot.
        solve_j5 = mount_constraint is not None and is_v13
        camera_pos_range_m = None
        if mount_constraint is not None:
            mount_cam_init = list(mount_constraint["mount_to_cam"])
            if mount_cam_from_step1_5 is not None:
                # The angles held are Step 1.5's: its optical-axis rotation comes from the head sweeps alone,
                # while here it trades with both J1 (2026-10-16: -0.65 deg there, -0.87 deg when freed here).
                mount_cam_init[3:] = [float(v) for v in mount_cam_from_step1_5[3:]]
            camera_pos_range_m = mount_constraint["position_range_m"]
            self.log_msg(f"[INFO] Camera mount held near its design pose ({mount_constraint['model']}): "
                         f"{format_mount_constraint(mount_constraint, j6_solved=solve_j6, j5_solved=solve_j5)}")

        # Pass 1: the original validated joint optimization (arms + head +
        # camera together, J0 common-mode damping + head soft anchor active).
        # NOT arms-only: an earlier version of this 2-pass split held
        # head/camera completely fixed in Pass 1, which removed their
        # ability to absorb a bad Step 1.5 starting estimate at all --  with
        # nowhere else to go, that error dumped into the least-protected arm
        # joints (J1/J4, which have no anchor), badly regressing arm
        # accuracy (confirmed on live-sim data: J1 error grew from ~0.1 deg
        # to ~2.4-2.8 deg). Keeping head/camera free here is what lets them
        # keep absorbing error instead of the arms, exactly as validated
        # before this 2-pass change.
        self.log_msg("\n[INFO] --- Pass 1/2: joint arm+head+camera solve (validated baseline) ---")
        optimizer_pass1 = QPCalibrationOptimizer(
            robot=self.robot,
            arm_idx=cfg_both["arm_idx"],
            ee_links=cfg_both["ee_links"],
            mount_to_cam_nom=mount_cam_init,
            head_base_to_cam_nom=cfg_both.get("head_base_to_cam_nom"),
            ee_to_marker_nom=ee_to_marker_nom,
            active_arms=["right", "left"],
            optimize_arm=True,
            optimize_head=optimize_head,
            optimize_camera=optimize_camera,
            head_idx=head_idx,
            use_head_kinematics=use_head_kinematics,
            lambda_cam_pos=lambda_cam_pos,
            lambda_cam_rot=lambda_cam_rot,
            use_sag=use_sag,
            estimate_measurement_noise=True,
            apply_joint_offset_limits=apply_limits,
            joint_offsets_to_apply=joint_offsets,
            camera_pos_bound_m=camera_pos_bound_m,
            camera_rot_bound_rad=3.0 * D2R,
            eps=1e-7,
            max_iter=200,
            min_rot_noise_std_rad=min_rot_noise_rad,
            head_anchor_weight_tilt=head_tilt_anchor_weight,
            camera_pos_range_m=camera_pos_range_m,
            anchor_joint6=not solve_j6,
            anchor_joint5=not solve_j5,
        )
        optimizer_pass1.stop_event = self.stop_event
        self.stop_check()
        q_arm_offset, q_head_offset_p1, xi_cam_p1, _pass1_mc, _pass1_hc = optimizer_pass1.optimize(
            q_arm_list, q_head_list, T_meas_list, q_head_offset_init=q_head_offset_init
        )
        self.checkpoint("optimizer_pass1", {
            "q_arm_offset": q_arm_offset, "q_head_offset": q_head_offset_p1,
            "xi_cam": xi_cam_p1, "mount_to_cam": _pass1_mc, "head_base_to_cam": _pass1_hc,
        })
        self.stop_check()
        self.log_msg(f"[INFO] Pass 1 arm joint offsets (deg): {np.rad2deg(q_arm_offset)}")
        if q_head_offset_p1 is not None:
            self.log_msg(f"[INFO] Pass 1 head joint offsets (deg): {np.rad2deg(q_head_offset_p1)}")

        # Pass 2: freeze arms at the Pass 1 result, refine head/camera
        # starting from Pass 1's own head/camera solution (not from scratch)
        # using the SAME 64-pose dataset. With arms fixed, the marker's
        # position at each pose is fully determined by known kinematics, so
        # the "which point on the sweep circle" ambiguity that makes Head
        # Pan exactly unidentifiable from Step 1.5's single-marker sweep
        # does not exist here -- richer pose diversity can further refine it
        # beyond Pass 1's result.
        self.log_msg("\n[INFO] --- Pass 2/2: head/camera refinement (arms frozen at Pass 1) ---")
        optimizer = QPCalibrationOptimizer(
            robot=self.robot,
            arm_idx=cfg_both["arm_idx"],
            ee_links=cfg_both["ee_links"],
            mount_to_cam_nom=mount_cam_init,
            head_base_to_cam_nom=cfg_both.get("head_base_to_cam_nom"),
            ee_to_marker_nom=ee_to_marker_nom,
            active_arms=["right", "left"],
            optimize_arm=False,
            optimize_head=optimize_head,
            optimize_camera=optimize_camera,
            head_idx=head_idx,
            use_head_kinematics=use_head_kinematics,
            lambda_cam_pos=lambda_cam_pos,
            lambda_cam_rot=lambda_cam_rot,
            use_sag=use_sag,
            estimate_measurement_noise=True,
            apply_joint_offset_limits=apply_limits,
            joint_offsets_to_apply=joint_offsets,
            camera_pos_bound_m=camera_pos_bound_m,
            camera_rot_bound_rad=3.0 * D2R,
            eps=1e-7,
            max_iter=200,
            min_rot_noise_std_rad=min_rot_noise_rad,
            head_anchor_weight_tilt=head_tilt_anchor_weight,
            camera_pos_range_m=camera_pos_range_m,
            anchor_joint6=not solve_j6,
            anchor_joint5=not solve_j5,
        )
        optimizer.stop_event = self.stop_event
        self.stop_check()
        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
            q_arm_list, q_head_list, T_meas_list,
            q_arm_offset_init=q_arm_offset,
            q_head_offset_init=q_head_offset_p1,
            xi_mount_cam_init=xi_cam_p1,
        )
    else:
        mount_constraint = None
        self.log_msg("\n[INFO] === SINGLE-ARM JOINT-CAMERA CALIBRATION WORKFLOW ===")
        mount_cam_init = with_nominal_head_axis_rotation(mount_cam_from_step1_5 or self.marker_calibrator.camera_config.get("mount_to_cam", cfg["mount_to_cam_nom"]))
        if mount_cam_from_step1_5:
            self.log_msg(f"[INFO] Using Step 1.5 calibrated mount_to_cam as fixed baseline: {mount_cam_init}")
        opt_single = QPCalibrationOptimizer(
            robot=self.robot,
            arm_idx=cfg["arm_idx"],
            ee_links=ee_links,
            mount_to_cam_nom=mount_cam_init,
            head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
            ee_to_marker_nom=ee_to_marker_nom,
            active_arms=active_arms,
            optimize_arm=True,
            optimize_head=optimize_head,
            optimize_camera=optimize_camera,
            head_idx=head_idx,
            use_head_kinematics=use_head_kinematics,
            lambda_cam_pos=lambda_cam_pos,
            lambda_cam_rot=lambda_cam_rot,
            use_sag=use_sag,
            estimate_measurement_noise=True,
            apply_joint_offset_limits=apply_limits,
            joint_offsets_to_apply=joint_offsets,
            camera_pos_bound_m=0.005,
            camera_rot_bound_rad=2.0 * D2R,
            eps=1e-7,
            max_iter=200,
            min_rot_noise_std_rad=min_rot_noise_rad,
            head_anchor_weight_tilt=head_tilt_anchor_weight,
        )
        opt_single.stop_event = self.stop_event
        self.stop_check()
        q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = opt_single.optimize(
            q_arm_list, q_head_list, T_meas_list
        )
        optimizer = opt_single

    if len(active_arms) == 1:
        if active_arms[0] == "right":
            right_arm_offset = q_arm_offset
            left_arm_offset = None
        else:
            right_arm_offset = None
            left_arm_offset = q_arm_offset
    else:
        right_arm_offset = q_arm_offset[:7]
        left_arm_offset = q_arm_offset[7:]

    head_base_to_cam_new = [float(x) for x in head_base_to_cam_new] if head_base_to_cam_new else None
    mount_to_cam_new = [float(x) for x in mount_to_cam_new] if mount_to_cam_new else None

    self.log_msg("\n===== RESULT =====")
    self.log_msg(f"lambda_cam_pos = {lambda_cam_pos}")
    self.log_msg(f"lambda_cam_rot = {lambda_cam_rot}")
    self.log_msg(f"measurement_noise = {optimizer.noise_estimator.format()}")

    if right_arm_offset is not None:
        self.log_msg(f"Right arm joint offset (deg): {np.rad2deg(right_arm_offset)}")

    if left_arm_offset is not None:
        self.log_msg(f"Left arm joint offset (deg): {np.rad2deg(left_arm_offset)}")
    if q_head_offset is not None:
        self.log_msg(f"Head joint offset (deg): {np.rad2deg(q_head_offset)}")

    if use_head_kinematics:
        self.log_msg(f"mount_to_cam xi: {xi_cam}")
        self.log_msg(f"mount_to_cam_new: {mount_to_cam_new}")
    else:
        self.log_msg(f"head_base-to-camera xi: {xi_cam}")
        self.log_msg(f"head_base_to_cam_new: {head_base_to_cam_new}")

    result_dict = {
        "joint_offset_deg": np.rad2deg(q_arm_offset).tolist(),
        "right_arm_joint_offset_deg": np.rad2deg(right_arm_offset).tolist() if right_arm_offset is not None else None,
        "left_arm_joint_offset_deg": np.rad2deg(left_arm_offset).tolist() if left_arm_offset is not None else None,
        "head_joint_offset_deg": np.rad2deg(q_head_offset).tolist() if q_head_offset is not None else None,
        "xi_cam": np.array(xi_cam).tolist(),
        "measurement_noise": optimizer.noise_estimator.as_dict(),
        # The brackets this solve actually used. They live in memory until the operator clicks
        # APPLY BRACKETS, so recording them here keeps the result reproducible on its own.
        "ee_to_marker_used": {side: [float(v) for v in ee_to_marker_nom[side]] for side in active_arms},
    }
    if mount_to_cam_new is not None:
        result_dict["mount_to_cam_new"] = mount_to_cam_new
    if head_base_to_cam_new is not None:
        result_dict["head_base_to_cam_new"] = head_base_to_cam_new

    if mount_constraint is not None:
        delta_m = np.array(mount_to_cam_new[:3]) - np.array(mount_constraint["mount_to_cam"][:3])
        ends = mount_range_ends(delta_m, mount_constraint["position_range_m"])
        self.log_msg(f"[INFO] Camera position against its design: x {delta_m[0] * 1000:+.2f}, y {delta_m[1] * 1000:+.2f}, "
                     f"z {delta_m[2] * 1000:+.2f} mm")
        for end in ends:
            self.log_msg(f"[WARN] Camera {end} of its design range: the Step 2 poses want it further. "
                         f"Check the design value or the range in camera_info.yaml.")
        # joint_offsets_store has the opposite sign to the result.
        step1_j6 = {side: -float(self.joint_offsets_store.get(side, {}).get("joint6", 0.0)) for side in ("right", "left")}
        step2_j6 = {"right": float(np.rad2deg(right_arm_offset[6])), "left": float(np.rad2deg(left_arm_offset[6]))}
        self.log_msg(f"[INFO] J6 {'solved in' if solve_j6 else 'held at Step 1 in'} Step 2 (Step 1 value -> Step 2): "
                     + ", ".join(f"{side} {step1_j6[side]:+.3f} -> {step2_j6[side]:+.3f} deg" for side in ("right", "left")))
        step1_j5 = {side: -float(self.joint_offsets_store.get(side, {}).get("joint5", 0.0)) for side in ("right", "left")}
        if solve_j5:
            step2_j5 = {"right": float(np.rad2deg(right_arm_offset[5])), "left": float(np.rad2deg(left_arm_offset[5]))}
            self.log_msg("[INFO] J5 solved in Step 2 (Step 1 value -> Step 2): "
                         + ", ".join(f"{side} {step1_j5[side]:+.3f} -> {step2_j5[side]:+.3f} deg" for side in ("right", "left")))
        result_dict["camera_mount_constraint"] = {
            "model": mount_constraint["model"],
            "mount_to_cam": mount_constraint["mount_to_cam"],
            "position_range_m": mount_constraint["position_range_m"].tolist(),
            "camera_offset_from_design_mm": (delta_m * 1000.0).tolist(),
            "camera_at_range_end": ends,
            "joint6_step1_deg": step1_j6,
            "joint5_step1_deg": step1_j5,
        }

    if self.last_home_reset_path is not None and Path(self.last_home_reset_path).exists():
        result_dict["home_reset_baseline_path"] = str(self.last_home_reset_path)

    # The state this solve ran in, kept inside the result: Home Offset Apply later zeroes setting.yaml's
    # joint offsets and the baseline file, and the exposure is never saved.
    result_dict["camera_exposure"] = self.marker_calibrator.camera_exposure_record()
    try:
        result_dict["setting_snapshot"] = ConfigStorage.load(CONFIG_PATHS["setting_yaml"])
        if "home_reset_baseline_path" in result_dict:
            result_dict["home_reset_baseline"] = ResultStorage.load(self.last_home_reset_path)
    except Exception as error:
        self.log_msg(f"[WARN] setting.yaml / home reset baseline could not be copied into the result: {error}")

    if use_head_kinematics:
        result_dict["xi_mount_cam"] = result_dict["xi_cam"]
    else:
        result_dict["xi_head_base_cam"] = result_dict["xi_cam"]

    # Which offset combinations these poses barely constrain, and how far each moves the hands at the
    # check pose (2026-09-29: small Step 2 residual, check pose off fore-aft). Report only.
    try:
        reference_q_arm = None
        try:
            # get_ready_pose returns radians.
            reference_q_arm = np.concatenate([
                self.marker_calibrator.get_ready_pose(f"v{self.get_robot_version()}", "check_calib", None, side)
                for side in ("right", "left") if side in active_arms])
        except Exception as error:
            self.log_msg(f"[WARN] Step 2 observability: check pose not available ({error}); reporting without it.")
        observability = step2_weak_directions(optimizer, q_arm_list, q_head_list, T_meas_list, q_arm_offset,
                                              q_head_offset, xi_cam, active_arms, reference_q_arm=reference_q_arm)
        for line in format_weak_directions(observability):
            self.log_msg(line)
        result_dict["observability"] = observability
    except Exception as error:
        self.log_msg(f"[WARN] Step 2 observability report failed: {error}")

    self.stop_check()
    ResultStorage.save(result_path, result_dict)

    history_path = os.path.join(os.path.dirname(result_path), "calibration_history.txt")
    try:
        with FileStorage.open(history_path, "a") as f:
            f.write(f"\n--- Calibration Iteration: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Result Path: {result_path}\n")
            f.write(f"Right Arm Joint Offset (deg): {result_dict.get('right_arm_joint_offset_deg')}\n")
            f.write(f"Left Arm Joint Offset (deg): {result_dict.get('left_arm_joint_offset_deg')}\n")
            f.write(f"Head Joint Offset (deg): {result_dict.get('head_joint_offset_deg')}\n")
            f.write(f"Camera xi: {result_dict.get('xi_cam')}\n")
            f.write(f"Measurement Noise: {json.dumps(result_dict.get('measurement_noise'))}\n")
    except Exception as e:
        self.log_msg(f"[ERROR] Failed to append to history: {e}")

    self.last_result_path = result_path
    self.log_msg(f"Result saved to {result_path}")
    self.log_msg(f"History appended to {history_path}")

    # Baseline Comparison Output
    baseline_file = StoragePaths.root / "config" / "home_reset_baseline.json"
    if os.path.exists(baseline_file):
        try:
            b_data = ResultStorage.load(baseline_file)
            self.log_msg("\n=========================================================")
            self.log_msg(f"  BASE LINE COMPARISON ({baseline_file})")
            self.log_msg("=========================================================")
            if right_arm_offset is not None and "right_arm_joint_offset_deg" in b_data:
                calc_r = np.rad2deg(right_arm_offset)
                base_r = np.array(b_data["right_arm_joint_offset_deg"])
                diff_r = np.abs(calc_r - base_r)
                self.log_msg(" [RIGHT ARM]")
                for i in range(len(calc_r)):
                    self.log_msg(f"   J{i}: Calc = {calc_r[i]:+8.4f}° | Baseline = {base_r[i]:+8.4f}° | Diff = {diff_r[i]:6.4f}°")
            if left_arm_offset is not None and "left_arm_joint_offset_deg" in b_data:
                calc_l = np.rad2deg(left_arm_offset)
                base_l = np.array(b_data["left_arm_joint_offset_deg"])
                diff_l = np.abs(calc_l - base_l)
                self.log_msg(" [LEFT ARM]")
                for i in range(len(calc_l)):
                    self.log_msg(f"   J{i}: Calc = {calc_l[i]:+8.4f}° | Baseline = {base_l[i]:+8.4f}° | Diff = {diff_l[i]:6.4f}°")
            self.log_msg("=========================================================\n")
        except Exception as e:
            self.log_msg(f"[WARN] Failed to compare with baseline: {e}")

    # Simulation Ground-Truth Comparison Output (Only runs in simulation mode)
    is_sim = bool(getattr(self.observer, "sim", False))

    if is_sim:
        try:
            sim_model = getattr(self.observer, 'simulation_model', None)
            sim_gt = sim_model.config.get('offsets', {}) if (sim_model and hasattr(sim_model, 'config')) else {}
            is_v13 = (self.get_robot_version() == "1.3")
            ver_key = "1.3" if is_v13 else "1.2"

            self.log_msg("\n=========================================================")
            self.log_msg("  SIMULATION GROUND-TRUTH COMPARISON REPORT")
            self.log_msg("=========================================================")

            # 1. Joint Offsets Comparison
            if right_arm_offset is not None and "right" in sim_gt:
                r_calc = np.rad2deg(right_arm_offset)
                r_gt = [
                    sim_gt["right"].get("joint0", 0.0),
                    sim_gt["right"].get("joint1", 0.0),
                    sim_gt["right"].get("joint2", 0.0),
                    sim_gt["right"].get("joint3", 0.0),
                    sim_gt["right"].get("joint4", 0.0),
                    sim_gt["right"].get("joint5_v13" if is_v13 else "joint5_v12", 0.0),
                    sim_gt["right"].get("joint6", 0.0),
                ]
                self.log_msg(" [RIGHT ARM JOINTS]")
                for i in range(7):
                    diff = abs(r_calc[i] - r_gt[i])
                    self.log_msg(f"   J{i}: Calc = {r_calc[i]:+8.4f}° | GT = {r_gt[i]:+8.4f}° | Error = {diff:6.4f}°")

            if left_arm_offset is not None and "left" in sim_gt:
                l_calc = np.rad2deg(left_arm_offset)
                l_gt = [
                    sim_gt["left"].get("joint0", 0.0),
                    sim_gt["left"].get("joint1", 0.0),
                    sim_gt["left"].get("joint2", 0.0),
                    sim_gt["left"].get("joint3", 0.0),
                    sim_gt["left"].get("joint4", 0.0),
                    sim_gt["left"].get("joint5_v13" if is_v13 else "joint5_v12", 0.0),
                    sim_gt["left"].get("joint6", 0.0),
                ]
                self.log_msg(" [LEFT ARM JOINTS]")
                for i in range(7):
                    diff = abs(l_calc[i] - l_gt[i])
                    self.log_msg(f"   J{i}: Calc = {l_calc[i]:+8.4f}° | GT = {l_gt[i]:+8.4f}° | Error = {diff:6.4f}°")

            # 2. Head Joint Offsets Comparison
            if q_head_offset is not None and "head" in sim_gt:
                h_calc = np.rad2deg(q_head_offset)
                h_gt = [
                    sim_gt["head"].get("pan", 0.0),
                    sim_gt["head"].get("tilt", 0.0),
                ]
                self.log_msg(" [HEAD JOINTS]")
                self.log_msg(f"   Pan:  Calc = {h_calc[0]:+8.4f}° | GT = {h_gt[0]:+8.4f}° | Error = {abs(h_calc[0] - h_gt[0]):6.4f}°")
                self.log_msg(f"   Tilt: Calc = {h_calc[1]:+8.4f}° | GT = {h_gt[1]:+8.4f}° | Error = {abs(h_calc[1] - h_gt[1]):6.4f}°")

            # 3. Marker Bracket Offsets Comparison (relative to Nominal)
            for side in ["right", "left"]:
                key = f"Tf_to_marker_{side}"
                if key in self.marker_calibrator.camera_config and side in sim_gt:
                    calc_val = self.marker_calibrator.camera_config[key]
                    nom_val = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][side]
                    T_nom = BaseCalibrator.make_transform(nom_val)
                    T_cal = BaseCalibrator.make_transform(calc_val)

                    # T_bracket_calc represents the actual translation/rotation of the bracket relative to flange
                    T_bracket_calc = T_cal @ np.linalg.inv(T_nom)
                    calc_pos_offset = T_bracket_calc[:3, 3]

                    calc_rot_offset = R_scipy.from_matrix(T_bracket_calc[:3, :3]).as_euler('ZYX', degrees=True)[::-1]

                    # Normalize rotation differences to [-180, 180]
                    calc_rot_offset = (calc_rot_offset + 180) % 360 - 180

                    gt_pos_offset = np.array(sim_gt[side]["bracket_pos"])
                    gt_rot_offset = np.array(sim_gt[side]["bracket_rpy"])

                    self.log_msg(f" [{side.upper()} ARM BRACKET OFFSETS]")
                    pos_norm_mm = np.linalg.norm(calc_pos_offset) * 1000.0
                    if pos_norm_mm > 40.0:
                        self.log_msg(f"   [ERROR] Bracket position offset exceeded safety threshold: {pos_norm_mm:.1f}mm > 40.0mm!")
                    # Position in mm
                    self.log_msg(f"   Pos X (mm): Calc = {calc_pos_offset[0]*1000.0:+7.2f} | GT = {gt_pos_offset[0]*1000.0:+7.2f} | Error = {abs(calc_pos_offset[0] - gt_pos_offset[0])*1000.0:5.2f}")
                    self.log_msg(f"   Pos Y (mm): Calc = {calc_pos_offset[1]*1000.0:+7.2f} | GT = {gt_pos_offset[1]*1000.0:+7.2f} | Error = {abs(calc_pos_offset[1] - gt_pos_offset[1])*1000.0:5.2f}")
                    self.log_msg(f"   Pos Z (mm): Calc = {calc_pos_offset[2]*1000.0:+7.2f} | GT = {gt_pos_offset[2]*1000.0:+7.2f} | Error = {abs(calc_pos_offset[2] - gt_pos_offset[2])*1000.0:5.2f}")
                    # Rotation in deg
                    self.log_msg(f"   Rot R (deg): Calc = {calc_rot_offset[0]:+7.2f}° | GT = {gt_rot_offset[0]:+7.2f}° | Error = {abs(calc_rot_offset[0] - gt_rot_offset[0]):5.2f}°")
                    self.log_msg(f"   Rot P (deg): Calc = {calc_rot_offset[1]:+7.2f}° | GT = {gt_rot_offset[1]:+7.2f}° | Error = {abs(calc_rot_offset[1] - gt_rot_offset[1]):5.2f}°")
                    self.log_msg(f"   Rot Y (deg): Calc = {calc_rot_offset[2]:+7.2f}° | GT = {gt_rot_offset[2]:+7.2f}° | Error = {abs(calc_rot_offset[2] - gt_rot_offset[2]):5.2f}°")

            self.log_msg("=========================================================\n")
        except Exception as e:
            self.log_msg(f"[WARN] Failed to print simulation GT comparison: {e}")

    return result_dict
