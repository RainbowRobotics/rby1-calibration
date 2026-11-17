"""Step 2 observability report: offset combinations the collected poses barely constrain, and how far
each one moves the hands at a reference pose (the check pose).

2026-09-29 (D405): the 64 Step 2 samples cover about 10 deg per joint around the auto ready pose.
Combinations of J2/J4/J6 changed the weighted residual by only 2-6 sigma-units per degree over 768
residuals, yet moved the hands at the check pose by 1-2 mm per degree fore-aft, left and right in
opposite directions -- the operator saw the left hand ahead of the right after applying the result.

Compute layer: no SDK import. The fitted QPCalibrationOptimizer supplies the model
(evaluate_sample, noise weights, dynamics for the reference-pose FK).
"""
import numpy as np

from .calibration_optimizer import prepare_q_full, se3_log

TORSO_LINK = "link_torso_5"
ARM_SIDES = ("right", "left")
CAMERA_LABELS = ("camRx", "camRy", "camRz", "camX", "camY", "camZ")
# Finite-difference steps: joint/rotation parameters in rad, camera translation in m.
FD_STEP_RAD = 1e-6
FD_STEP_M = 1e-7
# A combination counts as weak when one unit step (1 deg or 1 mm) raises the weighted residual RMS by
# less than this many noise sigmas, and as mattering when it then moves a hand at the reference pose by
# at least this much.
WEAK_RMS_SIGMA_PER_UNIT = 0.3
HAND_MOVE_WARN_MM_PER_UNIT = 1.0


def _parameter_layout(opt, q_arm_offset, q_head_offset, active_arms):
    """(labels, unit scale to rad/m, where each parameter goes) for the parameters Step 2 estimated."""
    labels, scale, slots = [], [], []
    n_arm = len(q_arm_offset)
    arm_sides = [s for s in ARM_SIDES if s in active_arms]
    for k in range(n_arm):
        side = arm_sides[k // 7] if len(arm_sides) > 1 else arm_sides[0]
        labels.append(f"{side[0].upper()}J{k % 7}")
        scale.append(np.pi / 180.0)
        slots.append(("arm", k))
    if q_head_offset is not None and opt.use_head_kinematics:
        for k, name in enumerate(("pan", "tilt")):
            labels.append(f"head_{name}")
            scale.append(np.pi / 180.0)
            slots.append(("head", k))
    if opt.optimize_camera:
        free = [2, 3, 4, 5] if (opt.lock_camera_head_axis_rotation and opt.use_head_kinematics) else range(6)
        for k in free:
            labels.append(CAMERA_LABELS[k])
            scale.append(np.pi / 180.0 if k < 3 else 1e-3)
            slots.append(("cam", k))
    return labels, np.array(scale), slots


def _noise_weights(opt):
    """1/sigma from the fitted noise estimate, even when the optimizer ran unweighted (weights() is then
    all ones, which would mix rad and m)."""
    est = opt.noise_estimator
    return np.array([1.0 / est.rot_std_rad] * 3 + [1.0 / est.pos_std_m] * 3)


def _residuals(opt, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_cam, active_arms):
    weights = _noise_weights(opt)
    q_head_iter = [None] * len(q_arm_list) if q_head_list is None else q_head_list
    out = []
    for q_arm, q_head, T_pair in zip(q_arm_list, q_head_iter, T_meas_list):
        T_pair = np.asarray(T_pair)
        for side_idx, side in enumerate(ARM_SIDES):
            if side not in active_arms:
                continue
            T_meas = T_pair[side_idx] if T_pair.shape == (2, 4, 4) else T_pair
            _, _, _, T_model = opt.evaluate_sample(q_arm, q_head, side, q_arm_offset, q_head_offset, xi_cam)
            out.append(se3_log(np.linalg.inv(T_model) @ T_meas) * weights)
    return np.concatenate(out)


def _hand_positions_mm(opt, reference_q_arm, q_arm_offset):
    """Hand (ee link) positions in the torso frame at the reference arm pose with the given offsets."""
    q_full = prepare_q_full(q_nominal=opt.q_nominal, arm_idx=opt.arm_idx, q_cmd=reference_q_arm, q_offset=q_arm_offset)
    out = []
    for side in ARM_SIDES:
        if side not in opt.ee_links:
            continue
        state = opt.dyn_model.make_state([TORSO_LINK, opt.ee_links[side]], opt.model.robot_joint_names)
        state.set_q(q_full)
        opt.dyn_model.compute_forward_kinematics(state)
        out.append(np.asarray(opt.dyn_model.compute_transformation(state, 0, 1))[:3, 3] * 1000.0)
    return np.concatenate(out)


def step2_weak_directions(opt, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_cam,
                          active_arms, reference_q_arm=None, n_dirs=4):
    """SVD of the noise-weighted Step 2 Jacobian at the solution, in units of 1 deg (angles) / 1 mm
    (camera translation). Returns the n_dirs weakest combinations with, per unit step, the rise of the
    weighted residual RMS (in noise sigmas) and, if reference_q_arm is given, the hand displacement at
    that pose (torso frame, mm). Head and camera terms do not move the hands, only arm offsets do.
    """
    q_arm_offset = np.asarray(q_arm_offset, dtype=float)
    q_head_offset = None if q_head_offset is None else np.asarray(q_head_offset, dtype=float)
    xi_cam = np.zeros(6) if xi_cam is None else np.asarray(xi_cam, dtype=float)
    labels, scale, slots = _parameter_layout(opt, q_arm_offset, q_head_offset, active_arms)

    def unpack(delta):
        qa, qh, xi = q_arm_offset.copy(), None if q_head_offset is None else q_head_offset.copy(), xi_cam.copy()
        for (kind, k), d in zip(slots, delta):
            if kind == "arm":
                qa[k] += d
            elif kind == "head":
                qh[k] += d
            else:
                xi[k] += d
        return qa, qh, xi

    args = (opt, q_arm_list, q_head_list, T_meas_list)
    r0 = _residuals(*args, q_arm_offset, q_head_offset, xi_cam, active_arms)
    J = np.zeros((len(r0), len(slots)))
    for j, (kind, _) in enumerate(slots):
        h = FD_STEP_M if (kind == "cam" and scale[j] == 1e-3) else FD_STEP_RAD
        delta = np.zeros(len(slots))
        delta[j] = h
        J[:, j] = (_residuals(*args, *unpack(delta), active_arms) - r0) / h * scale[j]

    H = None
    if reference_q_arm is not None:
        reference_q_arm = np.asarray(reference_q_arm, dtype=float)
        p0 = _hand_positions_mm(opt, reference_q_arm, q_arm_offset)
        H = np.zeros((len(p0), len(slots)))
        for j, (kind, k) in enumerate(slots):
            if kind != "arm":
                continue
            qa = q_arm_offset.copy()
            qa[k] += FD_STEP_RAD
            H[:, j] = (_hand_positions_mm(opt, reference_q_arm, qa) - p0) / FD_STEP_RAD * scale[j]

    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    n_res = len(r0)
    hand_sides = [s for s in ARM_SIDES if s in opt.ee_links]
    directions = []
    for idx in np.argsort(sv)[:n_dirs]:
        v = Vt[idx]
        v = v if v[np.argmax(np.abs(v))] > 0 else -v          # sign convention: largest component positive
        top = np.argsort(-np.abs(v))[:6]
        entry = {
            "residual_rms_sigma_per_unit": float(sv[idx] / np.sqrt(n_res)),
            "components": {labels[i]: float(v[i]) for i in top},
        }
        if H is not None:
            move = H @ v
            entry["reference_hand_move_mm_per_unit"] = {
                side: [float(x) for x in move[3 * s:3 * s + 3]] for s, side in enumerate(hand_sides)}
        directions.append(entry)
    return {
        "n_residuals": int(n_res),
        "residual_rms_sigma": float(np.sqrt(np.mean(r0 ** 2))),
        "parameters": labels,
        "weak_directions": directions,
    }


def weak_directions_needing_attention(report):
    """Weak combinations (see WEAK_RMS_SIGMA_PER_UNIT) that move a hand at the reference pose by at
    least HAND_MOVE_WARN_MM_PER_UNIT per unit step."""
    flagged = []
    for d in report.get("weak_directions", []):
        moves = d.get("reference_hand_move_mm_per_unit") or {}
        largest = max((float(np.linalg.norm(m)) for m in moves.values()), default=0.0)
        if d["residual_rms_sigma_per_unit"] < WEAK_RMS_SIGMA_PER_UNIT and largest >= HAND_MOVE_WARN_MM_PER_UNIT:
            flagged.append((d, largest))
    return flagged


def format_weak_directions(report):
    """Log lines for the report (English, like the other Step 2 logs)."""
    lines = [f"[INFO] Step 2 observability: {report['n_residuals']} weighted residuals, RMS "
             f"{report['residual_rms_sigma']:.2f} sigma. Weakest offset combinations (unit = 1 deg / 1 mm):"]
    for d in report["weak_directions"]:
        comp = ", ".join(f"{k} {v:+.2f}" for k, v in d["components"].items())
        line = f"  * +{d['residual_rms_sigma_per_unit']:.3f} sigma RMS per unit | {comp}"
        moves = d.get("reference_hand_move_mm_per_unit")
        if moves:
            line += " | check-pose hand move per unit: " + "; ".join(
                f"{side} x{m[0]:+.1f} y{m[1]:+.1f} z{m[2]:+.1f} mm" for side, m in moves.items())
        lines.append(line)
    flagged = weak_directions_needing_attention(report)
    if flagged:
        worst = max(m for _, m in flagged)
        lines.append(f"[WARN] Step 2 poses barely constrain {len(flagged)} offset combination(s) that move a hand at the "
                     f"check pose by up to {worst:.1f} mm per degree; the check pose can be off even with a small "
                     f"Step 2 residual (poses outside the Step 2 sample range).")
    return lines
