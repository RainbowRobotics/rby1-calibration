"""Wizard auto exposure: right arm to its J5 ready pose, then try exposure times on the right marker.

D405 only (2026-11-12, user); any other camera is put on its own auto exposure. Each candidate is measured
held still, the steadiest few again with the wrist pitched towards and away from the camera (glare and
blur change with the marker angle), and data.choose_auto_exposure picks the value. Detection is raw (no
filter), sampled like the marker monitor, so the jitter is what the detector gives.
"""
import time

import numpy as np

from ..data import (AUTO_EXPOSURE_CANDIDATES_US, AUTO_EXPOSURE_MIN_RATE, choose_auto_exposure,
                    load_max_marker_jitter_mm, marker_position_jitter_mm)

SAMPLE_PERIOD_S = 0.1
SETTLE_S = 0.5          # frames after an exposure change still carry the old exposure
STILL_S = 2.0
TURNED_S = 1.0
TURN_DEG = 15.0          # J5 (wrist pitch) either way from the ready pose
TURN_MOVE_S = 1.5
TURNED_CANDIDATES = 3
SIDE = "right"


def measure(observer, duration_s, check_cancelled):
    """Rate and RMS jitter (mm) of the right marker over duration_s."""
    positions = []
    end = time.monotonic() + duration_s
    while True:
        check_cancelled()
        result = observer.get_marker_transform(sampling_time=0, side=SIDE, use_filter=False)
        positions.append(np.asarray(result[0], dtype=np.float64).reshape(4, 4)[:3, 3]
                         if isinstance(result, list) and result else None)
        if time.monotonic() >= end:
            break
        time.sleep(SAMPLE_PERIOD_S)
    detected = [p for p in positions if p is not None]
    jitter = marker_position_jitter_mm(detected) if len(detected) >= 2 else None
    return {"rate": len(detected) / len(positions), "jitter_mm": jitter["rms_mm"] if jitter else None}


def run_auto_exposure(core, ctx, settle_s=SETTLE_S, still_s=STILL_S, turned_s=TURNED_S):
    observer, robot, log = core.observer, core.robot, core.log_msg
    calibrator = core.joint_calibrator
    mode = "wrist_pitch_v13" if core.robot_version == "1.3" else "wrist_pitch"
    if core.include_head_motion and not calibrator.movej(robot, head=[0.0, 0.0], minimum_time=3.0, apply_offsets=False):
        ctx.check_cancelled()
        raise RuntimeError("Head zero move failed")
    ctx.check_cancelled()
    if not calibrator.perform_move_to_ready_pose(arm_side=SIDE, mode=mode, log_callback=log):
        ctx.check_cancelled()
        raise RuntimeError("Right arm ready pose failed")
    ctx.check_cancelled()

    family = str((observer.get_camera_info() or {}).get("camera_model") or "").upper()
    if family != "D405":
        observer.set_camera_exposure(AUTO_EXPOSURE_CANDIDATES_US[3], auto_exposure=True)
        log(f"[Auto exposure] Camera {family or 'unknown'} is not a D405: its own auto exposure is used.")
        return {"applied": "auto", "camera_model": family, "trials": []}

    max_jitter, _ = load_max_marker_jitter_mm()
    trials = []
    for exposure in AUTO_EXPOSURE_CANDIDATES_US:
        observer.set_camera_exposure(exposure, auto_exposure=False)
        time.sleep(settle_s)
        trial = {"exposure": exposure, **measure(observer, still_s, ctx.check_cancelled), "turned": []}
        trials.append(trial)
        log(f"[Auto exposure] {exposure} us held still: detected {trial['rate'] * 100:.0f}%, "
            f"jitter {trial['jitter_mm'] if trial['jitter_mm'] is None else round(trial['jitter_mm'], 3)} mm")

    steady = sorted((t for t in trials if t["rate"] >= AUTO_EXPOSURE_MIN_RATE and t["jitter_mm"] is not None),
                    key=lambda t: t["jitter_mm"])[:TURNED_CANDIDATES]
    arm_idx = list(robot.model().right_arm_idx)
    q_ready = np.asarray(robot.get_state().position, dtype=np.float64)[arm_idx]
    try:
        for turn in (-TURN_DEG, TURN_DEG):
            q = q_ready.copy()
            q[5] += np.radians(turn)
            if not calibrator.movej(robot, right_arm=q, minimum_time=TURN_MOVE_S, apply_offsets=False):
                ctx.check_cancelled()
                raise RuntimeError("Wrist turn for the auto exposure failed")
            for trial in steady:
                observer.set_camera_exposure(trial["exposure"], auto_exposure=False)
                time.sleep(settle_s)
                m = measure(observer, turned_s, ctx.check_cancelled)
                trial["turned"].append(m)
                log(f"[Auto exposure] {trial['exposure']} us, J5 {turn:+.0f} deg: detected {m['rate'] * 100:.0f}%, "
                    f"jitter {m['jitter_mm'] if m['jitter_mm'] is None else round(m['jitter_mm'], 3)} mm")
    finally:
        if not ctx.stop_event.is_set():
            calibrator.movej(robot, right_arm=q_ready, minimum_time=TURN_MOVE_S, apply_offsets=False)

    best = choose_auto_exposure(trials, max_jitter)
    if best is None:
        observer.set_camera_exposure(AUTO_EXPOSURE_CANDIDATES_US[3], auto_exposure=True)
        log(f"[Auto exposure] No exposure kept the right marker detected and within {max_jitter} mm; "
            "the camera's own auto exposure is back on. Adjust the exposure by hand.")
        return {"applied": None, "camera_model": family, "trials": trials, "max_jitter_mm": max_jitter}
    observer.set_camera_exposure(best["exposure"], auto_exposure=False)
    log(f"[Auto exposure] Applied {best['exposure']} us.")
    return {"applied": "manual", "exposure": best["exposure"], "camera_model": family, "trials": trials,
            "max_jitter_mm": max_jitter}
