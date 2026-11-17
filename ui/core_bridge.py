"""Qt boundary: core never imports a widget or reads UI fields."""
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from functools import wraps


def idle_core_action(method):
    """Reject a second UI action before it can reset shared stop flags/state."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        app = getattr(self, "parent_app", self)
        if app.core.is_busy:
            app.log_msg("[INFO] Another core action is running. Stop it and wait for its result first.")
            return False
        return method(self, *args, **kwargs)
    return guarded


class CoreBridge(QObject):
    log = Signal(str)
    progress = Signal(object)
    result = Signal(object)
    bracket = Signal(dict)
    joint = Signal(dict)
    marker_gap = Signal(object)
    marker_monitor = Signal(object)

    def __init__(self, core, parent=None):
        super().__init__(parent)
        self.core = core
        self.core.on_event = self._relay

    def _relay(self, kind, value):
        signal = getattr(self, kind, None)
        if signal is not None and hasattr(signal, "emit"):
            signal.emit(value)

    def cancel(self):
        self.core.cancel()


class SequenceWorker(QThread):
    finished_result = Signal(object)
    _running_workers = set()

    def __init__(self, core, name, options=None, parent=None):
        super().__init__(parent)
        self.core, self.name, self.options = core, name, dict(options or {})
        self.core.prepare_run()
        self.result = None
        self.finished.connect(self._release_running_reference, Qt.QueuedConnection)

    def start(self, *args, **kwargs):
        # Completion signals can make the UI drop its worker reference before
        # QThread.run has actually returned. Keep it alive until Qt confirms exit.
        self._running_workers.add(self)
        try:
            return super().start(*args, **kwargs)
        except Exception:
            self._running_workers.discard(self)
            raise

    @Slot()
    def _release_running_reference(self):
        self._running_workers.discard(self)

    @classmethod
    def active_for_core(cls, core):
        return [worker for worker in cls._running_workers if worker.core is core]

    def run(self):
        from core.calibration.sequences.result import SequenceResult
        try:
            self.result = self.core.run(self.name, prepared=True, **self.options)
        except Exception as error:
            self.result = SequenceResult(self.name).finish("failed", str(error))
        self.finished_result.emit(self.result)


class MarkerCalibrationWorker(SequenceWorker):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)

    def __init__(self, core, arm_side, use_head_tracking=False, tolerance=0.5, save_debug=False):
        super().__init__(core, "marker", dict(arm_side=arm_side, use_head_tracking=use_head_tracking,
                                            tolerance=tolerance, save_debug=save_debug))

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.completed.get("marker") if self.result.success else None)


class JointCalibrationWorker(SequenceWorker):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)

    def __init__(self, core, arm_side, mode, ui_only=False, current_offset_deg=0.0, sweep_duration=15.0, save_debug=False):
        super().__init__(core, "joint", dict(arm_side=arm_side, mode=mode, current_offset_deg=current_offset_deg,
                                           sweep_duration=sweep_duration, save_debug=save_debug))

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.completed.get("joint") if self.result.success else None)


class HeadCamSweepWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, dict)

    def __init__(self, core, pan_range=15.0, tilt_range=10.0, num_steps=11, stop_event=None, parent=None):
        super().__init__(core, "step1_5", dict(prepare=False, arm_side="auto", pan_range_deg=pan_range,
                                             tilt_range_deg=tilt_range, num_steps=num_steps, step_delay=0.6), parent)

    def run(self):
        super().run()
        value = self.result.completed.get("head_camera", {})
        if not self.result.success:
            value = {"status": self.result.status, "error": self.result.error, "partial": self.result.partial}
        self.finished_signal.emit(self.result.success, value)


class Step2CalculateWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, core, active_arms, optimize_head, optimize_camera, q_arm_list, q_head_list,
                 T_meas_list, result_path, lambda_cam_pos, lambda_cam_rot, data_home_epoch=None):
        # data_home_epoch: CalibrationCore.home_epoch the samples were collected at, or None for
        # an npz dataset (then the result is analysis only and cannot be applied).
        from copy import deepcopy
        options = dict(active_arms=active_arms, optimize_head=optimize_head, optimize_camera=optimize_camera,
                       q_arm_list=q_arm_list, q_head_list=q_head_list, T_meas_list=T_meas_list,
                       result_path=result_path, lambda_cam_pos=lambda_cam_pos, lambda_cam_rot=lambda_cam_rot)
        super().__init__(core, "optimize", {"kwargs": deepcopy(options), "data_home_epoch": data_home_epoch})

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error)


class Step2AutoMotionWorker(SequenceWorker):
    log_signal = Signal(str)
    sample_signal = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, core, plan=None, start_index=0, parent=None):
        from copy import deepcopy
        super().__init__(core, "collect", dict(plan=deepcopy(plan), start_index=start_index), parent)

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error)


class FullAutoWorker(SequenceWorker):
    log_msg = Signal(str)
    status_signal = Signal(bool)
    bracket_finished_signal = Signal(dict)
    joint_finished_signal = Signal(dict)
    finished_signal = Signal()

    def __init__(self, core, sequence="full", options=None):
        super().__init__(core, sequence, options)
        self.error_msg = None

    def run(self):
        super().run()
        self.error_msg = self.result.error if self.result.status == "failed" else None
        self.finished_signal.emit()


class HomeOffsetActionWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str, dict)
    dict_finished_signal = Signal(dict)

    def __init__(self, core, task_type, **options):
        super().__init__(core, "home", dict(task_type=task_type, **options))

    def run(self):
        super().run()
        value = self.result.completed.get("home", self.result.partial.get("home", {}))
        value.update(success=self.result.success, error=self.result.error, sequence_status=self.result.status)
        self.dict_finished_signal.emit(value)
        self.finished_signal.emit(self.result.success, self.result.error, value)


class Step2ApplyHomeOffsetWorker(HomeOffsetActionWorker):
    def __init__(self, app, task_type, **options):
        # Called on the GUI thread: read values now, never retain the QWidget.
        super().__init__(app._sync_core_state(), task_type, **options)


class HomeOffsetResetWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, core, model, model_name, include_head):
        super().__init__(core, "home", dict(task_type="reset", model_name=model_name,
                                          include_head=include_head, arm="both"))

    def run(self):
        super().run()
        value = self.result.completed.get("home", self.result.partial.get("home", {}))
        value.update(success=self.result.success, error=self.result.error, sequence_status=self.result.status)
        self.finished_signal.emit(value)


class Step2InitPoseWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, core, active_arms, priority, include_head_motion=True, parent=None):
        super().__init__(core, "step2_ready", dict(active_arms=active_arms, priority=priority), parent)

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error)


class GripperPoseWorker(SequenceWorker):
    """Wizard: both arms to the gripper-change pose, then the tool flange output off.
    finished_signal(success, error, {side: volts read back})."""
    log_signal = Signal(str)
    finished_signal = Signal(bool, str, dict)

    def __init__(self, core):
        super().__init__(core, "gripper_pose", {})

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error,
                                  dict(self.result.completed.get("gripper_pose") or {}))


class AutoExposureWorker(SequenceWorker):
    """Wizard exposure slide: right arm test pose, then the auto exposure (D405).
    finished_signal(success, error, result dict of sequences.auto_exposure.run_auto_exposure)."""
    log_signal = Signal(str)
    finished_signal = Signal(bool, str, dict)

    def __init__(self, core):
        super().__init__(core, "auto_exposure", {})

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error,
                                  dict(self.result.completed.get("auto_exposure") or {}))


class MoveToReadyWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, core, arm_side, mode="marker", head_zero=False):
        kind = "marker" if mode == "marker" else "joint"
        super().__init__(core, "ready", dict(kind=kind, arm_side=arm_side, mode=mode, head_zero=head_zero))

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error)


class ManualHeadWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, core, yaw_rad, pitch_rad):
        super().__init__(core, "manual_head", dict(position=[yaw_rad, pitch_rad]))

    def run(self):
        super().run()
        self.finished_signal.emit()


class HeadCamReadyWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, core, stop_event=None, parent=None):
        super().__init__(core, "ready", dict(kind="head", arm_side="both"), parent)

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error)


class FullAutoReadyWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, core):
        super().__init__(core, "full_ready")
        self.error_msg = None

    def run(self):
        super().run()
        self.error_msg = self.result.error
        self.finished_signal.emit()


class Step2ZeroPoseCheckWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, core, arm, include_head):
        super().__init__(core, "zero_pose", dict(arm=arm, include_head=include_head))

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error)


class CheckCalibrationStateWorker(SequenceWorker):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, task_type, core, model_name, active_arms, data, offset, skip_ready=False):
        options = dict(active_arms=active_arms, offset=offset)
        if task_type == "move":
            options.update(model_name=model_name, data=data, skip_ready=skip_ready)
        super().__init__(core, "check_state" if task_type == "move" else "draw_square", options)

    def run(self):
        super().run()
        self.finished_signal.emit(self.result.success, self.result.error)
