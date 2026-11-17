# =============================================================================
# 1. Standard Python Libraries (시스템, 파일 I/O, 시간, 수학, 동시성 등 기본 유틸리티)
# =============================================================================
import sys
import os
import argparse
import threading
import traceback
import datetime
from pathlib import Path

# =============================================================================
# 2. Path Configuration & Asset Helper (PyInstaller 번들 및 core 모듈 경로 등록)
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
else:
    app_dir = current_dir


# Seed config/ next to the executable before anything reads it. core.language loads
# ui_dropdowns.yaml and dark_theme.qss at import time, so on the very first run in a new folder
# they were read before this copy happened -- the window came up unthemed that one time.
from core.storage import StoragePaths as _StoragePaths
_StoragePaths.initialize_defaults()


def get_asset_path(relative_path: str) -> str:
    """PyInstaller 번들 실행 및 로컬 파이썬 실행 환경에서 이미지/설정 등의 절대 경로를 반환합니다."""
    return StoragePaths.asset(relative_path)

# =============================================================================
# 3. Third-Party Scientific & Robotics Libraries (수치연산, 영상처리, 로봇 제어 SDK)
# =============================================================================
import yaml
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 충돌 방지를 위해 Non-interactive 백엔드로 고정

# Degree to Radian 단위 변환 상수 (전역 공용)
D2R = np.pi / 180.0

# =============================================================================
# 4. GUI Framework - PySide6 / Qt (위젯, 레이아웃, 그래픽스, 비동기 스레드)
# =============================================================================
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QDialog, QMessageBox, QInputDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QCheckBox,
    QLineEdit, QSpinBox, QSlider, QTabWidget, QTableWidget, QHeaderView,
    QTableWidgetItem, QSizePolicy, QStackedWidget, QButtonGroup, QScrollArea, QFrame
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QFont, QPixmap, QImage, QGuiApplication, QFontDatabase
)

# =============================================================================
# 5. Core Application Modules (설정 경로, 다국어 I18n, 테마 스타일시트, 위자드)
# =============================================================================
from core.storage import CONFIG_PATHS, SessionLog, StoragePaths
from core.language import UI_DROPDOWNS, DARK_STYLESHEET, LanguageManager, tr
from ui.wizard_widget import CalibrationWizardWidget
from ui.marker_monitor_panel import MarkerMonitorPanel
from ui.busy_indicator import BusyIndicator

# =============================================================================
# 6. Computer Vision & Calibration Logic (마커 검출 및 카메라/관절 캘리브레이터)
# =============================================================================
from ui.core_bridge import Step2ZeroPoseCheckWorker, CheckCalibrationStateWorker, idle_core_action
from ui.core_bridge import FullAutoReadyWorker, HeadCamReadyWorker, ManualHeadWorker, MoveToReadyWorker, GripperPoseWorker
from ui.core_bridge import AutoExposureWorker
from ui.core_bridge import HomeOffsetResetWorker, Step2ApplyHomeOffsetWorker, Step2InitPoseWorker
from core.storage import ResultStorage, ConfigStorage
from core.calibration.calibration_core import CalibrationCore
from ui.core_bridge import CoreBridge, SequenceWorker, FullAutoWorker, HeadCamSweepWorker, JointCalibrationWorker, MarkerCalibrationWorker, Step2AutoMotionWorker, Step2CalculateWorker
from core.storage import FileStorage
from core.storage import save_camera_intrinsics
from core.calibration import BaseCalibrator, IntrinsicsCalibrator
from core.robot.home_offset import load_offset_from_json

from core.calibration.data import (get_arm_config, get_both_arm_config, get_head_config, load_npz_dataset,
                                   load_step2_plan_options, load_step2_wrist_diversity, save_npz_dataset,
                                   validate_dataset)
from core.robot.motion import AutoCollectionConfig, build_incremental_motion_plan, reset_motion_state


# =========================================================================
# Modular UI Dialog & Confirmation Helper Functions
# =========================================================================
def confirm_dialog(parent: QWidget, title: str, message: str) -> bool:
    """Standardized confirmation question dialog."""
    res = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    return res == QMessageBox.Yes

def show_warning_dialog(parent: QWidget, title: str, message: str):
    """Standardized warning popup dialog."""
    QMessageBox.warning(parent, title, message)

def show_info_dialog(parent: QWidget, title: str, message: str):
    """Standardized information popup dialog."""
    QMessageBox.information(parent, title, message)

def show_error_dialog(parent: QWidget, title: str, message: str):
    """Standardized critical/error popup dialog."""
    QMessageBox.critical(parent, title, message)


def enforce_nominal_taught_joints(taught_pose, norm_mode, nom_pose):
    """Put back the joint a re-taught joint-stage posture must keep at its ready-pose value.

    The marker (bracket) posture is deliberately left alone: a taught pose is replayed as raw
    encoder values (apply_offsets=False), so writing the nominal J5 into it would drop the J5
    calibration offset and put J5 physically off by that offset.
    """
    if norm_mode == "elbow":
        taught_pose[3] = nom_pose[3]
    elif norm_mode in ("wrist_pitch", "wrist_roll", "wrist_yaw2"):
        taught_pose[5] = nom_pose[5]
    return taught_pose


# --- Custom UI Widgets ---
class IndicatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.is_detected = False

    def set_detected(self, detected):
        self.is_detected = detected
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(0, 230, 118) if self.is_detected else QColor(255, 23, 68)
        painter.setBrush(color)
        painter.setPen(QPen(Qt.black, 1.5))
        painter.drawEllipse(2, 2, 26, 26)

class CameraFeedDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Live Feed")
        self.resize(640, 480)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)
        self.lbl_feed = QLabel("Waiting for camera frame...")
        self.lbl_feed.setObjectName("live_feed_display")
        self.lbl_feed.setAlignment(Qt.AlignCenter)
        self.lbl_feed.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.lbl_feed)

    def closeEvent(self, event):
        if self.parent() and hasattr(self.parent(), "on_feed_dialog_closed"):
            self.parent().on_feed_dialog_closed()
        super().closeEvent(event)

class PlotViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Plot Viewer")
        self.resize(950, 750)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)

        # Navigation layout
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(10, 5, 10, 5)
        nav_layout.setSpacing(10)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setObjectName("plot_btn_prev")
        self.btn_prev.setFixedSize(40, 30)

        self.lbl_title = QLabel("No Plot Loaded")
        self.lbl_title.setObjectName("plot_lbl_title")
        self.lbl_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_title.setAlignment(Qt.AlignCenter)

        self.btn_next = QPushButton("▶")
        self.btn_next.setObjectName("plot_btn_next")
        self.btn_next.setFixedSize(40, 30)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_title, 1)
        nav_layout.addWidget(self.btn_next)

        self.plot_label = QLabel("No plots to display")
        self.plot_label.setObjectName("plot_lbl_canvas")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addLayout(nav_layout)
        layout.addWidget(self.plot_label)

        if parent:
            self.btn_prev.clicked.connect(parent.show_prev_plot)
            self.btn_next.clicked.connect(parent.show_next_plot)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent() and hasattr(self.parent(), "display_current_plot"):
            self.parent().display_current_plot()

class ZeroPoseCheckDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.zero_pose_check.title"))
        self.resize(760, 800)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        msg_lbl = QLabel(tr("dialogs.zero_pose_check.guide_msg"))
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl)

        # Load warning pose check images
        for path_name in ["img/warning_pose_check.png", "img/warning_pose.png", "warning_pose_check.png", "warning_pose.png"]:
            img_path = get_asset_path(path_name)
            if os.path.exists(img_path):
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    lbl = QLabel()
                    lbl.setPixmap(pixmap.scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    lbl.setAlignment(Qt.AlignCenter)
                    layout.addWidget(lbl)

        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

        self.setLayout(layout)

class MarkerRecognitionProblemDialog(QDialog):
    def __init__(self, parent=None, is_ko=None, arm_label=""):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle(tr("dialogs.marker_problem.title"))
        self.resize(1060, 760)
        self.setStyleSheet(DARK_STYLESHEET)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # 1. Title Header
        lbl_title = QLabel(tr("dialogs.marker_problem.warn_header", arm=arm_label))
        lbl_title.setObjectName("dialog_title_warn")
        lbl_title.setWordWrap(True)
        lbl_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl_title)

        # 2. Body Layout: Left Guidance Box | Right Live Camera Feed
        body_layout = QHBoxLayout()
        body_layout.setSpacing(15)

        # --- Left Box: Guidance Images & Instructions ---
        left_box = QGroupBox(tr("dialogs.marker_problem.steps_box_title"))
        left_box.setObjectName("dialog_blue_box")
        left_layout = QVBoxLayout(left_box)
        left_layout.setSpacing(10)

        img_row = QHBoxLayout()
        img_row.setSpacing(8)

        step_items = [
            ("img/marker_problem_before.png", "dialogs.marker_problem.step1_cap"),
            ("img/marker_problem_teaching.png", "dialogs.marker_problem.step2_cap"),
            ("img/marker_problem_after.png", "dialogs.marker_problem.step3_cap"),
        ]

        for img_path, cap_key in step_items:
            v_box = QVBoxLayout()
            lbl_img = QLabel()
            pix = QPixmap(get_asset_path(img_path))
            if not pix.isNull():
                lbl_img.setPixmap(pix.scaled(150, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                lbl_img.setText(f"[{img_path}]")
            lbl_img.setAlignment(Qt.AlignCenter)

            lbl_cap = QLabel(tr(cap_key))
            lbl_cap.setObjectName("dialog_step_cap")
            lbl_cap.setAlignment(Qt.AlignCenter)

            v_box.addWidget(lbl_img)
            v_box.addWidget(lbl_cap)
            img_row.addLayout(v_box)

        left_layout.addLayout(img_row)

        lbl_desc = QLabel(tr("dialogs.marker_problem.desc"))
        lbl_desc.setObjectName("dialog_desc")
        lbl_desc.setWordWrap(True)
        left_layout.addWidget(lbl_desc)
        body_layout.addWidget(left_box, stretch=5)

        # --- Right Box: Embedded Real-time Live Feed ---
        right_box = QGroupBox(tr("dialogs.marker_problem.feed_box_title"))
        right_box.setObjectName("dialog_green_box")
        right_layout = QVBoxLayout(right_box)
        right_layout.setSpacing(10)

        self.lbl_live_feed = QLabel(tr("dialogs.marker_problem.feed_loading"))
        self.lbl_live_feed.setObjectName("live_feed_display")
        self.lbl_live_feed.setAlignment(Qt.AlignCenter)
        self.lbl_live_feed.setMinimumSize(420, 320)
        right_layout.addWidget(self.lbl_live_feed)

        body_layout.addWidget(right_box, stretch=4)
        main_layout.addLayout(body_layout)

        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        btn_done = QPushButton(tr("dialogs.marker_problem.btn_done"))
        btn_done.setObjectName("btn_success_lg")
        btn_done.setMinimumHeight(52)
        btn_done.clicked.connect(self.accept)

        btn_cancel = QPushButton(tr("dialogs.marker_problem.btn_cancel"))
        btn_cancel.setObjectName("btn_danger_lg")
        btn_cancel.setMinimumHeight(52)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_done, stretch=2)
        btn_layout.addWidget(btn_cancel, stretch=1)
        main_layout.addLayout(btn_layout)

class MarkerSpacingDialog(QDialog):
    """Step 2 init pose: the operator widens the arms while the live marker gap is shown.

    Red while the gap is below the minimum (or a marker is out of view), green once it is wide
    enough; Done is enabled only while green. The gap itself is measured by the core and passed
    in through update_gap()."""
    GAP_OK_STYLE = "font-size: 30px; font-weight: bold; color: #66bb6a; padding: 8px;"
    GAP_BAD_STYLE = "font-size: 30px; font-weight: bold; color: #ef5350; padding: 8px;"
    CLOSE_CHECK_MS = 200

    def __init__(self, parent, min_gap_m, session, stop_event=None):
        super().__init__(parent)
        self.min_gap_m = float(min_gap_m)
        self.session = session
        self.stop_event = stop_event
        self.last_gap_m = None
        self.setWindowTitle(tr("dialogs.markers_too_close.monitor_title"))
        self.resize(900, 720)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl_title = QLabel(tr("dialogs.markers_too_close.monitor_header"))
        lbl_title.setObjectName("dialog_title_warn")
        lbl_title.setWordWrap(True)
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(tr("dialogs.markers_too_close.monitor_desc", min_gap=f"{self.min_gap_m * 100:.1f}"))
        lbl_desc.setObjectName("dialog_desc")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        self.lbl_gap = QLabel()
        self.lbl_gap.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_gap)

        feed_box = QGroupBox(tr("dialogs.marker_problem.feed_box_title"))
        feed_box.setObjectName("dialog_green_box")
        feed_layout = QVBoxLayout(feed_box)
        self.lbl_live_feed = QLabel(tr("dialogs.marker_problem.feed_loading"))
        self.lbl_live_feed.setObjectName("live_feed_display")
        self.lbl_live_feed.setAlignment(Qt.AlignCenter)
        self.lbl_live_feed.setMinimumSize(560, 360)
        feed_layout.addWidget(self.lbl_live_feed)
        layout.addWidget(feed_box, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.btn_done = QPushButton(tr("dialogs.markers_too_close.btn_done"))
        self.btn_done.setObjectName("btn_success_lg")
        self.btn_done.setMinimumHeight(52)
        self.btn_done.clicked.connect(self.accept)
        btn_cancel = QPushButton(tr("dialogs.markers_too_close.btn_cancel"))
        btn_cancel.setObjectName("btn_danger_lg")
        btn_cancel.setMinimumHeight(52)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_done, stretch=2)
        btn_layout.addWidget(btn_cancel, stretch=1)
        layout.addLayout(btn_layout)

        # The core stops monitoring on Stop or an error: close instead of waiting for the operator.
        self.close_timer = QTimer(self)
        self.close_timer.timeout.connect(self._close_if_abandoned)
        self.close_timer.start(self.CLOSE_CHECK_MS)

    @staticmethod
    def format_gap_cm(gap_m):
        # Rounded down: a 10.99 cm gap must not read "11.0 cm" while it is still shown red.
        return f"{np.floor(gap_m * 1000.0) / 10.0:.1f}"

    def update_gap(self, info):
        gap = info.get("gap_m") if isinstance(info, dict) else None
        min_gap = f"{self.min_gap_m * 100:.1f}"
        if gap is None:
            self.last_gap_m = None
            self.lbl_gap.setText(tr("dialogs.markers_too_close.gap_hidden", min_gap=min_gap))
            wide_enough = False
        else:
            self.last_gap_m = float(gap)
            self.lbl_gap.setText(tr("dialogs.markers_too_close.gap_value", gap=self.format_gap_cm(gap), min_gap=min_gap))
            wide_enough = gap >= self.min_gap_m
        self.lbl_gap.setStyleSheet(self.GAP_OK_STYLE if wide_enough else self.GAP_BAD_STYLE)
        self.btn_done.setEnabled(wide_enough)

    def accept(self):
        # Only a gap measured at or above the minimum may be confirmed.
        if self.last_gap_m is None or self.last_gap_m < self.min_gap_m:
            return
        super().accept()

    def _close_if_abandoned(self):
        if self.session.get("closed") or (self.stop_event is not None and self.stop_event.is_set()):
            self.reject()

    def done(self, result):
        self.close_timer.stop()
        super().done(result)

class ApplyHomeOffsetDialog(QDialog):
    def __init__(self, parent, result_path, baseline_path, arm, include_head, compare_summary=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.apply_home_offset.title"))
        self.resize(900, 600)
        self.setStyleSheet(DARK_STYLESHEET)

        self.parent_app = parent
        self.result_path = result_path
        self.baseline_path = baseline_path
        self.arm = arm
        self.include_head = include_head
        self._drag_pos = None

        self.current_apply_arm = parent.infer_home_offset_apply_arm(arm, result_path)

        if compare_summary is None:
            compare_summary = parent.format_home_offset_compare_summary(result_path, baseline_path)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        msg_lbl = QLabel(tr("dialogs.apply_home_offset.guide_msg"))
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl)

        # Summary text box
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setText(compare_summary)
        self.summary_box.setFont(QFont("Consolas", 10))
        layout.addWidget(self.summary_box, 1)

        # State Switcher (Big Toggle Buttons with external QSS)
        state_layout = QHBoxLayout()
        self.btn_group = QButtonGroup(self)

        self.btn_baseline = QPushButton(tr("dialogs.apply_home_offset.btn_baseline"))
        self.btn_opt = QPushButton(tr("dialogs.apply_home_offset.btn_optimized"))

        for btn in [self.btn_baseline, self.btn_opt]:
            btn.setObjectName("btn_state_toggle")
            btn.setCheckable(True)
            btn.setMinimumHeight(60)
            self.btn_group.addButton(btn)
            state_layout.addWidget(btn)

        self.btn_baseline.setChecked(True)

        # The optimized result can always be previewed (zero / check pose) for comparison; it may be
        # applied only once, and only when it was solved from samples collected at the current robot
        # zero in this run (CalibrationCore.home_epoch).
        self.opt_exists = bool(result_path is not None and os.path.exists(result_path))
        self.opt_allowed = self.opt_exists and parent.core.result_apply_allowed(result_path)
        self.btn_opt.setEnabled(self.opt_exists)

        layout.addLayout(state_layout)

        self.lbl_opt_locked = QLabel(tr("dialogs.apply_home_offset.opt_locked"))
        self.lbl_opt_locked.setWordWrap(True)
        self.lbl_opt_locked.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 13px;")
        self.lbl_opt_locked.setVisible(not self.opt_allowed)
        layout.addWidget(self.lbl_opt_locked)

        # Movement Buttons
        move_layout = QHBoxLayout()
        self.btn_move_zero = QPushButton(tr("dialogs.apply_home_offset.btn_move_zero"))
        self.btn_move_zero.clicked.connect(self.on_move_zero)
        move_layout.addWidget(self.btn_move_zero)

        self.btn_move_check = QPushButton(tr("dialogs.apply_home_offset.btn_move_check"))
        self.btn_move_check.clicked.connect(self.on_move_check)
        move_layout.addWidget(self.btn_move_check)
        layout.addLayout(move_layout)

        # Action buttons row
        btn_layout = QHBoxLayout()

        self.btn_apply = QPushButton(tr("dialogs.apply_home_offset.btn_apply_selected"))
        self.btn_apply.setObjectName("btn_apply_offset")
        self.btn_apply.clicked.connect(self.on_apply_selected)

        self.btn_apply.setEnabled(self.apply_possible())

        btn_layout.addWidget(self.btn_apply)

        btn_close = QPushButton(tr("dialogs.apply_home_offset.btn_close"))
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self._drag_pos = pos - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, '_drag_pos') and self._drag_pos is not None:
            pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self.move(pos - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def on_apply_selected(self):
        state, path = self.get_current_target()
        if confirm_dialog(self, tr("dialogs.apply_home_offset.confirm_title"), tr("dialogs.apply_home_offset.confirm_msg", state=state.upper())):
            self.on_apply(state)

    def get_current_target(self):
        if self.btn_baseline.isChecked():
            return "baseline", self.baseline_path
        else:
            return "optimized", self.result_path

    @idle_core_action
    def _execute_preview_motion(self, task_type: str, action_label: str, success_target: str, error_title: str):
        state, path = self.get_current_target()
        if not path or not os.path.exists(path):
            show_warning_dialog(self, tr("common.status_warning", default="Warning"), f"No {state} JSON found.")
            return

        self.set_buttons_enabled(False)
        self.worker = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            task_type,
            json_path=path,
            label=f"{state.capitalize()} {action_label}",
            arm=self.arm,
            include_head=self.include_head
        )
        self.worker.log_signal.connect(self.parent_app.log_msg)

        def on_finished(success, error_msg, res):
            self.set_buttons_enabled(True)
            if success:
                self.current_apply_arm = res.get("arm", self.arm)
                show_info_dialog(self, tr("dialogs.apply_home_offset.preview_complete_title"), tr("dialogs.apply_home_offset.preview_complete_msg", state=state, target=success_target))
            else:
                show_error_dialog(self, tr("dialogs.apply_home_offset.preview_error_title"), error_msg)

        self.worker.finished_signal.connect(on_finished)
        self.worker.start()

    def on_move_zero(self):
        self._execute_preview_motion("move_zero", "Zero", "zero", "Preview Error")

    def on_move_check(self):
        self._execute_preview_motion("move_check", "Check Position", "check position", "Check Preview Error")

    @idle_core_action
    def on_apply(self, state):
        state, path = self.get_current_target()
        if not path or not os.path.exists(path):
            show_warning_dialog(self, tr("common.status_warning", default="Warning"), f"No {state} JSON found.")
            return
        if self.optimized_blocked(state):
            return

        self.set_buttons_enabled(False)
        self.parent_app.log_msg(f"[INFO] Moving robot to '{state.upper()}' Zero Pose before applying home offset...")

        # Step 1: Move to Zero Pose first
        self.worker_move = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_zero",
            json_path=path,
            label=f"{state.capitalize()} Zero",
            arm=self.arm,
            include_head=self.include_head
        )
        self.worker_move.log_signal.connect(self.parent_app.log_msg)

        def on_move_finished(success, error_msg, res):
            if not success:
                self.set_buttons_enabled(True)
                show_error_dialog(self, "Zero Pose Move Error", f"Failed to move to zero pose before applying: {error_msg}")
                return

            if "arm" in res:
                self.current_apply_arm = res["arm"]

            self.parent_app.log_msg(f"[INFO] Arrived at '{state.upper()}' Zero Pose. Now resetting and applying home offset...")

            # Step 2: Apply Home Offset from the Zero Pose
            self.worker_apply = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "apply",
                arm=self.current_apply_arm,
                include_head=self.include_head,
                json_path=self.result_path if state == "optimized" else None
            )
            self.worker_apply.log_signal.connect(self.parent_app.log_msg)

            def on_apply_finished(app_success, app_error_msg, app_res):
                self.set_buttons_enabled(True)
                if app_success:
                    if app_res.get("needs_reconnect", False):
                        self.parent_app.log_msg("Re-connecting and initializing robot...")
                        if self.parent_app.robot:
                            self.parent_app.connect_robot()
                            QApplication.processEvents()
                        self.parent_app.connect_robot()
                        self.parent_app.log_msg("Current pose home offset apply complete.")

                    if app_res.get("success", False) or app_res.get("needs_reconnect", False):
                        # Reset software joint offsets to 0.0 for the applied arm(s) since they are now physically absorbed
                        for arm in ["left", "right"]:
                            if self.current_apply_arm == "both" or self.current_apply_arm == arm:
                                self.parent_app.joint_offsets_store[arm]["joint3"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint5"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint6"] = 0.0

                                self.parent_app.joint_offsets[arm]["wrist_pitch"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_roll"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_yaw2"] = 0.0
                                self.parent_app.joint_offsets[arm]["elbow"] = 0.0
                        if self.include_head:
                            # Head offsets are physically absorbed by the applied home offset too.
                            self.parent_app.joint_offsets_store["head"] = {"pan": 0.0, "tilt": 0.0}

                        # Save zeroed offsets to setting.yaml and update GUI
                        self.parent_app.save_offsets_to_yaml()
                        self.parent_app.update_applied_offset_label()

                        # Zero out baseline json if it exists to prevent accidental unsafe rollback later
                        if self.baseline_path and os.path.exists(self.baseline_path):
                            try:
                                data = ResultStorage.load(self.baseline_path)

                                if "right_arm_joint_offset_deg" in data and (self.current_apply_arm == "both" or self.current_apply_arm == "right"):
                                    data["right_arm_joint_offset_deg"] = [0.0] * len(data["right_arm_joint_offset_deg"])
                                if "left_arm_joint_offset_deg" in data and (self.current_apply_arm == "both" or self.current_apply_arm == "left"):
                                    data["left_arm_joint_offset_deg"] = [0.0] * len(data["left_arm_joint_offset_deg"])
                                if "head_joint_offset_deg" in data and data["head_joint_offset_deg"] is not None and self.include_head:
                                    data["head_joint_offset_deg"] = [0.0] * len(data["head_joint_offset_deg"])

                                if "right_arm_joint_offset_deg" in data and "left_arm_joint_offset_deg" in data:
                                    data["joint_offset_deg"] = data["right_arm_joint_offset_deg"] + data["left_arm_joint_offset_deg"]
                                elif "joint_offset_deg" in data:
                                    data["joint_offset_deg"] = [0.0] * len(data["joint_offset_deg"])

                                ResultStorage.save(self.baseline_path, data)
                                self.parent_app.log_msg(f"[INFO] Zeroed out applied arm offsets in baseline json: {self.baseline_path}")
                            except Exception as e:
                                self.parent_app.log_msg(f"[WARN] Failed to zero out baseline json: {e}")

                        show_info_dialog(self, tr("dialogs.apply_home_offset.apply_success_title"), tr("dialogs.apply_home_offset.apply_success_msg", state=state.upper()))
                        self.accept()
                    else:
                        show_warning_dialog(self, tr("common.status_warning", default="Warning"), tr("dialogs.apply_home_offset.apply_partial_warn"))
                else:
                    show_error_dialog(self, "Apply Pose Error", app_error_msg)

            self.worker_apply.finished_signal.connect(on_apply_finished)
            self.worker_apply.start()

        self.worker_move.finished_signal.connect(on_move_finished)
        self.worker_move.start()

    def apply_possible(self):
        """Rollback to the baseline is always possible; the optimized result only while allowed."""
        baseline_ok = self.baseline_path is not None and os.path.exists(self.baseline_path)
        return baseline_ok or self.opt_allowed

    def optimized_blocked(self, state):
        """Refuse an optimized application that the core would refuse, with the reason."""
        if state != "optimized" or self.parent_app.core.result_apply_allowed(self.result_path):
            return False
        show_warning_dialog(self, tr("dialogs.apply_home_offset.opt_locked_title"),
                            tr("dialogs.apply_home_offset.opt_locked"))
        return True

    def set_buttons_enabled(self, enabled):
        # An apply that failed part way may still have moved the robot zero: re-read the core.
        self.opt_allowed = self.opt_exists and self.parent_app.core.result_apply_allowed(self.result_path)
        self.lbl_opt_locked.setVisible(not self.opt_allowed)
        self.btn_move_zero.setEnabled(enabled)
        self.btn_move_check.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled and self.apply_possible())
        self.btn_opt.setEnabled(enabled and self.opt_exists)
        self.btn_baseline.setEnabled(enabled)

class CheckCalibrationStateDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Check Calibration State")
        self.resize(450, 300)
        self.setStyleSheet(DARK_STYLESHEET)

        self.parent_app = parent
        self.check_state_moved = False
        self._worker = None  # QThread 참조 유지 (GC 방지)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        grid = QGridLayout()
        grid.addWidget(QLabel("X Position (m):"), 0, 0)
        self.x_input = QLineEdit("0.35")
        self.x_input.setObjectName("dialog_input_field")
        grid.addWidget(self.x_input, 0, 1)

        grid.addWidget(QLabel("Y Position (m):"), 1, 0)
        self.y_input = QLineEdit("0.0")
        self.y_input.setObjectName("dialog_input_field")
        grid.addWidget(self.y_input, 1, 1)

        grid.addWidget(QLabel("Z Position (m):"), 2, 0)
        self.z_input = QLineEdit("0.0")
        self.z_input.setObjectName("dialog_input_field")
        grid.addWidget(self.z_input, 2, 1)

        grid.addWidget(QLabel("Y Offset (m):"), 3, 0)
        self.offset_input = QLineEdit("0.175")
        self.offset_input.setObjectName("dialog_input_field")
        grid.addWidget(self.offset_input, 3, 1)

        layout.addLayout(grid)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: #2979ff; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        btn_layout = QHBoxLayout()
        self.btn_move = QPushButton("Move")
        self.btn_move.clicked.connect(self.on_move)
        btn_layout.addWidget(self.btn_move)

        self.btn_draw = QPushButton("Draw Square")
        self.btn_draw.clicked.connect(self.on_draw_square)
        btn_layout.addWidget(self.btn_draw)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _set_buttons_enabled(self, enabled):
        self.btn_move.setEnabled(enabled)
        self.btn_draw.setEnabled(enabled)

    @idle_core_action
    def on_move(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            offset = float(self.offset_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid floating-point numbers.")
            return

        if not self.parent_app.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return

        self.lbl_status.setText("Status: Moving...")
        self.lbl_status.setStyleSheet("color: #ff9800;")
        self._set_buttons_enabled(False)

        # threading.Thread 대신 QThread(CheckCalibrationStateWorker) 사용
        # rby1_sdk C++ 라이브러리는 일반 Python 스레드와 호환되지 않아 segfault 발생
        worker = CheckCalibrationStateWorker(
            task_type="move",
            core=self.parent_app._sync_core_state(),
            model_name=self.parent_app.model_input.currentText().strip(),
            active_arms=["right", "left"],
            data=[x, y, z],
            offset=offset,
            skip_ready=self.check_state_moved,
        )
        worker.log_signal.connect(lambda msg: self.parent_app.log_msg(f"[Check State] {msg}"))

        def on_move_finished(success, error_msg):
            self._set_buttons_enabled(True)
            self._worker = None
            if success:
                self.check_state_moved = True
                self.parent_app.log_msg("[Check State] Symmetrical move completed successfully.")
                self.lbl_status.setText("Status: Move OK")
                self.lbl_status.setStyleSheet("color: #00e676;")
            else:
                self.parent_app.log_msg(f"[Check State Error] {error_msg}")
                self.lbl_status.setText("Status: Error")
                self.lbl_status.setStyleSheet("color: #ff1744;")

        worker.finished_signal.connect(on_move_finished)
        self._worker = worker
        worker.start()

    @idle_core_action
    def on_draw_square(self):
        if not self.check_state_moved:
            QMessageBox.warning(self, "Error", "Please click 'Move' first to reach the initial check state.")
            return

        if not self.parent_app.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return

        try:
            offset = float(self.offset_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid floating-point numbers for Y Offset.")
            return

        self.lbl_status.setText("Status: Drawing...")
        self.lbl_status.setStyleSheet("color: #ff9800;")
        self._set_buttons_enabled(False)

        # threading.Thread 대신 QThread(CheckCalibrationStateWorker) 사용
        worker = CheckCalibrationStateWorker(
            task_type="draw_square",
            core=self.parent_app._sync_core_state(),
            model_name=self.parent_app.model_input.currentText().strip(),
            active_arms=["right", "left"],
            data=[0.35, 0.0, 0.0],  # 기준 포지션 (draw_square 내부에서 포인트 순회)
            offset=offset,
            skip_ready=True,
        )
        worker.log_signal.connect(lambda msg: self.parent_app.log_msg(f"[Draw Square] {msg}"))

        def on_draw_finished(success, error_msg):
            self._set_buttons_enabled(True)
            self._worker = None
            if success:
                self.parent_app.log_msg("[Draw Square] Square drawing sequence completed successfully.")
                self.lbl_status.setText("Status: Draw OK")
                self.lbl_status.setStyleSheet("color: #00e676;")
            else:
                self.parent_app.log_msg(f"[Draw Square Error] {error_msg}")
                self.lbl_status.setText("Status: Draw Error")
                self.lbl_status.setStyleSheet("color: #ff1744;")

        worker.finished_signal.connect(on_draw_finished)
        self._worker = worker
        worker.start()

# --- Common Worker Threads ---


Step1_5ReadyWorker = HeadCamReadyWorker


Step1_5HeadSweepWorker = HeadCamSweepWorker


# Step2ApplyHomeOffsetWorker는 통합 HomeOffsetActionWorker 클래스로 연결


# --- Specialized Calibration Workers ---


# --- Unified Calibration App ---
class UnifiedCalibrationApp(QWidget):
    log_signal_safe = Signal(str)
    update_ui_signal_safe = Signal(str)
    marker_problem_signal = Signal(str, object, object)
    marker_spacing_signal = Signal(float, float, object)

    # Tab layout (2026-09-30): Wizard / Camera (intrinsics, brightness) / Advanced (Step 1, Step 2).
    # Code checks tabs by these names, never by a bare index.
    TAB_WIZARD, TAB_CAMERA, TAB_ADVANCED = 0, 1, 2
    CAMERA_SUB_INTRINSICS, CAMERA_SUB_EXPOSURE = 0, 1
    ADV_SUB_STEP1, ADV_SUB_STEP2 = 0, 1

    def is_camera_tab_active(self):
        return hasattr(self, 'left_tabs') and self.left_tabs.currentIndex() == self.TAB_CAMERA

    def is_intrinsics_subtab_active(self):
        return (self.is_camera_tab_active() and hasattr(self, 'camera_tabs')
                and self.camera_tabs.currentIndex() == self.CAMERA_SUB_INTRINSICS)

    def is_step2_view_active(self):
        return (hasattr(self, 'left_tabs') and self.left_tabs.currentIndex() == self.TAB_ADVANCED
                and hasattr(self, 'advanced_tabs') and self.advanced_tabs.currentIndex() == self.ADV_SUB_STEP2)

    @property
    def is_ko_ui(self) -> bool:
        return LanguageManager.instance().is_korean

    def _sync_core_state(self):
        """Read UI choices before the worker starts; workers never read widgets."""
        if self.core.is_busy:
            raise RuntimeError("Cannot change core inputs while a sequence is running")
        if self.core.robot is not self.robot:
            self.core.bind_robot(self.robot, self.get_robot_version())
        self.core.robot_version = self.get_robot_version()
        self.core.model = self.model
        self.core.include_head_motion = self.include_head_motion
        self.core.apply_joint_offset_flag = self.apply_joint_offset_flag
        self.core.robot_address = self.ip_input.text().strip()
        self.core.robot_model_name = self.model_input.currentText().strip().lower()
        self.core.joint_offsets_store = self.joint_offsets_store
        self.core.last_home_reset_path = getattr(self, "last_home_reset_path", None)
        self.core.auto_config = self.auto_config
        self.core.prompt_teaching = self.prompt_marker_problem_teaching
        self.core.prompt_marker_spacing = self.prompt_marker_spacing
        for calibrator in self.core.calibrators:
            calibrator.include_head_motion = self.include_head_motion
            calibrator.marker_problem_callback = self.prompt_marker_problem_teaching
        return self.core

    def _on_core_result(self, result):
        self.last_sequence_result = result
        self.last_result_path = self.core.last_result_path
        self.last_home_reset_path = self.core.last_home_reset_path
        self.log_msg(f"[{result.name}] {result.status}" + (f": {result.error}" if result.error else ""))
        def display_step(step):
            samples = step.completed.get("samples", step.partial.get("samples", []))
            for sample in samples:
                self.shared_arm_q_list.append(sample["q_arm"])
                self.shared_sample_epochs.append(sample.get("home_epoch"))
                if sample["q_head"] is not None:
                    self.shared_head_q_list.append(sample["q_head"])
                self.shared_T_list.append(sample["marker"])
            if samples:
                self.head_move_count = step.partial.get("next_motion_index", self.head_move_count)
                self.update_sample_counts()
                self.update_head_pose_status()
            head = step.completed.get("head_camera")
            if head and not head.get("skipped"):
                self._update_step1_5_tables(head)
            if step.name == "full":
                for name in ("step1", "step1_5", "step2"):
                    child = step.completed.get(name, step.partial.get(name))
                    if child is not None:
                        display_step(child)
        display_step(result)
        step1 = result if result.name == "step1" else result.completed.get("step1")
        if step1 is not None and step1.success:
            # The core has written Step 1 to setting.yaml; show what was saved.
            self.update_applied_offset_label()
            self.load_bracket_design_values()

    def prompt_marker_spacing(self, gap_m, min_gap_m):
        """Core callback (worker thread): open the widen-the-arms dialog and return at once.

        The core keeps measuring the gap and sends it through CoreBridge.marker_gap; the session
        is completed by the dialog (done/accepted/gap_m) or closed by the core (closed)."""
        session = {"done": threading.Event(), "accepted": False, "gap_m": None, "closed": False}
        self.marker_spacing_signal.emit(float(gap_m), float(min_gap_m), session)
        return session

    def _on_marker_spacing_requested(self, gap_m, min_gap_m, session):
        try:
            self.show_message_box(
                tr("dialogs.markers_too_close.title"),
                tr("dialogs.markers_too_close.notice_msg", gap=MarkerSpacingDialog.format_gap_cm(gap_m),
                   min_gap=f"{min_gap_m * 100:.1f}"),
                QMessageBox.Warning)
            dlg = MarkerSpacingDialog(self, min_gap_m, session, stop_event=self.core.stop_event)
            dlg.update_gap({"gap_m": gap_m, "min_gap_m": min_gap_m})
            self.marker_spacing_dlg = dlg
            self.on_left_tab_changed(self.left_tabs.currentIndex())
            accepted = dlg.exec() == QDialog.Accepted
            session["gap_m"] = dlg.last_gap_m
            session["accepted"] = accepted
        finally:
            self.marker_spacing_dlg = None
            self.on_left_tab_changed(self.left_tabs.currentIndex())
            session["done"].set()

    def _on_marker_gap(self, info):
        dlg = getattr(self, "marker_spacing_dlg", None)
        if dlg is not None:
            dlg.update_gap(info)

    def marker_monitor_panels(self):
        """The wizard exposure slide's panel and the Camera > Brightness panel (same core monitor)."""
        wizard = getattr(self, "wizard_widget", None)
        panels = (getattr(wizard, "marker_monitor_panel", None), getattr(self, "marker_monitor_panel", None))
        return [panel for panel in panels if panel is not None]

    def _on_marker_monitor(self, info):
        for panel in self.marker_monitor_panels():
            panel.update_monitor(info)

    def reset_marker_monitor_confirmation(self):
        """Call whenever the camera exposure changes: earlier marker confirmations no longer count,
        and neither do the monitor's samples taken at the old exposure."""
        window = self.core.restart_marker_monitor_window()
        for panel in self.marker_monitor_panels():
            panel.reset_confirmation(window)

    def is_marker_monitor_view_active(self):
        wizard = getattr(self, "wizard_widget", None)
        if (wizard is not None and wizard.isVisible()
                and wizard.stacked_widget.currentIndex() == wizard.SLIDE_EXPOSURE):
            return True
        return (self.is_camera_tab_active() and hasattr(self, 'camera_tabs')
                and self.camera_tabs.currentIndex() == self.CAMERA_SUB_EXPOSURE)

    def sync_marker_monitor(self):
        """Run the core marker monitor exactly while a view that shows it is on screen."""
        core = getattr(self, "core", None)
        if core is None or not hasattr(core, "start_marker_monitor"):
            return
        wanted = self.is_marker_monitor_view_active() and core.observer is not None
        try:
            if wanted and not core.marker_monitor_running and not core.is_busy:
                core.start_marker_monitor()
            elif not wanted and core.marker_monitor_running:
                core.stop_marker_monitor()
        except Exception as error:
            self.log_msg(f"[WARN] Marker monitor: {error}")

    def __init__(self, core=None, robot=None, arm_side="right", ui_only=False):
        super().__init__()
        # The UI-only mode is for trying the screens without hardware; its log is not a run record.
        self.save_gui_log = not ui_only
        self.log_signal_safe.connect(self._log_msg_slot)
        self.update_ui_signal_safe.connect(self._update_ui_slot)
        self.marker_problem_signal.connect(self._on_marker_problem_requested)
        self.marker_spacing_signal.connect(self._on_marker_spacing_requested)
        self.marker_spacing_dlg = None

        self.core = core if isinstance(core, CalibrationCore) else CalibrationCore(observer=core, robot=robot)
        self.core_bridge = CoreBridge(self.core, self)
        self.core_bridge.log.connect(self.log_msg)
        self.core_bridge.result.connect(self._on_core_result)
        self.core_bridge.bracket.connect(self.handle_full_auto_bracket_finished)
        self.core_bridge.joint.connect(self.handle_full_auto_joint_finished)
        self.core_bridge.progress.connect(self._on_core_progress)
        self.core_bridge.marker_gap.connect(self._on_marker_gap)
        self.core_bridge.marker_monitor.connect(self._on_marker_monitor)
        self.last_sequence_result = None
        self.robot = self.core.robot
        self.arm_side = arm_side
        self.ui_only = ui_only

        # Core Calibrator Instances
        self.marker_calibrator = self.core.marker_calibrator
        self.joint_calibrator = self.core.joint_calibrator
        self.head_camera_calibrator = self.core.head_camera_calibrator
        self.robot_version = "1.2"

        # Intrinsics Calibrator (Tab 3 용)
        self.intrinsics_calibrator = self.core.intrinsics_calibrator
        # Default: 8x5 squares, 36mm x 27mm, DICT_5X5_100
        self.intrinsics_calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 36.0, 27.0, "DICT_5X5_100")

        self.monitor_enabled = False
        self.captured_images = []
        self.current_guide_idx = 0
        self.intrinsics_busy = False
        self._intrinsics_button_states = {}
        self.output_yaml = CONFIG_PATHS["camera_intrinsics"]

        # Saved Calibration Results
        self.marker_data_4 = None
        self.marker_data_5 = None
        self.marker_data_6 = None
        self.joint_sweep_data = None
        self.generated_plots = []
        self.current_plot_idx = -1

        # Cumulative Joint Offsets for iterative sweeps
        self.joint_offsets = {
            "left": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
            "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
        }
        self.wrist_roll_calibrated = {"right": False, "left": False}
        self.ready_done_joint = False
        self.ready_done_marker = False
        self.load_offsets_from_yaml()

        # Step 2 calibration state
        self.apply_joint_offset_flag = True
        self.include_head_motion = True
        self.shared_arm_q_list = []
        self.shared_head_q_list = []
        self.shared_T_list = []
        # Robot zero (CalibrationCore.home_epoch) each shared sample was collected at.
        self.shared_sample_epochs = []
        self.head_move_count = 0

        self.auto_config = AutoCollectionConfig()
        self.auto_motion_plan = None
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.auto_motion_running = False
        self.auto_stop_requested = False
        self.auto_motion_thread = None
        self.sim_q_full = np.zeros(30)

        self.last_result_path = None
        self.last_home_reset_path = None
        self.last_dataset_path = None
        self.dataset_saved_in_session = False
        self.current_session_dataset_path = None

        self.check_state_moved = False

        self.model = None
        self.dyn_model = None
        if self.robot and hasattr(self.robot, "model") and hasattr(self.robot, "get_dynamics"):
            try:
                self.model = self.robot.model()
                self.dyn_model = self.robot.get_dynamics()
            except Exception:
                self.model = None
                self.dyn_model = None

        if self.model is None or self.dyn_model is None:
            if hasattr(self, 'joint_calibrator') and getattr(self.joint_calibrator, 'robot', None) is not None:
                try:
                    self.model = self.joint_calibrator.robot.model()
                    self.dyn_model = self.joint_calibrator.robot.get_dynamics()
                except Exception:
                    pass

        if self.model and hasattr(self.model, "robot_joint_names"):
            self.sim_q_full = np.zeros(len(self.model.robot_joint_names))
        else:
            self.sim_q_full = np.zeros(30)

        self.recommended_joint_offset = None

        self.marker_calibrator.joint_offsets = self.joint_offsets
        self.joint_calibrator.joint_offsets = self.joint_offsets
        if self.robot:
            try:
                self.robot.joint_offsets = self.joint_offsets
            except AttributeError:
                pass

        self.setWindowTitle("Unified Robot Calibration Suite")
        self.resize(1050, 620)
        self.setStyleSheet(DARK_STYLESHEET)

        # 1. 200ms poll timer (탭 1, 2, 4 용)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera_status)

        # 2. 33ms video timer (탭 3용)
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)

        # 3. Dedicated Temperature Monitor Timer (runs continuously every 5 seconds)
        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.poll_camera_temperature)
        self.temp_timer.start(5000)

        self.init_ui()
        self.load_bracket_design_values()
        self.update_applied_offset_label()

        # 초기화 시 탭 상태에 맞춰 타이머 활성화
        self.on_left_tab_changed(self.left_tabs.currentIndex())

        self.active_worker = None

        self.warn_camera_config_gaps()

        self.init_camera_exposure_state()

    def warn_camera_config_gaps(self):
        """Report anything the connected camera model has no stored data for.

        The observer resolves the bracket extrinsics and the intrinsics from the model it detected
        and records what it could not find. Neither gap stops the app: it runs on the camera's
        factory intrinsics and the extrinsics already in setting.yaml, both of which give a wrong
        calibration, so say so plainly instead of failing silently.
        """
        observer = self.core.observer
        if not observer:
            return
        connected_model = getattr(observer, 'camera_model', '') or "?"

        if getattr(observer, 'extrinsics_missing', False):
            QMessageBox.critical(
                self,
                tr("dialogs.camera_extrinsics_missing.title"),
                tr("dialogs.camera_extrinsics_missing.text").format(connected_model=connected_model) + "\n\n" +
                tr("dialogs.camera_extrinsics_missing.informative_text").format(connected_model=connected_model)
            )

        if getattr(observer, 'intrinsics_missing', False):
            QMessageBox.critical(
                self,
                tr("dialogs.camera_intrinsics_missing.title"),
                tr("dialogs.camera_intrinsics_missing.text").format(
                    connected_model=connected_model, store=connected_model.lower()) + "\n\n" +
                tr("dialogs.camera_intrinsics_missing.informative_text")
            )
        elif getattr(observer, 'intrinsics_mismatch', False):
            calib_device = getattr(observer, 'calib_device_name', '')
            QMessageBox.warning(
                self,
                tr("dialogs.camera_intrinsics_warning.title"),
                tr("dialogs.camera_intrinsics_warning.text").format(connected_model=connected_model, calib_device=calib_device) + "\n\n" +
                tr("dialogs.camera_intrinsics_warning.informative_text")
            )

    def init_camera_exposure_state(self):
        # Load exposure config from setting.yaml with fallback to defaults
        setting_path = CONFIG_PATHS.get("setting_yaml") or CONFIG_PATHS.get("setting")
        exp_val = 6000
        auto_mode = True
        if setting_path and os.path.exists(setting_path):
            try:
                cfg = ConfigStorage.load(setting_path)
                cam_cfg = cfg.get("camera", {})
                if "exposure" in cam_cfg:
                    exp_val = int(cam_cfg["exposure"])
                if "auto_exposure" in cam_cfg:
                    auto_mode = bool(cam_cfg["auto_exposure"])
            except Exception as e:
                self.log_msg(f"[Camera] Warning: Failed to load exposure from setting: {e}")

        self.camera_auto_exposure = auto_mode
        self.camera_exposure_value = exp_val
        self.applied_camera_auto_exposure = auto_mode
        self.applied_camera_exposure_value = exp_val
        self.saved_camera_auto_exposure = auto_mode
        self.saved_camera_exposure_value = exp_val

        if self.core.observer is not None and hasattr(self.core.observer, 'set_camera_exposure'):
            self.core.set_camera_exposure(exp_val, auto_exposure=auto_mode)
            self.log_msg(f"[Camera] Initialized camera exposure: auto={auto_mode}, exposure={exp_val}")
        self.reset_marker_monitor_confirmation()

        # Reconnecting can bring up a different model, whose exposure option has its own range.
        self.sync_exposure_limits_from_camera()

    def set_camera_auto_mode(self):
        self.applied_camera_auto_exposure = True
        if hasattr(self, 'chk_auto_exposure'):
            self.chk_auto_exposure.blockSignals(True)
            self.chk_auto_exposure.setChecked(True)
            self.chk_auto_exposure.blockSignals(False)
        if hasattr(self, 'spin_exposure'):
            self.spin_exposure.setEnabled(False)
        if hasattr(self, 'slider_exposure'):
            self.slider_exposure.setEnabled(False)

        cur_exp = getattr(self, 'saved_camera_exposure_value', 6000)
        if self.core.observer is not None and hasattr(self.core.observer, 'set_camera_exposure'):
            self.core.set_camera_exposure(cur_exp, auto_exposure=True)
        self.reset_marker_monitor_confirmation()
        self.log_msg("[Camera] Switched to AUTO exposure mode.")

    def on_auto_exposure_toggled(self, checked):
        if hasattr(self, 'spin_exposure'):
            self.spin_exposure.setEnabled(not checked)
        if hasattr(self, 'slider_exposure'):
            self.slider_exposure.setEnabled(not checked)
        if checked:
            self.set_camera_auto_mode()
        else:
            if hasattr(self, 'spin_exposure'):
                self.spin_exposure.setFocus()

    def sync_exposure_limits_from_camera(self):
        """Bound the exposure controls by what the connected camera's colour sensor accepts.

        The range is per model and cannot be assumed: a D435's RGB camera bottoms out well below
        the 100 these controls used to allow, so short exposures could not be set on it at all.
        Falls back to the widened built-in bounds when no camera can be asked.
        """
        observer = self.core.observer
        limits = None
        if observer is not None and hasattr(observer, 'get_camera_exposure_range'):
            try:
                limits = observer.get_camera_exposure_range()
            except Exception as error:
                self.log_msg(f"[Camera][WARN] Could not read the exposure range: {error}")
        if not limits:
            return
        low, high, step, _default = (int(limits[0]), int(limits[1]), int(limits[2]), limits[3])
        step = max(1, step)
        unit_us = 1.0
        if hasattr(observer, 'get_camera_exposure_unit_us'):
            try:
                unit_us = float(observer.get_camera_exposure_unit_us() or 1.0)
            except Exception:
                unit_us = 1.0
        self.exposure_unit_us = unit_us
        wizard = getattr(self, 'wizard_widget', None)
        widgets = [getattr(self, 'spin_exposure', None), getattr(self, 'slider_exposure', None),
                   getattr(wizard, 'spin_wiz_exp', None), getattr(wizard, 'slider_wiz_exp', None)]
        for widget in widgets:
            if widget is None:
                continue
            widget.blockSignals(True)
            widget.setRange(low, high)
            widget.setSingleStep(max(step, min(500, max(1, (high - low) // 200))))
            widget.blockSignals(False)
        range_text = (f"Range {low}–{high}  (1 = {unit_us:g} µs → "
                      f"{self.exposure_ms(low):.1f}–{self.exposure_ms(high):.0f} ms)")
        for label in (getattr(self, 'lbl_exposure_range', None), getattr(wizard, 'lbl_wiz_exp_range', None)):
            if label is not None:
                label.setText(range_text)
        self.log_msg(f"[Camera] Exposure range for this camera: {low}..{high} (step {step}, "
                     f"1 unit = {unit_us:g} us, i.e. {self.exposure_ms(low):.1f}..{self.exposure_ms(high):.0f} ms)")
        # Start the controls from what the camera is actually set to: a setpoint remembered in
        # another camera's units (the 6000 default was microseconds) is 600 ms on a D435.
        if hasattr(observer, 'get_camera_exposure'):
            try:
                _auto, cam_value = observer.get_camera_exposure()
                if cam_value is not None and low <= int(cam_value) <= high and hasattr(self, 'spin_exposure'):
                    self.on_exposure_value_changed(int(cam_value))
                    if wizard is not None and hasattr(wizard, 'spin_wiz_exp'):
                        for widget in (wizard.spin_wiz_exp, wizard.slider_wiz_exp):
                            widget.blockSignals(True)
                            widget.setValue(int(cam_value))
                            widget.blockSignals(False)
            except Exception:
                pass
        current = self.spin_exposure.value() if hasattr(self, 'spin_exposure') else low
        if not low <= current <= high:
            self.log_msg(f"[Camera] Stored exposure {current} is outside that range; using {min(max(current, low), high)}.")
            self.on_exposure_value_changed(min(max(current, low), high))
        self.refresh_measured_exposure()

    def exposure_ms(self, value):
        """A setpoint in the camera's own exposure units, in milliseconds."""
        return float(value) * float(getattr(self, 'exposure_unit_us', 1.0) or 1.0) / 1000.0

    def refresh_measured_exposure(self):
        """Show the exposure the camera actually reports in its frame metadata, in ms.

        This is the one figure that means the same thing on every model -- unlike the setpoint,
        whose unit belongs to the sensor.
        """
        if not hasattr(self, 'lbl_exposure_ms'):
            return
        try:
            measured = self.core.get_monitor_snapshot().get("exposure")
        except Exception:
            measured = None
        self.lbl_exposure_ms.setText(f"≈ {float(measured) / 1000.0:.1f} ms" if measured else "")

    def on_exposure_value_changed(self, value):
        if hasattr(self, 'slider_exposure') and self.slider_exposure.value() != value:
            self.slider_exposure.blockSignals(True)
            self.slider_exposure.setValue(value)
            self.slider_exposure.blockSignals(False)
        if hasattr(self, 'spin_exposure') and self.spin_exposure.value() != value:
            self.spin_exposure.blockSignals(True)
            self.spin_exposure.setValue(value)
            self.spin_exposure.blockSignals(False)

    def apply_camera_exposure(self):
        auto_mode = self.chk_auto_exposure.isChecked() if hasattr(self, 'chk_auto_exposure') else True
        exp_val = self.spin_exposure.value() if hasattr(self, 'spin_exposure') else 6000
        self.applied_camera_auto_exposure = auto_mode
        self.applied_camera_exposure_value = exp_val
        self.saved_camera_auto_exposure = auto_mode
        self.saved_camera_exposure_value = exp_val

        if self.core.observer is not None and hasattr(self.core.observer, 'set_camera_exposure'):
            self.core.set_camera_exposure(exp_val, auto_exposure=auto_mode)
        self.reset_marker_monitor_confirmation()

        if auto_mode:
            self.log_msg("[Camera] Applied AUTO exposure setting.")
        else:
            self.log_msg(f"[Camera] Applied manual exposure: {exp_val}")
        self.refresh_measured_exposure()

    def cancel_camera_exposure(self):
        # Restore previous applied/saved state
        self.applied_camera_auto_exposure = self.saved_camera_auto_exposure
        self.applied_camera_exposure_value = self.saved_camera_exposure_value

        if hasattr(self, 'chk_auto_exposure'):
            self.chk_auto_exposure.blockSignals(True)
            self.chk_auto_exposure.setChecked(self.saved_camera_auto_exposure)
            self.chk_auto_exposure.blockSignals(False)
        if hasattr(self, 'spin_exposure'):
            self.spin_exposure.blockSignals(True)
            self.spin_exposure.setValue(self.saved_camera_exposure_value)
            self.spin_exposure.setEnabled(not self.saved_camera_auto_exposure)
            self.spin_exposure.blockSignals(False)
        if hasattr(self, 'slider_exposure'):
            self.slider_exposure.blockSignals(True)
            self.slider_exposure.setValue(self.saved_camera_exposure_value)
            self.slider_exposure.setEnabled(not self.saved_camera_auto_exposure)
            self.slider_exposure.blockSignals(False)
        if self.core.observer is not None and hasattr(self.core.observer, 'set_camera_exposure'):
            self.core.set_camera_exposure(self.saved_camera_exposure_value, auto_exposure=self.saved_camera_auto_exposure)
        self.reset_marker_monitor_confirmation()
        self.refresh_measured_exposure()
        self.log_msg("[Camera] Exposure changes cancelled. Restored previous setting.")

    def save_camera_exposure(self):
        self.saved_camera_auto_exposure = self.applied_camera_auto_exposure
        self.saved_camera_exposure_value = self.applied_camera_exposure_value
        self.log_msg(f"[SUCCESS] Confirmed and saved exposure for calibration: auto={self.saved_camera_auto_exposure}, exposure={self.saved_camera_exposure_value}")

    def reconnect_camera(self, show_dialog=True):
        # The connect blocks this thread; clicks made meanwhile used to queue up and each ran another
        # reconnect. Ignore them: the button stays disabled until the queued clicks have been delivered.
        if getattr(self, "camera_reconnecting", False):
            return
        self.camera_reconnecting = True
        button = getattr(self, "btn_reconnect_cam", None)
        if button is not None:
            button.setEnabled(False)
            button.setText(tr("common.reconnecting_camera"))
            QApplication.processEvents()
        self.log_msg("[INFO] Reconnecting RealSense camera...")
        error = None
        try:
            self.core.connect_camera()
            self.init_camera_exposure_state()
            self.refresh_camera_tab_info()
            self.sync_marker_monitor()
            self.log_msg("[SUCCESS] RealSense Camera connected and streaming successfully!")
        except Exception as e:
            error = e
            self.log_msg(f"[ERROR] Failed to connect RealSense camera: {e}")
        finally:
            QApplication.processEvents()
            if button is not None:
                button.setText(tr("common.reconnect_camera"))
                button.setEnabled(True)
            self.camera_reconnecting = False
        if show_dialog and error is None:
            QMessageBox.information(self, tr("common.camera_connected_title"), tr("common.camera_connected_msg"))
        elif show_dialog:
            QMessageBox.warning(self, tr("common.camera_failed_title"), tr("common.camera_failed_msg", error=error))

    def load_offsets_from_yaml(self):
        self.joint_offsets_store = {
            "left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
            "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0}
        }
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if os.path.exists(config_path):
                data = ConfigStorage.load(config_path)
                jo = data.get("joint_offset", {})
                for arm in ["left", "right"]:
                    arm_data = jo.get(arm, {})
                    if isinstance(arm_data, dict):
                        self.joint_offsets_store[arm]["joint3"] = float(arm_data.get("joint3", 0.0))
                        self.joint_offsets_store[arm]["joint5"] = float(arm_data.get("joint5", 0.0))
                        self.joint_offsets_store[arm]["joint6"] = float(arm_data.get("joint6", 0.0))
                if "head" in jo and isinstance(jo["head"], dict):
                    self.joint_offsets_store["head"] = {
                        "pan": float(jo["head"].get("pan", 0.0)),
                        "tilt": float(jo["head"].get("tilt", 0.0)),
                    }
                head_str = f" Head[Pan={self.joint_offsets_store['head']['pan']:.4f}°, Tilt={self.joint_offsets_store['head']['tilt']:.4f}°]" if "head" in self.joint_offsets_store else ""
                self.log_msg(f"[INFO] Loaded joint offsets from setting.yaml: "
                             f"R[J3={self.joint_offsets_store['right']['joint3']:.4f}°, "
                             f"J5={self.joint_offsets_store['right']['joint5']:.4f}°, "
                             f"J6={self.joint_offsets_store['right']['joint6']:.4f}°] "
                             f"L[J3={self.joint_offsets_store['left']['joint3']:.4f}°, "
                             f"J5={self.joint_offsets_store['left']['joint5']:.4f}°, "
                             f"J6={self.joint_offsets_store['left']['joint6']:.4f}°]{head_str}")
            else:
                self.log_msg("[INFO] setting.yaml not found. Initialized all joint offsets to 0.0°.")
        except Exception as e:
            self.log_msg(f"[WARNING] Failed to load joint offsets from setting.yaml: {e}. Using 0.0° defaults.")

        is_v13 = self.get_robot_version() == "1.3"
        self.joint_offsets = {
            "left": {
                "wrist_pitch": self.joint_offsets_store["left"]["joint5"],
                "wrist_roll": self.joint_offsets_store["left"]["joint6"] if is_v13 else 0.0,
                "wrist_yaw2": self.joint_offsets_store["left"]["joint6"] if not is_v13 else 0.0,
                "elbow": self.joint_offsets_store["left"]["joint3"]
            },
            "right": {
                "wrist_pitch": self.joint_offsets_store["right"]["joint5"],
                "wrist_roll": self.joint_offsets_store["right"]["joint6"] if is_v13 else 0.0,
                "wrist_yaw2": self.joint_offsets_store["right"]["joint6"] if not is_v13 else 0.0,
                "elbow": self.joint_offsets_store["right"]["joint3"]
            }
        }
        self.marker_calibrator.joint_offsets = self.joint_offsets
        self.joint_calibrator.joint_offsets = self.joint_offsets
        self.marker_calibrator.marker_problem_callback = self.prompt_marker_problem_teaching
        self.joint_calibrator.marker_problem_callback = self.prompt_marker_problem_teaching

    def save_offsets_to_yaml(self):
        """Persist the staged joint offsets. Parsed-and-rewritten rather than line-patched so
        the values keep full precision and the rest of the file is left untouched."""
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            store = self.joint_offsets_store
            changes = {}
            for arm in ("left", "right"):
                for joint in ("joint3", "joint5", "joint6"):
                    changes[("joint_offset", arm, joint)] = float(store[arm].get(joint, 0.0))
            if "head" in store:
                for axis in ("pan", "tilt"):
                    changes[("joint_offset", "head", axis)] = float(store["head"].get(axis, 0.0))
            duplicates = ConfigStorage.update_values(config_path, changes)
            if duplicates:
                self.log_msg(f"[WARN] setting.yaml had duplicate keys {sorted(set(duplicates))}; "
                    "only the last of each was kept.")
            self.log_msg("[SUCCESS] Saved offsets permanently to setting.yaml!")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save offsets to setting.yaml: {e}")

    def on_cell_double_clicked(self, row, col):
        arm = "right" if row == 0 else "left"
        if col == 0:
            joint_key = "joint6"
            joint_label = "Joint 6"
        elif col == 1:
            joint_key = "joint5"
            joint_label = "Joint 5"
        else:
            joint_key = "joint3"
            joint_label = "Joint 3"

        current_val = self.joint_offsets_store[arm][joint_key]
        new_val, ok = QInputDialog.getDouble(
            self,
            "Manual Offset Override",
            f"Enter manual staged offset for {arm.upper()} Arm {joint_label} (degrees):",
            current_val, -45.0, 45.0, 4
        )
        if ok:
            self.joint_offsets_store[arm][joint_key] = new_val
            self.update_applied_offset_label()
            self.log_msg(f"[MANUAL OVERRIDE] Staged {arm.upper()} Arm {joint_label} offset manually to {new_val:.4f}°. (Not saved to disk yet. Click APPLY OFFSET to save)")

    def update_joint_modes(self):
        if not hasattr(self, 'joint_mode_sel'):
            return
        is_v13 = self.get_robot_version() == "1.3"
        self.joint_mode_sel.blockSignals(True)
        self.joint_mode_sel.clear()
        if is_v13:
            self.joint_mode_sel.addItems(UI_DROPDOWNS["joint_modes_v13"])
        else:
            self.joint_mode_sel.addItems(UI_DROPDOWNS["joint_modes_v12"])
        self.joint_mode_sel.blockSignals(False)


    def get_selected_joint_mode(self):
        if not hasattr(self, 'joint_mode_sel'):
            return "wrist_pitch"
        mode_str = self.joint_mode_sel.currentText()
        if "wrist_pitch_v13" in mode_str:
            return "wrist_pitch_v13"
        elif "wrist_roll_v13" in mode_str:
            return "wrist_roll_v13"
        elif "wrist_yaw2" in mode_str:
            return "wrist_yaw2"
        elif "wrist_pitch" in mode_str:
            return "wrist_pitch"
        else:
            return "elbow"

    def get_offset_key_for_mode(self, mode):
        if mode == "wrist_pitch_v13":
            return "wrist_pitch"
        elif mode == "wrist_roll_v13":
            return "wrist_roll"
        elif mode == "wrist_yaw2":
            return "wrist_yaw2"
        else:
            return mode

    def init_ui(self):
        # Instantiate the dialog first
        self.plot_dialog = PlotViewerDialog(self)
        # Re-map plot widgets to the floating dialog
        self.lbl_plot_title = self.plot_dialog.lbl_title
        self.plot_label_combined = self.plot_dialog.plot_label
        self.btn_plot_prev = self.plot_dialog.btn_prev
        self.btn_plot_next = self.plot_dialog.btn_next

        # Main horizontal layout
        main_layout = QHBoxLayout()

        # --- Top-Level Step Tabs ---
        self.left_tabs = QTabWidget()
        self.left_tabs.setObjectName("mainTabs")   # larger, high-contrast tab bar (dark_theme.qss)
        self.left_tabs.currentChanged.connect(self.on_left_tab_changed)

        # ==========================================
        # 1. Main Tab (로봇 동작 및 캘리브레이션 모듈)
        # ==========================================
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout()
        main_tab_layout.setContentsMargins(5, 5, 5, 5)

        # --- COLUMN 1 (Robot Connection, Head & Home, Workflows) ---
        col1_layout = QVBoxLayout()

        # Robot Connection Box (head movement controls removed per user request)
        conn_head_box = QGroupBox("Robot Connection")
        conn_head_box.setFixedHeight(130)
        conn_head_layout = QVBoxLayout()
        conn_head_layout.setSpacing(4)
        conn_head_layout.setContentsMargins(6, 6, 6, 6)

        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP/Port:"))
        is_sim_or_ui = self.ui_only or getattr(self.core.observer, 'sim', False)
        default_ip = "127.0.0.1:50051" if is_sim_or_ui else "192.168.30.1:50051"
        self.ip_input = QLineEdit(default_ip)
        self.ip_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        ip_row.addWidget(self.ip_input)
        conn_head_layout.addLayout(ip_row)

        model_row = QHBoxLayout()
        self.lbl_model_tag = QLabel("Model:")
        model_row.addWidget(self.lbl_model_tag)
        self.model_input = QComboBox()
        self.model_input.addItems(UI_DROPDOWNS["robot_models"])
        model_row.addWidget(self.model_input)
        conn_head_layout.addLayout(model_row)

        # Hide model selection UI as it is auto-detected and updated dynamically
        self.lbl_model_tag.hide()
        self.model_input.hide()

        connect_head_row = QHBoxLayout()
        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.setStyleSheet("background-color: #2b5278; color: #ffffff; font-weight: bold; padding: 4px 8px; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_connect.clicked.connect(self.connect_robot)
        self.btn_connect.setFixedHeight(28)
        connect_head_row.addWidget(self.btn_connect)

        # Head checkbox — controls whether head servos are enabled on connect
        self.chk_servo_head = QCheckBox("Head")
        self.chk_servo_head.setChecked(True)
        self.chk_servo_head.setStyleSheet("color: #b0bec5;")
        self.chk_servo_head.toggled.connect(self.on_head_checkbox_changed)
        connect_head_row.addWidget(self.chk_servo_head)
        conn_head_layout.addLayout(connect_head_row)

        conn_head_box.setLayout(conn_head_layout)

        # Calibration Workflows Box
        workflow_box = QGroupBox("Calibration Workflows")
        workflow_layout = QVBoxLayout()

        # Target Arm Selection
        self.arm_sel = QComboBox()
        self.arm_sel.addItems(UI_DROPDOWNS["arm_sides"])
        idx = 1 if self.arm_side == "left" else 0
        self.arm_sel.setCurrentIndex(idx)
        self.arm_sel.currentTextChanged.connect(self.on_arm_side_changed)

        self.joint_arm_sel = self.arm_sel
        self.marker_arm_sel = self.arm_sel

        arm_side_layout = QHBoxLayout()
        arm_side_layout.addWidget(QLabel("Active Arm Side:"))
        arm_side_layout.addWidget(self.arm_sel)
        workflow_layout.addLayout(arm_side_layout)

        debug_row = QHBoxLayout()
        self.chk_save_debug = QCheckBox("Save Debug Data")
        self.chk_save_debug.setChecked(True)
        debug_row.addWidget(self.chk_save_debug)
        workflow_layout.addLayout(debug_row)

        self.btn_stop_motion = QPushButton("STOP MOTION")
        self.btn_stop_motion.setStyleSheet("background-color: #c0392b; color: #ffffff; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_stop_motion.clicked.connect(self.stop_motion)
        self.btn_stop_motion.setFixedHeight(26)
        workflow_layout.addWidget(self.btn_stop_motion)

        self.workflow_tabs = QTabWidget()

        # Sub-tab 1: Joint Calibration
        joint_subtab = QWidget()
        joint_sublayout = QVBoxLayout()

        self.joint_mode_sel = QComboBox()
        self.update_joint_modes()
        self.joint_mode_sel.currentIndexChanged.connect(self.update_applied_offset_label)

        self.btn_joint_ready = QPushButton("MOVE TO READY")
        self.btn_joint_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_ready.clicked.connect(self.move_to_ready_pose_joint)

        self.btn_joint_start = QPushButton("START SWEEP")
        self.btn_joint_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_start.clicked.connect(self.start_calibration_joint)

        joint_sublayout.addWidget(QLabel("Joint Sweeps for Polarities & Kinematics:"))
        joint_sublayout.addWidget(self.joint_mode_sel)
        joint_sublayout.addWidget(self.btn_joint_ready)
        joint_sublayout.addWidget(self.btn_joint_start)
        joint_subtab.setLayout(joint_sublayout)

        # Sub-tab 2: Marker Bracket Calibration
        marker_subtab = QWidget()
        marker_sublayout = QVBoxLayout()

        self.lbl_marker_axis = QLabel("Marker Bracket Alignment Sweeps:")
        self.lbl_marker_axis.hide()
        self.marker_axis_sel = QComboBox()
        self.marker_axis_sel.addItems(UI_DROPDOWNS["marker_axes"])
        self.marker_axis_sel.hide()

        self.tolerance_input = QLineEdit("0.5")

        self.btn_marker_ready = QPushButton("MOVE TO READY")
        self.btn_marker_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_marker_ready.clicked.connect(self.move_to_ready_pose_marker)

        self.btn_marker_start = QPushButton("START SWEEP")
        self.btn_marker_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_marker_start.clicked.connect(self.start_calibration_marker)

        self.btn_marker_result = QPushButton("UNIFIED RESULT")
        self.btn_marker_result.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_marker_result.clicked.connect(self.show_unified_result_marker)

        marker_sublayout.addWidget(self.lbl_marker_axis)
        marker_sublayout.addWidget(self.marker_axis_sel)
        marker_sublayout.addWidget(self.btn_marker_ready)
        marker_sublayout.addWidget(self.btn_marker_start)
        marker_subtab.setLayout(marker_sublayout)

        # Sub-tab 3: Full Auto Calibration
        full_auto_subtab = QWidget()
        full_auto_sublayout = QVBoxLayout()

        self.btn_full_auto_ready = QPushButton("MOVE TO READY")
        self.btn_full_auto_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_full_auto_ready.clicked.connect(self.move_to_ready_full_auto)

        self.btn_full_auto_start = QPushButton("START FULL AUTO")
        self.btn_full_auto_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_full_auto_start.clicked.connect(self.start_full_auto)
        self.btn_step1_auto = QPushButton("STEP 1 AUTO")
        self.btn_step1_auto.clicked.connect(lambda: self.start_full_auto(step1_only=True))

        self.btn_full_auto_apply = QPushButton("APPLY FULL AUTO RESULTS")
        self.btn_full_auto_apply.setStyleSheet("background-color: #d35400; color: #ffffff; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_full_auto_apply.clicked.connect(self.apply_full_auto_results)
        self.btn_full_auto_apply.setEnabled(False) # Enabled after full auto finishes

        full_auto_sublayout.addWidget(QLabel("Full Auto Sequential Calibration:"))
        full_auto_sublayout.addWidget(self.btn_full_auto_ready)
        full_auto_sublayout.addWidget(self.btn_step1_auto)
        full_auto_sublayout.addWidget(self.btn_full_auto_start)
        full_auto_sublayout.addWidget(self.btn_full_auto_apply)
        full_auto_sublayout.addStretch()
        full_auto_subtab.setLayout(full_auto_sublayout)

        # Sub-tab 4: Head & Camera Extrinsics Calibration
        head_cam_subtab = QWidget()
        head_cam_sublayout = QVBoxLayout()
        head_cam_sublayout.setSpacing(6)

        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("Pan Range (±deg):"))
        self.step1_5_pan_range = QLineEdit("10.0")
        self.step1_5_pan_range.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        pan_row.addWidget(self.step1_5_pan_range)
        head_cam_sublayout.addLayout(pan_row)

        tilt_row = QHBoxLayout()
        tilt_row.addWidget(QLabel("Tilt Range (±deg):"))
        self.step1_5_tilt_range = QLineEdit("10.0")
        self.step1_5_tilt_range.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        tilt_row.addWidget(self.step1_5_tilt_range)
        head_cam_sublayout.addLayout(tilt_row)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Sweep Steps:"))
        self.step1_5_num_steps = QLineEdit("11")
        self.step1_5_num_steps.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        steps_row.addWidget(self.step1_5_num_steps)
        head_cam_sublayout.addLayout(steps_row)

        self.btn_step1_5_ready = QPushButton("1) MOVE TO READY")
        self.btn_step1_5_ready.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step1_5_ready.setFixedHeight(28)
        self.btn_step1_5_ready.clicked.connect(self.move_to_ready_pose_step1_5)
        head_cam_sublayout.addWidget(self.btn_step1_5_ready)

        self.btn_step1_5_start = QPushButton("2) START HEAD SWEEP")
        self.btn_step1_5_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step1_5_start.setFixedHeight(28)
        self.btn_step1_5_start.clicked.connect(self.start_calibration_step1_5)
        head_cam_sublayout.addWidget(self.btn_step1_5_start)

        self.btn_step1_5_apply = QPushButton("3) APPLY RESULTS")
        self.btn_step1_5_apply.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step1_5_apply.setFixedHeight(28)
        self.btn_step1_5_apply.setEnabled(False)
        self.btn_step1_5_apply.clicked.connect(self.apply_results_step1_5)
        head_cam_sublayout.addWidget(self.btn_step1_5_apply)

        head_cam_sublayout.addStretch()
        head_cam_subtab.setLayout(head_cam_sublayout)

        # Add workflow subtabs in order of: Full Auto, Joint Calib, Marker Calib, Head & Cam
        self.workflow_tabs.addTab(full_auto_subtab, "Auto")
        self.workflow_tabs.addTab(joint_subtab, "Joint")
        self.workflow_tabs.addTab(marker_subtab, "Marker")
        self.workflow_tabs.addTab(head_cam_subtab, "Head & Cam")
        self.workflow_tabs.currentChanged.connect(self._on_workflow_tab_changed)

        workflow_layout.addWidget(self.workflow_tabs)
        workflow_box.setLayout(workflow_layout)

        # Store shared boxes as instance attributes for reparenting
        self.conn_head_box = conn_head_box
        self.home_offset_box = None  # Will be set below after creation
        self.status_box = None  # Will be set below after creation
        self.log_box = None  # Will be set below after creation

        # Assemble Column 1
        col1_layout.addWidget(conn_head_box)
        col1_layout.addWidget(workflow_box, 1)

        # --- COLUMN 2 (Calibration Status & Monitoring) ---
        col2_layout = QVBoxLayout()

        # Standalone Robot Home Offset Reset Box
        home_offset_box = QGroupBox("Robot Home Offset")
        home_offset_box.setFixedHeight(160)
        home_offset_layout = QVBoxLayout()
        home_offset_layout.setSpacing(6)
        home_offset_layout.setContentsMargins(8, 8, 8, 8)

        desc_label = QLabel("Reset joint offsets to zero to restore factory alignment:")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #b0bec5; font-size: 11px;")
        home_offset_layout.addWidget(desc_label)

        btn_row = QHBoxLayout()
        self.btn_home_reset = QPushButton("Home Offset Reset")
        self.btn_home_reset.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_home_reset.clicked.connect(lambda: self.home_offset_reset(confirm_dialog=True))
        self.btn_home_reset.setFixedHeight(28)
        btn_row.addWidget(self.btn_home_reset)

        self.btn_step2_zero_pose = QPushButton("Zero Pose")
        self.btn_step2_zero_pose.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_zero_pose.setFixedHeight(28)
        self.btn_step2_zero_pose.clicked.connect(self.step2_zero_pose_check)
        btn_row.addWidget(self.btn_step2_zero_pose)

        home_offset_layout.addLayout(btn_row)

        hint_label = QLabel("Tip: Double-click any cell in the table below to manually stage individual offsets.")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        home_offset_layout.addWidget(hint_label)

        home_offset_box.setLayout(home_offset_layout)
        self.home_offset_box = home_offset_box

        dash_box = QGroupBox("Calibration Status & Monitoring")
        dash_layout = QVBoxLayout()
        dash_layout.setSpacing(4)
        dash_layout.setContentsMargins(8, 4, 8, 4)

        # Monitoring Table (Arm Joint Offsets)
        self.tbl_offset_monitor = QTableWidget(2, 3)
        self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Roll/Yaw 2)", "Joint 5 (Wrist Pitch)", "Joint 3 (Elbow)"])
        self.tbl_offset_monitor.setVerticalHeaderLabels(["Right Arm", "Left Arm"])
        self.tbl_offset_monitor.setFixedHeight(110)
        self.tbl_offset_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_offset_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.cellDoubleClicked.connect(self.on_cell_double_clicked)

        # Marker Bracket Design Offset UI GroupBox (Nested inside dash_box)
        bracket_box = QGroupBox("Marker Bracket Design Offset (Tf_to_marker)")
        bracket_layout = QVBoxLayout()
        bracket_layout.setSpacing(4)
        bracket_layout.setContentsMargins(6, 6, 6, 6)

        input_style = "background-color: #1c1c1c; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;"

        grid = QGridLayout()
        grid.setSpacing(6)

        # Column headers
        grid.addWidget(QLabel("Parameter"), 0, 0)
        grid.addWidget(QLabel("Left Arm"), 0, 1)
        grid.addWidget(QLabel("Right Arm"), 0, 2)

        # Row 1: X
        grid.addWidget(QLabel("X (m):"), 1, 0)
        self.txt_bracket_l_x = QLineEdit()
        self.txt_bracket_l_x.setStyleSheet(input_style)
        self.txt_bracket_r_x = QLineEdit()
        self.txt_bracket_r_x.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_x, 1, 1)
        grid.addWidget(self.txt_bracket_r_x, 1, 2)

        # Row 2: Y
        grid.addWidget(QLabel("Y (m):"), 2, 0)
        self.txt_bracket_l_y = QLineEdit()
        self.txt_bracket_l_y.setStyleSheet(input_style)
        self.txt_bracket_r_y = QLineEdit()
        self.txt_bracket_r_y.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_y, 2, 1)
        grid.addWidget(self.txt_bracket_r_y, 2, 2)

        # Row 3: Z
        grid.addWidget(QLabel("Z (m):"), 3, 0)
        self.txt_bracket_l_z = QLineEdit()
        self.txt_bracket_l_z.setStyleSheet(input_style)
        self.txt_bracket_r_z = QLineEdit()
        self.txt_bracket_r_z.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_z, 3, 1)
        grid.addWidget(self.txt_bracket_r_z, 3, 2)

        # Row 4: Roll
        grid.addWidget(QLabel("Roll (deg):"), 4, 0)
        self.txt_bracket_l_roll = QLineEdit()
        self.txt_bracket_l_roll.setStyleSheet(input_style)
        self.txt_bracket_r_roll = QLineEdit()
        self.txt_bracket_r_roll.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_roll, 4, 1)
        grid.addWidget(self.txt_bracket_r_roll, 4, 2)

        # Row 5: Pitch
        grid.addWidget(QLabel("Pitch (deg):"), 5, 0)
        self.txt_bracket_l_pitch = QLineEdit()
        self.txt_bracket_l_pitch.setStyleSheet(input_style)
        self.txt_bracket_r_pitch = QLineEdit()
        self.txt_bracket_r_pitch.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_pitch, 5, 1)
        grid.addWidget(self.txt_bracket_r_pitch, 5, 2)

        # Row 6: Yaw
        grid.addWidget(QLabel("Yaw (deg):"), 6, 0)
        self.txt_bracket_l_yaw = QLineEdit()
        self.txt_bracket_l_yaw.setStyleSheet(input_style)
        self.txt_bracket_r_yaw = QLineEdit()
        self.txt_bracket_r_yaw.setStyleSheet(input_style)
        grid.addWidget(self.txt_bracket_l_yaw, 6, 1)
        grid.addWidget(self.txt_bracket_r_yaw, 6, 2)

        bracket_layout.addLayout(grid)

        self.btn_apply_bracket = QPushButton("APPLY BRACKETS")
        self.btn_apply_bracket.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_apply_bracket.setFixedHeight(24)
        self.btn_apply_bracket.clicked.connect(self.apply_bracket_design_values)
        bracket_layout.addWidget(self.btn_apply_bracket)

        bracket_box.setLayout(bracket_layout)

        # Apply & Clear buttons for Joint offsets
        btn_joint_layout = QHBoxLayout()
        btn_joint_layout.setSpacing(6)

        self.btn_joint_apply = QPushButton("APPLY OFFSET")
        self.btn_joint_apply.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_apply.clicked.connect(self.apply_joint_offset)
        self.btn_joint_apply.setFixedHeight(24)

        self.btn_joint_clear = QPushButton("CLEAR OFFSET")
        self.btn_joint_clear.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_joint_clear.clicked.connect(self.clear_joint_offset)
        self.btn_joint_clear.setFixedHeight(24)

        btn_joint_layout.addWidget(self.btn_joint_apply)
        btn_joint_layout.addWidget(self.btn_joint_clear)

        # Page 0: Arm & Marker Monitoring View
        dash_page_arm_marker = QWidget()
        dash_am_layout = QVBoxLayout()
        dash_am_layout.setContentsMargins(0, 0, 0, 0)
        dash_am_layout.setSpacing(4)
        dash_am_layout.addWidget(self.tbl_offset_monitor)
        dash_am_layout.addWidget(bracket_box)
        dash_am_layout.addLayout(btn_joint_layout)
        dash_page_arm_marker.setLayout(dash_am_layout)

        # Page 1: Head & Camera Parameter Status View
        dash_page_head_cam = QWidget()
        dash_hc_layout = QVBoxLayout()
        dash_hc_layout.setContentsMargins(0, 0, 0, 0)
        dash_hc_layout.setSpacing(6)

        lbl_head_tbl = QLabel("Head Joint Offsets:")
        lbl_head_tbl.setStyleSheet("font-weight: bold; color: #2979ff;")
        dash_hc_layout.addWidget(lbl_head_tbl)

        self.tbl_step1_5_head_monitor = QTableWidget(2, 2)
        self.tbl_step1_5_head_monitor.setHorizontalHeaderLabels(["Nominal / Current", "Calibrated Offset"])
        self.tbl_step1_5_head_monitor.setVerticalHeaderLabels(["Head Pan (Joint 0)", "Head Tilt (Joint 1)"])
        self.tbl_step1_5_head_monitor.setFixedHeight(85)
        self.tbl_step1_5_head_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_step1_5_head_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_step1_5_head_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        dash_hc_layout.addWidget(self.tbl_step1_5_head_monitor)

        lbl_cam_tbl = QLabel("Camera Extrinsic Parameters (Mount to Cam):")
        lbl_cam_tbl.setStyleSheet("font-weight: bold; color: #00e676;")
        dash_hc_layout.addWidget(lbl_cam_tbl)

        self.tbl_step1_5_cam_monitor = QTableWidget(6, 2)
        self.tbl_step1_5_cam_monitor.setHorizontalHeaderLabels(["Nominal CAD", "Calibrated Value"])
        self.tbl_step1_5_cam_monitor.setVerticalHeaderLabels(["Roll (°)", "Pitch (°)", "Yaw (°)", "X (mm)", "Y (mm)", "Z (mm)"])
        self.tbl_step1_5_cam_monitor.setFixedHeight(180)
        self.tbl_step1_5_cam_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_step1_5_cam_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_step1_5_cam_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        dash_hc_layout.addWidget(self.tbl_step1_5_cam_monitor)

        step1_5_diag_box = QGroupBox("Fit Quality Diagnostics")
        step1_5_diag_box.setStyleSheet("QGroupBox { border: 1px solid #333333; border-radius: 4px; } QGroupBox::title { color: #b0bec5; font-size: 11px; }")
        diag_layout = QGridLayout()
        diag_layout.setSpacing(4)
        diag_layout.addWidget(QLabel("Tilt Plane RMSE:"), 0, 0)
        self.lbl_step1_5_rmse_tilt = QLabel("-")
        self.lbl_step1_5_rmse_tilt.setStyleSheet("color: #00e5ff; font-weight: bold;")
        diag_layout.addWidget(self.lbl_step1_5_rmse_tilt, 0, 1)

        diag_layout.addWidget(QLabel("Pan Plane RMSE:"), 0, 2)
        self.lbl_step1_5_rmse_pan = QLabel("-")
        self.lbl_step1_5_rmse_pan.setStyleSheet("color: #00e5ff; font-weight: bold;")
        diag_layout.addWidget(self.lbl_step1_5_rmse_pan, 0, 3)

        diag_layout.addWidget(QLabel("Axis Orthogonality Error:"), 1, 0)
        self.lbl_step1_5_ortho_err = QLabel("-")
        self.lbl_step1_5_ortho_err.setStyleSheet("color: #ffca28; font-weight: bold;")
        diag_layout.addWidget(self.lbl_step1_5_ortho_err, 1, 1)

        step1_5_diag_box.setLayout(diag_layout)
        dash_hc_layout.addWidget(step1_5_diag_box)
        dash_hc_layout.addStretch()

        dash_page_head_cam.setLayout(dash_hc_layout)

        # Stack to hold both pages
        self.dash_stack = QStackedWidget()
        self.dash_stack.addWidget(dash_page_arm_marker)
        self.dash_stack.addWidget(dash_page_head_cam)
        dash_layout.addWidget(self.dash_stack)
        dash_box.setLayout(dash_layout)

        col2_layout.addWidget(home_offset_box)
        col2_layout.addWidget(dash_box, 1)

        # --- COLUMN 3 (Camera Status & System Log/Plots) ---
        col3_layout = QVBoxLayout()

        # Status Indicator Box (Constructed here for Col 3)
        status_box = QGroupBox("Camera & Marker Status")
        status_box.setFixedHeight(160)
        status_layout = QVBoxLayout()
        status_layout.setSpacing(6)
        status_layout.setContentsMargins(8, 8, 8, 8)

        ind_layout = QHBoxLayout()
        self.indicator = IndicatorWidget()
        ind_layout.addWidget(self.indicator)
        self.status_label = QLabel("Not Detected")
        self.status_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.status_label.setStyleSheet("color: #ff1744;")
        ind_layout.addWidget(self.status_label)
        ind_layout.addStretch()
        status_layout.addLayout(ind_layout)

        self.temp_label = QLabel(self.get_temp_label_text())
        self.temp_label.setStyleSheet("color: #fb8c00; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.temp_label)

        btn_layout = QHBoxLayout()
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        self.btn_monitor.setFixedHeight(26)

        self.btn_camera_feed = QPushButton("Camera Feed")
        self.btn_camera_feed.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_camera_feed.clicked.connect(self.toggle_camera_feed_dialog)
        self.btn_camera_feed.setFixedHeight(26)

        btn_layout.addWidget(self.btn_monitor)
        btn_layout.addWidget(self.btn_camera_feed)
        status_layout.addLayout(btn_layout)

        status_box.setLayout(status_layout)
        self.status_box = status_box

        # System Log GroupBox
        log_box = QGroupBox("System Log & Control")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(6, 6, 6, 6)

        log_header = QHBoxLayout()
        console_title = QLabel("Execution Console Logs")
        console_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        console_title.setStyleSheet("color: #ffffff; margin-bottom: 2px;")

        self.btn_show_plot = QPushButton("Show Calibration Plot")
        self.btn_show_plot.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; font-size: 11px; border-radius: 4px; border: 1px solid #111111;")
        self.btn_show_plot.setFixedHeight(24)
        self.btn_show_plot.clicked.connect(self.open_plot_dialog)

        log_header.addWidget(console_title)
        log_header.addStretch()
        log_header.addWidget(self.btn_show_plot)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        font_log = QFont("Noto Sans Mono CJK KR", 10)
        font_log.setStyleHint(QFont.Monospace)
        font_log.setStyleStrategy(QFont.PreferAntialias | QFont.PreferQuality)
        self.log_text.setFont(font_log)

        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_text)
        log_box.setLayout(log_layout)
        self.log_box = log_box

        col3_layout.addWidget(status_box)
        col3_layout.addWidget(log_box, 1)

        # Assemble side-by-side 3 Columns (1:3:3 weight)
        main_tab_columns = QHBoxLayout()
        main_tab_columns.addLayout(col1_layout, 1)
        main_tab_columns.addLayout(col2_layout, 3)
        main_tab_columns.addLayout(col3_layout, 3)

        main_tab_layout.addLayout(main_tab_columns)


        main_tab.setLayout(main_tab_layout)

        # ==========================================
        # Camera Tab (카메라 내부 파라미터 보정 전용)
        # ==========================================
        camera_tab = QWidget()
        camera_tab_layout = QHBoxLayout()

        int_left = QVBoxLayout()
        self.video_label = QLabel("Camera Feed Loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 300)
        self.video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        int_left.addWidget(self.video_label, 3)

        instr_box = QGroupBox("Calibration Guidelines")
        instr_box.setStyleSheet("QGroupBox::title { color: #ff1744; font-weight: bold; }")
        instr_layout = QVBoxLayout()
        instructions = [
            "1. Ensure the calibration board is recognized correctly (green overlay).",
            "2. Tilt the board at various angles while capturing.",
            "3. Acquire data covering the entire camera field of view.",
            "4. Keep the board as steady as possible during each capture."
        ]
        for text in instructions:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #ff5252; font-weight: bold;")
            instr_layout.addWidget(lbl)
        instr_box.setLayout(instr_layout)
        int_left.addWidget(instr_box, 1)

        # Move Calibration Controls from left column to right column underneath Stats Box
        controls_box = QGroupBox("Calibration Controls")
        controls_layout = QVBoxLayout()

        self.chk_int_guide = QCheckBox("Show Guide Overlay")
        self.chk_int_guide.setChecked(True)
        self.chk_int_guide.setStyleSheet("color: #00e5ff; font-weight: bold;")
        self.chk_int_guide.stateChanged.connect(self.on_guide_changed)
        controls_layout.addWidget(self.chk_int_guide)

        self.btn_int_capture = QPushButton("CAPTURE FRAME (C)")
        self.btn_int_capture.setMinimumHeight(45)
        self.btn_int_capture.setStyleSheet("background-color: #1e88e5; color: white; font-size: 13px; font-weight: bold; border-radius: 4px;")
        self.btn_int_capture.clicked.connect(self.capture_intrinsics_frame)
        controls_layout.addWidget(self.btn_int_capture)

        self.btn_int_calibrate = QPushButton("RUN CALIBRATION")
        self.btn_int_calibrate.setMinimumHeight(45)
        self.btn_int_calibrate.setStyleSheet("background-color: #43a047; color: white; font-size: 13px; font-weight: bold; border-radius: 4px;")
        self.btn_int_calibrate.clicked.connect(self.run_intrinsics_calibration)
        controls_layout.addWidget(self.btn_int_calibrate)

        self.btn_int_save = QPushButton("SAVE PARAMETERS")
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #fb8c00; color: #000000; font-size: 13px; font-weight: bold; border-radius: 4px;")
        self.btn_int_save.clicked.connect(self.save_intrinsics_calibration)
        self.btn_int_save.setEnabled(False)
        controls_layout.addWidget(self.btn_int_save)

        self.btn_int_reset = QPushButton("RESET CAPTURES")
        self.btn_int_reset.setMinimumHeight(30)
        self.btn_int_reset.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 12px; border-radius: 4px;")
        self.btn_int_reset.clicked.connect(self.reset_intrinsics_captures)
        controls_layout.addWidget(self.btn_int_reset)

        self.int_busy = BusyIndicator("main_tabs.intrinsics_calculating")
        controls_layout.addWidget(self.int_busy)

        controls_box.setLayout(controls_layout)

        int_right = QVBoxLayout()

        stats_box2 = QGroupBox("Capture Stats")
        stats_layout2 = QHBoxLayout()
        self.lbl_captured = QLabel("Captured Frames: 0")
        self.lbl_captured.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #2979ff;")

        self.lbl_temp = QLabel(self.get_temp_label_text())
        self.lbl_temp.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #ff5500;")

        stats_layout2.addWidget(self.lbl_captured)
        stats_layout2.addStretch()
        stats_layout2.addWidget(self.lbl_temp)
        stats_box2.setLayout(stats_layout2)

        self.btn_reconnect_cam = QPushButton(tr("common.reconnect_camera"))
        self.btn_reconnect_cam.setMinimumHeight(34)
        self.btn_reconnect_cam.setStyleSheet("background-color: #00838f; color: white; font-size: 11px; font-weight: bold; border-radius: 4px;")
        self.btn_reconnect_cam.clicked.connect(self.reconnect_camera)

        # Exposure / Brightness Adjustment Box
        exposure_box = QGroupBox("Camera Exposure / Brightness")
        exposure_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 13px;}")
        exp_layout = QVBoxLayout()
        exp_layout.setSpacing(8)

        self.lbl_exposure_guide = QLabel(tr("wizard.slides.slide_exposure.brightness_guide"))
        self.lbl_exposure_guide.setStyleSheet("color: #dddddd; font-size: 12px; font-weight: bold;")
        self.lbl_exposure_guide.setWordWrap(True)
        exp_layout.addWidget(self.lbl_exposure_guide)

        init_auto = getattr(self, 'saved_camera_auto_exposure', True)
        init_exp = getattr(self, 'saved_camera_exposure_value', 6000)

        self.chk_auto_exposure = QCheckBox(tr("common.auto_exposure"))
        self.chk_auto_exposure.setChecked(init_auto)
        self.chk_auto_exposure.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.chk_auto_exposure.toggled.connect(self.on_auto_exposure_toggled)
        exp_layout.addWidget(self.chk_auto_exposure)

        exp_val_row = QHBoxLayout()
        # No unit in the label: the exposure option's unit is the colour sensor's own and is not
        # the same across models (a D405 takes microseconds, a D435's RGB camera does not), so the
        # number here only means something next to the range the camera itself reports. The
        # measured time in ms comes from frame metadata instead, which is comparable.
        exp_val_row.addWidget(QLabel("Exposure:"))
        self.spin_exposure = QSpinBox()
        # Widened from (100, 100000): the old floor of 100 was above what a D435's RGB camera
        # needs, so short exposures were unreachable on it. sync_exposure_limits_from_camera()
        # narrows this to what the connected camera actually accepts.
        self.spin_exposure.setRange(1, 200000)
        self.spin_exposure.setSingleStep(500)
        self.spin_exposure.setValue(init_exp)
        self.spin_exposure.setEnabled(not init_auto)
        self.spin_exposure.valueChanged.connect(self.on_exposure_value_changed)
        exp_val_row.addWidget(self.spin_exposure)

        self.lbl_exposure_ms = QLabel("")
        self.lbl_exposure_ms.setStyleSheet("color: #ffd700; font-weight: bold; min-width: 90px;")
        exp_val_row.addWidget(self.lbl_exposure_ms)
        exp_layout.addLayout(exp_val_row)

        self.slider_exposure = QSlider(Qt.Horizontal)
        self.slider_exposure.setRange(1, 200000)
        self.slider_exposure.setSingleStep(500)
        self.slider_exposure.setPageStep(5000)
        self.slider_exposure.setValue(init_exp)
        self.slider_exposure.setEnabled(not init_auto)
        self.slider_exposure.valueChanged.connect(self.on_exposure_value_changed)
        exp_layout.addWidget(self.slider_exposure)

        self.lbl_exposure_range = QLabel("")
        self.lbl_exposure_range.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        exp_layout.addWidget(self.lbl_exposure_range)

        # Action Buttons: APPLY and CANCEL
        btn_row1 = QHBoxLayout()
        self.btn_apply_exp = QPushButton(tr("common.apply").upper())
        self.btn_apply_exp.setMinimumHeight(35)
        self.btn_apply_exp.setStyleSheet("background-color: #388e3c; color: white; font-size: 12px; font-weight: bold; border-radius: 4px;")
        self.btn_apply_exp.clicked.connect(self.apply_camera_exposure)
        btn_row1.addWidget(self.btn_apply_exp)

        self.btn_cancel_exp = QPushButton(tr("common.cancel").upper())
        self.btn_cancel_exp.setMinimumHeight(35)
        self.btn_cancel_exp.setStyleSheet("background-color: #546e7a; color: white; font-size: 12px; font-weight: bold; border-radius: 4px;")
        self.btn_cancel_exp.clicked.connect(self.cancel_camera_exposure)
        btn_row1.addWidget(self.btn_cancel_exp)
        exp_layout.addLayout(btn_row1)

        # The exposure is kept in memory only (setting.yaml is not written).
        self.lbl_exposure_not_saved = QLabel(tr("wizard.slides.slide_exposure.not_saved_warn"))
        self.lbl_exposure_not_saved.setStyleSheet("color: #ffb74d; font-size: 12px; font-weight: bold;")
        self.lbl_exposure_not_saved.setWordWrap(True)
        exp_layout.addWidget(self.lbl_exposure_not_saved)

        exposure_box.setLayout(exp_layout)

        int_right.addWidget(stats_box2)
        int_right.addWidget(controls_box)
        int_right.addStretch()

        camera_tab_layout.addLayout(int_left, 2)
        camera_tab_layout.addLayout(int_right, 1)
        camera_tab.setLayout(camera_tab_layout)

        # Brightness sub-tab: its own live view (a widget can sit in one tab only) + exposure box.
        exposure_tab = QWidget()
        exposure_tab_layout = QHBoxLayout(exposure_tab)
        self.video_label_exposure = QLabel("Camera Feed Loading...")
        self.video_label_exposure.setAlignment(Qt.AlignCenter)
        self.video_label_exposure.setMinimumSize(480, 300)
        self.video_label_exposure.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        # Same marker recognition/jitter panel as the wizard exposure slide.
        exp_left = QVBoxLayout()
        exp_left.addWidget(self.video_label_exposure, 1)
        self.marker_monitor_panel = MarkerMonitorPanel()
        self.marker_monitor_panel.restart_requested.connect(self.sync_marker_monitor)
        exp_left.addWidget(self.marker_monitor_panel)
        exposure_tab_layout.addLayout(exp_left, 2)
        exp_right = QVBoxLayout()
        exp_right.addWidget(exposure_box)
        exp_right.addStretch()
        exposure_tab_layout.addLayout(exp_right, 1)

        # Top-level Camera tab (2026-09-30): connected camera line + reconnect, then the two sub-tabs.
        camera_top_tab = QWidget()
        camera_top_layout = QVBoxLayout(camera_top_tab)
        camera_top_layout.setContentsMargins(6, 6, 6, 6)
        cam_header = QHBoxLayout()
        self.lbl_camera_tab_info = QLabel("")
        self.lbl_camera_tab_info.setStyleSheet("color: #dddddd; font-weight: bold; font-size: 13px;")
        cam_header.addWidget(self.lbl_camera_tab_info)
        self.lbl_camera_serial_status = QLabel("")
        self.lbl_camera_serial_status.setWordWrap(True)
        cam_header.addWidget(self.lbl_camera_serial_status, 1)
        cam_header.addWidget(self.btn_reconnect_cam)
        camera_top_layout.addLayout(cam_header)
        self.camera_tabs = QTabWidget()
        self.camera_tabs.addTab(camera_tab, "Intrinsics Calibration")
        self.camera_tabs.addTab(exposure_tab, "Brightness")
        self.camera_tabs.currentChanged.connect(lambda _index: self.on_left_tab_changed(self.left_tabs.currentIndex()))
        camera_top_layout.addWidget(self.camera_tabs, 1)

        # ==========================================
        # Overview Tab
        # ==========================================
        overview_tab = QWidget()
        overview_layout = QVBoxLayout()
        # Tight margins: the wizard slides must fit a maximized 1536x960 (1920x1200 at 125 %) window.
        overview_layout.setContentsMargins(10, 6, 10, 6)

        self.overview_container = QWidget()
        container_layout = QVBoxLayout(self.overview_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)  # Top Stretch to push contents down to center

        self.overview_title = QLabel("Calibration Process Overview")
        self.overview_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        self.overview_title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.overview_title)

        self.overview_link = QLabel('GitHub Repository: <a href="https://github.com/RainbowRobotics/rby1-calibration" style="color: #00e5ff; font-weight: bold;">https://github.com/RainbowRobotics/rby1-calibration</a>')
        self.overview_link.setStyleSheet("font-size: 15px;")
        self.overview_link.setAlignment(Qt.AlignCenter)
        self.overview_link.setOpenExternalLinks(True)
        container_layout.addWidget(self.overview_link)

        self.overview_duration = QLabel("Estimated Execution Time: ~20 minutes")
        self.overview_duration.setStyleSheet("font-size: 16px; font-weight: bold; color: #00e5ff;")
        self.overview_duration.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.overview_duration)

        container_layout.addSpacing(25)  # Add spacing before the start button

        self.btn_start_wizard = QPushButton(tr("wizard.btn_start_wizard"))
        self.btn_start_wizard.setStyleSheet("background-color: #d84315; color: white; font-weight: bold; font-size: 18px; padding: 10px;")
        self.btn_start_wizard.setFixedWidth(200)
        self.btn_start_wizard.clicked.connect(self.show_wizard_ui)
        container_layout.addWidget(self.btn_start_wizard, alignment=Qt.AlignCenter)

        container_layout.addStretch(1)  # Bottom Stretch to push contents up to center

        self.overview_img = None
        overview_layout.addWidget(self.overview_container)

        self.wizard_widget = CalibrationWizardWidget(self)
        self.wizard_widget.setVisible(False)
        overview_layout.addWidget(self.wizard_widget, stretch=1)

        overview_tab.setLayout(overview_layout)

        # ==========================================
        # Top-level tabs (2026-09-30): Wizard / Camera / Advanced (Step 1 + Step 2 sub-tabs)
        # ==========================================
        # 2026-11-17: the Advanced (840) and Camera tabs set the window's minimum height; on a screen lower
        # than that, Linux window managers refuse to maximize and only the height fits. They scroll instead.
        def scrollable(page):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            return scroll

        self.left_tabs.addTab(overview_tab, "Wizard")
        self.left_tabs.addTab(scrollable(camera_top_tab), "Camera")
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_tabs = QTabWidget()
        self.advanced_tabs.addTab(main_tab, "Step 1")
        advanced_layout.addWidget(self.advanced_tabs)
        self.left_tabs.addTab(scrollable(advanced_tab), "Advanced")

        # ==========================================
        # Step 2 Tab: Shared widgets + empty Box1
        # ==========================================
        step2_tab = QWidget()
        step2_layout = QVBoxLayout()
        step2_layout.setContentsMargins(5, 5, 5, 5)

        # Step 2 Top Row for shared widgets (conn_head_box and home_offset_box)
        self.step2_top_row = QHBoxLayout()
        self.step2_top_row.setSpacing(10)
        step2_layout.addLayout(self.step2_top_row)

        # Step 2 columns: Left (Config + Actions), Right (status + log)
        step2_columns = QHBoxLayout()

        # Step 2 Left Column
        self.step2_left_col = QVBoxLayout()
        # Placeholders for reparented widgets — Config and Actions boxes go here

        # Config Box (replaces Box1 — mirrors calibration_ui Config section)
        config_box = QGroupBox("Config")
        config_box.setStyleSheet("QGroupBox { border: 1px solid #555; border-radius: 4px; } QGroupBox::title { color: #2979ff; font-weight: bold; }")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(4)
        config_layout.setContentsMargins(6, 6, 6, 6)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.step2_mode_sel = QComboBox()
        self.step2_mode_sel.addItems(["live", "npz", "sim"])
        self.step2_mode_sel.setStyleSheet("background-color: #2a2a2a; color: white;")
        if self.ui_only or self.robot is None:
            self.step2_mode_sel.setCurrentText("sim")
        mode_row.addWidget(self.step2_mode_sel)
        config_layout.addLayout(mode_row)

        # Path input
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self.step2_path_input = QLineEdit("result/result_step2/dataset_YYYYMMDD_HHMMSS.npz")
        self.step2_path_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        path_row.addWidget(self.step2_path_input)
        config_layout.addLayout(path_row)

        # Estimated samples label
        self.step2_est_samples_label = QLabel("Est. Samples: 0")
        self.step2_est_samples_label.setStyleSheet("color: #2979ff; font-weight: bold; font-size: 11px;")
        config_layout.addWidget(self.step2_est_samples_label)

        # Auto Motion Step parameters
        auto_step_row = QHBoxLayout()
        auto_step_row.addWidget(QLabel("Angle(deg):"))
        self.step2_angle_step = QLineEdit("5.0")
        self.step2_angle_step.setFixedWidth(45)
        self.step2_angle_step.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row.addWidget(self.step2_angle_step)

        auto_step_row.addWidget(QLabel("Pos(m):"))
        self.step2_pos_step = QLineEdit("0.03")
        self.step2_pos_step.setFixedWidth(45)
        self.step2_pos_step.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row.addWidget(self.step2_pos_step)
        config_layout.addLayout(auto_step_row)

        auto_step_row2 = QHBoxLayout()
        auto_step_row2.addWidget(QLabel("Step(m):"))
        self.step2_step_x = QLineEdit("0.03")
        self.step2_step_x.setFixedWidth(45)
        self.step2_step_x.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row2.addWidget(self.step2_step_x)

        auto_step_row2.addWidget(QLabel("Max X(m):"))
        self.step2_max_x = QLineEdit("0.4")
        self.step2_max_x.setFixedWidth(45)
        self.step2_max_x.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        auto_step_row2.addWidget(self.step2_max_x)
        config_layout.addLayout(auto_step_row2)

        # Head status label
        self.step2_head_status_label = QLabel("Auto Motion: 0/0")
        self.step2_head_status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        config_layout.addWidget(self.step2_head_status_label)

        # Apply Joint Offset checkbox (instead of full joint offset box)
        jo_row = QHBoxLayout()
        self.chk_apply_joint_offset = QCheckBox("Apply Joint Offset")
        self.chk_apply_joint_offset.setChecked(True)
        self.chk_apply_joint_offset.setStyleSheet("color: #cccccc; font-weight: bold;")
        self.chk_apply_joint_offset.toggled.connect(self._on_apply_joint_offset_toggled)
        jo_row.addWidget(self.chk_apply_joint_offset)

        self.lbl_jo_status = QLabel("ACTIVE")
        self.lbl_jo_status.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px;")
        jo_row.addWidget(self.lbl_jo_status)
        jo_row.addStretch()
        config_layout.addLayout(jo_row)

        config_layout.addStretch()
        config_box.setLayout(config_layout)
        self.step2_left_col.addWidget(config_box, 1)

        self.step2_angle_step.textChanged.connect(self.update_step2_est_samples)
        self.step2_pos_step.textChanged.connect(self.update_step2_est_samples)
        self.step2_step_x.textChanged.connect(self.update_step2_est_samples)
        self.step2_max_x.textChanged.connect(self.update_step2_est_samples)
        self.step2_mode_sel.currentTextChanged.connect(self.on_step2_mode_changed)
        QTimer.singleShot(100, self.update_step2_est_samples)

        # Actions Box (mirrors calibration_ui Actions section)
        actions_box = QGroupBox("Actions")
        actions_box.setStyleSheet("QGroupBox { border: 1px solid #333333; border-radius: 4px; } QGroupBox::title { color: #ffffff; font-weight: bold; }")
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(4)
        actions_layout.setContentsMargins(6, 6, 6, 6)

        # Top row: Stop (Zero Pose is located in Calibration Status above)
        top_action_row = QHBoxLayout()
        self.btn_step2_stop = QPushButton("Stop")
        self.btn_step2_stop.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_stop.setFixedHeight(28)
        self.btn_step2_stop.clicked.connect(self.stop_motion)
        top_action_row.addWidget(self.btn_step2_stop)
        actions_layout.addLayout(top_action_row)

        # Numbered actions row 1
        act_row1 = QHBoxLayout()
        self.btn_step2_init_pose = QPushButton("1) Init Pose")
        self.btn_step2_init_pose.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_init_pose.setFixedHeight(28)
        self.btn_step2_init_pose.clicked.connect(self.step2_init_pose)
        act_row1.addWidget(self.btn_step2_init_pose)

        self.btn_step2_auto_motion = QPushButton("2) Auto Motion")
        self.btn_step2_auto_motion.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_auto_motion.setFixedHeight(28)
        self.btn_step2_auto_motion.clicked.connect(self.step2_auto_motion)
        act_row1.addWidget(self.btn_step2_auto_motion)
        actions_layout.addLayout(act_row1)

        # Numbered actions row 2
        act_row2 = QHBoxLayout()
        self.btn_step2_calculate = QPushButton("3) Calculate")
        self.btn_step2_calculate.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_calculate.setFixedHeight(28)
        self.btn_step2_calculate.clicked.connect(self.step2_calculate)
        act_row2.addWidget(self.btn_step2_calculate)

        self.btn_step2_clear = QPushButton("4) Clear Samples")
        self.btn_step2_clear.setStyleSheet("background-color: #34495e; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_clear.setFixedHeight(28)
        self.btn_step2_clear.clicked.connect(self.step2_clear_samples)
        act_row2.addWidget(self.btn_step2_clear)
        actions_layout.addLayout(act_row2)

        # Numbered actions row 3
        act_row3 = QHBoxLayout()
        self.btn_step2_apply_home = QPushButton("5) Apply Home Offset")
        self.btn_step2_apply_home.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_apply_home.setFixedHeight(28)
        self.btn_step2_apply_home.clicked.connect(self.step2_apply_home_offset)
        act_row3.addWidget(self.btn_step2_apply_home)

        self.btn_step2_check_state = QPushButton("6) Check Calibration State")
        self.btn_step2_check_state.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #111111;")
        self.btn_step2_check_state.setFixedHeight(28)
        self.btn_step2_check_state.clicked.connect(self.step2_check_calibration_state)
        act_row3.addWidget(self.btn_step2_check_state)
        actions_layout.addLayout(act_row3)

        # Sample count label
        self.step2_sample_count_label = QLabel("Shared Samples: 0")
        self.step2_sample_count_label.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 11px;")
        actions_layout.addWidget(self.step2_sample_count_label)

        actions_layout.addStretch()
        actions_box.setLayout(actions_layout)
        self.step2_left_col.addWidget(actions_box, 1)

        # Step 2 Right Column
        self.step2_right_col = QVBoxLayout()
        # status_box and log_box (without plot button) go here

        step2_columns.addLayout(self.step2_left_col, 1)
        step2_columns.addLayout(self.step2_right_col, 1)

        step2_layout.addLayout(step2_columns)


        step2_tab.setLayout(step2_layout)

        self.advanced_tabs.addTab(step2_tab, "Step 2")
        self.advanced_tabs.currentChanged.connect(lambda _index: self.on_left_tab_changed(self.left_tabs.currentIndex()))

        # Keep references for reparenting logic
        # Step 1 Main tab column layout indices for reinserting shared widgets
        self._step1_col1 = col1_layout
        self._step1_col2 = col2_layout
        self._step1_col3 = col3_layout

        # Assemble full-width tabs
        main_layout.addWidget(self.left_tabs)

        # Create outer vertical layout with Top Language Header
        outer_layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(5, 5, 5, 5)

        self.lbl_lang_select = QLabel(tr("common.language") + ":")
        self.lbl_lang_select.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffeb3b;")

        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["English", "한국어"])
        self.combo_lang.setCurrentIndex(0 if LanguageManager.instance().current_lang == "en" else 1)
        self.combo_lang.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px 12px; background-color: #2b2b2b; color: #ffffff; border-radius: 4px; min-width: 140px;")
        self.combo_lang.currentIndexChanged.connect(self.on_language_combo_changed)

        top_bar.addWidget(self.lbl_lang_select)
        top_bar.addWidget(self.combo_lang)
        top_bar.addStretch()

        outer_layout.addLayout(top_bar)
        outer_layout.addLayout(main_layout)

        self.btn_quit_global = QPushButton("QUIT")
        self.btn_quit_global.setMinimumHeight(40)
        self.btn_quit_global.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold; font-size: 14px; border-radius: 6px; border: 1px solid #111111;")
        self.btn_quit_global.clicked.connect(self.close)
        outer_layout.addWidget(self.btn_quit_global)

        self.setLayout(outer_layout)

        # Initial UI translation sync
        self.retranslate_ui()

        # Startup info
        self.log_msg("="*60)
        self.log_msg("  UNIFIED ROBOT CALIBRATION SUITE LOADED")
        self.log_msg("="*60)
        self.log_msg("[RECOMMENDED SEQUENCE]")
        self.log_msg("  1. Calibrate camera intrinsics first if needed (Step 1 > Camera tab).")
        self.log_msg("  2. Calibrate joint offsets using Joint subtab.")
        self.log_msg("  3. Perform marker bracket sweeps using Marker subtab.")
        self.log_msg("  4. Control head and verify offsets as a final check.")
        self.log_msg("="*60)

    def is_temp_supported(self):
        if self.ui_only or self.core.observer is None:
            return False
        return self.core.get_monitor_snapshot()["temperature"] is not None

    def get_temp_label_text(self, temp_val=None):
        if not self.is_temp_supported():
            return tr("common.camera_temp_not_supported")
        else:
            if temp_val is not None:
                return tr("common.camera_temp_val", temp=f"{temp_val:.1f}")
            else:
                return tr("common.camera_temp_unknown")

    def on_language_combo_changed(self, index):
        lang_code = "en" if index == 0 else "ko"
        LanguageManager.instance().set_language(lang_code)
        self.retranslate_ui()

    def retranslate_ui(self):
        if hasattr(self, 'lbl_lang_select'):
            self.lbl_lang_select.setText(tr("common.language") + ":")
        if hasattr(self, 'left_tabs'):
            self.left_tabs.setTabText(self.TAB_WIZARD, tr("main_tabs.tab_overview"))
            self.left_tabs.setTabText(self.TAB_CAMERA, tr("main_tabs.tab_camera"))
            self.left_tabs.setTabText(self.TAB_ADVANCED, tr("main_tabs.tab_advanced"))
        if hasattr(self, 'camera_tabs'):
            self.camera_tabs.setTabText(self.CAMERA_SUB_INTRINSICS, tr("main_tabs.camera_intrinsics"))
            self.camera_tabs.setTabText(self.CAMERA_SUB_EXPOSURE, tr("main_tabs.camera_exposure"))
        if hasattr(self, 'marker_monitor_panel'):
            self.marker_monitor_panel.retranslate()
        if hasattr(self, 'int_busy'):
            self.int_busy.retranslate()
        if hasattr(self, 'lbl_exposure_not_saved'):
            self.lbl_exposure_not_saved.setText(tr("wizard.slides.slide_exposure.not_saved_warn"))
        if hasattr(self, 'lbl_exposure_guide'):
            self.lbl_exposure_guide.setText(tr("wizard.slides.slide_exposure.brightness_guide"))
        if hasattr(self, 'advanced_tabs'):
            self.advanced_tabs.setTabText(self.ADV_SUB_STEP1, tr("main_tabs.tab_step1"))
            self.advanced_tabs.setTabText(self.ADV_SUB_STEP2, tr("main_tabs.tab_step2"))
        self.refresh_camera_tab_info()
        if hasattr(self, 'btn_start_wizard'):
            self.btn_start_wizard.setText(tr("wizard.btn_start_wizard"))
        if hasattr(self, 'btn_reconnect_cam'):
            self.btn_reconnect_cam.setText(tr("common.reconnect_camera"))
        if hasattr(self, 'chk_auto_exposure'):
            self.chk_auto_exposure.setText(tr("common.auto_exposure"))
        if hasattr(self, 'btn_apply_exp'):
            self.btn_apply_exp.setText(tr("common.apply").upper())
        if hasattr(self, 'btn_cancel_exp'):
            self.btn_cancel_exp.setText(tr("common.cancel").upper())

        # Translate temperature labels
        last_temp = getattr(self, 'last_temp', None)
        temp_text = self.get_temp_label_text(last_temp)
        if hasattr(self, 'temp_label') and self.temp_label is not None:
            self.temp_label.setText(temp_text)
        if hasattr(self, 'lbl_temp') and self.lbl_temp is not None:
            self.lbl_temp.setText(temp_text)
        if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
            if hasattr(self.wizard_widget, 'lbl_temp') and self.wizard_widget.lbl_temp is not None:
                self.wizard_widget.lbl_temp.setText(temp_text)

    @property
    def sim(self) -> bool:
        return bool(getattr(self.core.observer, 'sim', False))

    # --- Common Helper Functions ---
    def _log_msg_slot(self, msg):
        if self.save_gui_log:
            try:
                SessionLog.append(msg)
            except Exception as error:
                # Say it once and stop trying, instead of failing on every later line.
                self.save_gui_log = False
                msg = f"{msg}\n[WARN] The GUI log is no longer being saved to {CONFIG_PATHS['log_dir']}: {error}"
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(msg)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        else:
            print(msg)

    def _update_ui_slot(self, action):
        if action == "head_pose":
            self.update_head_pose_status()
        elif action == "samples":
            self.update_step2_est_samples()
        elif action == "sample_counts":
            self.update_sample_counts()

    def show_message_box(self, title, text, icon=QMessageBox.Information, buttons=QMessageBox.Ok):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStandardButtons(buttons)

        # Enforce minimum size on QMessageBox dialog
        msg_box.setMinimumSize(550, 220)

        # Enforce wrap-around and minimum label constraints
        label = msg_box.findChild(QLabel, "qt_msgbox_label")
        if label:
            label.setWordWrap(True)
            label.setMinimumHeight(100)
            label.setStyleSheet("QLabel { font-size: 13px; color: #ffffff; padding: 10px; line-height: 1.4; }")

        return msg_box.exec()

    def log_msg(self, msg):
        if QThread.currentThread() == QApplication.instance().thread():
            self._log_msg_slot(msg)
        else:
            self.log_signal_safe.emit(msg)

    def _write_step2_log(self, msg):
        log_file = os.path.join(CONFIG_PATHS["txt_dir"], "step2_capture_log.txt")
        try:
            FileStorage.ensure_dir(CONFIG_PATHS["txt_dir"], exist_ok=True)
            with FileStorage.open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            self.log_msg(f"Failed to write step2 log: {e}")

    def safe_cancel_control(self):
        self.core.robot_address = self.ip_input.text().strip()
        self.core.robot_model_name = self.model_input.currentText().strip().lower()
        self.core.cancel()

    @idle_core_action
    def on_head_checkbox_changed(self, checked):
        self.include_head_motion = checked
        if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
            self.joint_calibrator.include_head_motion = checked
        if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
            self.marker_calibrator.include_head_motion = checked
        if hasattr(self, 'head_camera_calibrator') and self.head_camera_calibrator:
            self.head_camera_calibrator.include_head_motion = checked

        if hasattr(self, 'btn_step1_5_ready'):
            self.btn_step1_5_ready.setEnabled(checked)
        if hasattr(self, 'btn_step1_5_start'):
            self.btn_step1_5_start.setEnabled(checked)

        if hasattr(self, 'chk_servo_head'):
            self.chk_servo_head.blockSignals(True)
            self.chk_servo_head.setChecked(checked)
            self.chk_servo_head.blockSignals(False)

        if hasattr(self, 'wizard_widget') and hasattr(self.wizard_widget, 'wizard_chk_head'):
            self.wizard_widget.wizard_chk_head.blockSignals(True)
            self.wizard_widget.wizard_chk_head.setChecked(checked)
            self.wizard_widget.wizard_chk_head.blockSignals(False)

        self.sync_connection_settings('main')
        if hasattr(self, 'wizard_widget') and self.wizard_widget:
            self.wizard_widget.sync_bracket_radio()
        self.log_msg(f"[INFO] Additional camera bracket: {'YES' if not checked else 'NO'} (Head motion: {'ENABLED' if checked else 'DISABLED'})")

    def update_robot_connection_status(self):
        is_connected = self.robot is not None
        if is_connected:
            self.btn_connect.setText("CONNECTED")
            self.btn_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(True)
            if hasattr(self, 'wizard_widget') and self.wizard_widget:
                self.wizard_widget.btn_wizard_connect.setText(tr("wizard.btn_connected"))
                self.wizard_widget.btn_wizard_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px;")
                self.wizard_widget.mark_step_completed(self.wizard_widget.SLIDE_ROBOT_CONNECT, True, ("wizard.st.connected", {}))
        else:
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(True)
            if hasattr(self, 'wizard_widget') and self.wizard_widget:
                self.wizard_widget.btn_wizard_connect.setText(tr("wizard.btn_connect"))
                self.wizard_widget.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
                self.wizard_widget.mark_step_completed(self.wizard_widget.SLIDE_ROBOT_CONNECT, False, ("wizard.st.disconnected", {}))

    def sync_connection_settings(self, source='main'):
        if not hasattr(self, 'wizard_widget') or not self.wizard_widget:
            return
        if source == 'main':
            if hasattr(self, 'ip_input') and hasattr(self.wizard_widget, 'wizard_ip_input'):
                self.wizard_widget.wizard_ip_input.blockSignals(True)
                self.wizard_widget.wizard_ip_input.setText(self.ip_input.text().strip())
                self.wizard_widget.wizard_ip_input.blockSignals(False)
            if hasattr(self, 'chk_servo_head') and hasattr(self.wizard_widget, 'wizard_chk_head'):
                self.wizard_widget.wizard_chk_head.blockSignals(True)
                self.wizard_widget.wizard_chk_head.setChecked(self.chk_servo_head.isChecked())
                self.wizard_widget.wizard_chk_head.blockSignals(False)
        elif source == 'wizard':
            if hasattr(self, 'ip_input') and hasattr(self.wizard_widget, 'wizard_ip_input'):
                self.ip_input.blockSignals(True)
                self.ip_input.setText(self.wizard_widget.wizard_ip_input.text().strip())
                self.ip_input.blockSignals(False)
            if hasattr(self, 'chk_servo_head') and hasattr(self.wizard_widget, 'wizard_chk_head'):
                self.chk_servo_head.blockSignals(True)
                self.chk_servo_head.setChecked(self.wizard_widget.wizard_chk_head.isChecked())
                self.chk_servo_head.blockSignals(False)

    def clear_user_taught_ready_poses(self, arm_side=None, mode=None):
        if hasattr(self, 'user_taught_ready_poses') and isinstance(self.user_taught_ready_poses, dict):
            if arm_side and mode:
                norm_mode = "wrist_pitch" if mode == "wrist_pitch_v13" else ("wrist_roll" if mode == "wrist_roll_v13" else mode)
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].pop(norm_mode, None)
            elif arm_side:
                if arm_side in self.user_taught_ready_poses and isinstance(self.user_taught_ready_poses[arm_side], dict):
                    self.user_taught_ready_poses[arm_side].clear()
            else:
                self.user_taught_ready_poses.clear()
        if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
            self.marker_calibrator.clear_user_taught_ready_poses(arm_side, mode)
        if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
            self.joint_calibrator.clear_user_taught_ready_poses(arm_side, mode)

    def _on_marker_problem_requested(self, arm_side, evt, res):
        # Stop pressed after the worker asked (the signal is queued): no readjustment window for a stopped run.
        if self.core.stop_event.is_set():
            res['resolved'] = False
            evt.set()
            return
        # Operators did not understand the teaching window: say first which arm lost its marker.
        arm_label = tr(f"dialogs.marker_problem.arm_{arm_side}") if arm_side in ("right", "left") else str(arm_side)
        self.show_message_box(tr("dialogs.marker_problem.notice_title"),
                              tr("dialogs.marker_problem.notice_msg", arm=arm_label), QMessageBox.Warning)
        dlg = MarkerRecognitionProblemDialog(self, arm_label=arm_label)
        self.marker_problem_dlg = dlg
        self.on_left_tab_changed(self.left_tabs.currentIndex())

        while True:
            resolved = (dlg.exec() == QDialog.Accepted)
            if not resolved:
                res['resolved'] = False
                break

            if self.robot:
                try:
                    state = self.robot.get_state()
                    model = self.robot.model()
                    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                    taught_pose = list(state.position[arm_idx])
                    active_mode = getattr(self, 'active_teaching_mode', None)
                    if not active_mode:
                        if hasattr(self, 'marker_calibrator') and self.marker_calibrator and getattr(self.marker_calibrator, 'current_calib_mode', None):
                            active_mode = self.marker_calibrator.current_calib_mode
                        elif hasattr(self, 'joint_calibrator') and self.joint_calibrator and getattr(self.joint_calibrator, 'current_calib_mode', None):
                            active_mode = self.joint_calibrator.current_calib_mode
                        elif getattr(self, 'current_calib_mode', None):
                            active_mode = self.current_calib_mode
                        else:
                            active_mode = "marker"
                    norm_mode = "wrist_pitch" if active_mode == "wrist_pitch_v13" else ("wrist_roll" if active_mode == "wrist_roll_v13" else active_mode)

                    version_key = "v1.3" if (hasattr(self, 'marker_calibrator') and self.marker_calibrator and self.marker_calibrator.is_v13()) else "v1.2"
                    if norm_mode == "marker":
                        type_key = "marker"
                        ready_mode = None
                    else:
                        type_key = "joint"
                        if norm_mode == "wrist_pitch":
                            ready_mode = "wrist_pitch"
                        elif norm_mode in ("wrist_roll", "wrist_yaw2"):
                            ready_mode = "wrist_roll_v13" if version_key == "v1.3" else "wrist_yaw2"
                        elif norm_mode == "elbow":
                            ready_mode = "elbow"
                        else:
                            ready_mode = "wrist_pitch"

                    if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
                        try:
                            nom_pose = self.marker_calibrator.get_ready_pose(version_key, type_key, ready_mode, arm_side)
                            enforce_nominal_taught_joints(taught_pose, norm_mode, nom_pose)
                        except Exception as e:
                            self.log_msg(f"[WARN] Could not enforce nominal target angle on taught pose: {e}")

                    # Check if the joint values are valid within the robot's operating range
                    dyn_model = self.robot.get_dynamics()
                    state_lim = dyn_model.make_state([f"ee_{arm_side}"], model.robot_joint_names)
                    q_lower_all = np.array(dyn_model.get_limit_q_lower(state_lim))
                    q_upper_all = np.array(dyn_model.get_limit_q_upper(state_lim))

                    invalid_joints = []
                    for i in range(7):
                        j_val = taught_pose[i]
                        g_idx = arm_idx[i]
                        low_lim = q_lower_all[g_idx]
                        upp_lim = q_upper_all[g_idx]
                        if j_val < low_lim or j_val > upp_lim:
                            invalid_joints.append((i, np.degrees(j_val), np.degrees(low_lim), np.degrees(upp_lim)))

                    if len(invalid_joints) > 0:
                        err_lines = []
                        for i, val, low, upp in invalid_joints:
                            err_lines.append(tr("dialogs.joint_limit_exceeded.joint_line",
                                                joint=i+1, val=f"{val:.2f}", low=f"{low:.1f}", upp=f"{upp:.1f}"))

                        details = "\n".join(err_lines)
                        title = tr("dialogs.joint_limit_exceeded.title")
                        msg = tr("dialogs.joint_limit_exceeded.msg", details=details)

                        self.show_message_box(title, msg, QMessageBox.Warning)
                        continue  # Keep the teaching guide dialog open for readjustment

                    res['resolved'] = True
                    if not hasattr(self, 'user_taught_ready_poses') or not isinstance(self.user_taught_ready_poses, dict):
                        self.user_taught_ready_poses = {}
                    if arm_side not in self.user_taught_ready_poses or not isinstance(self.user_taught_ready_poses[arm_side], dict):
                        self.user_taught_ready_poses[arm_side] = {}

                    self.user_taught_ready_poses[arm_side][norm_mode] = taught_pose

                    if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
                        self.marker_calibrator.user_taught_ready_poses = self.user_taught_ready_poses

                    if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
                        self.joint_calibrator.user_taught_ready_poses = self.user_taught_ready_poses

                    self.log_msg(f"[INFO] Preserved user-taught ready pose for {arm_side} arm ({norm_mode}).")
                    break
                except Exception as e:
                    self.log_msg(f"[WARN] Failed to preserve user-taught ready pose: {e}")
                    res['resolved'] = True
                    break
            else:
                res['resolved'] = True
                break

        self.marker_problem_dlg = None
        self.on_left_tab_changed(self.left_tabs.currentIndex())
        evt.set()

    def prompt_marker_problem_teaching(self, arm_side, active_mode=None, **kwargs):
        if not active_mode and 'mode' in kwargs:
            active_mode = kwargs['mode']
        # 2026-11-17 (user): a sweep cut short by Stop loses its marker; that is not a posture problem.
        if self.core.stop_event.is_set():
            self.log_msg("[INFO] Stopped by the operator: no posture readjustment window.")
            return False
        self.active_teaching_mode = active_mode
        evt = threading.Event()
        res = {'resolved': False}
        try:
            self.marker_problem_signal.emit(arm_side, evt, res)
            while not evt.wait(0.1):
                if self.core.stop_event.is_set():
                    return False
            return res['resolved']
        finally:
            self.active_teaching_mode = None

    def connect_robot(self):
        if self.core.is_busy:
            self.log_msg("Stop the current sequence before changing the connection.")
            return
        if self.robot:
            self.log_msg("[INFO] Disconnecting from robot...")
            self.core.disconnect_robot()
            self.robot = None
            self.model = None
            self.dyn_model = None
            self.robot_version = "1.2"
            self.marker_calibrator.robot = None
            self.marker_calibrator.robot_version = "1.2"
            self.joint_calibrator.robot = None
            self.joint_calibrator.robot_version = "1.2"
            self.head_camera_calibrator.robot = None
            self.head_camera_calibrator.robot_version = "1.2"
            if self.core.observer is not None:
                self.core.bind_robot(None, "1.2")
            self.update_joint_modes()
            self.load_offsets_from_yaml()
            self.update_applied_offset_label()
            self.update_robot_connection_status()
            if hasattr(self, 'chk_servo_head'):
                self.chk_servo_head.setEnabled(True)
            self.log_msg("[INFO] Robot disconnected.")
            return

        try:
            self.sync_connection_settings('main')
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip()

            # Read head checkbox state (like calibration_ui's servo_head)
            head_enabled = self.chk_servo_head.isChecked() if hasattr(self, 'chk_servo_head') else True
            self.include_head_motion = head_enabled
            if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
                        self.marker_calibrator.include_head_motion = head_enabled
            if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
                        self.joint_calibrator.include_head_motion = head_enabled

            # Update connection button to loading state
            self.btn_connect.setText("CONNECTING...")
            self.btn_connect.setStyleSheet("background-color: #ffb74d; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(False)
            QApplication.processEvents()

            robot = self.core.connect_robot(addr, model, include_head=head_enabled,
                                            simulated=self.ui_only or self.core.is_simulated)

            self.robot = robot

            if self.robot:
                self.model = self.robot.model()
                self.dyn_model = self.robot.get_dynamics()
                if self.model and hasattr(self.model, "robot_joint_names"):
                    self.sim_q_full = np.zeros(len(self.model.robot_joint_names))
                self.marker_calibrator.robot = self.robot
                self.joint_calibrator.robot = self.robot
                self.head_camera_calibrator.robot = self.robot

                # Determine version classification automatically
                detected_version = "1.2"
                try:
                    robot_info = self.robot.get_robot_info()
                    actual_model_name = robot_info.robot_model_name
                    if actual_model_name.lower() != model.lower():
                        self.log_msg(f"[INFO] Auto-updating UI model selection to match robot model: '{actual_model_name}'")
                        found = False
                        for i in range(self.model_input.count()):
                            if self.model_input.itemText(i).lower() == actual_model_name.lower():
                                self.model_input.blockSignals(True)
                                self.model_input.setCurrentIndex(i)
                                self.model_input.blockSignals(False)
                                found = True
                                break
                        if not found:
                            self.model_input.blockSignals(True)
                            self.model_input.addItem(actual_model_name)
                            self.model_input.setCurrentIndex(self.model_input.count() - 1)
                            self.model_input.blockSignals(False)

                    raw_version = robot_info.robot_model_version
                    self.log_msg(f"[INFO] Connected robot model version string: '{raw_version}'")
                    print(f"[INFO] Connected robot model version string: '{raw_version}'")

                    if "1.3" in raw_version:
                        detected_version = "1.3"
                    else:
                        detected_version = "1.2"
                except Exception as e:
                    self.log_msg(f"[WARNING] Failed to query version from robot: {e}")
                    detected_version = "1.2"

                # Cache the version classification on the app instance
                self.robot_version = detected_version
                if self.robot and hasattr(self.robot, "robot_version"):
                    self.robot.robot_version = detected_version
                if self.robot:
                    try:
                        self.robot.joint_offsets = self.joint_offsets
                    except AttributeError:
                        pass

                # Configure calibrators version
                self.marker_calibrator.robot_version = detected_version
                self.joint_calibrator.robot_version = detected_version
                self.head_camera_calibrator.robot_version = detected_version

                # Update UI modes and offsets based on version classification
                self.update_joint_modes()
                self.load_offsets_from_yaml()
                self.update_applied_offset_label()

                # Bind robot and version to marker detector (handles both real and sim)
                if self.core.observer is not None:
                    self.core.bind_robot(self.robot, detected_version)

                # After the bind, not before: binding is what re-points the marker brackets in
                # setting.yaml at the version this robot actually is, and this reads that file.
                self.load_bracket_design_values()

                if self.ui_only:
                    self.step2_mode_sel.setCurrentText("sim")
                elif self.step2_mode_sel.currentText() != "npz":
                    self.step2_mode_sel.setCurrentText("live")

                self.log_msg(f"[INFO] Robot successfully connected and initialized (Classified Version: {detected_version}).")
                self.update_robot_connection_status()
            else:
                self.log_msg("[ERROR] Robot initialization failed. Check IP.")
                self.update_robot_connection_status()
        except Exception as e:
            traceback.print_exc()
            self.log_msg(f"[ERROR] Connection failure: {e}")
            self.update_robot_connection_status()

    def on_arm_side_changed(self, text):
        new_side = "left" if "Left" in text else "right"
        if self.arm_side != new_side:
            self.arm_side = new_side
            self.ready_done_joint = False
            self.ready_done_marker = False
            self.log_msg(f"[INFO] Changed active arm to {self.arm_side.upper()}. Cleared loaded datasets.")
            self.marker_data_4 = None
            self.marker_data_5 = None
            self.marker_data_6 = None
            self.joint_sweep_data = None

            # Sync current offsets with active arm_side from memory store (do not reload from yaml disk)
            is_v13 = self.get_robot_version() == "1.3"
            for arm in ["left", "right"]:
                self.joint_offsets[arm]["wrist_pitch"] = self.joint_offsets_store.get(arm, {}).get("joint5", 0.0)
                if is_v13:
                    self.joint_offsets[arm]["wrist_roll"] = self.joint_offsets_store.get(arm, {}).get("joint6", 0.0)
                    self.joint_offsets[arm]["wrist_yaw2"] = 0.0
                else:
                    self.joint_offsets[arm]["wrist_roll"] = 0.0
                    self.joint_offsets[arm]["wrist_yaw2"] = self.joint_offsets_store.get(arm, {}).get("joint6", 0.0)
                self.joint_offsets[arm]["elbow"] = self.joint_offsets_store.get(arm, {}).get("joint3", 0.0)
            self.marker_calibrator.joint_offsets = self.joint_offsets
            self.joint_calibrator.joint_offsets = self.joint_offsets
            self.update_applied_offset_label()

            self.load_bracket_design_values()

            # Sync dropdown indexes between controls (blocking signals to avoid cycles)
            self.arm_sel.blockSignals(True)
            idx = 1 if self.arm_side == "left" else 0
            self.arm_sel.setCurrentIndex(idx)
            self.arm_sel.blockSignals(False)

    def on_monitor_toggled(self, checked):
        if checked:
            self.btn_monitor.setText("Marker Monitor: ON")
            self.btn_monitor.setStyleSheet("background-color: #ffeb3b; color: black; font-weight: bold;")
        else:
            self.btn_monitor.setText("Marker Monitor: OFF")
            self.btn_monitor.setStyleSheet("")

    def show_wizard_ui(self):
        if hasattr(self, 'overview_container') and self.overview_container:
            self.overview_container.setVisible(False)
        else:
            if hasattr(self, 'overview_title') and self.overview_title:
                self.overview_title.setVisible(False)
            if hasattr(self, 'overview_link') and self.overview_link:
                self.overview_link.setVisible(False)
            if hasattr(self, 'overview_duration') and self.overview_duration:
                self.overview_duration.setVisible(False)
            if self.overview_img:
                self.overview_img.setVisible(False)
            self.btn_start_wizard.setVisible(False)
        self.wizard_widget.setVisible(True)
        self.wizard_widget.skip_passed_over_start()
        self.on_left_tab_changed(self.left_tabs.currentIndex())


    def toggle_camera_feed_dialog(self):
        if hasattr(self, 'feed_dialog') and self.feed_dialog is not None:
            self.feed_dialog.close()
            return

        self.feed_dialog = CameraFeedDialog(self)
        self.feed_dialog.show()
        self.on_left_tab_changed(self.left_tabs.currentIndex())

    def on_feed_dialog_closed(self):
        self.feed_dialog = None
        self.on_left_tab_changed(self.left_tabs.currentIndex())

    def update_marker_indicator(self, detected):
        self.indicator.set_detected(detected)
        if detected:
            self.status_label.setText("Detected")
            self.status_label.setStyleSheet("color: #00e676;")
        else:
            self.status_label.setText("Not Detected")
            self.status_label.setStyleSheet("color: #ff1744;")

    def poll_camera_status(self):
        # Monitoring consumes camera snapshots; calibration alone requests observations.
        return

    def poll_camera_temperature(self):
        if self.ui_only or self.core.observer is None:
            return
        try:
            if not self.is_temp_supported():
                text = self.get_temp_label_text()
                if hasattr(self, 'temp_label') and self.temp_label is not None:
                    self.temp_label.setText(text)
                if hasattr(self, 'lbl_temp') and self.lbl_temp is not None:
                    self.lbl_temp.setText(text)
                if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                    if hasattr(self.wizard_widget, 'lbl_temp') and self.wizard_widget.lbl_temp is not None:
                        self.wizard_widget.lbl_temp.setText(text)
                return

            if hasattr(self.core.observer, 'camera') and self.core.observer.camera is not None:
                temp = self.core.get_monitor_snapshot()["temperature"]
                self.last_temp = temp
                if temp is not None:
                    text = self.get_temp_label_text(temp)
                    if hasattr(self, 'temp_label') and self.temp_label is not None:
                        self.temp_label.setText(text)
                    if hasattr(self, 'lbl_temp') and self.lbl_temp is not None:
                        self.lbl_temp.setText(text)
                    if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                        if hasattr(self.wizard_widget, 'lbl_temp') and self.wizard_widget.lbl_temp is not None:
                            self.wizard_widget.lbl_temp.setText(text)
        except Exception:
            pass

    def is_any_camera_dialog_visible(self):
        feed_vis = (hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible())
        prob_vis = (hasattr(self, 'marker_problem_dlg') and self.marker_problem_dlg is not None)
        spacing_vis = getattr(self, 'marker_spacing_dlg', None) is not None
        return feed_vis or prob_vis or spacing_vis

    def is_wizard_video_active(self):
        if not hasattr(self, 'wizard_widget') or self.wizard_widget is None:
            return False
        if self.wizard_widget.isHidden():
            return False
        cur_idx = self.wizard_widget.stacked_widget.currentIndex()
        return cur_idx in self.wizard_widget.VIDEO_SLIDES

    def on_left_tab_changed(self, index):
        # 방어적 코드: 타이머 객체가 아직 미생성된 상태이면 처리를 생략
        if not hasattr(self, 'poll_timer') or not hasattr(self, 'video_timer'):
            return

        if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
            self.wizard_widget.check_pose_init_done = False

        # Reparent shared widgets between Step 1 and Step 2 (both under the Advanced tab)
        self._reparent_shared_widgets(self.is_step2_view_active())
        self.refresh_camera_tab_info()

        dialog_visible = self.is_any_camera_dialog_visible()

        if index == self.TAB_CAMERA:
            # Both camera sub-tabs show the live feed
            if self.poll_timer.isActive():
                self.poll_timer.stop()
            self.video_timer.start(50)
        elif index == self.TAB_ADVANCED:
            # Step 1 / Step 2 have no camera feed of their own -- poll, unless a feed dialog is open
            if self.video_timer.isActive():
                self.video_timer.stop()
            if dialog_visible:
                if self.poll_timer.isActive():
                    self.poll_timer.stop()
                self.video_timer.start(50)
            elif not self.ui_only and self.core.observer is not None:
                self.poll_timer.start(200)
        else:
            # Overview tab (index 0) or others: check if wizard is on video slides (0, 2, 4)
            wizard_video_active = self.is_wizard_video_active()
            if wizard_video_active or dialog_visible:
                if self.poll_timer.isActive():
                    self.poll_timer.stop()
                self.video_timer.start(50)
            else:
                if self.video_timer.isActive():
                    self.video_timer.stop()
                if not self.ui_only and self.core.observer is not None:
                    self.poll_timer.start(200)
        self.sync_marker_monitor()

    @staticmethod
    def camera_serial_status(info):
        """(text, colour): whether this camera unit (S/N) has its own saved intrinsics.

        Green only when the unit's own file is in use; otherwise the model's shared file, the stored
        file or the factory values are, which is what the red line calls the default values."""
        if not info.get("serial_number"):
            return tr("main_tabs.camera_serial_unknown"), "#ff1744"
        if info.get("intrinsics_serial_matched"):
            return tr("main_tabs.camera_serial_registered"), "#00e676"
        return tr("main_tabs.camera_serial_unregistered"), "#ff1744"

    def refresh_camera_tab_info(self):
        """Camera tab header: model, S/N and whether that S/N has its own saved intrinsics."""
        if not hasattr(self, 'lbl_camera_tab_info'):
            return
        observer = getattr(getattr(self, 'core', None), 'observer', None)
        info = None
        if observer is not None and hasattr(observer, 'get_camera_info'):
            try:
                info = observer.get_camera_info()
            except Exception:
                info = None
        if not info:
            self.lbl_camera_tab_info.setText(tr("main_tabs.camera_info_none"))
            self.lbl_camera_tab_info.setStyleSheet("color: #f44336; font-weight: bold; font-size: 13px;")
            self.lbl_camera_serial_status.setText("")
            return
        unknown = tr("wizard.slides.slide_0.cam_unknown")
        self.lbl_camera_tab_info.setText(tr(
            "main_tabs.camera_unit", model=(info.get("camera_model") or unknown).upper(),
            serial=info.get("serial_number") or unknown))
        self.lbl_camera_tab_info.setStyleSheet("color: #dddddd; font-weight: bold; font-size: 13px;")
        status, color = self.camera_serial_status(info)
        self.lbl_camera_serial_status.setText(status)
        self.lbl_camera_serial_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")

    def _on_workflow_tab_changed(self, index):
        """Automatically toggle Column 2 Dashboard between Arm/Marker and Head/Camera."""
        if hasattr(self, 'dash_stack') and self.dash_stack is not None:
            if index == 3:  # Head & Cam subtab
                self.dash_stack.setCurrentIndex(1)
            else:
                self.dash_stack.setCurrentIndex(0)

    def _reparent_shared_widgets(self, step2_active):
        """Move shared GroupBoxes between the Step 1 and Step 2 layouts (Advanced tab)."""
        if not hasattr(self, 'conn_head_box') or self.conn_head_box is None:
            return
        if not hasattr(self, 'step2_left_col'):
            return

        if step2_active:  # Switching TO Step 2
            # Move shared widgets into Step 2 layout
            self.step2_top_row.insertWidget(0, self.conn_head_box)
            self.step2_top_row.insertWidget(1, self.home_offset_box)
            self.step2_top_row.insertWidget(2, self.status_box)

            self.step2_right_col.insertWidget(0, self.log_box)
            self.step2_right_col.setStretchFactor(self.log_box, 1)

            if hasattr(self, 'btn_show_plot'):
                self.btn_show_plot.hide()

        else:  # Switching TO Step 1 (or any other tab)
            self._step1_col1.insertWidget(0, self.conn_head_box)
            self._step1_col2.insertWidget(0, self.home_offset_box)
            self._step1_col3.insertWidget(0, self.status_box)
            self._step1_col3.insertWidget(1, self.log_box)
            self._step1_col3.setStretchFactor(self.log_box, 1)

            if hasattr(self, 'btn_show_plot'):
                self.btn_show_plot.show()

    # =============================================
    # Head & Camera Extrinsics Action Handlers
    # =============================================

    @idle_core_action
    def move_to_ready_pose_step1_5(self):
        if not getattr(self, 'include_head_motion', True):
            self.log_msg("[INFO] Headless mode: Robot has no head. Step 1.5 Head & Camera calibration is not required.")
            return

        self.log_msg("\n[Head & Camera] Moving to Ready Pose (Dual-arm Init Pose)...")
        self.btn_step1_5_ready.setEnabled(False)
        self.btn_step1_5_start.setEnabled(False)
        self.stop_requested = False
        if not hasattr(self, 'head_cam_stop_event') or self.head_cam_stop_event is None:
            self.head_cam_stop_event = self.core.stop_event
        self.head_cam_stop_event.clear()

        self.step1_5_ready_worker = HeadCamReadyWorker(self._sync_core_state(), self.head_cam_stop_event)
        self.step1_5_ready_worker.log_signal.connect(self.log_msg)
        self.step1_5_ready_worker.finished_signal.connect(self._on_step1_5_ready_finished)
        self.step1_5_ready_worker.start()

    def _on_step1_5_ready_finished(self, success, err_msg):
        self.btn_step1_5_ready.setEnabled(True)
        self.btn_step1_5_start.setEnabled(True)
        if success:
            self.log_msg("[Head & Camera] [SUCCESS] Ready Pose reached. Stationary arm markers verified.")
        else:
            self.log_msg(f"[Head & Camera] [ERROR] Ready pose move failed: {err_msg}")

    @idle_core_action
    def start_calibration_step1_5(self):
        if not getattr(self, 'include_head_motion', True):
            self.log_msg("[INFO] Headless mode: Robot has no head. Step 1.5 Head & Camera calibration is not required.")
            return

        try:
            pan_r = float(self.step1_5_pan_range.text())
            tilt_r = float(self.step1_5_tilt_range.text())
            n_steps = int(self.step1_5_num_steps.text())
        except ValueError:
            self.log_msg("[ERROR] Invalid numeric parameters for Head Sweep.")
            return

        self.btn_step1_5_ready.setEnabled(False)
        self.btn_step1_5_start.setEnabled(False)
        self.btn_step1_5_apply.setEnabled(False)
        self.stop_requested = False
        if not hasattr(self, 'head_cam_stop_event') or self.head_cam_stop_event is None:
            self.head_cam_stop_event = self.core.stop_event
        self.head_cam_stop_event.clear()

        self.step1_5_sweep_worker = HeadCamSweepWorker(
            self._sync_core_state(), pan_r, tilt_r, n_steps, self.core.stop_event
        )
        self.step1_5_sweep_worker.log_signal.connect(self.log_msg)
        self.step1_5_sweep_worker.finished_signal.connect(self._on_step1_5_sweep_finished)
        self.step1_5_sweep_worker.start()

    def _on_step1_5_sweep_finished(self, success, results):
        self.btn_step1_5_ready.setEnabled(True)
        self.btn_step1_5_start.setEnabled(True)
        if not success or not results or not results.get("success"):
            self.log_msg("[Head & Camera] [ERROR] Head Sweep Calibration failed or aborted.")
            return

        self.btn_step1_5_apply.setEnabled(True)
        self.log_msg("[Head & Camera] Calibration complete. Updating monitoring tables...")
        self._update_step1_5_tables(results)

    def _update_step1_5_tables(self, results):
        head_offsets = results.get("head_offsets_deg", {})
        pan_off = head_offsets.get("pan", 0.0)
        tilt_off = head_offsets.get("tilt", 0.0)

        self.tbl_step1_5_head_monitor.setItem(0, 0, QTableWidgetItem("0.000°"))
        self.tbl_step1_5_head_monitor.setItem(0, 1, QTableWidgetItem(f"{pan_off:+.3f}°"))

        self.tbl_step1_5_head_monitor.setItem(1, 0, QTableWidgetItem("0.000°"))
        self.tbl_step1_5_head_monitor.setItem(1, 1, QTableWidgetItem(f"{tilt_off:+.3f}°"))

        nom = results.get("nominal_mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        cal = results.get("calibrated_mount_to_cam", nom)

        rpy_rows = [
            ("Roll (°)", nom[3], cal[3]),
            ("Pitch (°)", nom[4], cal[4]),
            ("Yaw (°)", nom[5], cal[5]),
            ("X (mm)", nom[0]*1000.0, cal[0]*1000.0),
            ("Y (mm)", nom[1]*1000.0, cal[1]*1000.0),
            ("Z (mm)", nom[2]*1000.0, cal[2]*1000.0),
        ]
        for row_idx, (name, val_nom, val_cal) in enumerate(rpy_rows):
            self.tbl_step1_5_cam_monitor.setItem(row_idx, 0, QTableWidgetItem(f"{val_nom:+.3f}"))
            self.tbl_step1_5_cam_monitor.setItem(row_idx, 1, QTableWidgetItem(f"{val_cal:+.3f}"))

        q = results.get("quality", {})
        self.lbl_step1_5_rmse_tilt.setText(f"{q.get('rmse_tilt_plane_mm', 0.0):.3f} mm")
        self.lbl_step1_5_rmse_pan.setText(f"{q.get('rmse_pan_plane_mm', 0.0):.3f} mm")
        self.lbl_step1_5_ortho_err.setText(f"{q.get('ortho_error_deg', 0.0):.3f}°")

    @idle_core_action
    def apply_results_step1_5(self):
        ok = self.head_camera_calibrator.apply_calibration_results(log_callback=self.log_msg)
        if ok:
            self.log_msg("[Head & Camera] [SUCCESS] Applied calibrated parameters to setting.yaml and memory.")
            self.btn_step1_5_apply.setEnabled(False)
        else:
            self.log_msg("[Head & Camera] [ERROR] Failed to apply calibration results.")

    # =============================================
    # Step 2 Action Handlers
    # =============================================

    def on_step2_mode_changed(self, text):
        self.log_msg(f"[INFO] Step 2 Mode changed to: '{text}'")
        self.update_step2_est_samples()

    def update_step2_est_samples(self, *args):
        if QThread.currentThread() != QApplication.instance().thread():
            self.update_ui_signal_safe.emit("samples")
            return
        try:
            p = float(self.step2_pos_step.text())
            step_x = float(self.step2_step_x.text())
            m = float(self.step2_max_x.text())
            if p <= 0 or step_x <= 0:
                return

            current_x = 0.0
            if self.robot is not None and self.dyn_model is not None:
                try:
                    # Sync config values for build_incremental_motion_plan
                    self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
                    self.auto_config.position_step_m = float(self.step2_pos_step.text())
                    self.auto_config.step_x_m = float(self.step2_step_x.text())
                    self.auto_config.max_x = float(self.step2_max_x.text())
                    self.auto_config.max_loops = 1

                    active_arms = ["right", "left"]
                    temp_plan = build_incremental_motion_plan(
                        self.robot, self.dyn_model, self.auto_config, active_arms,
                        include_head_motion=self.include_head_motion, wrist_diversity=self.step2_wrist_diversity(),
                        skip_half_steps=load_step2_plan_options()["skip_half_steps"]
                    )
                    cnt = len(temp_plan)

                    # Compute current_x for display
                    q_full = self.robot.get_state().position
                    T_fk = BaseCalibrator.compute_fk(self.robot, self.dyn_model, q_full, "ee_right", "link_torso_5")
                    current_x = T_fk[0, 3]

                    self.step2_est_samples_label.setText(f"Est. Samples: {cnt} (from X={current_x:.3f})")
                    return
                except:
                    pass

            # Fallback/Offline estimation:
            # We exactly generate 33 steps per X-step (12 joint steps + 1 restore step + 12 RPY steps + 8 YZ steps = 33)
            current_x = 0.3
            if m > current_x:
                cnt = 33 * (int((m - current_x) / step_x) + 1)
            else:
                cnt = 0
            self.step2_est_samples_label.setText(f"Est. Samples: {cnt} (approx)")
        except:
            pass

    def get_capture_head_idx(self):
        if self.model is None:
            return None
        return get_head_config(self.model)["head_idx"]

    def get_active_arms(self):
        arm_text = self.arm_sel.currentText()
        if "Left" in arm_text:
            return ["left"]
        elif "Right" in arm_text:
            return ["right"]
        return ["right", "left"]

    def get_target_arm_str(self):
        active_arms = self.get_active_arms()
        if len(active_arms) == 1:
            return active_arms[0]
        return "both"

    def ensure_home_offset_robot(self):
        if self.robot is None or self.model is None:
            self.log_msg("[INFO] Robot is not connected. Connecting before home offset operation...")
            self.connect_robot()
        if self.robot is None or self.model is None:
            raise RuntimeError("Robot is not connected.")

    def resolve_input_path(self, raw_path):
        input_path = Path(raw_path).expanduser()
        if input_path.is_absolute():
            return input_path
        return Path(current_dir) / input_path

    def ensure_result_dir(self):
        path = Path(CONFIG_PATHS["result_dir"])
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_output_paths(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.ensure_result_dir()
        dataset_path = result_dir / f"dataset_{timestamp}.npz"
        result_path = result_dir / f"result_{timestamp}.json"
        return dataset_path, result_path

    # NOTE: get_latest_result_path / get_home_reset_path_for_result are defined
    # once, further down this class (near apply_home_offset). They prefer the
    # path the just-completed sequence actually wrote (self.last_result_path /
    # self.last_home_reset_path, synced from self.core in _on_core_result) and
    # fall back to scanning the result/config directories.

    def format_home_offset_compare_summary(self, result_path, baseline_path):
        lines = [
            "Preview moves use the same convention as Apply Home Offset:",
            "the robot moves to zero pose first, then to -joint_offset.",
            "",
            f"Optimized result: {result_path if result_path else 'None'}",
            f"Baseline reset: {baseline_path if baseline_path else 'None'}",
            ""
        ]

        def get_offsets(path):
            if path is None or not os.path.exists(path):
                return None, None, None
            try:
                data = ResultStorage.load(path)
                right = np.array(data.get("right_arm_joint_offset_deg", []))
                left = np.array(data.get("left_arm_joint_offset_deg", []))
                head = data.get("head_joint_offset_deg", [])
                if head is not None:
                    head = np.array(head)
                return right, left, head
            except Exception:
                return None, None, None

        opt_r, opt_l, opt_h = get_offsets(result_path)
        base_r, base_l, base_h = get_offsets(baseline_path)

        def format_section(title, opt, base):
            section = [f"--- {title} ---"]
            if base is not None and len(base) > 0:
                section.append(f"  Baseline  : {np.array2string(np.round(base, 4), separator=', ')}")
            else:
                section.append(f"  Baseline  : Unavailable")

            if opt is not None and len(opt) > 0:
                section.append(f"  Optimized : {np.array2string(np.round(opt, 4), separator=', ')}")
            else:
                section.append(f"  Optimized : Unavailable")

            if base is not None and opt is not None and len(base) == len(opt) and len(base) > 0:
                diff = base - opt
                section.append(f"  Diff (B-O): {np.array2string(np.round(diff, 4), separator=', ')}")
            else:
                section.append(f"  Diff (B-O): Unavailable")
            section.append("")
            return section

        lines.extend(format_section("RIGHT ARM (deg)", opt_r, base_r))
        lines.extend(format_section("LEFT ARM (deg)", opt_l, base_l))
        lines.extend(format_section("HEAD (deg)", opt_h, base_h))

        return "\n".join(lines)
        return "\n".join(lines)

    def infer_home_offset_apply_arm(self, requested_arm, json_path):
        try:
            offset_rad, _ = load_offset_from_json(str(json_path))
        except Exception:
            return requested_arm

        if self.model is not None:
            both_dof = len(self.model.right_arm_idx) + len(self.model.left_arm_idx)
            if len(offset_rad) == both_dof:
                return "both"

        if len(offset_rad) == 14:
            return "both"
        return requested_arm


    @idle_core_action
    def move_to_all_auto_motions(self):
        if not self.auto_ready_done:
            self.auto_ready_done = True

        if self.ui_only and self.step2_mode_sel.currentText() not in ["live", "sim"]:
            self.step2_mode_sel.setCurrentText("sim")

        try:
            if hasattr(self, 'step2_angle_step'): self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
            if hasattr(self, 'step2_pos_step'): self.auto_config.position_step_m = float(self.step2_pos_step.text())
            if hasattr(self, 'step2_step_x'): self.auto_config.step_x_m = float(self.step2_step_x.text())
            if hasattr(self, 'step2_max_x'): self.auto_config.max_x = float(self.step2_max_x.text())
            self.auto_config.max_loops = 1
        except Exception as e:
            self.log_msg(f"Failed to read auto config: {e}. Using current values.")

        if self.auto_motion_plan is None or len(self.auto_motion_plan) == 0:
            self.log_msg("Motion plan is missing or empty. Re-building...")
            active_arms = ["right", "left"]
            try:
                self.auto_motion_plan = build_incremental_motion_plan(
                    self.robot, self.dyn_model, self.auto_config, active_arms, include_head_motion=self.include_head_motion,
                    wrist_diversity=self.step2_wrist_diversity(),
                    skip_half_steps=load_step2_plan_options()["skip_half_steps"]
                )
            except Exception as e:
                self.log_msg(f"[ERROR] Failed to build motion plan: {e}")
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Motion Plan Error")
                msg_box.setText(f"Failed to build motion plan:\n{e}")
                msg_box.setStyleSheet("QLabel { min-height: 70px; font-size: 13px; padding: 12px; } QMessageBox { min-width: 500px; }")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec()
                return

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log_msg("Auto motions have already been executed.")
            return

        if self.auto_motion_running or self.auto_motion_thread is not None:
            self.log_msg("Another robot operation is already running.")
            return

        self.auto_stop_requested = False
        self.auto_motion_running = True
        self.log_msg("Auto Motion started in a background thread. Press Stop to cancel.")

        self.auto_motion_thread = Step2AutoMotionWorker(self._sync_core_state(), self.auto_motion_plan, self.head_move_count, parent=self)
        self.auto_motion_thread.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            self.auto_motion_running = False
            self.auto_motion_thread = None
            self.auto_save_current_dataset()
            if success:
                self.log_msg("Auto motions sequence completed.")
                self.step2_calculate()
            else:
                self.log_msg(f"Auto motion error: {error_msg}")
        self.auto_motion_thread.finished_signal.connect(on_finished)
        self.auto_motion_thread.start()

    def stop_all_auto_motion_internal(self, cancel_robot=False, reset_stop_requested=True):
        self.auto_motion_running = False
        if reset_stop_requested:
            self.auto_stop_requested = False

        if cancel_robot:
            self.safe_cancel_control()

        # 백그라운드 스레드가 완전히 종료되어 finished_signal을 통해 on_finished()가 호출될 때까지
        # self.auto_motion_thread를 None으로 설정하지 않고 유지합니다.
        # 이를 통해 이전 작업이 완전히 끝나지 않은 상태에서 새로운 로봇 명령이 병렬로 전송되는 것을 방지합니다.
        pass

        reset_motion_state()

    def request_stop_all_auto_motion(self):
        self.core.cancel()
        if not self.auto_motion_running and self.auto_motion_thread is None:
            self.log_msg("No Auto Motion sequence is running.")
            self.stop_all_auto_motion_internal(cancel_robot=True)
            return

        self.auto_stop_requested = True
        self.stop_all_auto_motion_internal(cancel_robot=True, reset_stop_requested=False)
        self.log_msg("Stop requested. Sent robot.cancel_control(); the auto motion sequence stops after the current step.")

    def auto_save_current_dataset(self):
        if len(self.shared_arm_q_list) == 0:
            return

        q_arm_list = np.array(self.shared_arm_q_list)
        q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
        T_meas_list = np.array(self.shared_T_list)

        # Determine active arms based on joint dimensions
        if q_arm_list.shape[1] == 14:
            active_arms = ["right", "left"]
        else:
            active_arms = self.get_active_arms()
        optimize_head = self.include_head_motion

        try:
            validate_dataset(q_arm_list, q_head_list, T_meas_list, optimize_head, active_arms)

            if not self.dataset_saved_in_session or self.current_session_dataset_path is None:
                dataset_path, _ = self.build_output_paths()
                self.current_session_dataset_path = dataset_path
                self.dataset_saved_in_session = True

            save_npz_dataset(self.current_session_dataset_path, q_arm=q_arm_list, q_head=q_head_list, T_meas=T_meas_list)
            self.last_dataset_path = self.current_session_dataset_path
            self.log_msg(f"[Auto-Save] Dataset saved/updated in: {self.current_session_dataset_path}")
        except Exception as e:
            self.log_msg(f"[Auto-Save Error] {e}")

    def _on_core_progress(self, value):
        """Live counters while a core sequence collects samples (Qt-queued, GUI thread)."""
        if not isinstance(value, dict) or value.get("stage") != "collect":
            return
        total = value.get("pose_total") or 0
        if hasattr(self, 'step2_head_status_label'):
            label = f"Auto Motion: {value.get('pose_index', 0)}/{total}"
            if not self.include_head_motion:
                label += " (headless)"
            self.step2_head_status_label.setText(label)
        if hasattr(self, 'step2_sample_count_label'):
            self.step2_sample_count_label.setText(f"Shared Samples: {value.get('samples', 0)}")

    def update_sample_counts(self):
        if QThread.currentThread() != QApplication.instance().thread():
            self.update_ui_signal_safe.emit("sample_counts")
            return
        sample_count = len(self.shared_arm_q_list)
        if hasattr(self, 'step2_sample_count_label'):
            self.step2_sample_count_label.setText(f"Shared Samples: {sample_count}")

    def get_auto_pose_target_count(self):
        if self.auto_motion_plan is not None:
            return len(self.auto_motion_plan)
        return 0

    def update_head_pose_status(self):
        if QThread.currentThread() != QApplication.instance().thread():
            self.update_ui_signal_safe.emit("head_pose")
            return
        pose_target_count = self.get_auto_pose_target_count()
        pose_idx = min(self.head_move_count, pose_target_count)
        label = f"Auto Motion: {pose_idx}/{pose_target_count}"
        if not self.include_head_motion:
            label += " (headless)"
        if hasattr(self, 'step2_head_status_label'):
            self.step2_head_status_label.setText(label)


    def run_optimizer(self, *args, **kwargs):
        self._sync_core_state()
        result = self.core.run("optimize", args=args, kwargs=kwargs)
        self.last_sequence_result = result
        self.last_result_path = self.core.last_result_path
        if result.status == "failed":
            raise RuntimeError(result.error)
        return result

    def _on_apply_joint_offset_toggled(self, checked):
        """Toggle apply joint offset flag and update status label."""
        self.apply_joint_offset_flag = checked
        if checked:
            self.lbl_jo_status.setText("ACTIVE")
            self.lbl_jo_status.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_jo_status.setText("INACTIVE")
            self.lbl_jo_status.setStyleSheet("color: #ff1744; font-weight: bold; font-size: 11px;")
        self.log_msg(f"[INFO] Apply Joint Offset: {'ACTIVE' if checked else 'INACTIVE'}")

    @idle_core_action
    def step2_zero_pose_check(self):
        self.log_msg("[Step2] Zero Pose Check requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        arm = "both"

        self.zero_pose_worker = Step2ZeroPoseCheckWorker(
            self._sync_core_state(),
            arm,
            self.include_head_motion
        )
        self.zero_pose_worker.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            if success:
                self.log_msg("\n===== ZERO POSE CHECK COMPLETE =====")
                dialog = ZeroPoseCheckDialog(self)
                dialog.exec()
            else:
                self.log_msg(f"Zero pose check failed: {error_msg}")
        self.zero_pose_worker.finished_signal.connect(on_finished)
        self.zero_pose_worker.start()

    @idle_core_action
    def move_to_zero_for_exposure(self, on_finished):
        """Wizard exposure slide: back to zero after the markers were shown to the camera. Its Next
        waits for this move, so the home offset reset starts from the zero pose."""
        self.log_msg("[Wizard 3-1] Move to Zero Pose requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return False
        self.exposure_pose_worker = Step2ZeroPoseCheckWorker(self._sync_core_state(), "both", self.include_head_motion)
        self.exposure_pose_worker.log_signal.connect(self.log_msg)
        self.exposure_pose_worker.finished_signal.connect(on_finished)
        self.exposure_pose_worker.start()
        return True

    @idle_core_action
    def run_auto_exposure(self, on_finished):
        """Wizard exposure slide: right arm to its J5 ready pose, then the auto exposure (D405 only).
        on_finished(success, error, result)."""
        self.log_msg("[Wizard] Test pose + auto exposure requested.")
        if not self.robot or self.core.observer is None:
            self.log_msg("[ERROR] Robot and camera must be connected!")
            return False
        self.auto_exposure_worker = AutoExposureWorker(self._sync_core_state())
        self.auto_exposure_worker.log_signal.connect(self.log_msg)
        self.auto_exposure_worker.finished_signal.connect(on_finished)
        self.auto_exposure_worker.start()
        return True

    @idle_core_action
    def move_to_gripper_pose(self, on_finished):
        """Wizard gripper slide: both arms to the gripper-change pose of the robot version, then the
        tool flange output off (v1.2; v1.3 keeps it on). on_finished(success, error, {side: volts})."""
        self.log_msg("[Wizard] Gripper-change pose requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return False
        self.gripper_pose_worker = GripperPoseWorker(self._sync_core_state())
        self.gripper_pose_worker.log_signal.connect(self.log_msg)
        self.gripper_pose_worker.finished_signal.connect(on_finished)
        self.gripper_pose_worker.start()
        return True

    def step2_stop_auto_motion(self):
        self.log_msg("[Step2] Stop Auto Motion requested.")
        self.request_stop_all_auto_motion()
        self.auto_save_current_dataset()

    @idle_core_action
    def step2_init_pose(self, silent=False):
        self.log_msg("[Step2] Init Pose requested.")
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        if self.auto_motion_running or self.auto_motion_thread is not None:
            if not silent:
                QMessageBox.critical(self, "Execution Error", "Another robot operation is currently running.")
            return

        self.auto_motion_running = True
        active_arms = ["right", "left"]
        self.auto_motion_thread = Step2InitPoseWorker(
            self._sync_core_state(),
            active_arms,
            self.auto_config.priority if hasattr(self, 'auto_config') else 0,
            include_head_motion=self.include_head_motion,
            parent=self
        )
        self.auto_motion_thread.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            self.auto_motion_running = False
            self.auto_motion_thread = None
            if success:
                self.auto_ready_done = True
                try:
                    if hasattr(self, 'step2_angle_step'):
                        self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
                    if hasattr(self, 'step2_pos_step'):
                        self.auto_config.position_step_m = float(self.step2_pos_step.text())
                    if hasattr(self, 'step2_step_x'):
                        self.auto_config.step_x_m = float(self.step2_step_x.text())
                    if hasattr(self, 'step2_max_x'):
                        self.auto_config.max_x = float(self.step2_max_x.text())
                    self.auto_config.max_loops = 1
                except Exception as e:
                    self.log_msg(f"Failed to read auto config: {e}. Using default values.")

                self.auto_motion_plan = None
                if getattr(self, 'include_head_motion', False) and self.robot:
                    head_cfg = get_head_config(self.model)
                    if head_cfg.get("head_idx") is not None:
                        state = self.robot.get_state()
                        if state is not None and getattr(state, 'position', None) is not None:
                            q_full = np.array(state.position)
                            h_idx = list(head_cfg["head_idx"])
                            self.auto_base_head_q = np.array([float(q_full[i]) for i in h_idx], dtype=np.float64)
                            self.log_msg(f"Auto base head pose (deg): {np.round(np.rad2deg(self.auto_base_head_q), 3)}")
                        else:
                            self.auto_base_head_q = None
                    else:
                        self.auto_base_head_q = None
                        self.include_head_motion = False

                self.head_move_count = 0
                self.update_head_pose_status()
                self.update_step2_est_samples()

                if not silent:
                    self.show_message_box(
                        "Teaching Required",
                        "Robot has moved to the initial pose.\n\n"
                        "Please adjust the robot's pose so that the marker is clearly visible to the camera.\n"
                        "Once adjusted, press '2) Auto Motion' to start the sequence.",
                        QMessageBox.Information
                    )
            else:
                self.log_msg(f"Init pose failed: {error_msg}")
                if not getattr(self, 'auto_stop_requested', False) and not silent:
                    QMessageBox.critical(self, "Init Error", error_msg)

        self.auto_motion_thread.finished_signal.connect(on_finished)
        self.auto_motion_thread.start()

    def step2_auto_motion(self):
        self.log_msg("[Step2] Auto Motion requested.")
        mode = self.step2_mode_sel.currentText()
        if mode not in ["live", "sim"]:
            self.log_msg("[Step2] Auto motion is only available in live or sim mode.")
            return
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        try:
            self.move_to_all_auto_motions()
        except Exception as e:
            if not self.ui_only:
                QMessageBox.critical(self, "Auto Motion Error", str(e))
            self.log_msg(f"Auto motion failed: {e}")

    @idle_core_action
    def step2_calculate(self):
        self.log_msg("[Step2] Calculate requested.")

        # Check if calculation is already running
        if hasattr(self, 'calc_worker') and self.calc_worker is not None and self.calc_worker.isRunning():
            self.log_msg("[Step2] [WARNING] Optimization calculation is ALREADY running!")
            QMessageBox.information(
                self,
                "Calculation In Progress",
                "Optimization calculation is already running in the background.\n\nPlease wait for the current calculation to complete."
            )
            return

        if not self.robot and not self.ui_only:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        try:
            mode = self.step2_mode_sel.currentText().strip()
            self.log_msg(f"\n[Step2] Calculate requested (Active Mode: '{mode}').")
            active_arms = ["right", "left"]
            has_step1_5 = self.include_head_motion and (
                hasattr(self, 'head_camera_calibrator') and
                self.head_camera_calibrator is not None and
                self.head_camera_calibrator.calibrated_results is not None and
                not self.head_camera_calibrator.calibrated_results.get("skipped", False)
            )

            if has_step1_5:
                optimize_head = self.include_head_motion
                optimize_camera = True
                lambda_cam_pos = 1.0
                lambda_cam_rot = 100.0
                self.log_msg("[INFO] Step 1.5 Camera parameters detected. Step 1.5 camera rotation is used as baseline (lambda_cam_rot=100.0), and camera XYZ translation is optimized within ±10mm in Step 2.")
            else:
                optimize_head = self.include_head_motion
                if not self.include_head_motion and optimize_head:
                    optimize_head = False
                    self.log_msg("Headless mode selected; optimize_head changed to False.")

                camera_cfg = getattr(self.marker_calibrator, "camera_config", {})
                if not optimize_head:
                    optimize_camera = False
                    lambda_cam_pos = 1.0
                    lambda_cam_rot = 1e6
                    self.log_msg("[INFO] Headless mode: Camera extrinsics optimization is DISABLED (Locked to CAD nominal).")
                else:
                    optimize_camera = True
                    lambda_cam_pos = 1.0
                    lambda_cam_rot = 1e6

            if len(active_arms) == 1:
                cfg = get_arm_config(self.model, active_arms[0], version=self.get_robot_version())
                ee_links = {active_arms[0]: cfg["ee_link"]}
                ee_to_marker_nom = {active_arms[0]: cfg["ee_to_marker_nom"]}
            else:
                cfg = get_both_arm_config(self.model, version=self.get_robot_version())
                ee_links = cfg["ee_links"]
                ee_to_marker_nom = cfg["ee_to_marker_nom"]

            head_cfg = get_head_config(self.model)
            # Robot zero the samples were collected at; None (npz, mixed or unknown) makes the
            # result analysis only: it cannot be applied as home offset (CalibrationCore.home_epoch).
            data_home_epoch = None

            if mode in ["live", "sim"]:
                if len(self.shared_arm_q_list) == 0:
                    QMessageBox.warning(self, "Warning", "No recorded samples in memory.")
                    return
                epochs = set(self.shared_sample_epochs)
                if len(epochs) == 1 and len(self.shared_sample_epochs) == len(self.shared_arm_q_list):
                    data_home_epoch = epochs.pop()
                if data_home_epoch != self.core.home_epoch:
                    data_home_epoch = None
                    self.log_msg("[Step2][WARN] The samples in memory were not all collected at the current robot zero "
                                 "(a home offset was reset or applied since). The result will be analysis only; "
                                 "clear the samples and collect again to get an applicable result.")

                self.log_msg(f"[Step2] Using live recorded dataset ({len(self.shared_arm_q_list)} samples in memory).")
                q_arm_list = np.array(self.shared_arm_q_list)
                q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
                T_meas_list = np.array(self.shared_T_list)

                if q_arm_list.shape[1] == 7 and len(active_arms) == 2:
                    active_arms = ["right"]
                    self.log_msg("[INFO] Single-arm dataset detected (7 joints). Switching active_arms to ['right'].")

                if len(active_arms) == 1:
                    if q_arm_list.shape[1] == 14:
                        if active_arms[0] == "right":
                            q_arm_list = q_arm_list[:, :7]
                        else:
                            q_arm_list = q_arm_list[:, 7:]
                    if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                        if active_arms[0] == "right":
                            T_meas_list = T_meas_list[:, 0]
                        else:
                            T_meas_list = T_meas_list[:, 1]

            elif mode == "npz":
                npz_raw = self.step2_path_input.text().strip()
                npz_path = self.resolve_input_path(npz_raw)
                self.log_msg(f"[Step2] Loading NPZ dataset from: {npz_path}")
                q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)
                self.log_msg(f"[Step2] Loaded {len(q_arm_list)} samples from NPZ dataset.")

                if q_arm_list.shape[1] == 7 and len(active_arms) == 2:
                    active_arms = ["right"]
                    self.log_msg("[INFO] Single-arm NPZ dataset detected (7 joints). Switching active_arms to ['right'].")

                if len(active_arms) == 1:
                    if q_arm_list.shape[1] == 14:
                        if active_arms[0] == "right":
                            q_arm_list = q_arm_list[:, :7]
                        else:
                            q_arm_list = q_arm_list[:, 7:]
                    if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
                        if active_arms[0] == "right":
                            T_meas_list = T_meas_list[:, 0]
                        else:
                            T_meas_list = T_meas_list[:, 1]

            dataset_path, result_path = self.build_output_paths()

            # Disable calculate button and update UI status
            if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
                self.btn_step2_calculate.setEnabled(False)
                self.btn_step2_calculate.setText("Calculating...")
                self.btn_step2_calculate.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold;")

            self.log_msg("[Step2] Optimization calculation started in background thread...")

            self.calc_worker = Step2CalculateWorker(
                self._sync_core_state(),
                active_arms,
                optimize_head,
                optimize_camera,
                q_arm_list,
                q_head_list,
                T_meas_list,
                result_path,
                lambda_cam_pos,
                lambda_cam_rot,
                data_home_epoch=data_home_epoch,
            )
            self.calc_worker.log_signal.connect(self.log_msg)

            def on_finished(success, error_msg):
                if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
                    self.btn_step2_calculate.setEnabled(True)
                    self.btn_step2_calculate.setText("3) Calculate")
                    self.btn_step2_calculate.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")

                if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                    self.wizard_widget.stop_step5(success, error_msg if not success else "")

                if success:
                    self.log_msg("Optimization finished successfully.")
                    QMessageBox.information(self, "Step 2 Calculation", "Optimization finished successfully!\nCheck the logs and Result Output for details.")
                else:
                    self.log_msg(f"[Error] Optimization failed: {error_msg}")
                    QMessageBox.warning(self, "Step 2 Calculation Failed", f"Optimization failed:\n{error_msg}")

            self.calc_worker.finished_signal.connect(on_finished)
            self.calc_worker.start()

        except Exception as e:
            if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
                self.btn_step2_calculate.setEnabled(True)
                self.btn_step2_calculate.setText("3) Calculate")
                self.btn_step2_calculate.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
            QMessageBox.critical(self, "Calculate Error", str(e))
            self.log_msg(f"Calculate failed: {e}")
            QMessageBox.critical(self, "Calculate Error", str(e))
            self.log_msg(f"Calculate failed: {e}")

    def step2_clear_samples(self):
        self.log_msg("[Step2] Clear Samples requested.")
        self.stop_all_auto_motion_internal(cancel_robot=True)
        reset_motion_state()
        self.shared_arm_q_list.clear()
        self.shared_head_q_list.clear()
        self.shared_T_list.clear()
        self.shared_sample_epochs.clear()
        self.head_move_count = 0
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.dataset_saved_in_session = False
        self.current_session_dataset_path = None
        self.update_sample_counts()
        self.update_head_pose_status()
        self.log_msg("Shared samples cleared.")

    def step2_apply_home_offset(self):
        self.log_msg("[Step2] Apply Home Offset requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        try:
            result_path = self.get_latest_result_path()
            baseline_path = self.get_home_reset_path_for_result(result_path)
            arm = "both"

            compare_summary = self.format_home_offset_compare_summary(result_path, baseline_path)

            if hasattr(self, 'apply_offset_dialog') and self.apply_offset_dialog is not None and self.apply_offset_dialog.isVisible():
                self.apply_offset_dialog.raise_()
                self.apply_offset_dialog.activateWindow()
                return

            self.apply_offset_dialog = ApplyHomeOffsetDialog(
                parent=self,
                result_path=result_path,
                baseline_path=baseline_path,
                arm=arm,
                include_head=self.include_head_motion,
                compare_summary=compare_summary
            )
            self.apply_offset_dialog.show()
            self.apply_offset_dialog.raise_()
            self.apply_offset_dialog.activateWindow()

        except Exception as e:
            QMessageBox.critical(self, "Apply Home Offset Error", str(e))
            self.log_msg(f"Apply home offset failed: {e}")

    def step2_check_calibration_state(self):
        self.log_msg("[Step2] Check Calibration State requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        dialog = CheckCalibrationStateDialog(self)
        dialog.exec()

    def update_applied_offset_label(self):
        self.ready_done_joint = False
        if not hasattr(self, 'tbl_offset_monitor') or not hasattr(self, 'btn_joint_apply'):
            return

        is_v13 = self.get_robot_version() == "1.3"
        self.tbl_offset_monitor.setColumnCount(3)
        if is_v13:
            self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Roll)", "Joint 5 (Pitch)", "Joint 3 (Elbow)"])
        else:
            self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Yaw 2)", "Joint 5 (Wrist Pitch)", "Joint 3 (Elbow)"])

        for row_idx, arm in enumerate(["right", "left"]):
            for col_idx, joint_key in enumerate(["joint6", "joint5", "joint3"]):
                val = self.joint_offsets_store.get(arm, {}).get(joint_key, 0.0)
                item = QTableWidgetItem(f"{val:.4f}°")
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl_offset_monitor.setItem(row_idx, col_idx, item)

    @idle_core_action
    def apply_joint_offset(self):
        is_v13 = self.get_robot_version() == "1.3"

        for arm in ["left", "right"]:
            self.joint_offsets[arm]["wrist_pitch"] = self.joint_offsets_store[arm]["joint5"]
            if is_v13:
                self.joint_offsets[arm]["wrist_roll"] = self.joint_offsets_store[arm]["joint6"]
                self.joint_offsets[arm]["wrist_yaw2"] = 0.0
            else:
                self.joint_offsets[arm]["wrist_roll"] = 0.0
                self.joint_offsets[arm]["wrist_yaw2"] = self.joint_offsets_store[arm]["joint6"]
            self.joint_offsets[arm]["elbow"] = self.joint_offsets_store[arm]["joint3"]
        self.joint_calibrator.joint_offsets = self.joint_offsets
        self.marker_calibrator.joint_offsets = self.joint_offsets

        self.save_offsets_to_yaml()
        self.update_applied_offset_label()

        self.log_msg(f"\n" + "="*50)
        self.log_msg(f"[APPLY] Applied current staged joint offsets for BOTH arms:")
        for arm in ["left", "right"]:
            self.log_msg(f"  --- {arm.upper()} ARM ---")
            if is_v13:
                self.log_msg(f"    * Joint 6 (Wrist Roll) : {self.joint_offsets[arm]['wrist_roll']:.4f}°")
            else:
                self.log_msg(f"    * Joint 6 (Wrist Yaw 2): {self.joint_offsets[arm]['wrist_yaw2']:.4f}°")
            self.log_msg(f"    * Joint 5 (Wrist Pitch): {self.joint_offsets[arm]['wrist_pitch']:.4f}°")
            self.log_msg(f"    * Joint 3 (Elbow)      : {self.joint_offsets[arm]['elbow']:.4f}°")
        self.log_msg("[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!")
        self.log_msg("="*50 + "\n")


    def stop_motion(self):
        self.core.cancel()
        self.log_msg("[STOP] Stop requested by user.")

        # 1. Stop Step 1 sweep calibrations
        self.joint_calibrator.stop_requested = True
        self.marker_calibrator.stop_requested = True
        if hasattr(self, 'stop_event_mc') and self.stop_event_mc:
            self.stop_event_mc.set()

        # 2. Stop Step 2 auto collection/motion
        if self.auto_motion_running or self.auto_motion_thread is not None:
            self.request_stop_all_auto_motion()
            self.auto_save_current_dataset()
        else:
            # If not running Step 2 auto motion, we still want to make sure
            # any robot motion is cancelled if robot is connected.
            if self.robot:
                self.log_msg("[STOP] Sending cancel_control to robot!")
                self.safe_cancel_control()
            else:
                self.log_msg("[STOP] No robot connected to cancel control.")

        # 3. Stop Full Auto calibration
        if hasattr(self, 'full_auto_stop_event') and self.full_auto_stop_event:
            self.full_auto_stop_event.set()

        # 4. Stop Head & Camera calibration
        if hasattr(self, 'head_cam_stop_event') and self.head_cam_stop_event:
            self.head_cam_stop_event.set()

    @idle_core_action
    def clear_joint_offset(self):
        reply = QMessageBox.question(
            self,
            "Clear Joint Offset",
            "Are you sure you want to reset all staged/saved joint offsets for BOTH arms to 0.0?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for arm in ["right", "left"]:
                self.joint_offsets_store[arm]["joint5"] = 0.0
                self.joint_offsets_store[arm]["joint6"] = 0.0
                self.joint_offsets_store[arm]["joint3"] = 0.0

                self.joint_offsets[arm]["wrist_pitch"] = 0.0
                self.joint_offsets[arm]["wrist_roll"] = 0.0
                self.joint_offsets[arm]["wrist_yaw2"] = 0.0
                self.joint_offsets[arm]["elbow"] = 0.0

            self.joint_offsets_store["head"] = {"pan": 0.0, "tilt": 0.0}
            if hasattr(self, 'head_camera_calibrator') and self.head_camera_calibrator:
                self.head_camera_calibrator.calibrated_results = None

            self.joint_calibrator.joint_offsets = self.joint_offsets
            self.marker_calibrator.joint_offsets = self.joint_offsets

            self.save_offsets_to_yaml()
            self.update_applied_offset_label()

            self.log_msg("[CLEAR] Staged and saved offsets cleared to 0.0 for BOTH Arms and Head.")

    def load_bracket_design_values(self):
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if os.path.exists(config_path):
                with FileStorage.open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                    marker_data = data.get("marker", {})
                    # Tf_to_marker_<side> is shown as it stands. Which robot version's bracket it
                    # is belongs to the observer, which re-points it at the connected robot's
                    # version on bind and writes that to setting.yaml before this runs. Guessing
                    # here off `abs(x) > 0.05` used to overwrite a v1.3 bracket with the v1.2
                    # nominal whenever the robot was not connected yet and the version still
                    # read as the "1.2" default.
                    # Load Left Arm values
                    val_left = marker_data.get("Tf_to_marker_left", None)
                    if val_left and len(val_left) == 6:
                        self.txt_bracket_l_x.setText(f"{val_left[0]:.4f}")
                        self.txt_bracket_l_y.setText(f"{val_left[1]:.4f}")
                        self.txt_bracket_l_z.setText(f"{val_left[2]:.4f}")
                        self.txt_bracket_l_roll.setText(f"{val_left[3]:.2f}")
                        self.txt_bracket_l_pitch.setText(f"{val_left[4]:.2f}")
                        self.txt_bracket_l_yaw.setText(f"{val_left[5]:.2f}")
                        # Sync back to memory configs. camera_config carries its own merged copy
                        # and is what the Step 1 / Step 2 sequences read, so it has to follow too
                        # or a bracket that just changed with the robot version stays stale there.
                        for calibrator in (self.marker_calibrator, self.joint_calibrator):
                            calibrator.markers_config["Tf_to_marker_left"] = val_left
                            calibrator.camera_config["Tf_to_marker_left"] = val_left

                    # Load Right Arm values
                    val_right = marker_data.get("Tf_to_marker_right", None)
                    if val_right and len(val_right) == 6:
                        self.txt_bracket_r_x.setText(f"{val_right[0]:.4f}")
                        self.txt_bracket_r_y.setText(f"{val_right[1]:.4f}")
                        self.txt_bracket_r_z.setText(f"{val_right[2]:.4f}")
                        self.txt_bracket_r_roll.setText(f"{val_right[3]:.2f}")
                        self.txt_bracket_r_pitch.setText(f"{val_right[4]:.2f}")
                        self.txt_bracket_r_yaw.setText(f"{val_right[5]:.2f}")
                        # Sync back to memory configs
                        for calibrator in (self.marker_calibrator, self.joint_calibrator):
                            calibrator.markers_config["Tf_to_marker_right"] = val_right
                            calibrator.camera_config["Tf_to_marker_right"] = val_right

                    self.log_msg(f"[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory")
                    return
            self.log_msg("[WARNING] Could not load bracket design values from setting.yaml.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to load setting.yaml: {e}")


    @idle_core_action
    def apply_bracket_design_values(self, silent=False):
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            try:
                l_x = float(self.txt_bracket_l_x.text())
                l_y = float(self.txt_bracket_l_y.text())
                l_z = float(self.txt_bracket_l_z.text())
                l_roll = float(self.txt_bracket_l_roll.text())
                l_pitch = float(self.txt_bracket_l_pitch.text())
                l_yaw = float(self.txt_bracket_l_yaw.text())

                r_x = float(self.txt_bracket_r_x.text())
                r_y = float(self.txt_bracket_r_y.text())
                r_z = float(self.txt_bracket_r_z.text())
                r_roll = float(self.txt_bracket_r_roll.text())
                r_pitch = float(self.txt_bracket_r_pitch.text())
                r_yaw = float(self.txt_bracket_r_yaw.text())
            except ValueError:
                if not silent:
                    QMessageBox.critical(self, "Invalid Inputs", "Please enter valid numeric values for all bracket design fields.")
                return

            duplicates = ConfigStorage.update_values(config_path, {
                ("marker", "Tf_to_marker_left"): [l_x, l_y, l_z, l_roll, l_pitch, l_yaw],
                ("marker", "Tf_to_marker_right"): [r_x, r_y, r_z, r_roll, r_pitch, r_yaw],
            })
            if duplicates:
                self.log_msg(f"[WARN] setting.yaml had duplicate keys {sorted(set(duplicates))}; "
                    "only the last of each was kept.")

            self.log_msg(f"[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml")
            if not silent:
                QMessageBox.information(self, "Success", "Bracket design values saved for both arms!")

            if not self.ui_only and self.core.observer is not None:
                self.core.update_marker_transforms(
                    [l_x, l_y, l_z, l_roll, l_pitch, l_yaw],
                    [r_x, r_y, r_z, r_roll, r_pitch, r_yaw])
                self.log_msg("[INFO] Updated marker transforms through core.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save bracket values: {e}")
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to save bracket values: {e}")

    def set_controls_enabled(self, enabled):
        if hasattr(self, 'btn_full_auto_start'):
            self.btn_full_auto_start.setEnabled(enabled)
            self.btn_step1_auto.setEnabled(enabled)
        self.btn_joint_ready.setEnabled(enabled)
        self.btn_joint_start.setEnabled(enabled)
        self.btn_joint_apply.setEnabled(enabled)
        if hasattr(self, 'btn_joint_clear'):
            self.btn_joint_clear.setEnabled(enabled)
        if hasattr(self, 'btn_apply_bracket'):
            self.btn_apply_bracket.setEnabled(enabled)

        if hasattr(self, 'txt_bracket_l_x'):
            self.txt_bracket_l_x.setEnabled(enabled)
            self.txt_bracket_l_y.setEnabled(enabled)
            self.txt_bracket_l_z.setEnabled(enabled)
            self.txt_bracket_l_roll.setEnabled(enabled)
            self.txt_bracket_l_pitch.setEnabled(enabled)
            self.txt_bracket_l_yaw.setEnabled(enabled)

            self.txt_bracket_r_x.setEnabled(enabled)
            self.txt_bracket_r_y.setEnabled(enabled)
            self.txt_bracket_r_z.setEnabled(enabled)
            self.txt_bracket_r_roll.setEnabled(enabled)
            self.txt_bracket_r_pitch.setEnabled(enabled)
            self.txt_bracket_r_yaw.setEnabled(enabled)

        self.btn_marker_ready.setEnabled(enabled)
        self.btn_marker_start.setEnabled(enabled)
        if hasattr(self, 'btn_marker_result'):
            self.btn_marker_result.setEnabled(enabled)

        if hasattr(self, 'btn_step2_calculate') and self.btn_step2_calculate is not None:
            is_calculating = hasattr(self, 'calc_worker') and self.calc_worker is not None and self.calc_worker.isRunning()
            self.btn_step2_calculate.setEnabled(enabled and not is_calculating)

        self.btn_int_capture.setEnabled(enabled)
        self.btn_int_calibrate.setEnabled(enabled)
        self.btn_int_reset.setEnabled(enabled)

        if hasattr(self, 'chk_servo_head'):
            self.chk_servo_head.setEnabled(enabled)
        if hasattr(self, 'btn_step1_5_ready'):
            self.btn_step1_5_ready.setEnabled(enabled)
        if hasattr(self, 'btn_step1_5_start'):
            self.btn_step1_5_start.setEnabled(enabled)

        self.btn_connect.setEnabled(enabled)
        self.model_input.setEnabled(enabled)
        # The tab bar stays navigable while a sequence runs: every action button inside
        # these tabs is already disabled individually above, and idle_core_action rejects
        # a second start anyway. Disabling the whole widget only trapped the operator on
        # whichever tab started the run.
        self.arm_sel.setEnabled(enabled)
        self.joint_mode_sel.setEnabled(enabled)
        if hasattr(self, 'marker_axis_sel'):
            self.marker_axis_sel.setEnabled(enabled)
        if hasattr(self, 'btn_camera_feed'):
            self.btn_camera_feed.setEnabled(True) # Keep camera feed button enabled always!

    def on_action_finished(self):
        self.set_controls_enabled(True)

    def on_move_ready_joint_finished(self, success, error=""):
        self.ready_done_joint = success
        if not success:
            self.log_msg(f"[ERROR] Joint ready pose failed: {error}")
        self.on_action_finished()

    def on_move_ready_marker_finished(self, success, error=""):
        self.ready_done_marker = success
        if not success:
            self.log_msg(f"[ERROR] Marker ready pose failed: {error}")
        self.on_action_finished()

    def get_robot_version(self) -> str:
        return str(getattr(self, "robot_version", "1.2"))

    def step2_wrist_diversity(self):
        """Extra Step 2 wrist poses (head robots only, setting.yaml step2.wrist_diversity_poses)."""
        if not self.include_head_motion:
            return []
        steps, enabled = load_step2_wrist_diversity(self.get_robot_version())
        if enabled:
            self.log_msg(f"[INFO] Step 2 wrist-diversity poses: {len(steps)} (v{self.get_robot_version()}).")
        return steps

    @idle_core_action
    def move_to_ready_full_auto(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.active_worker = FullAutoReadyWorker(self._sync_core_state())

        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_full_auto_ready_finished)
        self.active_worker.start()

    def on_full_auto_ready_finished(self):
        self.set_controls_enabled(True)
        error_msg = getattr(self.active_worker, 'error_msg', None) if self.active_worker else None
        if self.active_worker is not None:
            self.active_worker.wait()
        self.active_worker = None

        # Restart poll_timer if appropriate
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        camera_subtab_active = self.is_camera_tab_active()
        if not camera_subtab_active and not dialog_visible:
            if not self.poll_timer.isActive():
                self.poll_timer.start(200)

        if error_msg:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Movement Failed")
            msg_box.setText(f"Ready pose movement failed!\n\nReason:\n{error_msg}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            msg_box.exec()
        else:
            self.ready_done_joint = True
            self.ready_done_marker = True
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Movement Complete")
            msg_box.setText("Robot arms have moved to the initial ready poses successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            msg_box.exec()

    def get_latest_result_path(self):
        # Prefer the exact file the most recently completed sequence wrote
        # (kept in sync with self.core.last_result_path in _on_core_result).
        # Falling back to a directory mtime scan is a best-effort guess and
        # can pick the wrong file if an older result's mtime is touched by a
        # copy/sync/checkout, so only use it when we have no better answer.
        if self.last_result_path is not None and Path(self.last_result_path).exists():
            return Path(self.last_result_path)

        result_dir = Path(CONFIG_PATHS["result_dir"])
        if not result_dir.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
        result_files = sorted(
            result_dir.glob("result_*.json"),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )
        if not result_files:
            return None
        self.last_result_path = result_files[0]
        return self.last_result_path

    def get_latest_home_reset_path(self, required=True):
        path = Path(os.path.abspath(os.path.join(app_dir, "config", "home_reset_baseline.json")))
        if path.exists():
            return path
        if required:
            raise RuntimeError(f"No home reset baseline JSON found at {path}")
        return None

    def get_home_reset_path_for_result(self, result_path):
        if self.last_home_reset_path is not None and Path(self.last_home_reset_path).exists():
            return Path(self.last_home_reset_path)
        path = self.get_latest_home_reset_path(required=False)
        if path is not None:
            self.last_home_reset_path = path
        return path

    def apply_home_offset(self):
        try:
            result_path = self.get_latest_result_path()
            baseline_path = self.get_home_reset_path_for_result(result_path)

            if hasattr(self, 'apply_offset_dialog') and self.apply_offset_dialog is not None and self.apply_offset_dialog.isVisible():
                self.apply_offset_dialog.raise_()
                self.apply_offset_dialog.activateWindow()
                return

            self.apply_offset_dialog = ApplyHomeOffsetDialog(
                self,
                result_path,
                baseline_path,
                "both",
                include_head=True
            )
            self.apply_offset_dialog.show()
            self.apply_offset_dialog.raise_()
            self.apply_offset_dialog.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Apply Home Offset Error", str(e))
            self.log_msg(f"[ERROR] Apply home offset failed: {e}")

    @idle_core_action
    def home_offset_reset(self, *args, confirm_dialog=True, **kwargs) -> bool:
        if not self.ui_only and not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return False

        if confirm_dialog:
            msg = tr("common.home_offset_reset_msg")
            dialog = QDialog(self)
            dialog.setWindowTitle(tr("common.home_offset_reset_title"))
            dialog.setStyleSheet(DARK_STYLESHEET)
            layout = QVBoxLayout(dialog)

            # Image
            img_label = QLabel()
            pixmap = QPixmap(get_asset_path("img/offset_reset_pose_direction.png"))
            if not pixmap.isNull():
                img_label.setPixmap(pixmap.scaled(820, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                img_label.setText("[img/offset_reset_pose_direction.png not found]")
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

            # Text
            msg_label = QLabel(msg)
            msg_label.setStyleSheet("font-size: 14px; color: white;")
            layout.addWidget(msg_label)

            # Buttons
            btn_layout = QHBoxLayout()
            btn_ok = QPushButton(tr("common.apply"))
            btn_ok.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
            btn_cancel = QPushButton(tr("common.cancel"))
            btn_cancel.setStyleSheet("background-color: #555; color: white; padding: 5px;")

            btn_ok.clicked.connect(dialog.accept)
            btn_cancel.clicked.connect(dialog.reject)

            btn_layout.addStretch()
            btn_layout.addWidget(btn_ok)
            btn_layout.addWidget(btn_cancel)
            layout.addLayout(btn_layout)

            if dialog.exec() != QDialog.Accepted:
                return False

        self.set_controls_enabled(False)
        self.btn_home_reset.setEnabled(False)

        # Start worker thread
        self.active_worker = HomeOffsetResetWorker(
            self._sync_core_state(),
            self.robot.model() if self.robot else None,
            self.model_input.currentText().strip() if hasattr(self, 'model_input') else "a",
            include_head=True
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_home_offset_reset_finished)
        self.active_worker.start()
        return True

    def on_home_offset_reset_finished(self, result):
        self.set_controls_enabled(True)
        self.btn_home_reset.setEnabled(True)
        if self.active_worker is not None:
            self.active_worker.wait()
        self.active_worker = None

        success = False
        error_msg = ""
        if result.get("success", False):
            # Reset software joint offsets to 0.0 for both arms since they are now physically absorbed
            for arm in ["left", "right"]:
                self.joint_offsets_store[arm]["joint3"] = 0.0
                self.joint_offsets_store[arm]["joint5"] = 0.0
                self.joint_offsets_store[arm]["joint6"] = 0.0

                self.joint_offsets[arm]["wrist_pitch"] = 0.0
                self.joint_offsets[arm]["wrist_roll"] = 0.0
                self.joint_offsets[arm]["wrist_yaw2"] = 0.0
                self.joint_offsets[arm]["elbow"] = 0.0

            # Save zeroed offsets to setting.yaml and update GUI
            self.save_offsets_to_yaml()
            self.update_applied_offset_label()


            self.log_msg("Re-connecting and initializing robot...")
            if self.robot:
                self.connect_robot() # Disconnects first
                QApplication.processEvents()
            self.connect_robot() # Connects again
            self.log_msg("Home Offset Reset complete!")
            QMessageBox.information(self, "Success", "Home Offset Reset, Power, and Servo Initialization completed successfully! Software joint offsets have been reset to 0.0.")
            success = True
        else:
            error_msg = result.get("error", "Some joints failed to reset")
            if result.get("failed_joints") or result.get("reset_joints"):
                QMessageBox.warning(self, "Warning", f"Home Offset Reset finished, but some joints failed to reset: {error_msg}")
            else:
                # Refused before any joint was touched (reset posture check): not a partial reset.
                QMessageBox.warning(self, "Warning", error_msg)
            success = False

        if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
            self.wizard_widget.set_wizard_busy(False)
            if success:
                self.wizard_widget.mark_step_completed(self.wizard_widget.SLIDE_HOME_OFFSET, True, ("wizard.st.reset_done", {}))
            else:
                self.wizard_widget.mark_step_completed(self.wizard_widget.SLIDE_HOME_OFFSET,False, error_msg)


    def clear_old_plots(self):
        self.generated_plots = []
        self.current_plot_idx = -1
        self.lbl_plot_title.setText("No Plot Loaded")
        self.plot_label_combined.setPixmap(QPixmap())
        self.btn_plot_prev.setEnabled(False)
        self.btn_plot_next.setEnabled(False)
        plot_dir = CONFIG_PATHS.get("plot_dir")
        if plot_dir and os.path.exists(plot_dir):
            for f_name in os.listdir(plot_dir):
                if f_name.endswith(".png") and "circle_fit_" in f_name:
                    try:
                        FileStorage.remove(os.path.join(plot_dir, f_name))
                    except Exception:
                        pass
        if plot_dir:
            txt_dir = os.path.abspath(os.path.join(os.path.dirname(plot_dir), "result_txt"))
            if os.path.exists(txt_dir):
                for f_name in os.listdir(txt_dir):
                    if f_name.endswith(".txt"):
                        try:
                            FileStorage.remove(os.path.join(txt_dir, f_name))
                        except Exception:
                            pass
    @idle_core_action
    def apply_full_auto_results(self, silent=False):
        if silent:
            do_apply = True
        else:
            reply = QMessageBox.question(self, 'Apply Full Auto Results',
                                         "Do you want to apply all calibrated Joint Offsets and Marker Brackets to setting.yaml?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            do_apply = (reply == QMessageBox.Yes)

        if do_apply:
            self.apply_joint_offset()
            self.apply_bracket_design_values(silent=True)
            self.log_msg("[APPLY] Full auto results (Joints & Brackets) applied successfully.")
            if not silent:
                QMessageBox.information(self, "Apply Complete", "All full auto calibration results have been applied successfully.")

            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None and hasattr(self.wizard_widget, 'on_step4_applied'):
                self.wizard_widget.on_step4_applied()


    @idle_core_action
    def start_full_auto(self, step1_only=False):
        try:
            head_options = {
                "pan_range_deg": float(self.step1_5_pan_range.text()),
                "tilt_range_deg": float(self.step1_5_tilt_range.text()),
                "num_steps": int(self.step1_5_num_steps.text()),
            } if not step1_only else {}
            if not step1_only:
                self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
                self.auto_config.position_step_m = float(self.step2_pos_step.text())
                self.auto_config.step_x_m = float(self.step2_step_x.text())
                self.auto_config.max_x = float(self.step2_max_x.text())
                self.auto_config.max_loops = 1
        except ValueError:
            self.log_msg("[ERROR] Invalid sequence parameters. Check Head Sweep and Step 2 inputs.")
            return False
        self.joint_offsets = {
            "left": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0},
            "right": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
        }
        self.joint_offsets_store = {
            "left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
            "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
            # Without an explicit head entry, save_offsets_to_yaml kept a stale head offset from
            # an earlier session (simulator values were found in setting.yaml on 2026-09-15).
            "head": {"pan": 0.0, "tilt": 0.0},
        }
        if hasattr(self, 'head_camera_calibrator') and self.head_camera_calibrator:
            self.head_camera_calibrator.calibrated_results = None

        if hasattr(self, 'joint_calibrator') and self.joint_calibrator:
            self.joint_calibrator.joint_offsets = self.joint_offsets
        if hasattr(self, 'marker_calibrator') and self.marker_calibrator:
            self.marker_calibrator.joint_offsets = self.joint_offsets

        self.update_applied_offset_label()
        self.log_msg("[FULL AUTO] Initial arm & head joint offsets reset to 0.0 before starting calibration.")
        if getattr(self.core.observer, 'sim', False):
            sim_model = getattr(self.core.observer, 'simulation_model', None)
            if sim_model and hasattr(sim_model, 'config'):
                sim_offsets = sim_model.config.get('offsets', {})
                self.log_msg("[SIM GT] Simulation Ground-Truth Offsets:")
                is_v13 = self.get_robot_version() == "1.3"
                for arm in ["right", "left"]:
                    if arm in sim_offsets:
                        sim_gt = sim_offsets[arm]
                        j6_gt = sim_gt.get("joint6", 0.0)
                        j5_gt = sim_gt.get("joint5_v13" if is_v13 else "joint5_v12", 0.0)
                        j3_gt = sim_gt.get("joint3", 0.0)
                        pos_gt = [x * 1000.0 for x in sim_gt.get("bracket_pos", [0, 0, 0])]
                        rpy_gt = sim_gt.get("bracket_rpy", [0, 0, 0])
                        self.log_msg(f"  --- {arm.upper()} ARM ---")
                        self.log_msg(f"  * Bracket Pos: X: {pos_gt[0]:+.1f}, Y: {pos_gt[1]:+.1f}, Z: {pos_gt[2]:+.1f} mm")
                        self.log_msg(f"  * Bracket Rot: R: {rpy_gt[0]:+.2f}, P: {rpy_gt[1]:+.2f}, Y: {rpy_gt[2]:+.2f} deg")
                        self.log_msg(f"  * Joint Offsets: Joint 6: {j6_gt:+.2f}°, Joint 5: {j5_gt:+.2f}°, Joint 3: {j3_gt:+.2f}°")

        self.set_controls_enabled(False)
        self.btn_full_auto_start.setEnabled(False)

        if self.poll_timer.isActive():
            self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False

        self.full_auto_stop_event = self.core.stop_event

        # update robot version on calibrators just in case
        v = self.get_robot_version()
        self.marker_calibrator.robot_version = v
        self.joint_calibrator.robot_version = v

        self._sync_core_state()
        options = {"save_debug": self.chk_save_debug.isChecked()}
        sequence = "step1" if step1_only else "full"
        if not step1_only:
            options = {"step1": options, "step1_5": head_options}
        self.active_worker = FullAutoWorker(self.core, sequence, options)


        self.active_worker.log_msg.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.bracket_finished_signal.connect(self.handle_full_auto_bracket_finished)
        self.active_worker.joint_finished_signal.connect(self.handle_full_auto_joint_finished)
        self.active_worker.finished_signal.connect(self.on_full_auto_finished)
        self.active_worker.start()

    def stop_full_auto(self):
        self.core.cancel()
        self.log_msg("[STOP] Stopping Full Auto Calibration...")
        if hasattr(self, 'full_auto_stop_event') and self.full_auto_stop_event:
            self.full_auto_stop_event.set()
        self.joint_calibrator.stop_requested = True
        self.marker_calibrator.stop_requested = True
        if self.robot:
            self.safe_cancel_control()

    def on_full_auto_finished(self):
        self.set_controls_enabled(True)
        if hasattr(self, 'btn_full_auto_start'):
            self.btn_full_auto_start.setEnabled(True)
        if hasattr(self, 'btn_full_auto_apply'):
            result = getattr(self.active_worker, "result", None)
            self.btn_full_auto_apply.setEnabled(bool(result and result.success))

        was_stopped = False
        if hasattr(self, 'full_auto_stop_event') and self.full_auto_stop_event is not None:
            was_stopped = self.full_auto_stop_event.is_set()

        error_msg = getattr(self.active_worker, 'error_msg', None) if self.active_worker else None
        self.last_full_auto_error = error_msg
        if self.active_worker is not None:
            self.active_worker.wait()
        self.active_worker = None
        self.log_msg("[INFO] Full Auto sequential calibration ended.")

        # Restart poll_timer if appropriate (not tab 2 and feed dialog closed)
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        camera_subtab_active = self.is_camera_tab_active()
        if not camera_subtab_active and not dialog_visible:
            if not self.poll_timer.isActive():
                self.poll_timer.start(200)

        if error_msg:
            self.log_msg(f"[ERROR] Full Auto Calibration FAILED: {error_msg}")
        elif not was_stopped:
            self.log_msg("[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.")

    def handle_full_auto_bracket_finished(self, bracket_res):
        arm_side = bracket_res['arm_side']

        # Update UI text boxes for corresponding arm
        if arm_side == "left":
            self.txt_bracket_l_x.setText(f"{bracket_res['x_e']/1000.0:.4f}")
            self.txt_bracket_l_y.setText(f"{bracket_res['y_e']/1000.0:.4f}")
            self.txt_bracket_l_z.setText(f"{bracket_res['z_e']/1000.0:.4f}")
            self.txt_bracket_l_roll.setText(f"{bracket_res['roll_e']:.2f}")
            self.txt_bracket_l_pitch.setText(f"{bracket_res['pitch_e']:.2f}")
            self.txt_bracket_l_yaw.setText(f"{bracket_res['yaw_e']:.2f}")
        else:
            self.txt_bracket_r_x.setText(f"{bracket_res['x_e']/1000.0:.4f}")
            self.txt_bracket_r_y.setText(f"{bracket_res['y_e']/1000.0:.4f}")
            self.txt_bracket_r_z.setText(f"{bracket_res['z_e']/1000.0:.4f}")
            self.txt_bracket_r_roll.setText(f"{bracket_res['roll_e']:.2f}")
            self.txt_bracket_r_pitch.setText(f"{bracket_res['pitch_e']:.2f}")
            self.txt_bracket_r_yaw.setText(f"{bracket_res['yaw_e']:.2f}")

        # Stage Joint 5 and Joint 6 offsets if solved (v1.3)
        if 'opt_delta_5' in bracket_res:
            self.joint_offsets_store[arm_side]["joint5"] = float(bracket_res['opt_delta_5'])
            self.joint_offsets_store[arm_side]["joint6"] = float(bracket_res['opt_delta_6'])
            self.update_applied_offset_label()
            self.log_msg(f"[INFO] Full Auto: Staged joint offsets for {arm_side.upper()} Arm - Joint 5: {bracket_res['opt_delta_5']:.4f}°, Joint 6: {bracket_res['opt_delta_6']:.4f}°")

        if 'plot_path_combined' in bracket_res and bracket_res.get('pass_idx', 1) == 2:
            self.add_and_show_plot(f"[{arm_side.upper()}] FullAuto - Marker Bracket", bracket_res['plot_path_combined'])

        self.log_msg(f"[INFO] Full Auto: Finished bracket calibration for {arm_side.upper()} arm. Values staged in UI (click APPLY BRACKETS to save).")

    def handle_full_auto_joint_finished(self, joint_res):
        arm_side = joint_res['arm_side']
        mode = joint_res.get('mode', 'elbow')

        recommended = joint_res['recommended_joint_offset']
        if mode in ("wrist_roll_v13", "wrist_yaw2"):
            joint_key = "joint6"
        elif mode in ("wrist_pitch_v13", "wrist_pitch"):
            joint_key = "joint5"
        else:
            joint_key = "joint3"
            # -3 -> -4 deg (2026-10-27): an elbow pushed fully back measured -3.48 deg and was cut to -3.
            clipped = float(np.clip(recommended, -4.0, 0.0))
            if clipped != float(recommended):
                self.log_msg(f"[WARN] {arm_side} elbow (J3) offset {float(recommended):+.4f}° is outside [-4, 0]°; "
                             f"stored {clipped:+.4f}°.")
            recommended = clipped

        # Update staged offsets store
        self.joint_offsets_store[arm_side][joint_key] = float(recommended)

        # Refresh offset monitor table view
        self.update_applied_offset_label()

        self.log_msg(f"[INFO] Full Auto: Finished joint calibration for {arm_side.upper()} {mode}. Staged: {recommended:.4f}° (click APPLY OFFSET to save).")

        if 'plot_path_combined' in joint_res and joint_res.get('pass_idx', 1) == 2:
            self.add_and_show_plot(f"[{arm_side.upper()}] FullAuto Joint - {mode}", joint_res['plot_path_combined'])

        if hasattr(self, 'stop_event_mc'):
            self.stop_event_mc.clear()

        # 탭 상태에 맞춰 타이머 활성화
        self.on_left_tab_changed(self.left_tabs.currentIndex())

    # --- Joint Calibration Workflows ---
    @idle_core_action
    def move_to_ready_pose_joint(self):
        if not self.robot and not self.ui_only:
            self.log_msg("[ERROR] Robot is not connected!")
            return


        mode = self.get_selected_joint_mode()

        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.ready_worker = MoveToReadyWorker(self._sync_core_state(), self.arm_side, mode)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_move_ready_joint_finished)
        self.ready_worker.start()

    @idle_core_action
    def start_calibration_joint(self):
        if not self.ready_done_joint:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Prerequisite Check")
            msg_box.setText("Please move the robot to the Ready pose first by clicking 'MOVE TO READY'!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        self.clear_old_plots()

        mode = self.get_selected_joint_mode()
        if mode == "wrist_pitch_v13" and not self.wrist_roll_calibrated.get(self.arm_side, False):
            reply = QMessageBox.warning(
                self,
                "Calibration Sequence Warning",
                "Wrist Roll (Joint 6) has not been calibrated yet.\n"
                "It is highly recommended to calibrate Joint 6 (wrist_roll_v13) first.\n"
                "Do you want to proceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        offset_key = self.get_offset_key_for_mode(mode)
        self.original_joint_offset = self.joint_offsets[self.arm_side].get(offset_key, 0.0)
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Joint Sweep: {mode.upper()}")
        if getattr(self.core.observer, 'sim', False):
            sim_model = getattr(self.core.observer, 'simulation_model', None)
            if sim_model and hasattr(sim_model, 'config'):
                sim_gt = sim_model.config.get('offsets', {}).get(self.arm_side, {})
                is_v13 = self.get_robot_version() == "1.3"
                j_gt = {
                    "wrist_roll_v13": sim_gt.get("joint6", 0.0),
                    "wrist_yaw2": sim_gt.get("joint6", 0.0),
                    "wrist_pitch_v13": sim_gt.get("joint5_v13", 0.0),
                    "wrist_pitch": sim_gt.get("joint5_v12", 0.0),
                    "elbow": sim_gt.get("joint3", 0.0)
                }
                gt_val = j_gt.get(mode, 0.0)
                self.log_msg(f"[SIM GT] Simulated Target Joint Offset: {gt_val:+.2f}°")

        curr_offset = self.joint_offsets[self.arm_side].get(offset_key, 0.0)
        sweep_time = 20.0 if mode in ["elbow", "wrist_yaw2", "wrist_roll_v13"] else 15.0
        self.active_worker = JointCalibrationWorker(
            self._sync_core_state(), self.arm_side, mode,
            ui_only=self.ui_only,
            current_offset_deg=curr_offset,
            sweep_duration=sweep_time,
            save_debug=self.chk_save_debug.isChecked()
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.finished_signal.connect(self.on_calibration_finished_joint)
        self.active_worker.start()

    def on_calibration_finished_joint(self, res):
        if not res:
            self.on_action_finished()
            return

        mode = res['mode']
        recommended = res.get('recommended_joint_offset', res['optimal_offset'])
        self.recommended_joint_offset = recommended
        self.finalize_joint_calibration_run(mode, res, converged=res.get('converged', True))

    def finalize_joint_calibration_run(self, mode, res, converged=True):
        self.on_action_finished()
        self.joint_sweep_data = res

        # Update Plot viewer if plots exist
        if 'plot_path_combined' in res and os.path.exists(res['plot_path_combined']):
            self.add_and_show_plot(f"[{self.arm_side.upper()}] Joint - {mode}", res['plot_path_combined'])

        # Update UI bracket design text fields if new values are calibrated
        if 'x_cal' in res and not np.isnan(res['x_cal']):
            arm = self.arm_side
            x_val, y_val, z_val = res['x_cal'], res['y_cal'], res['z_cal']
            r_val, p_val, yaw_val = res.get('roll_cal', float('nan')), res.get('pitch_cal', float('nan')), res.get('yaw_cal', float('nan'))
            if arm == "left":
                self.txt_bracket_l_x.setText(f"{x_val:.5f}")
                self.txt_bracket_l_y.setText(f"{y_val:.5f}")
                self.txt_bracket_l_z.setText(f"{z_val:.5f}")
                if not np.isnan(r_val):
                    self.txt_bracket_l_roll.setText(f"{r_val:.2f}")
                    self.txt_bracket_l_pitch.setText(f"{p_val:.2f}")
                    self.txt_bracket_l_yaw.setText(f"{yaw_val:.2f}")
            else:
                self.txt_bracket_r_x.setText(f"{x_val:.5f}")
                self.txt_bracket_r_y.setText(f"{y_val:.5f}")
                self.txt_bracket_r_z.setText(f"{z_val:.5f}")
                if not np.isnan(r_val):
                    self.txt_bracket_r_roll.setText(f"{r_val:.2f}")
                    self.txt_bracket_r_pitch.setText(f"{p_val:.2f}")
                    self.txt_bracket_r_yaw.setText(f"{yaw_val:.2f}")
            if not np.isnan(r_val):
                self.log_msg(f"[INFO] Staged calibrated nominal marker values in UI for {arm} arm: X={x_val:.5f}, Y={y_val:.5f}, Z={z_val:.5f}, R={r_val:.2f}, P={p_val:.2f}, Y={yaw_val:.2f}")
            else:
                self.log_msg(f"[INFO] Staged calibrated nominal marker values in UI for {arm} arm: X={x_val:.5f}, Y={y_val:.5f}, Z={z_val:.5f}")

        if mode in ("wrist_roll_v13", "wrist_yaw2"):
            joint_key = "joint6"
            if converged:
                self.wrist_roll_calibrated[self.arm_side] = True
        elif mode in ("wrist_pitch_v13", "wrist_pitch"):
            joint_key = "joint5"
        else:
            joint_key = "joint3"
        self.joint_offsets_store[self.arm_side][joint_key] = float(self.recommended_joint_offset)

        # Revert active offsets to nominal original values in model (until user clicks APPLY)
        offset_key = self.get_offset_key_for_mode(mode)
        self.joint_offsets[self.arm_side][offset_key] = self.original_joint_offset
        self.joint_calibrator.joint_offsets[self.arm_side][offset_key] = self.original_joint_offset
        self.marker_calibrator.joint_offsets[self.arm_side][offset_key] = self.original_joint_offset
        self.update_applied_offset_label()

        self.log_msg(f"\n" + "="*50)
        if converged:
            self.log_msg(f"   [SUCCESS] 3-STEP POLARITY CALIBRATION CONVERGED SUCCESSFULLY!")
        else:
            self.log_msg(f"   [INFO] 3-STEP CALIBRATION COMPLETED")
        self.log_msg(f"   * Recommended Absolute Offset : {self.recommended_joint_offset:.4f}°")
        self.log_msg(f"   * Current Active Offset       : {self.original_joint_offset:.4f}° (REVERTED)")
        self.log_msg(f"   --> Click 'APPLY OFFSET' on the UI panel to apply this new calibration.")
        self.log_msg("="*50 + "\n")

        self.show_result_joint()

    def show_result_joint(self):
        self.log_msg("\n" + "="*50)
        self.log_msg("       JOINT CALIBRATION ESTIMATED RESULTS")
        self.log_msg("="*50)

        if not self.joint_sweep_data:
            self.log_msg("\n[ERROR] No joint sweep data loaded! Perform a sweep first.")
            return

        mode = self.joint_sweep_data['mode']
        self.log_msg(f"\n[1] Calibration Target: {mode}")

        recommended = self.joint_sweep_data.get('recommended_joint_offset', self.joint_sweep_data['optimal_offset'])
        if mode == "wrist_roll_v13":
            joint_name = "Joint 6 (Wrist Roll)"
        elif mode == "wrist_yaw2":
            joint_name = "Joint 6 (Wrist Yaw 2)"
        elif mode == "wrist_pitch_v13":
            joint_name = "Joint 5 (Wrist Pitch)"
        elif mode == "wrist_pitch":
            joint_name = "Joint 5 (Wrist Pitch)"
        else:
            joint_name = "Joint 3 (Elbow Pitch)"
        self.log_msg(f"    - Target Swept Joint       : {joint_name}")
        self.log_msg(f"    - Estimated Optimal Offset : {recommended:.4f} deg")

        self.log_msg("\n[2] Suggested Joint Home Offset update:")
        self.log_msg(f"  Add offset: {recommended:.4f} deg to calibration config.")

        # display bracket design verification results
        if mode in ("wrist_pitch_v13", "wrist_roll_v13", "wrist_yaw2"):
            d = self.joint_sweep_data
            sweep_axis_label = "Joint 6" if mode in ("wrist_roll_v13", "wrist_yaw2") else "Joint 5"
            self.log_msg(f"\n[3] Bracket Design Verification (Based on {sweep_axis_label} Axis)")
            perp_b = d.get('perp_dist_before', float('nan'))
            perp_a = d.get('perp_dist_after',  float('nan'))
            self.log_msg(f"    - c_B ~ {sweep_axis_label} axis perp. dist (before) : {perp_b:.4f} mm")
            self.log_msg(f"    - c_B ~ {sweep_axis_label} axis perp. dist (after)  : {perp_a:.4f} mm")
            r_A = d.get('r_A', float('nan'))
            axial  = d.get('axial_offset_mm',   float('nan'))
            lateral = d.get('lateral_offset_mm', float('nan'))
            self.log_msg(f"    - Sweep A fitting radius (r_A, lateral marker offset) : {r_A:.3f} mm")
            self.log_msg(f"    - Axial marker offset (c_B along {sweep_axis_label} axis)  : {axial:.3f} mm")
            self.log_msg(f"    - Lateral marker offset (c_B perp {sweep_axis_label} axis)  : {lateral:.3f} mm")
            axis_dir = "Z" if mode == "wrist_yaw2" else ("X" if mode == "wrist_roll_v13" else "Y")
            self.log_msg(f"    * Design Reference Offset Axis: {axis_dir}-axis")

        self.log_msg("="*50)


    # --- Marker Bracket Calibration Workflows ---
    @idle_core_action
    def move_to_ready_pose_marker(self):
        if not self.robot and not self.ui_only:
            self.log_msg("[ERROR] Robot is not connected!")
            return


        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.ready_worker = MoveToReadyWorker(self._sync_core_state(), self.arm_side)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_move_ready_marker_finished)
        self.ready_worker.start()

    # Move to Center is removed as it is no longer needed

    @idle_core_action
    def start_calibration_marker(self):
        # 1. Prerequisite: Joint 6 calibrated first. 2026-09-15: warn + confirm instead of blocking,
        # so the bracket sweeps can be run on their own (e.g. to check bracket geometry).
        if not self.wrist_roll_calibrated.get(self.arm_side, False):
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Prerequisite Warning")
            msg_box.setText(
                "Joint 6 (Wrist Roll / Wrist Yaw 2) has not been calibrated in this session.\n\n"
                "The bracket sweeps still run, but the bracket pose is computed against the current "
                "Joint 6 value, so an uncalibrated Joint 6 shows up as bracket yaw/roll error.\n\n"
                "Run the bracket calibration anyway?"
            )
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            if msg_box.exec() != QMessageBox.Yes:
                return
            self.log_msg("[WARN] Marker bracket calibration started without a calibrated Joint 6 (user confirmed).")

        # 2. Prerequisite Check: Move to Ready Pose first
        if not self.ready_done_marker:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Prerequisite Check")
            msg_box.setText("Please move the robot to the Ready pose first by clicking 'MOVE TO READY'!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            return

        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return

        self.clear_old_plots()

        # use_head = self.cb_head_tracking.isChecked()
        use_head = False
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()

        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.log_text.clear()
        self.log_msg(f"[INFO] Starting Unified Marker Sweep (Axis 6 & 5) (Head Tracking: {use_head})")
        if getattr(self.core.observer, 'sim', False):
            sim_model = getattr(self.core.observer, 'simulation_model', None)
            if sim_model and hasattr(sim_model, 'config'):
                sim_gt = sim_model.config.get('offsets', {}).get(self.arm_side, {})
                pos_gt = [x * 1000.0 for x in sim_gt.get("bracket_pos", [0, 0, 0])]
                rpy_gt = sim_gt.get("bracket_rpy", [0, 0, 0])
                self.log_msg(f"[SIM GT] Simulated Bracket Offset (Tf_to_marker):")
                self.log_msg(f"  * Pos: X: {pos_gt[0]:+.1f}, Y: {pos_gt[1]:+.1f}, Z: {pos_gt[2]:+.1f} mm")
                self.log_msg(f"  * Rot: R: {rpy_gt[0]:+.2f}, P: {rpy_gt[1]:+.2f}, Y: {rpy_gt[2]:+.2f} deg")

        try:
            tolerance = float(self.tolerance_input.text())
        except ValueError:
            tolerance = 0.5
        self.active_worker = MarkerCalibrationWorker(
            self._sync_core_state(), self.arm_side,
            use_head_tracking=use_head, tolerance=tolerance,
            save_debug=self.chk_save_debug.isChecked()
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.finished_signal.connect(self.on_calibration_finished_marker)
        self.active_worker.start()

    def on_calibration_finished_marker(self, res):
        self.on_action_finished()

        if res:
            self.marker_data_unified = res
            self.marker_data_5 = res['res_5']
            self.marker_data_6 = res['res_6']
            self.marker_data_4 = res.get('res_4', None)

            # Stage Joint 5 and Joint 6 offsets if solved (v1.3)
            if 'opt_delta_5' in res:
                self.joint_offsets_store[self.arm_side]["joint5"] = float(res['opt_delta_5'])
                self.joint_offsets_store[self.arm_side]["joint6"] = float(res['opt_delta_6'])
                self.update_applied_offset_label()
                self.log_msg(f"[INFO] Staged joint offsets for {self.arm_side.upper()} Arm - Joint 5: {res['opt_delta_5']:.4f}°, Joint 6: {res['opt_delta_6']:.4f}°")

            # Update UI bracket design text fields
            arm_side = self.arm_side
            x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0
            if arm_side == "left":
                self.txt_bracket_l_x.setText(f"{x_m:.5f}")
                self.txt_bracket_l_y.setText(f"{y_m:.5f}")
                self.txt_bracket_l_z.setText(f"{z_m:.5f}")
                self.txt_bracket_l_roll.setText(f"{res['roll_e']:.2f}")
                self.txt_bracket_l_pitch.setText(f"{res['pitch_e']:.2f}")
                self.txt_bracket_l_yaw.setText(f"{res['yaw_e']:.2f}")
            else:
                self.txt_bracket_r_x.setText(f"{x_m:.5f}")
                self.txt_bracket_r_y.setText(f"{y_m:.5f}")
                self.txt_bracket_r_z.setText(f"{z_m:.5f}")
                self.txt_bracket_r_roll.setText(f"{res['roll_e']:.2f}")
                self.txt_bracket_r_pitch.setText(f"{res['pitch_e']:.2f}")
                self.txt_bracket_r_yaw.setText(f"{res['yaw_e']:.2f}")

            # Sync to memory configs
            new_vals = [x_m, y_m, z_m, res['roll_e'], res['pitch_e'], res['yaw_e']]
            key = f"Tf_to_marker_{arm_side}"
            self.marker_calibrator.camera_config[key] = new_vals
            self.joint_calibrator.camera_config[key] = new_vals
            self.log_msg(f"[INFO] Staged calibrated nominal marker values in UI for {arm_side} arm. (Click APPLY BRACKETS to save)")

            if 'plot_path_combined' in res and os.path.exists(res['plot_path_combined']):
                self.add_and_show_plot(f"[{self.arm_side.upper()}] Marker Bracket", res['plot_path_combined'])
            self.show_unified_result_marker_direct(res)
        else:
            self.log_msg("[ERROR] Marker sweep failed.")

    def show_unified_result_marker_direct(self, res):
        self.log_msg("\n" + "="*50)
        self.log_msg("       UNIFIED BRACKET CALIBRATION RESULTS")
        self.log_msg("="*50)

        self.log_msg("\n[1] Cartesian Offset (EE Link Frame)")
        self.log_msg(f"    - X-Offset: {res['x_e']:.2f} mm")
        self.log_msg(f"    - Y-Offset: {res['y_e']:.2f} mm")
        self.log_msg(f"    - Z-Offset: {res['z_e']:.2f} mm")
        r4_str = f", R4: {res['radius_4']:.2f} mm" if res.get('radius_4', 0.0) > 0.0 else ""
        self.log_msg(f"       * (L_5_ee: {res['L_5_ee']:.1f} mm, R6: {res['radius_6']:.2f} mm, R5: {res['radius_5']:.2f} mm{r4_str})")
        if 'opt_delta_5' in res:
            self.log_msg(f"    - Opt Delta 5: {res['opt_delta_5']:.3f} deg")
            self.log_msg(f"    - Opt Delta 6: {res['opt_delta_6']:.3f} deg")
            self.log_msg(f"    - Min Circle Fitting Radius: {res['min_radius']:.2f} mm")

        self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
        self.log_msg(f"    - Roll : {res['roll_e']:.2f} deg")
        self.log_msg(f"    - Pitch: {res['pitch_e']:.2f} deg")
        self.log_msg(f"    - Yaw  : {res['yaw_e']:.2f} deg")

        self.log_msg("\n[3] setting.yaml Config Update values:")
        x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0

        if self.arm_side == "left":
            self.log_msg(f"  Tf_to_marker_left:  [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
        else:
            self.log_msg(f"  Tf_to_marker_right: [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")

        self.log_msg(f"\n[4] Confidence Metrics:")
        self.log_msg(f"    - Orthogonality Error  : {res['ortho_err']:.3f} deg")

        rmse_warn = res['rmse_6'] > 0.5 or res['rmse_5'] > 0.5 or res.get('rmse_4', 0.0) > 0.5
        if rmse_warn:
            self.log_msg("\n" + "!"*60)
            self.log_msg(" [WARNING] Fitting RMSE exceeds 0.5 mm!")
            self.log_msg("  The marker coordinates may have high noise. Check hardware.")
            self.log_msg("!"*60)
        self.log_msg("="*50)

    def show_unified_result_marker(self):
        self.log_msg("\n" + "="*50)
        self.log_msg("       UNIFIED BRACKET CALIBRATION RESULTS")
        self.log_msg("="*50)

        if not self.marker_data_5 or not self.marker_data_6:
            self.log_msg("\n[ERROR] Missing dataset!")
            if not self.marker_data_6: self.log_msg(" -> Axis 6 Sweep (Yaw) data is missing.")
            if not self.marker_data_5: self.log_msg(" -> Axis 5 Sweep (Pitch) data is missing.")
            return

        try:
            try:
                tolerance = float(self.tolerance_input.text())
            except ValueError:
                tolerance = 0.5
            marker_data_4_val = getattr(self, 'marker_data_4', None)
            res = self.marker_calibrator.compute_unified_bracket_calibration(
                self.marker_data_5, self.marker_data_6, self.arm_side, tolerance=tolerance, marker_data_4=marker_data_4_val, calib_roll_deg=0.0, calib_pitch_deg=0.0
            )

            self.log_msg("\n[1] Cartesian Offset (EE Link Frame)")
            self.log_msg(f"    - X-Offset: {res['x_e']:.2f} mm")
            self.log_msg(f"    - Y-Offset: {res['y_e']:.2f} mm")
            self.log_msg(f"    - Z-Offset: {res['z_e']:.2f} mm")
            r4_str = f", R4: {res['radius_4']:.2f} mm" if res.get('radius_4', 0.0) > 0.0 else ""
            self.log_msg(f"       * (L_5_ee: {res['L_5_ee']:.1f} mm, R6: {res['radius_6']:.2f} mm, R5: {res['radius_5']:.2f} mm{r4_str})")
            if 'opt_delta_5' in res:
                self.log_msg(f"    - Opt Delta 5 (joint 5 offset): {res['opt_delta_5']:.3f} deg")
                self.log_msg(f"    - Opt Delta 6 (joint 6 offset): {res['opt_delta_6']:.3f} deg")
                self.log_msg(f"    - Min Circle Fitting Radius: {res['min_radius']:.2f} mm")

            self.log_msg("\n[2] Angular Misalignment (EE Link Frame)")
            self.log_msg(f"    - Roll : {res['roll_e']:.2f} deg")
            self.log_msg(f"    - Pitch: {res['pitch_e']:.2f} deg")
            self.log_msg(f"    - Yaw  : {res['yaw_e']:.2f} deg")

            self.log_msg("\n[3] setting.yaml Config Update values:")
            x_m, y_m, z_m = res['x_e']/1000.0, res['y_e']/1000.0, res['z_e']/1000.0

            if self.arm_side == "left":
                self.log_msg(f"  Tf_to_marker_left:  [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")
            else:
                self.log_msg(f"  Tf_to_marker_right: [{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, {res['roll_e']:.2f}, {res['pitch_e']:.2f}, {res['yaw_e']:.2f}]")

            # Update UI bracket design text fields
            arm_side = self.arm_side
            if arm_side == "left":
                self.txt_bracket_l_x.setText(f"{x_m:.5f}")
                self.txt_bracket_l_y.setText(f"{y_m:.5f}")
                self.txt_bracket_l_z.setText(f"{z_m:.5f}")
                self.txt_bracket_l_roll.setText(f"{res['roll_e']:.2f}")
                self.txt_bracket_l_pitch.setText(f"{res['pitch_e']:.2f}")
                self.txt_bracket_l_yaw.setText(f"{res['yaw_e']:.2f}")
            else:
                self.txt_bracket_r_x.setText(f"{x_m:.5f}")
                self.txt_bracket_r_y.setText(f"{y_m:.5f}")
                self.txt_bracket_r_z.setText(f"{z_m:.5f}")
                self.txt_bracket_r_roll.setText(f"{res['roll_e']:.2f}")
                self.txt_bracket_r_pitch.setText(f"{res['pitch_e']:.2f}")
                self.txt_bracket_r_yaw.setText(f"{res['yaw_e']:.2f}")

            # Sync to memory configs
            new_vals = [x_m, y_m, z_m, res['roll_e'], res['pitch_e'], res['yaw_e']]
            key = f"Tf_to_marker_{arm_side}"
            self.marker_calibrator.camera_config[key] = new_vals
            self.joint_calibrator.camera_config[key] = new_vals
            self.log_msg(f"[INFO] Staged calibrated nominal marker values in UI for {arm_side} arm. (Click APPLY BRACKETS to save)")

            self.log_msg(f"\n[4] Confidence Metrics:")
            self.log_msg(f"    - Orthogonality Error  : {res['ortho_err']:.3f} deg")

            rmse_warn = res['rmse_6'] > 0.5 or res['rmse_5'] > 0.5 or res.get('rmse_4', 0.0) > 0.5
            if rmse_warn:
                self.log_msg("\n" + "!"*60)
                self.log_msg(" [WARNING] Fitting RMSE exceeds 0.5 mm!")
                self.log_msg("  The marker coordinates may have high noise. Check hardware.")
                self.log_msg("!"*60)
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to calculate bracket calibration: {e}")

        self.log_msg("="*50)

    def open_plot_dialog(self):
        if not hasattr(self, 'plot_dialog') or self.plot_dialog is None:
            self.plot_dialog = PlotViewerDialog(self)
        self.plot_dialog.show()
        self.plot_dialog.raise_()
        self.plot_dialog.activateWindow()
        self.display_current_plot()

    def add_and_show_plot(self, friendly_name, file_path):
        if not file_path or not os.path.exists(file_path):
            return

        display_name = friendly_name

        # Check if file_path already in list
        existing_idx = -1
        for idx, (_, path) in enumerate(self.generated_plots):
            if path == file_path:
                existing_idx = idx
                break

        if existing_idx == -1:
            self.generated_plots.append((display_name, file_path))
            self.current_plot_idx = len(self.generated_plots) - 1
        else:
            self.current_plot_idx = existing_idx

        # Do not automatically show the plot. It will be shown only when Full Auto ends/errors, or by manual click.

    def update_navigation_buttons(self):
        self.btn_plot_prev.setEnabled(self.current_plot_idx > 0)
        self.btn_plot_next.setEnabled(self.current_plot_idx < len(self.generated_plots) - 1)

    def show_prev_plot(self):
        if self.current_plot_idx > 0:
            self.current_plot_idx -= 1
            self.display_current_plot()

    def show_next_plot(self):
        if self.current_plot_idx < len(self.generated_plots) - 1:
            self.current_plot_idx += 1
            self.display_current_plot()

    def display_current_plot(self):
        if 0 <= self.current_plot_idx < len(self.generated_plots):
            display_name, file_path = self.generated_plots[self.current_plot_idx]
            self.lbl_plot_title.setText(display_name)
            self.display_plot_image(file_path)
            self.update_navigation_buttons()
        else:
            self.lbl_plot_title.setText("No Plot Loaded")
            self.plot_label_combined.setPixmap(QPixmap())
            self.btn_plot_prev.setEnabled(False)
            self.btn_plot_next.setEnabled(False)

    def display_plot_image(self, file_path):
        if os.path.exists(file_path):
            # Scale to fit the current dialog size dynamically
            if hasattr(self, 'plot_dialog') and self.plot_dialog is not None and self.plot_dialog.isVisible():
                target_w = max(800, self.plot_dialog.width() - 40)
                target_h = max(500, self.plot_dialog.height() - 100)
            else:
                target_w = 900
                target_h = 550
            pix = QPixmap(file_path).scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.plot_label_combined.setPixmap(pix)

    # --- Head Control and Manual Operations ---
    @idle_core_action
    def move_head_manually(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        try:
            yaw = float(self.txt_head_yaw.text())
            pitch = float(self.txt_head_pitch.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Inputs", "Head angles must be valid numbers.")
            return

        self.log_msg(f"[MANUAL HEAD] Commands sent - Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°")
        if self.robot is None:
            self.log_msg("[INFO] Robot is not connected. Head movement skipped.")
            return

        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.head_worker = ManualHeadWorker(self._sync_core_state(), np.radians(yaw), np.radians(pitch))
        self.head_worker.log_signal.connect(self.log_msg)
        self.head_worker.finished_signal.connect(self.on_action_finished)
        self.head_worker.start()

    # --- Camera Intrinsics Calibration Workflows (Tab 3) ---
    def toggle_intrinsics_monitoring(self, checked):
        self.monitor_enabled = checked
        if checked:
            self.btn_int_monitor.setText("STOP MONITORING")
            self.btn_int_monitor.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold;")
        else:
            self.btn_int_monitor.setText("ENABLE MONITORING")
            self.btn_int_monitor.setStyleSheet("background-color: #1e1e1e; color: white;")
            if hasattr(self, 'lbl_marker_pos'):
                self.lbl_marker_pos.setText("Position: X: 0.0, Y: 0.0, Z: 0.0 mm")

    def update_video_frame(self):
        # Camera 서브탭(Step1 > Camera)이 활성화되어 있거나, Camera Feed 대화상자 또는 마커 미인식 대화상자, 위자드 비디오 슬라이드가 열려있을 때 업데이트
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        prob_dlg_visible = hasattr(self, 'marker_problem_dlg') and self.marker_problem_dlg is not None
        spacing_dlg = getattr(self, 'marker_spacing_dlg', None)
        camera_tab_active = self.is_camera_tab_active()
        wizard_active = hasattr(self, 'wizard_widget') and self.wizard_widget is not None and not self.wizard_widget.isHidden()
        wizard_slide_idx = self.wizard_widget.stacked_widget.currentIndex() if wizard_active else -1
        wizard_slide_mount = wizard_active and (wizard_slide_idx == self.wizard_widget.SLIDE_CAMERA_MOUNT)
        wizard_slide_exp = wizard_active and (wizard_slide_idx == self.wizard_widget.SLIDE_EXPOSURE)
        wizard_slide_calib = wizard_active and (wizard_slide_idx == self.wizard_widget.SLIDE_INTRINSICS_CALIB)

        if not camera_tab_active and not dialog_visible and not wizard_slide_mount and not wizard_slide_exp and not wizard_slide_calib and not prob_dlg_visible and spacing_dlg is None:
            return

        if not self.ui_only and self.core.observer is not None:
            snapshot = self.core.get_monitor_snapshot()
            img = snapshot["image"]

            # Sync real-time auto exposure value to UI if auto mode is enabled
            is_auto = hasattr(self, 'chk_auto_exposure') and self.chk_auto_exposure.isChecked()
            if is_auto:
                act_exp = snapshot["exposure"] or 0.0
                act_exp_int = int(round(act_exp / float(getattr(self, 'exposure_unit_us', 1.0) or 1.0)))
                # snapshot["exposure"] is the measured time in microseconds, while the controls
                # hold a setpoint in the sensor's own units. They coincide on a D405 and do not
                # on a D435, so only mirror the reading into the controls when it is a value they
                # could actually be set to -- otherwise it would be silently clamped and then
                # carried over as the manual setpoint on the next switch out of auto.
                in_range = (hasattr(self, 'spin_exposure')
                            and self.spin_exposure.minimum() <= act_exp_int <= self.spin_exposure.maximum())
                if in_range and hasattr(self, 'spin_exposure'):
                    self.spin_exposure.blockSignals(True)
                    self.spin_exposure.setValue(act_exp_int)
                    self.spin_exposure.blockSignals(False)
                if in_range and hasattr(self, 'slider_exposure'):
                    self.slider_exposure.blockSignals(True)
                    self.slider_exposure.setValue(act_exp_int)
                    self.slider_exposure.blockSignals(False)
                if hasattr(self, 'lbl_exposure_ms'):
                    self.lbl_exposure_ms.setText(f"{act_exp / 1000.0:.1f} ms (Auto)")

                # Sync Wizard exposure widgets if wizard slide 2 is active
                if wizard_slide_exp and hasattr(self.wizard_widget, 'chk_wiz_auto_exp') and self.wizard_widget.chk_wiz_auto_exp.isChecked():
                    if hasattr(self.wizard_widget, 'spin_wiz_exp'):
                        self.wizard_widget.spin_wiz_exp.blockSignals(True)
                        self.wizard_widget.spin_wiz_exp.setValue(act_exp_int)
                        self.wizard_widget.spin_wiz_exp.blockSignals(False)
                    if hasattr(self.wizard_widget, 'slider_wiz_exp'):
                        self.wizard_widget.slider_wiz_exp.blockSignals(True)
                        self.wizard_widget.slider_wiz_exp.setValue(act_exp_int)
                        self.wizard_widget.slider_wiz_exp.blockSignals(False)
                    if hasattr(self.wizard_widget, 'lbl_wiz_exp_ms'):
                        self.wizard_widget.lbl_wiz_exp_ms.setText(f"{act_exp / 1000.0:.1f} ms (Auto)")

            if img is None:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(img, "No Camera Detected", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
        else:
            # Simulation / UI-only placeholder image
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "SIMULATION / UI MODE", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 100, 100), 3)

        if img is None:
            return

        self.current_frame = img.copy()
        display_img = img.copy()

        guide_checked = (hasattr(self, 'chk_int_guide') and self.chk_int_guide.isChecked()) or (hasattr(self, 'wizard_widget') and hasattr(self.wizard_widget, 'chk_int_guide') and self.wizard_widget.chk_int_guide.isChecked())
        if guide_checked and (self.is_intrinsics_subtab_active() or wizard_slide_calib):
            num_steps = len(IntrinsicsCalibrator.CALIB_GUIDELINES)
            if self.current_guide_idx < num_steps:
                guideline = IntrinsicsCalibrator.CALIB_GUIDELINES[self.current_guide_idx]
                h, w = display_img.shape[:2]
                pts_pixel = np.array(guideline["pts"] * [w, h], dtype=np.int32)

                # Draw filled transparent guide poly
                overlay = display_img.copy()
                cv2.fillPoly(overlay, [pts_pixel], (255, 229, 0)) # Neon Cyan in BGR
                cv2.addWeighted(overlay, 0.15, display_img, 0.85, 0, display_img)

                # Draw border poly
                cv2.polylines(display_img, [pts_pixel], isClosed=True, color=(255, 229, 0), thickness=3)

                # Draw guide labels
                guide_name = guideline["name"]
                text_title = f"Guide {self.current_guide_idx + 1}/{num_steps}: {guide_name}"
                cv2.putText(display_img, text_title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_img, "Align checkerboard and press CAPTURE (C)", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(display_img, f"All {num_steps} steps captured! Press RUN CALIBRATION", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert to QImage and display
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if camera_tab_active:
            target = self.video_label if self.is_intrinsics_subtab_active() else getattr(self, 'video_label_exposure', None)
            if target is not None:
                target.setPixmap(pixmap.scaled(target.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if wizard_slide_exp and hasattr(self.wizard_widget, 'wizard_exposure_video_label'):
            self.wizard_widget.wizard_exposure_video_label.setPixmap(pixmap.scaled(self.wizard_widget.wizard_exposure_video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if (wizard_slide_mount or wizard_slide_calib) and hasattr(self.wizard_widget, 'wizard_video_label'):
            self.wizard_widget.wizard_video_label.setPixmap(pixmap.scaled(self.wizard_widget.wizard_video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if dialog_visible:
            w_lbl = max(20, self.feed_dialog.lbl_feed.width())
            h_lbl = max(20, self.feed_dialog.lbl_feed.height())
            self.feed_dialog.lbl_feed.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))
        if prob_dlg_visible and hasattr(self.marker_problem_dlg, 'lbl_live_feed'):
            w_lbl = max(20, self.marker_problem_dlg.lbl_live_feed.width())
            h_lbl = max(20, self.marker_problem_dlg.lbl_live_feed.height())
            scaled = pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation)
            # Centre dot the operator brings the marker to (the teaching guide refers to it).
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor("#00e5ff"), 2))
            centre = scaled.rect().center()
            painter.drawEllipse(centre, 7, 7)
            painter.drawLine(centre.x() - 20, centre.y(), centre.x() - 11, centre.y())
            painter.drawLine(centre.x() + 11, centre.y(), centre.x() + 20, centre.y())
            painter.drawLine(centre.x(), centre.y() - 20, centre.x(), centre.y() - 11)
            painter.drawLine(centre.x(), centre.y() + 11, centre.x(), centre.y() + 20)
            painter.end()
            self.marker_problem_dlg.lbl_live_feed.setPixmap(scaled)
        if spacing_dlg is not None:
            w_lbl = max(20, spacing_dlg.lbl_live_feed.width())
            h_lbl = max(20, spacing_dlg.lbl_live_feed.height())
            spacing_dlg.lbl_live_feed.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))

    def keyPressEvent(self, event):
        intrinsics_tab_active = self.is_intrinsics_subtab_active()
        wizard_slide4_active = (hasattr(self, 'wizard_widget') and self.wizard_widget.isVisible() and self.wizard_widget.stacked_widget.currentIndex() == self.wizard_widget.SLIDE_INTRINSICS_CALIB)
        if event.key() == Qt.Key_C and (intrinsics_tab_active or wizard_slide4_active):
            self.capture_intrinsics_frame()
        super().keyPressEvent(event)

    def capture_intrinsics_frame(self):
        # The C key reaches here even while the buttons are locked.
        if self.intrinsics_busy:
            return
        if hasattr(self, 'current_frame'):
            self.captured_images.append(self.current_frame.copy())
            frames = len(self.captured_images)
            self.lbl_captured.setText(f"Captured Frames: {frames}")
            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                if hasattr(self.wizard_widget, 'lbl_captured') and self.wizard_widget.lbl_captured is not None:
                    self.wizard_widget.set_status(self.wizard_widget.lbl_captured, "wizard.st.int_captured", frames=frames)
            self.log_msg(f"[INTRINSICS] Frame {frames} captured.")

            num_steps = len(IntrinsicsCalibrator.CALIB_GUIDELINES)
            if hasattr(self, 'chk_int_guide') and self.chk_int_guide.isChecked() and self.current_guide_idx < num_steps:
                self.current_guide_idx += 1
                if self.current_guide_idx == num_steps:
                    self.log_msg(f"[INTRINSICS] All {num_steps} guided frames captured! You can now run calibration.")

            if frames == 16:
                self.log_msg("[INTRINSICS] 16 frames collected! Automatically running calibration...")
                QApplication.processEvents()
                self.run_intrinsics_calibration()

    def reset_intrinsics_captures(self):
        self.captured_images.clear()
        self.current_guide_idx = 0
        self.lbl_captured.setText(f"Captured Frames: 0")
        self.btn_int_save.setEnabled(False)
        self.log_msg("[INTRINSICS] Capture memory cleared.")

    def on_guide_changed(self, state):
        checked = (state == Qt.Checked or state == 2)
        self.log_msg(f"[INTRINSICS] Guidance overlay {'ENABLED' if checked else 'DISABLED'}.")
        if checked:
            num_steps = len(IntrinsicsCalibrator.CALIB_GUIDELINES)
            self.current_guide_idx = min(num_steps, len(self.captured_images))

    @idle_core_action
    def run_intrinsics_calibration(self):
        if len(self.captured_images) < 16:
            self.log_msg("[ERROR] Need all 16 valid frames to run calibration!")
            QMessageBox.warning(self, "Insufficient Data", f"Cannot run calibration: Only {len(self.captured_images)} / 16 frames collected.\nPlease capture all 16 frames first.")
            return

        self.log_msg(f"\n[INTRINSICS] Running calibration on {len(self.captured_images)} images. Please wait...")

        self._sync_core_state()
        self.intrinsics_worker = SequenceWorker(self.core, "intrinsics", {
            "images": [image.copy() for image in self.captured_images]})
        self.intrinsics_worker.finished_result.connect(self._on_intrinsics_finished)
        self.intrinsics_worker.start()
        self.set_intrinsics_busy(True)

    def set_intrinsics_busy(self, busy):
        """Lock the intrinsics buttons and show the busy indicators while the solve runs in its worker.

        The buttons get back the state they had before the run, so Save stays off until a run succeeded."""
        panels = [self]
        if getattr(self, 'wizard_widget', None) is not None:
            panels.append(self.wizard_widget)
            self.wizard_widget.lbl_step1_status.setVisible(not busy)
        if busy:
            self._intrinsics_button_states = {}
        for panel in panels:
            panel.int_busy.setVisible(busy)
            for name in ("btn_int_capture", "btn_int_calibrate", "btn_int_save", "btn_int_reset"):
                button = getattr(panel, name)
                if busy:
                    self._intrinsics_button_states[button] = button.isEnabled()
                    button.setEnabled(False)
                else:
                    button.setEnabled(self._intrinsics_button_states.get(button, button.isEnabled()))
        self.intrinsics_busy = busy

    def _on_intrinsics_finished(self, result):
        self.set_intrinsics_busy(False)
        success = result.success
        if success:
            self.log_msg(f"[SUCCESS] Calibration complete! RMS Error: {self.intrinsics_calibrator.rms_error:.4f}")
            self.log_msg("[INTRINSICS] Parameter Standard Error (Uncertainty):")
            self.log_msg(f"  * Focal Length fx: {self.intrinsics_calibrator.std_fx:.4f} pixels")
            self.log_msg(f"  * Focal Length fy: {self.intrinsics_calibrator.std_fy:.4f} pixels")
            self.log_msg(f"  * Principal Point cx: {self.intrinsics_calibrator.std_cx:.4f} pixels")
            self.log_msg(f"  * Principal Point cy: {self.intrinsics_calibrator.std_cy:.4f} pixels")

            if self.intrinsics_calibrator.test_rmse is not None:
                self.log_msg(f"[INTRINSICS] Cross-Validation Test RMSE: {self.intrinsics_calibrator.test_rmse:.4f} pixels")
                if self.intrinsics_calibrator.test_rmse < 0.18:
                    self.log_msg("[INTRINSICS] Generalization check: EXCELLENT (low variance, high stability)")
                else:
                    self.log_msg("[INTRINSICS] Generalization check: WARNING (high variance, check board angles)")
            else:
                self.log_msg("[INTRINSICS] Cross-Validation: Not enough frames (min 6 frames needed)")

            self.log_msg("[INTRINSICS] Click 'SAVE PARAMETERS' to apply changes.")
            self.btn_int_save.setEnabled(True)
            self.show_intrinsics_verification()
            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                if hasattr(self.wizard_widget, 'lbl_step1_status') and self.wizard_widget.lbl_step1_status is not None:
                    self.wizard_widget.set_status(self.wizard_widget.lbl_step1_status, "wizard.st.int_ok",
                                                  "color: #ff9800; font-weight: bold; font-size: 16px;",
                                                  rms=f"{self.intrinsics_calibrator.rms_error:.4f}")
        else:
            self.log_msg("[ERROR] Calibration failed. Check images and board settings.")
            QMessageBox.critical(self, "Calibration Failed", "Calibration failed! Check board visibility and image quality.")
            if hasattr(self, 'wizard_widget') and self.wizard_widget is not None:
                if hasattr(self.wizard_widget, 'lbl_step1_status') and self.wizard_widget.lbl_step1_status is not None:
                    self.wizard_widget.set_status(self.wizard_widget.lbl_step1_status, "wizard.st.int_failed",
                                                  "color: #f44336; font-weight: bold; font-size: 16px;")

    def save_intrinsics_calibration(self):
        if len(self.captured_images) < 16:
            self.log_msg("[ERROR] Need all 16 frames to save parameters!")
            QMessageBox.warning(self, "Cannot Save", f"Cannot save parameters: Only {len(self.captured_images)} / 16 frames collected.")
            return

        if self.intrinsics_calibrator.cameraMatrix is None or float(self.intrinsics_calibrator.rms_error) <= 0.0:
            self.log_msg("[ERROR] No valid calibration data to save!")
            QMessageBox.warning(self, "Cannot Save", "Calibration has not been successfully executed yet.")
            return

        try:
            camera_model = getattr(self.core.observer, 'camera_model', "")
            data = {
                "device_name": camera_model,
                "camera_matrix": self.intrinsics_calibrator.cameraMatrix.tolist(),
                "dist_coeffs": self.intrinsics_calibrator.distCoeffs.flatten().tolist(),
                "rms_error": float(self.intrinsics_calibrator.rms_error),
                "width": int(self.captured_images[0].shape[1]),
                "height": int(self.captured_images[0].shape[0])
            }
            # Working file + this model's store (swapping cameras and coming back picks it up again)
            # + this unit's own serial file, which is always overwritten: the serial file wins at
            # the next start, so an old one would undo this calibration (2026-09-30).
            serial = getattr(self.core.observer, 'camera_serial', None) if self.core.observer is not None else None
            FileStorage.ensure_dir(os.path.dirname(self.output_yaml), exist_ok=True)
            for path in save_camera_intrinsics(data, family=camera_model or None, serial=serial):
                self.log_msg(f"[SUCCESS] Intrinsic parameters saved to: {path}")

            # Load what was just saved into the running detector. It used to take effect only on
            # the next start, so calibrating, saving and carrying straight on in the same session
            # ran every sweep on the old intrinsics while the file already held the new ones.
            # The core re-reads the files as a reconnect does: the status used to be written onto
            # core.observer, a wrapper, so the engine went on reporting "not registered" (2026-10-08).
            try:
                if self.core.reload_camera_intrinsics():
                    self.log_msg("[SUCCESS] New intrinsics are now in use by the camera (no restart needed).")
                else:
                    self.log_msg("[WARNING] Intrinsics saved, but not applied to a running camera "
                                 "(no camera connected). They load on the next camera connection.")
            except Exception as e:
                self.log_msg(f"[WARNING] Intrinsics saved, but could not be applied now ({e}). "
                             f"Reconnect the camera or restart before calibrating.")
            self.refresh_camera_tab_info()

            # Show save success message box
            self.show_message_box(
                tr("dialogs.intrinsics_save_complete.title"),
                tr("dialogs.intrinsics_save_complete.msg", path=self.output_yaml)
            )
        except Exception as e:
            self.log_msg(f"[ERROR] Save failed: {e}")

    def show_intrinsics_verification(self):
        if len(self.captured_images) == 0:
            return

        test_img = self.captured_images[-1]
        save_path = os.path.join(CONFIG_PATHS["plot_dir"], "camera_intrinsics_verification.png")

        # Delegate image generation to IntrinsicsCalibrator
        self.intrinsics_calibrator.generate_verification_image(test_img, save_path)

        # Load inside Plot viewer dialog history
        self.add_and_show_plot("[INTRINSICS] Verification Image", save_path)

        # Pop up visual verification dialog directly
        if os.path.exists(save_path):
            dialog = QDialog(self)
            dialog.setWindowTitle("Camera Intrinsics Calibration Verification (Original vs Undistorted)")
            dialog.setStyleSheet(DARK_STYLESHEET)

            main_layout = QVBoxLayout(dialog)

            # Left side: Image
            pixmap = QPixmap(save_path)
            scaled_pix = pixmap.scaled(1000, 750, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label = QLabel()
            img_label.setPixmap(scaled_pix)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border: 2px solid #2d2d2d; border-radius: 6px;")
            main_layout.addWidget(img_label)

            btn_close = QPushButton("CLOSE")
            btn_close.setMinimumHeight(40)
            btn_close.setStyleSheet("background-color: #37474f; color: white; font-weight: bold; font-size: 13px;")
            btn_close.clicked.connect(dialog.accept)
            main_layout.addWidget(btn_close)

            dialog.exec()
            self.log_msg(f"[INTRINSICS] Verification dialog shown. Image saved to: {save_path}")
        else:
            self.log_msg(f"[ERROR] Failed to generate verification image at: {save_path}")

    def closeEvent(self, event):
        if hasattr(self, 'feed_dialog') and self.feed_dialog is not None:
            try:
                self.feed_dialog.close()
            except Exception:
                pass
        if self.video_timer.isActive():
            self.video_timer.stop()
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        if hasattr(self, 'temp_timer') and self.temp_timer.isActive():
            self.temp_timer.stop()

        try:
            self.core.close()
        except RuntimeError as error:
            self.log_msg(str(error))
            event.ignore()
            return
        workers = (list(self.findChildren(QThread))
                   + [v for v in vars(self).values() if isinstance(v, QThread)]
                   + SequenceWorker.active_for_core(self.core))
        for worker in workers:
            if worker.isRunning() and not worker.wait(2000):
                event.ignore()
                return
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Unified Robot Calibration Suite GUI")
    parser.add_argument("--ui", action="store_true", help="Start only UI for debugging/simulation")
    args = parser.parse_args()

    # Enable High-DPI scaling and crisp pixmaps
    # Note: HighDPI scaling is natively enabled in Qt6. We guard against DeprecationWarning.
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)

    # Bundled Korean font (SIL OFL, config/fonts/OFL.txt): Linux PCs without a CJK font showed
    # Hangul as boxes. Families named in code fall back to it when the PC lacks them.
    for font_file in ("NotoSansKR-Regular.otf", "NotoSansKR-Bold.otf"):
        if QFontDatabase.addApplicationFont(get_asset_path(f"config/fonts/{font_file}")) < 0:
            print(f"[WARN] Could not load the bundled font config/fonts/{font_file}; Korean text may show as boxes.")
    for family in ("Segoe UI", "Consolas", "Noto Sans CJK KR", "Noto Sans Mono CJK KR"):
        QFont.insertSubstitution(family, "Noto Sans KR")

    # Global crisp Korean/English font with subpixel anti-aliasing
    default_font = QFont("Noto Sans KR", 10)
    default_font.setStyleStrategy(QFont.PreferAntialias | QFont.PreferQuality)
    app.setFont(default_font)

    robot = None
    core = CalibrationCore()

    camera_error = None
    if not args.ui:
        print("[INFO] Initializing Camera Marker Transform System...")
        try:
            core.connect_camera()
        except Exception as e:
            print(f"[ERROR] Failed to load camera marker system: {e}")
            print("[INFO] Fallback to UI-only mode.")
            camera_error = str(e)
            args.ui = True
    else:
        print("[INFO] Starting in simulation (UI-only) mode.")

    gui = UnifiedCalibrationApp(core, robot, "right", ui_only=args.ui)
    gui.show()
    # Windows' default placement left the bottom of the window off a 1920x1200 screen at 125 %.
    # Centre the frame on the monitor's work area; if it is taller, keep the title bar on screen.
    area = gui.screen().availableGeometry()
    frame = gui.frameGeometry()
    frame.moveCenter(area.center())
    gui.move(max(frame.left(), area.left()), max(frame.top(), area.top()))
    # Opens maximized (2026-11-12, user) so the wizard slides fit without scrolling; un-maximizing
    # returns to the centred window above.
    gui.showMaximized()
    if camera_error:
        # Used to go to the console only, so the window opened in UI-only mode with no word of
        # why -- including when the fix is simply a USB 3 cable.
        QMessageBox.critical(gui, "Camera Connection Failed",
                             f"Could not connect to RealSense camera:\n{camera_error}\n\n"
                             "Starting in UI-only mode.")

    try:
        sys.exit(app.exec())
    finally:
        if core:
            try:
                core.close()
                print("Camera resource released.")
            except Exception:
                pass

if __name__ == "__main__":
    main()
