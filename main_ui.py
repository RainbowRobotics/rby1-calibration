import sys
import os
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
import cv2
import numpy as np
import time
import argparse
import logging
import rby1_sdk as rby
import threading
import yaml
import traceback
import json
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QGroupBox, QComboBox, QCheckBox, 
                             QLineEdit, QDialog, QMessageBox, QTabWidget, QInputDialog, QGridLayout,
                             QTableWidget, QHeaderView, QTableWidgetItem, QSizePolicy, QRadioButton, QStackedWidget, QButtonGroup)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from pathlib import Path

# Import custom calibrator logic
from marker_detection import Marker_Detection, Marker_Transform
from calibration.Calibrator import MarkerCalibrator, JointCalibrator, BaseCalibrator
from calibration.IntrinsicsCalibrator import IntrinsicsCalibrator
from core.wizard_widget import CalibrationWizardWidget
from homeoffset_core import (
    reset_current_pose_home_offsets,
    save_home_reset_baseline_json,
    move_to_offset_candidate_from_json,
    load_offset_from_json,
    move_robot_to_zero_pose
)

try:
    from core.calibration_core import (
        capture_one_sample as capture_robot_sample,
        get_arm_config,
        get_both_arm_config,
        get_head_config,
        load_npz_dataset,
        save_npz_dataset,
        validate_dataset,
        check_calibration_state,
    )
    from core.calibration_optimizer import (
        DEFAULT_LAMBDA_CAM_POS,
        DEFAULT_LAMBDA_CAM_ROT,
        CalibrationOptimizer,
        QPCalibrationOptimizer,
    )
    from core.robot_motion import (
        AutoCollectionConfig,
        build_incremental_motion_plan,
        move_to_auto_ready_pose,
        execute_auto_motion_step,
        reset_motion_state,
    )
except ImportError:
    from calibration_core import (
        capture_one_sample as capture_robot_sample,
        get_arm_config,
        get_both_arm_config,
        get_head_config,
        load_npz_dataset,
        save_npz_dataset,
        validate_dataset,
        check_calibration_state,
    )
    from calibration_optimizer import (
        DEFAULT_LAMBDA_CAM_POS,
        DEFAULT_LAMBDA_CAM_ROT,
        CalibrationOptimizer,
        QPCalibrationOptimizer,
    )
    from robot_motion import (
        AutoCollectionConfig,
        build_incremental_motion_plan,
        move_to_auto_ready_pose,
        execute_auto_motion_step,
        reset_motion_state,
    )
current_dir = os.path.dirname(os.path.abspath(__file__))
calibration_dir = os.path.abspath(os.path.join(current_dir,"core","calibration"))
# --- Configuration & Paths ---
from core.paths import CONFIG_PATHS

UI_DROPDOWNS = {
    "robot_models": ["a", "m"],
    "arm_sides": ["Right Arm", "Left Arm"],
    "marker_axes": ["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"],
    "joint_modes_v13": ["wrist_roll_v13 (6-Axis Sweep)", "wrist_pitch_v13 (5-Axis Sweep)", "elbow (3-Axis Sweep)"],
    "joint_modes_v12": ["wrist_yaw2 (6-Axis Sweep)", "wrist_pitch (5-Axis Sweep)", "elbow (3-Axis Sweep)"]
}

# --- Premium Dark CSS Stylesheet ---
DARK_STYLESHEET = """
QWidget {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Malgun Gothic', Arial, sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 2px solid #2d2d2d;
    border-radius: 8px;
    margin-top: 15px;
    font-weight: bold;
    font-size: 13px;
    color: #2979ff;
    padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 5px;
}
QPushButton {
    background-color: #1e1e1e;
    color: #ffffff;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2c2c2c;
    border: 1px solid #2979ff;
}
QPushButton:pressed {
    background-color: #121212;
}
QPushButton:disabled {
    background-color: #1a1a1a;
    border: 1px solid #242424;
    color: #555555;
}
QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
    min-width: 120px;
}
QComboBox::drop-down {
    border: none;
}
QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
}
QTabWidget::pane {
    border: 1px solid #2d2d2d;
    background: #121212;
    border-radius: 6px;
}
QTabBar::tab {
    background: #1a1a1a;
    border: 1px solid #2d2d2d;
    border-bottom: none;
    padding: 8px 16px;
    font-weight: bold;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #888888;
}
QTabBar::tab:selected {
    background: #121212;
    color: #2979ff;
    border-bottom: 2px solid #2979ff;
}
QTabBar::tab:hover:!selected {
    background: #252525;
    color: #e0e0e0;
}
QCheckBox {
    spacing: 8px;
    font-weight: bold;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    background-color: #1e1e1e;
}
QCheckBox::indicator:checked {
    background-color: #2979ff;
    border: 1px solid #2979ff;
}
QTextEdit {
    background-color: #0e0e0e;
    color: #00e676;
    border: 2px solid #2d2d2d;
    border-radius: 6px;
}
"""

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
        
        layout = QVBoxLayout(self)
        self.lbl_feed = QLabel("Waiting for camera frame...")
        self.lbl_feed.setAlignment(Qt.AlignCenter)
        self.lbl_feed.setStyleSheet("background-color: black; color: white; border: 1px solid #2d2d2d; border-radius: 4px;")
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
        self.btn_prev.setFixedSize(40, 30)
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
            }
            QPushButton:disabled {
                background-color: #222222;
                color: #555555;
                border-color: #333333;
            }
        """)
        
        self.lbl_title = QLabel("No Plot Loaded")
        self.lbl_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("""
            background-color: #252525;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
        """)
        
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedSize(40, 30)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
            }
            QPushButton:disabled {
                background-color: #222222;
                color: #555555;
                border-color: #333333;
            }
        """)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_title, 1)
        nav_layout.addWidget(self.btn_next)
        
        self.plot_label = QLabel("No plots to display")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setStyleSheet("background-color: #1a1a1a; color: #888888; border: 2px solid #2d2d2d; border-radius: 8px;")
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
        self.setWindowTitle("Zero Pose Check")
        self.resize(760, 800)
        self.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        msg = (
            "The robot has moved to zero pose.\n\n"
            "Please compare the actual robot posture with the reference image.\n\n"
            "- If the posture matches the reference, you can proceed with data collection.\n"
            "- If the posture does not match and the two target joints appear outside the recommended range,\n"
            "  use direct teaching to move the robot to the recommended posture,\n"
            "  perform reset first, and then start data collection."
        )
        msg_lbl = QLabel(msg)
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl)
        
        # Load warning pose check images
        for path_name in ["warning_pose_check.png", "warning_pose.png"]:
            img_path = os.path.join(current_dir, path_name)
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

class ApplyHomeOffsetDialog(QDialog):
    def __init__(self, parent, result_path, baseline_path, arm, include_head, compare_summary=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Home Offset")
        self.resize(900, 600)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.parent_app = parent
        self.result_path = result_path
        self.baseline_path = baseline_path
        self.arm = arm
        self.include_head = include_head
        
        self.current_apply_arm = parent.infer_home_offset_apply_arm(arm, result_path)
        
        if compare_summary is None:
            compare_summary = parent.format_home_offset_compare_summary(result_path, baseline_path)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        msg = (
            "Compare the original baseline zero and the optimized zero before applying.\n\n"
            "1. Select Baseline or Optimized state.\n"
            "2. Move to Zero to inspect the zero pose before calibration reset.\n"
            "3. Move to Check Position to move the robot to the custom check pose.\n"
            "4. Apply the pose you want to keep using Rollback or Apply Optimized Result.\n\n"
            "Make sure the workspace is clear before each move."
        )
        msg_lbl = QLabel(msg)
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl)
        
        # Summary text box
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setText(compare_summary)
        self.summary_box.setFont(QFont("Consolas", 10))
        layout.addWidget(self.summary_box, 1)
        
        # State Switcher (Big Toggle Buttons)
        state_layout = QHBoxLayout()
        self.btn_group = QButtonGroup(self)
        
        self.btn_baseline = QPushButton("BASELINE\n(Rollback)")
        self.btn_opt = QPushButton("OPTIMIZED\n(Apply)")
        
        btn_style = """
        QPushButton {
            font-size: 16px;
            font-weight: bold;
            color: #aaaaaa;
            background-color: #333333;
            border: 2px solid #444444;
            border-radius: 8px;
        }
        QPushButton:checked {
            color: #ffffff;
            background-color: #d84315;
            border: 3px solid #ff9800;
        }
        QPushButton:disabled {
            background-color: #222222;
            color: #555555;
            border: 2px solid #333333;
        }
        """
        
        for btn in [self.btn_baseline, self.btn_opt]:
            btn.setCheckable(True)
            btn.setMinimumHeight(60)
            btn.setStyleSheet(btn_style)
            self.btn_group.addButton(btn)
            state_layout.addWidget(btn)
            
        self.btn_baseline.setChecked(True)
        
        if result_path is None or not os.path.exists(result_path):
            self.btn_opt.setEnabled(False)
            
        layout.addLayout(state_layout)
        
        # Movement Buttons
        move_layout = QHBoxLayout()
        self.btn_move_zero = QPushButton("Move to Zero")
        self.btn_move_zero.clicked.connect(self.on_move_zero)
        move_layout.addWidget(self.btn_move_zero)
        
        self.btn_move_check = QPushButton("Move to Check")
        self.btn_move_check.clicked.connect(self.on_move_check)
        move_layout.addWidget(self.btn_move_check)
        layout.addLayout(move_layout)
        
        # Action buttons row
        btn_layout = QHBoxLayout()
        
        self.btn_apply = QPushButton("Apply Selected Offset")
        self.btn_apply.setStyleSheet("background-color: #d84315; color: white; font-weight: bold; font-size: 16px; padding: 10px;")
        self.btn_apply.clicked.connect(self.on_apply_selected)
        
        if result_path is None or not os.path.exists(result_path):
            self.btn_apply.setEnabled(False)
            
        btn_layout.addWidget(self.btn_apply)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
    def on_apply_selected(self):
        state, path = self.get_current_target()
        
        # Add confirmation popup
        confirm = QMessageBox.question(
            self, 
            "Confirm Apply", 
            f"Are you sure you want to apply the '{state.upper()}' offsets?\n\nThis will write to the robot's configuration.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            self.on_apply(state)
        
    def get_current_target(self):
        if self.btn_baseline.isChecked():
            return "baseline", self.baseline_path
        else:
            return "optimized", self.result_path

    def on_move_zero(self):
        state, path = self.get_current_target()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"No {state} JSON found.")
            return
            
        self.set_buttons_enabled(False)
        self.worker = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_zero",
            json_path=path,
            label=f"{state.capitalize()} Zero",
            arm=self.arm,
            include_head=self.include_head
        )
        self.worker.log_signal.connect(self.parent_app.log_msg)
        def on_finished(success, error_msg, res):
            self.set_buttons_enabled(True)
            if success:
                self.current_apply_arm = res["arm"]
                QMessageBox.information(self, "Preview Complete", f"Moved to {state} zero candidate.")
            else:
                QMessageBox.critical(self, "Preview Error", error_msg)
        self.worker.finished_signal.connect(on_finished)
        self.worker.start()

    def on_move_check(self):
        state, path = self.get_current_target()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"No {state} JSON found.")
            return

        self.set_buttons_enabled(False)
        self.worker = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_check",
            json_path=path,
            label=f"{state.capitalize()} Check Position",
            arm=self.arm,
            include_head=self.include_head
        )
        self.worker.log_signal.connect(self.parent_app.log_msg)
        def on_finished(success, error_msg, res):
            self.set_buttons_enabled(True)
            if success:
                self.current_apply_arm = res["arm"]
                QMessageBox.information(self, "Preview Complete", f"Moved to {state} check position candidate.")
            else:
                QMessageBox.critical(self, "Check Preview Error", error_msg)
        self.worker.finished_signal.connect(on_finished)
        self.worker.start()

    def on_apply(self, state):
        msg = (
            f"This will redefine the selected joints' home offset using the robot's CURRENT pose as the new {state}.\n\n"
            "Only continue if the robot is currently at the zero pose you want to keep."
        )
        if QMessageBox.question(self, f"Apply {state.capitalize()} Pose", msg, QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        
        self.set_buttons_enabled(False)
        self.worker = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "apply",
            arm=self.current_apply_arm,
            include_head=self.include_head
        )
        self.worker.log_signal.connect(self.parent_app.log_msg)
        def on_finished(success, error_msg, res):
            self.set_buttons_enabled(True)
            if success:
                if res.get("needs_reconnect", False):
                    self.parent_app.log_msg("Re-connecting and initializing robot...")
                    self.parent_app.connect_robot()
                    self.parent_app.log_msg("Current pose home offset apply complete.")
                    
                if res.get("success", False) or res.get("needs_reconnect", False):
                    QMessageBox.information(self, "Success", f"Home offset applied from current pose ({state}).")
                    self.accept()
                else:
                    QMessageBox.warning(self, "Warning", "Home offset apply finished, but some joints failed to reset. Please check the logs.")
            else:
                QMessageBox.critical(self, "Apply Pose Error", error_msg)
        self.worker.finished_signal.connect(on_finished)
        self.worker.start()

    def set_buttons_enabled(self, enabled):
        self.btn_move_zero.setEnabled(enabled)
        self.btn_move_check.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        if self.result_path is not None and os.path.exists(self.result_path):
            self.btn_opt.setEnabled(enabled)
        self.btn_baseline.setEnabled(enabled)

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                                 QLabel, QStackedWidget, QGroupBox, QCheckBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap

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
        self.x_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        grid.addWidget(self.x_input, 0, 1)
        
        grid.addWidget(QLabel("Y Position (m):"), 1, 0)
        self.y_input = QLineEdit("0.0")
        self.y_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        grid.addWidget(self.y_input, 1, 1)
        
        grid.addWidget(QLabel("Z Position (m):"), 2, 0)
        self.z_input = QLineEdit("0.0")
        self.z_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        grid.addWidget(self.z_input, 2, 1)
        
        grid.addWidget(QLabel("Y Offset (m):"), 3, 0)
        self.offset_input = QLineEdit("0.175")
        self.offset_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
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
            robot=self.parent_app.robot,
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
            robot=self.parent_app.robot,
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

class MoveToReadyWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, calibrator, arm_side, mode=None):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode

    def run(self):
        if self.mode is not None:
            # Joint Calibrator
            self.calibrator.perform_move_to_ready_pose(self.arm_side, self.mode, log_callback=self.log_signal.emit)
        else:
            # Marker Calibrator
            self.calibrator.perform_move_to_ready_pose(self.arm_side, log_callback=self.log_signal.emit)
        self.finished_signal.emit()

class ManualHeadWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, calibrator, yaw_rad, pitch_rad):
        super().__init__()
        self.calibrator = calibrator
        self.yaw_rad = yaw_rad
        self.pitch_rad = pitch_rad

    def run(self):
        try:
            ok = self.calibrator.movej(
                self.calibrator.robot, 
                head=np.array([self.yaw_rad, self.pitch_rad]), 
                minimum_time=1.5
            )
            if ok:
                self.log_signal.emit("[MANUAL HEAD] Move head completed successfully.")
            else:
                self.log_signal.emit("[ERROR] Failed manual head move: command rejected by robot.")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Failed manual head move: {e}")
        self.finished_signal.emit()

class HomeOffsetResetWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, robot, model, model_name, include_head):
        super().__init__()
        self.robot = robot
        self.model = model
        self.model_name = model_name
        self.include_head = include_head

    def run(self):
        try:
            is_mock = (self.robot is None or self.robot == "mock_robot")
            if is_mock:
                self.log_signal.emit("[MOCK] Home offset baseline save simulated.")
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Home offset reset simulated successfully.")
                self.finished_signal.emit({"success": True})
                return

            config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))
            baseline_path, _ = save_home_reset_baseline_json(
                self.robot,
                self.model,
                config_dir,
                model_name=self.model_name,
                include_head=self.include_head,
            )
            self.log_signal.emit(f"Home reset baseline saved to: {baseline_path}")

            result = reset_current_pose_home_offsets(
                self.robot,
                self.model,
                arm="both",
                include_head=self.include_head,
                log_cb=self.log_signal.emit,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Home Offset Reset worker error: {e}")
            self.finished_signal.emit({"success": False, "error": str(e)})

class MoveHomeOffsetWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool)

    def __init__(self, robot, model, arm, json_path, include_head, label):
        super().__init__()
        self.robot = robot
        self.model = model
        self.arm = arm
        self.json_path = json_path
        self.include_head = include_head
        self.label = label

    def run(self):
        try:
            is_mock = (self.robot is None or self.robot == "mock_robot")
            if is_mock:
                self.log_signal.emit(f"\n[MOCK] ===== HOME OFFSET PREVIEW: {self.label} =====")
                self.log_signal.emit(f"[MOCK] JSON: {self.json_path}")
                self.log_signal.emit(f"[MOCK] Arm: {self.arm}")
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Preview move complete.")
                self.finished_signal.emit(True)
                return

            self.log_signal.emit(f"\n===== HOME OFFSET PREVIEW: {self.label} =====")
            self.log_signal.emit(f"JSON: {self.json_path}")
            
            result = move_to_offset_candidate_from_json(
                robot=self.robot,
                model=self.model,
                arm=self.arm,
                json_path=str(self.json_path),
                include_head=self.include_head,
                minimum_time=10,
                move_zero_first=True,
            )
            self.log_signal.emit(f"Arm: {result['arm']}")
            if result.get("right_offset_deg") is not None:
                self.log_signal.emit(f"Right move offset (deg): {result['right_offset_deg']}")
            if result.get("left_offset_deg") is not None:
                self.log_signal.emit(f"Left move offset (deg): {result['left_offset_deg']}")
            if result.get("head_offset_deg") is not None:
                self.log_signal.emit(f"Head move offset (deg): {result['head_offset_deg']}")
            self.log_signal.emit("Preview move complete. Inspect the robot pose before applying.")
            self.finished_signal.emit(True)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Preview move failed: {e}")
            self.finished_signal.emit(False)

class ApplyCurrentPoseWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, robot, model, arm, include_head):
        super().__init__()
        self.robot = robot
        self.model = model
        self.arm = arm
        self.include_head = include_head

    def run(self):
        try:
            is_mock = (self.robot is None or self.robot == "mock_robot")
            if is_mock:
                self.log_signal.emit("[MOCK] Starting Home Offset Reset from current pose...")
                time.sleep(1.0)
                self.log_signal.emit("[MOCK] Current pose home offset apply complete.")
                self.finished_signal.emit({"success": True})
                return

            self.log_signal.emit("Starting Home Offset Reset from current pose...")
            result = reset_current_pose_home_offsets(
                self.robot,
                self.model,
                arm=self.arm,
                include_head=self.include_head,
                log_cb=self.log_signal.emit,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Apply current pose failed: {e}")
            self.finished_signal.emit({"success": False, "error": str(e)})

class Step2InitPoseWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, robot, active_arms, priority, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.active_arms = active_arms
        self.priority = priority

    def run(self):
        try:
            from core.robot_motion import move_to_auto_ready_pose
            move_to_auto_ready_pose(
                robot=self.robot,
                active_arms=self.active_arms,
                minimum_time=10.0,
                priority=self.priority,
            )
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class Step2AutoMotionWorker(QThread):
    log_signal = Signal(str)
    sample_signal = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app

    def run(self):
        try:
            pose_target = self.app.get_auto_pose_target_count()
            while self.app.head_move_count < pose_target:
                if self.app.auto_stop_requested:
                    self.log_signal.emit("Auto Motion stopped by user.")
                    break

                ok = self.app.run_auto_motion_step_blocking()
                if not ok:
                    if self.app.auto_stop_requested:
                        self.log_signal.emit("Auto Motion stopped by user.")
                        break
                    self.log_signal.emit("Step capture failed and skipped. Continuing sequence...")

                self.sample_signal.emit(self.app.head_move_count)
                time.sleep(0.2)
            else:
                self.log_signal.emit("Auto motions completed.")
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class Step2ZeroPoseCheckWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, robot, model, arm, include_head):
        super().__init__()
        self.robot = robot
        self.model = model
        self.arm = arm
        self.include_head = include_head

    def run(self):
        try:
            from core.homeoffset_core import movej
            
            right_zero_pose = np.zeros(len(self.model.right_arm_idx))
            left_zero_pose = np.zeros(len(self.model.left_arm_idx))
            head_zero_pose = np.zeros(len(self.model.head_idx)) if self.include_head else None

            self.log_signal.emit("Moving robot to zero pose...")
            ok = movej(
                self.robot,
                right_arm=right_zero_pose,
                left_arm=left_zero_pose,
                head=head_zero_pose,
                minimum_time=5,
            )
            if not ok:
                raise RuntimeError("Failed to move robot to zero pose")
                
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class Step2ApplyHomeOffsetWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str, dict)

    def __init__(self, app, task_type, **kwargs):
        super().__init__()
        self.app = app
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == "move_zero":
                res = self.app.move_home_offset_candidate_path(
                    self.kwargs["json_path"],
                    self.kwargs["label"],
                    self.kwargs["arm"],
                    self.kwargs["include_head"]
                )
                self.finished_signal.emit(True, "", res)
            elif self.task_type == "move_check":
                res = self.app.move_to_check_position_candidate_path(
                    self.kwargs["json_path"],
                    self.kwargs["label"],
                    self.kwargs["arm"],
                    self.kwargs["include_head"]
                )
                self.finished_signal.emit(True, "", res)
            elif self.task_type == "apply":
                res = self.app.apply_current_pose_home_offset(
                    self.kwargs["arm"],
                    self.kwargs["include_head"]
                )
                self.finished_signal.emit(True, "", res)
        except Exception as e:
            self.finished_signal.emit(False, str(e), {})

class CheckCalibrationStateWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, task_type, robot, model_name, active_arms, data, offset, skip_ready=False):
        super().__init__()
        self.task_type = task_type
        self.robot = robot
        self.model_name = model_name
        self.active_arms = active_arms
        self.data = data
        self.offset = offset
        self.skip_ready = skip_ready

    def run(self):
        try:
            from core.robot_motion import check_calibration_state, make_dual_arm_head_cmd
            if self.task_type == "move":
                check_calibration_state(
                    self.robot,
                    self.model_name,
                    self.active_arms,
                    self.data,
                    self.offset,
                    log_cb=self.log_signal.emit,
                    skip_ready=self.skip_ready
                )
                self.finished_signal.emit(True, "")
            elif self.task_type == "draw_square":
                import rby1_sdk as rby
                import math
                square_points = [
                    [0.35, 0.07, 0.0],
                    [0.35, 0.0, 0.07],
                    [0.35, -0.07, 0.0],
                    [0.35, 0.0, -0.07],
                ]
                
                self.log_signal.emit("Starting square drawing sequence (2 loops)...")
                for loop_idx in range(2):
                    self.log_signal.emit(f"Loop {loop_idx + 1} / 2")
                    for pt_idx, pt in enumerate(square_points):
                        self.log_signal.emit(f"  Target Point {pt_idx + 1}: X={pt[0]}, Y={pt[1]}, Z={pt[2]}")
                        
                        roll_r = 90 * math.pi / 180
                        pitch_r = -90 * math.pi / 180
                        yaw_r = 0.0
                        cr_r = math.cos(roll_r); sr_r = math.sin(roll_r)
                        cp_r = math.cos(pitch_r); sp_r = math.sin(pitch_r)
                        cy_r = math.cos(yaw_r); sy_r = math.sin(yaw_r)
                        
                        T_right = np.eye(4, dtype=np.float64)
                        T_right[0, 0] = cy_r * cp_r
                        T_right[0, 1] = sr_r * sp_r * cy_r - cr_r * sy_r
                        T_right[0, 2] = cr_r * sp_r * cy_r + sr_r * sy_r
                        T_right[0, 3] = pt[0]
                        T_right[1, 0] = sy_r * cp_r
                        T_right[1, 1] = sr_r * sp_r * sy_r + cr_r * cy_r
                        T_right[1, 2] = cr_r * sp_r * sy_r - sr_r * cy_r
                        T_right[1, 3] = pt[1] - self.offset
                        T_right[2, 0] = -sp_r
                        T_right[2, 1] = cp_r * sr_r
                        T_right[2, 2] = cp_r * cr_r
                        T_right[2, 3] = pt[2]
                        
                        roll_l = -90 * math.pi / 180
                        pitch_l = -90 * math.pi / 180
                        yaw_l = 0.0
                        cr_l = math.cos(roll_l); sr_l = math.sin(roll_l)
                        cp_l = math.cos(pitch_l); sp_l = math.sin(pitch_l)
                        cy_l = math.cos(yaw_l); sy_l = math.sin(yaw_l)
                        
                        T_left = np.eye(4, dtype=np.float64)
                        T_left[0, 0] = cy_l * cp_l
                        T_left[0, 1] = sr_l * sp_l * cy_l - cr_l * sy_l
                        T_left[0, 2] = cr_l * sp_l * cy_l + sr_l * sy_l
                        T_left[0, 3] = pt[0]
                        T_left[1, 0] = sy_l * cp_l
                        T_left[1, 1] = sr_l * sp_l * sy_l + cr_l * cy_l
                        T_left[1, 2] = cr_l * sp_l * sy_l - sr_l * cy_l
                        T_left[1, 3] = pt[1] + self.offset
                        T_left[2, 0] = -sp_l
                        T_left[2, 1] = cp_l * sr_l
                        T_left[2, 2] = cp_l * cr_l
                        T_left[2, 3] = pt[2]
                        
                        cmd = make_dual_arm_head_cmd(
                            T_right=T_right,
                            T_left=T_left,
                            active_arms=self.active_arms,
                            head_position=None,
                            min_time=2.0,
                            hold_time=0.2
                        )
                        rv = self.robot.send_command(cmd, 10).get()
                        if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                            raise RuntimeError(f"Draw point move failed: {rv.finish_code}")
                        time.sleep(0.5)
                        
                self.log_signal.emit("Square drawing sequence completed successfully.")
                self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class Step2CalculateWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, app, active_arms, optimize_head, optimize_camera, q_arm_list, q_head_list, T_meas_list, result_path, lambda_cam_pos, lambda_cam_rot):
        super().__init__()
        self.app = app
        self.active_arms = active_arms
        self.optimize_head = optimize_head
        self.optimize_camera = optimize_camera
        self.q_arm_list = q_arm_list
        self.q_head_list = q_head_list
        self.T_meas_list = T_meas_list
        self.result_path = result_path
        self.lambda_cam_pos = lambda_cam_pos
        self.lambda_cam_rot = lambda_cam_rot

    def run(self):
        try:
            self.app.run_optimizer(
                active_arms=self.active_arms,
                optimize_head=self.optimize_head,
                optimize_camera=self.optimize_camera,
                q_arm_list=self.q_arm_list,
                q_head_list=self.q_head_list,
                T_meas_list=self.T_meas_list,
                result_path=self.result_path,
                lambda_cam_pos=self.lambda_cam_pos,
                lambda_cam_rot=self.lambda_cam_rot,
                solver_type="QP Solver",
                use_sag=False,
            )
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class FullAutoReadyWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, joint_calibrator, marker_calibrator, ui_only=False):
        super().__init__()
        self.joint_calibrator = joint_calibrator
        self.marker_calibrator = marker_calibrator
        self.ui_only = ui_only
        self.error_msg = None

    def run(self):
        try:
            self.error_msg = None
            self.log_signal.emit("Moving robot arms to Full Auto initial ready poses...")
            version_num = self.marker_calibrator.get_robot_version()
            is_v13 = self.marker_calibrator.is_v13()
            self.log_signal.emit(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")

            is_mock_run = (self.joint_calibrator.robot is None or self.joint_calibrator.robot == "mock_robot")
            
            for arm_side in ["right", "left"]:
                self.log_signal.emit(f"Preparing {arm_side.upper()} arm...")
                if is_mock_run:
                    time.sleep(1.0)
                    self.log_signal.emit(f"[MOCK] {arm_side.upper()} arm moved to ready pose.")
                else:
                    if not is_v13:
                        self.log_signal.emit(f"Moving {arm_side} arm to wrist pitch ready pose...")
                        if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch", log_callback=self.log_signal.emit):
                            raise RuntimeError(f"Failed to move {arm_side} arm to wrist pitch ready pose.")
                    else:
                        self.log_signal.emit(f"Moving {arm_side} arm to marker ready pose...")
                        if not self.marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=self.log_signal.emit):
                            raise RuntimeError(f"Failed to move {arm_side} arm to marker ready pose.")
            self.log_signal.emit("All arms moved to initial ready poses successfully.")
        except Exception as e:
            self.error_msg = str(e)
            self.log_signal.emit(f"[ERROR] Ready pose movement failed: {e}")
        self.finished_signal.emit()

# --- Specialized Calibration Workers ---
class MarkerCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)
    
    def __init__(self, calibrator, arm_side, use_head_tracking=True, tolerance=0.5, save_debug=False):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.use_head_tracking = use_head_tracking
        self.tolerance = tolerance
        self.save_debug = save_debug
        
    def run(self):
        try:
            version_num = self.calibrator.get_robot_version()
            is_v13 = self.calibrator.is_v13()
            
            is_mock_run = (self.calibrator.robot is None or self.calibrator.robot == "mock_robot")
            if not is_mock_run:
                # Automatically move to ready pose first to guarantee calibration starting pose consistency
                self.log_signal.emit("[INFO] Automatically moving active arm to marker ready pose...")
                success = self.calibrator.perform_move_to_ready_pose(self.arm_side, mode="marker", log_callback=self.log_signal.emit)
                if not success:
                    self.log_signal.emit("[ERROR] Failed to move to marker ready pose at startup. Aborting.")
                    self.finished_signal.emit(None)
                    return
                state = self.calibrator.robot.get_state()
                model = self.calibrator.robot.model()
                arm_idx = model.left_arm_idx if self.arm_side == "left" else model.right_arm_idx
                first_starting_pose = list(state.position[arm_idx])
            else:
                first_starting_pose = [0.0]*7
                res_4 = None
            if True: # Always sweep J4 for both v1.2 and v1.3 to get full 3D calibration
                # Stage 1 Axis 4 sweep starts immediately from the initial/current pose
                if getattr(self.calibrator, 'stop_requested', False):
                    self.finished_signal.emit(None)
                    return
                
                self.log_signal.emit("\n" + "="*50)
                self.log_signal.emit("   [Stage 1/3] Sweeping Axis 4 (Wrist Yaw)...")
                self.log_signal.emit("="*50 + "\n")
                res_4 = self.calibrator.perform_calibration_sweep(
                    self.arm_side, 4,
                    log_callback=self.log_signal.emit,
                    status_callback=self.status_signal.emit,
                    use_head_tracking=self.use_head_tracking,
                    save_debug=self.save_debug
                )
                if not res_4:
                    self.log_signal.emit("[ERROR] Stage 1 (Axis 4) sweep failed. Aborting.")
                    self.finished_signal.emit(None)
                    return
                res_4['axis_mode'] = 4
                res_4['axis'] = res_4['axis_opt']
                
                if getattr(self.calibrator, 'stop_requested', False):
                    self.finished_signal.emit(None)
                    return
                
                time.sleep(1.0)
                
                # Move back to initial starting pose
                if not is_mock_run:
                    self.log_signal.emit("\n" + "="*50)
                    self.log_signal.emit("   [Stage 2/3] Returning to Initial Starting Pose...")
                    self.log_signal.emit("="*50 + "\n")
                    
                    if self.arm_side == "right":
                        success_other = self.calibrator.movej(self.calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                    else:
                        success_other = self.calibrator.movej(self.calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                    
                    if not success_other:
                        self.log_signal.emit("[ERROR] Failed to move inactive arm to zero pose.")
                        self.finished_signal.emit(None)
                        return
                    
                    success = self.calibrator.movej(
                        self.calibrator.robot,
                        torso=[0.0]*6,
                        right_arm=first_starting_pose if self.arm_side == "right" else None,
                        left_arm=first_starting_pose if self.arm_side == "left" else None,
                        head=[0, 0],
                        minimum_time=5.0,
                        apply_offsets=False
                    )
                    if not success:
                        self.log_signal.emit("[ERROR] Failed to return to initial starting pose. Aborting.")
                        self.finished_signal.emit(None)
                        return
                else:
                    self.log_signal.emit("\n[MOCK] Returning to Initial Starting Pose...")
                    time.sleep(1.0)
                
                if getattr(self.calibrator, 'stop_requested', False):
                    self.finished_signal.emit(None)
                    return
                
                time.sleep(1.0)

            # Stage 2/3 Axis 6 Sweep
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit("   [Stage 2/3] Sweeping Axis 6 (Roll)...")
            self.log_signal.emit("="*50 + "\n")
            
            res_6 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 6, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking,
                save_debug=self.save_debug
            )
            if not res_6:
                self.log_signal.emit("[ERROR] Stage 6 sweep failed. Aborting.")
                self.finished_signal.emit(None)
                return
                
            res_6['axis_mode'] = 6
            res_6['axis'] = res_6['axis_opt']
            
            if getattr(self.calibrator, 'stop_requested', False):
                self.finished_signal.emit(None)
                return
                
            time.sleep(1.0)

            # Move back to initial starting pose before Axis 5 sweep
            if not is_mock_run:
                self.log_signal.emit("\n" + "="*50)
                self.log_signal.emit("   [Stage 3/3] Returning to Initial Starting Pose...")
                self.log_signal.emit("="*50 + "\n")
                
                if self.arm_side == "right":
                    success_other = self.calibrator.movej(self.calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                else:
                    success_other = self.calibrator.movej(self.calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                
                if not success_other:
                    self.log_signal.emit("[ERROR] Failed to move inactive arm to zero pose.")
                    self.finished_signal.emit(None)
                    return
                
                success = self.calibrator.movej(
                    self.calibrator.robot,
                    torso=[0.0]*6,
                    right_arm=first_starting_pose if self.arm_side == "right" else None,
                    left_arm=first_starting_pose if self.arm_side == "left" else None,
                    head=[0, 0],
                    minimum_time=5.0,
                    apply_offsets=False
                )
                if not success:
                    self.log_signal.emit("[ERROR] Failed to return to initial starting pose. Aborting.")
                    self.finished_signal.emit(None)
                    return
            else:
                self.log_signal.emit("\n[MOCK] Returning to Initial Starting Pose...")
                time.sleep(1.0)
            
            if getattr(self.calibrator, 'stop_requested', False):
                self.finished_signal.emit(None)
                return
            
            time.sleep(1.0)
            
            # Stage 3/3 Axis 5 Sweep
            self.log_signal.emit("\n" + "="*50)
            self.log_signal.emit("   [Stage 3/3] Sweeping Axis 5 (Pitch)...")
            self.log_signal.emit("="*50 + "\n")
            
            res_5 = self.calibrator.perform_calibration_sweep(
                self.arm_side, 5, 
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                use_head_tracking=self.use_head_tracking,
                save_debug=self.save_debug
            )
            if not res_5:
                self.log_signal.emit("[ERROR] Stage 5 sweep failed. Aborting.")
                self.finished_signal.emit(None)
                return
                
            res_5['axis_mode'] = 5
            res_5['axis'] = res_5['axis_opt']
            
            # Compute unified bracket calibration
            self.log_signal.emit("\n[PROCESSING] Computing unified bracket calibration parameters...")
            unified_res = self.calibrator.compute_unified_bracket_calibration(
                res_5, res_6, self.arm_side, tolerance=self.tolerance, marker_data_4=res_4, calib_roll_deg=0.0, calib_pitch_deg=0.0
            )
            
            unified_res['res_5'] = res_5
            unified_res['res_6'] = res_6
            if res_4 is not None:
                unified_res['res_4'] = res_4
            
            # Save plot using the calibrator method
            plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{self.arm_side}_marker_unified.png")
            plot_saved = self.calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, self.arm_side, is_v13, plot_path)
            
            if plot_saved:
                unified_res['plot_path_combined'] = plot_path
            self.finished_signal.emit(unified_res)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(None)

class JointCalibrationWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    finished_signal = Signal(dict)

    def __init__(self, calibrator, arm_side, mode, ui_only=False, current_offset_deg=0.0, sweep_duration=15.0, save_debug=False):
        super().__init__()
        self.calibrator = calibrator
        self.arm_side = arm_side
        self.mode = mode
        self.ui_only = ui_only
        self.current_offset_deg = current_offset_deg
        self.sweep_duration = sweep_duration
        self.save_debug = save_debug

    def run(self):
        try:
            res = self.calibrator.perform_joint_calibration(
                self.arm_side, self.mode,
                log_callback=self.log_signal.emit, 
                status_callback=self.status_signal.emit,
                current_offset_deg=self.current_offset_deg,
                sweep_duration=self.sweep_duration,
                save_debug=self.save_debug
            )

            if res:
                self.log_signal.emit("-" * 30)
                self.log_signal.emit(f"  [1] Calibration Target: {self.mode}")
                recommended = res.get('recommended_joint_offset', res['optimal_offset'])
                self.log_signal.emit(f"      Estimated Optimal Offset: {recommended:.3f} deg")
                self.log_signal.emit("-" * 30)
                self.log_signal.emit("\n[CALIBRATION COMPLETE]\n")
                self.finished_signal.emit(res)
            else:
                self.finished_signal.emit(None)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Worker exception: {e}")
            self.finished_signal.emit(None)


class SimulatedMarkerTransform:
    def __init__(self, robot, camera_config, robot_version="1.2"):
        self.robot = robot
        self.camera_config = camera_config
        self.robot_version = robot_version
        
        class DummyCamera:
            def stream_off(self): pass
        self.camera = DummyCamera()

    def get_marker_transform(self, sampling_time=0, side="right", use_filter=False):
        if side == "all":
            res = []
            for s in ["right", "left"]:
                try:
                    res_s = self.get_marker_transform(sampling_time, s, use_filter)
                    if res_s:
                        res.extend(res_s)
                except Exception:
                    pass
            return res

        version = self.robot_version
        is_v13 = (version == "1.3")
        try:
            q = self.robot.get_state().position
            dyn_model = self.robot.get_dynamics()
            
            ee_name = f"ee_{side}"
            
            # Apply simulated joint offsets to simulate actual robot kinematics
            q_actual = np.array(q)
            model = self.robot.model()
            arm_idx = model.left_arm_idx if side == "left" else model.right_arm_idx
            
            from core.calibration.CalibratorBase import BaseCalibrator
            if side not in BaseCalibrator.MOCK_GT_OFFSETS:
                raise KeyError(f"Mock ground-truth offsets not found for side: {side}")
            mock_gt = BaseCalibrator.MOCK_GT_OFFSETS[side]

            j6_gt = mock_gt.get("joint6")
            j5_gt = mock_gt.get("joint5_v13") if is_v13 else mock_gt.get("joint5_v12")
            j3_gt = mock_gt.get("joint3")
            if j6_gt is None or j5_gt is None or j3_gt is None:
                raise ValueError(f"Missing joint mock GT values in BaseCalibrator.MOCK_GT_OFFSETS for {side}")

            # Apply simulated joint offsets (ground-truth offset only) to simulate actual robot kinematics.
            # Staged user-applied offsets are already included in q from commanded ready poses.
            q_actual[arm_idx[6]] += np.radians(j6_gt)
            q_actual[arm_idx[5]] += np.radians(j5_gt)
            q_actual[arm_idx[3]] += np.radians(j3_gt)
                    
            T_t5_to_ee = BaseCalibrator.compute_fk(self.robot, dyn_model, q_actual, ee_name, "link_torso_5")
            
            try:
                T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q, "link_head_2", "link_torso_5")
            except Exception:
                try:
                    # Try alternative link name for v1.2 head camera mount
                    T_t5_to_head = BaseCalibrator.compute_fk(self.robot, dyn_model, q, "link_head", "link_torso_5")
                except Exception:
                    T_t5_to_head = np.eye(4)
            
            # Apply simulated bracket offset to simulate bracket misalignment
            bracket_pos = mock_gt.get("bracket_pos")
            bracket_rpy = mock_gt.get("bracket_rpy")
            if bracket_pos is None or bracket_rpy is None:
                raise ValueError(f"Missing bracket mock GT values in BaseCalibrator.MOCK_GT_OFFSETS for {side}")
            bracket_offset_vec = list(bracket_pos) + list(bracket_rpy)
            T_bracket_offset = BaseCalibrator.make_transform(bracket_offset_vec)
                
            mount_to_cam = self.camera_config.get("mount_to_cam", [0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
            T_head_to_cam = BaseCalibrator.make_transform(mount_to_cam)
            T_t5_to_cam = T_t5_to_head @ T_head_to_cam
            
            if is_v13:
                default_left = [0.097, 0.0, -0.005, 90.0, 0.0, -90.0]
                default_right = [0.097, 0.0, -0.005, 90.0, 0.0, -90.0]
                key_left = "Tf_to_marker_left_v13"
                key_right = "Tf_to_marker_right_v13"
            else:
                default_left = [0.0, 0.0775, -0.06677, 90.0, 0.0, 0.0]
                default_right = [0.0, -0.0775, -0.06677, 90.0, 0.0, 180.0]
                key_left = "Tf_to_marker_left"
                key_right = "Tf_to_marker_right"

            if side == "left":
                tf_vec = self.camera_config.get(key_left, default_left)
            else:
                tf_vec = self.camera_config.get(key_right, default_right)
            T_ee_to_marker = BaseCalibrator.make_transform(tf_vec)
            
            T_cam_to_t5 = np.linalg.inv(T_t5_to_cam)
            T_cam_to_marker = T_cam_to_t5 @ T_t5_to_ee @ T_bracket_offset @ T_ee_to_marker
            
            noise_t = np.random.normal(0, 0.0001, 3)
            T_cam_to_marker[:3, 3] += noise_t
            
            return [T_cam_to_marker.tolist()]
        except Exception as e:
            print(f"[SimulatedMarkerTransform] FK calculation failed: {e}")
            T = np.eye(4)
            T[2, 3] = 0.3
            return [T.tolist()]


class FullAutoWorker(QThread):
    log_msg = Signal(str)
    status_signal = Signal(bool)
    bracket_finished_signal = Signal(dict)
    joint_finished_signal = Signal(dict)
    finished_signal = Signal()

    def __init__(self, joint_calibrator, marker_calibrator, ui_only=False, stop_event=None, joint_offsets_store=None, save_debug=False):
        super().__init__()
        self.joint_calibrator = joint_calibrator
        self.marker_calibrator = marker_calibrator
        self.ui_only = ui_only
        self.stop_event = stop_event
        self.joint_offsets_store = joint_offsets_store if joint_offsets_store is not None else {}
        self.save_debug = save_debug
        self.error_msg = None

    def get_robot_version(self):
        return self.marker_calibrator.get_robot_version()

    def run(self):
        try:
            from scipy.spatial.transform import Rotation as R_scipy
            self.log_msg.emit("Starting FULL AUTO sequential calibration...")
            is_mock_run = (self.joint_calibrator.robot is None or self.joint_calibrator.robot == "mock_robot")
            version_num = self.get_robot_version()
            is_v13 = (version_num == "1.3")
            
            for arm_side in ["right", "left"]:
                pass1_joint_results = {"wrist_pitch": None, "elbow": None}
                # Backup of parameters before Pass 1 for early exit / change checking
                prev_j6 = self.joint_offsets_store[arm_side]["joint6"]
                prev_j5 = self.joint_offsets_store[arm_side]["joint5"]
                prev_j3 = self.joint_offsets_store[arm_side]["joint3"]
                
                # We need nominal bracket baseline to calculate bracket parameter changes
                ver_key = "1.3" if is_v13 else "1.2"
                nominal_vec = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
                tf_vec_init = self.marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
                if tf_vec_init is not None and len(tf_vec_init) == 6:
                    prev_bracket_pos = np.array(tf_vec_init[:3]) * 1000.0
                    prev_bracket_rot = np.array(tf_vec_init[3:])
                else:
                    prev_bracket_pos = np.array(nominal_vec[:3]) * 1000.0
                    prev_bracket_rot = np.array(nominal_vec[3:])

                res_4 = None
                res_5 = None
                res_6 = None

                for pass_idx in [1, 2]:
                    self.log_msg.emit("\n" + "="*50)
                    self.log_msg.emit(f"   STARTING PASS {pass_idx}/2 FOR {arm_side.upper()} ARM")
                    self.log_msg.emit("="*50 + "\n")
                    self.log_msg.emit(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")

                    for calibrator in [self.joint_calibrator, self.marker_calibrator]:
                        calibrator.joint_offsets[arm_side]["wrist_pitch"] = self.joint_offsets_store[arm_side]["joint5"]
                        if is_v13:
                            calibrator.joint_offsets[arm_side]["wrist_roll"] = self.joint_offsets_store[arm_side]["joint6"]
                            calibrator.joint_offsets[arm_side]["wrist_yaw2"] = 0.0
                        else:
                            calibrator.joint_offsets[arm_side]["wrist_roll"] = 0.0
                            calibrator.joint_offsets[arm_side]["wrist_yaw2"] = self.joint_offsets_store[arm_side]["joint6"]
                        calibrator.joint_offsets[arm_side]["elbow"] = self.joint_offsets_store[arm_side]["joint3"]

                    # --- Step 1: Marker Bracket Calibration ---
                    if True:
                        self.log_msg.emit(f"[FULL AUTO 1/2] Starting Marker Bracket Calibration for {arm_side} arm (Pass {pass_idx}/2)...")
                        if is_v13:
                            # 1. Marker Bracket Sweeps (Axis 4, 6, 5)
                            self.log_msg.emit(f"[FULL AUTO] Moving {arm_side} arm to ready pose...")
                            if not self.marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=self.log_msg.emit):
                                raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                            if self.stop_event.is_set(): return
                            
                            self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 4...")
                            res_4 = self.marker_calibrator.perform_calibration_sweep(
                                arm_side, 4, log_callback=self.log_msg.emit, status_callback=self.status_signal.emit,
                                save_debug=self.save_debug
                            )
                            if not res_4: raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
                            res_4['axis_mode'] = 4
                            res_4['axis'] = res_4['axis_opt']
                            if self.stop_event.is_set(): return
                            
                            if not is_mock_run:
                                self.log_msg.emit(f"[FULL AUTO] Returning to Initial Starting Pose...")
                                model = self.marker_calibrator.robot.model()
                                state = self.marker_calibrator.robot.get_state()
                                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                                first_starting_pose = list(state.position[arm_idx])
                                if arm_side == "right":
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                                else:
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                            else:
                                self.log_msg.emit("[FULL AUTO] [MOCK] Returning to Initial Starting Pose...")
                                time.sleep(1.0)
                            if self.stop_event.is_set(): return

                            self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 6...")
                            res_6 = self.marker_calibrator.perform_calibration_sweep(
                                arm_side, 6, log_callback=self.log_msg.emit, status_callback=self.status_signal.emit,
                                save_debug=self.save_debug
                            )
                            if not res_6: raise RuntimeError(f"Axis 6 marker sweep failed on {arm_side} arm")
                            res_6['axis_mode'] = 6
                            res_6['axis'] = res_6['axis_opt']
                            if self.stop_event.is_set(): return
                            
                            if not is_mock_run:
                                self.log_msg.emit(f"[FULL AUTO] Returning to Initial Starting Pose...")
                                model = self.marker_calibrator.robot.model()
                                state = self.marker_calibrator.robot.get_state()
                                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                                first_starting_pose = list(state.position[arm_idx])
                                if arm_side == "right":
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                                else:
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                            else:
                                self.log_msg.emit("[FULL AUTO] [MOCK] Returning to Initial Starting Pose...")
                                time.sleep(1.0)
                            if self.stop_event.is_set(): return

                            self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 5...")
                            res_5 = self.marker_calibrator.perform_calibration_sweep(
                                arm_side, 5, log_callback=self.log_msg.emit, status_callback=self.status_signal.emit,
                                save_debug=self.save_debug
                            )
                            if not res_5: raise RuntimeError(f"Axis 5 marker sweep failed on {arm_side} arm")
                            res_5['axis_mode'] = 5
                            res_5['axis'] = res_5['axis_opt']
                            if self.stop_event.is_set(): return
                        else:
                            # v1.2 sweeps
                            self.log_msg.emit(f"[FULL AUTO] Moving {arm_side} arm to ready pose...")
                            if not self.marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=self.log_msg.emit):
                                raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                            if self.stop_event.is_set(): return
                            
                            self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 4...")
                            res_4 = self.marker_calibrator.perform_calibration_sweep(
                                arm_side, 4, log_callback=self.log_msg.emit, status_callback=self.status_signal.emit,
                                save_debug=self.save_debug
                            )
                            if not res_4: raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
                            res_4['axis_mode'] = 4
                            res_4['axis'] = res_4['axis_opt']
                            if self.stop_event.is_set(): return
                            
                            if not is_mock_run:
                                self.log_msg.emit(f"[FULL AUTO] Returning to Initial Starting Pose...")
                                model = self.marker_calibrator.robot.model()
                                state = self.marker_calibrator.robot.get_state()
                                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                                first_starting_pose = list(state.position[arm_idx])
                                if arm_side == "right":
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                                else:
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                            else:
                                self.log_msg.emit("[FULL AUTO] [MOCK] Returning to Initial Starting Pose...")
                                time.sleep(1.0)
                            if self.stop_event.is_set(): return
                            
                            self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 6...")
                            res_6 = self.marker_calibrator.perform_calibration_sweep(
                                arm_side, 6, log_callback=self.log_msg.emit, status_callback=self.status_signal.emit,
                                save_debug=self.save_debug
                            )
                            if not res_6: raise RuntimeError(f"Axis 6 marker sweep failed on {arm_side} arm")
                            res_6['axis_mode'] = 6
                            res_6['axis'] = res_6['axis_opt']
                            if self.stop_event.is_set(): return
                            
                            if not is_mock_run:
                                self.log_msg.emit(f"[FULL AUTO] Returning to Initial Starting Pose...")
                                model = self.marker_calibrator.robot.model()
                                state = self.marker_calibrator.robot.get_state()
                                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                                first_starting_pose = list(state.position[arm_idx])
                                if arm_side == "right":
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                                else:
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, right_arm=[0.0]*7, head=None, minimum_time=3.0, apply_offsets=False)
                                    self.marker_calibrator.movej(self.marker_calibrator.robot, torso=[0.0]*6, left_arm=first_starting_pose, head=[0, 0], minimum_time=5.0, apply_offsets=False)
                            else:
                                self.log_msg.emit("[FULL AUTO] [MOCK] Returning to Initial Starting Pose...")
                                time.sleep(1.0)
                            if self.stop_event.is_set(): return
                            
                            self.log_msg.emit(f"[FULL AUTO] Sweeping Axis 5...")
                            res_5 = self.marker_calibrator.perform_calibration_sweep(
                                arm_side, 5, log_callback=self.log_msg.emit, status_callback=self.status_signal.emit,
                                save_debug=self.save_debug
                            )
                            if not res_5: raise RuntimeError(f"Axis 5 marker sweep failed on {arm_side} arm")
                            res_5['axis_mode'] = 5
                            res_5['axis'] = res_5['axis_opt']
                            if self.stop_event.is_set(): return


                    # 2. Calibrate J6 Wrist Roll/Yaw 2
                    self.log_msg.emit(f"\n[FULL AUTO] Calibrating J6 ({'Wrist Roll' if is_v13 else 'Wrist Yaw 2'}) first...")
                    dataset_A_6 = list(zip(res_6['captured_q_full'], res_6['captured_poses']))
                    dataset_B_5 = list(zip(res_5['captured_q_full'], res_5['captured_poses']))
                    model = self.joint_calibrator.robot.model()
                    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                    initial_joint_pos_roll = list(res_6['captured_q_full'][0][arm_idx])
                    
                    joint_res_roll = self.joint_calibrator.compute_calibration_results(
                        arm_side, "wrist_roll_v13" if is_v13 else "wrist_yaw2", dataset_A_6, dataset_B_5, initial_joint_pos_roll,
                        current_offset_deg=0.0, use_angle_based_fitting=True, log_callback=self.log_msg.emit
                    )
                    if not joint_res_roll:
                        raise RuntimeError(f"J6 calibration failed on {arm_side} arm")
                    opt_roll = joint_res_roll["recommended_joint_offset"]
                    self.log_msg.emit(f"[FULL AUTO] Staging J6 offset: {opt_roll:.4f}°")
                    self.joint_offsets_store[arm_side]["joint6"] = opt_roll
                    if is_v13:
                        self.joint_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                        self.marker_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                    else:
                        self.joint_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                        self.marker_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    
                    # Generate and save the J6 calibration comparison plot
                    plot_path = self.joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_roll_v13" if is_v13 else "wrist_yaw2", joint_res_roll, joint_res_roll, 
                        log_callback=self.log_msg.emit, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_roll['plot_path_combined'] = plot_path
                    
                    # Emitting the J6 calibration result to the UI plot
                    joint_res_roll['arm_side'] = arm_side
                    joint_res_roll['mode'] = "wrist_roll_v13" if is_v13 else "wrist_yaw2"
                    joint_res_roll['pass_idx'] = pass_idx
                    self.joint_finished_signal.emit(joint_res_roll)
                    time.sleep(0.5)
                    if self.stop_event.is_set(): return

                    # 3. Compute Marker Bracket (with J6 locked)
                    self.log_msg.emit("\n[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...")
                    staged_pitch = self.joint_offsets_store[arm_side]["joint5"]
                    unified_res = self.marker_calibrator.compute_unified_bracket_calibration(
                        res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_deg=opt_roll, calib_pitch_deg=staged_pitch
                    )
                    
                    unified_res['res_5'] = res_5
                    unified_res['res_6'] = res_6
                    if res_4 is not None:
                        unified_res['res_4'] = res_4
                    unified_res['arm_side'] = arm_side
                    unified_res['pass_idx'] = pass_idx
                    
                    plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                    # Always overwrite the plot to keep it up to date
                    plot_saved = self.marker_calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, arm_side, is_v13, plot_path)
                    if plot_saved:
                        unified_res['plot_path_combined'] = plot_path
                    
                    x_m, y_m, z_m = unified_res['x_e']/1000.0, unified_res['y_e']/1000.0, unified_res['z_e']/1000.0
                    new_vals = [x_m, y_m, z_m, unified_res['roll_e'], unified_res['pitch_e'], unified_res['yaw_e']]
                    key = f"Tf_to_marker_{arm_side}"
                    self.marker_calibrator.camera_config[key] = new_vals
                    self.joint_calibrator.camera_config[key] = new_vals
                    
                    # Prevent resetting J5 to 0.0 before physical sweep in Pass 1.
                    # We keep the previously loaded offset from setting.yaml.
                    
                    self.bracket_finished_signal.emit(unified_res)
                    time.sleep(0.5)
                    if self.stop_event.is_set(): return

                    # 4. Calibrate J5 Wrist Pitch
                    self.log_msg.emit("[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...")
                    # Restore/apply the newly calibrated offsets for physical sweeps in Pass 2
                    for calibrator in [self.joint_calibrator, self.marker_calibrator]:
                        calibrator.joint_offsets[arm_side]["wrist_pitch"] = self.joint_offsets_store[arm_side]["joint5"]
                        if is_v13:
                            calibrator.joint_offsets[arm_side]["wrist_roll"] = self.joint_offsets_store[arm_side]["joint6"]
                            calibrator.joint_offsets[arm_side]["wrist_yaw2"] = 0.0
                        else:
                            calibrator.joint_offsets[arm_side]["wrist_roll"] = 0.0
                            calibrator.joint_offsets[arm_side]["wrist_yaw2"] = self.joint_offsets_store[arm_side]["joint6"]
                        calibrator.joint_offsets[arm_side]["elbow"] = self.joint_offsets_store[arm_side]["joint3"]

                    if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch_v13" if is_v13 else "wrist_pitch", log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_pitch on {arm_side} arm")
                    if self.stop_event.is_set(): return
                    
                    pass1_res_pitch = pass1_joint_results.get("wrist_pitch")
                    joint_res_pitch = self.joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_pitch_v13" if is_v13 else "wrist_pitch",
                        log_callback=self.log_msg.emit,
                        status_callback=self.status_signal.emit,
                        current_offset_deg=self.joint_offsets_store[arm_side]["joint5"],
                        save_debug=self.save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_pitch
                    )
                    if not joint_res_pitch:
                        raise RuntimeError(f"Wrist pitch joint calibration failed on {arm_side} arm")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_pitch"] = joint_res_pitch
                    joint_res_pitch['arm_side'] = arm_side
                    joint_res_pitch['mode'] = "wrist_pitch_v13" if is_v13 else "wrist_pitch"
                    joint_res_pitch['pass_idx'] = pass_idx
                    
                    opt_pitch = joint_res_pitch["recommended_joint_offset"]
                    self.joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    self.marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    self.joint_offsets_store[arm_side]["joint5"] = opt_pitch
                    
                    self.joint_finished_signal.emit(joint_res_pitch)
                    time.sleep(0.5)
                    if self.stop_event.is_set(): return
                    
                    # 5. Calibrate J3 Elbow
                    self.log_msg.emit("[FULL AUTO] Sweeping Elbow (Joint 3)...")
                    if not self.joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=self.log_msg.emit):
                        raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                    if self.stop_event.is_set(): return
                    
                    pass1_res_elbow = pass1_joint_results.get("elbow")
                    joint_res_elbow = self.joint_calibrator.perform_joint_calibration(
                        arm_side, "elbow",
                        log_callback=self.log_msg.emit,
                        status_callback=self.status_signal.emit,
                        current_offset_deg=self.joint_offsets_store[arm_side]["joint3"],
                        save_debug=self.save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_elbow
                    )
                    if not joint_res_elbow:
                        raise RuntimeError(f"Elbow joint calibration failed on {arm_side} arm")
                    if pass_idx == 1:
                        pass1_joint_results["elbow"] = joint_res_elbow
                    joint_res_elbow['arm_side'] = arm_side
                    joint_res_elbow['mode'] = "elbow"
                    joint_res_elbow['pass_idx'] = pass_idx
                    
                    opt_elbow = joint_res_elbow["recommended_joint_offset"]
                    self.joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    self.marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    self.joint_offsets_store[arm_side]["joint3"] = opt_elbow
                    
                    self.joint_finished_signal.emit(joint_res_elbow)
                    time.sleep(0.5)

                    # Pass 1 Evaluation & Early Exit Check
                    if pass_idx == 1:
                        j6_change = abs(self.joint_offsets_store[arm_side]["joint6"] - prev_j6)
                        j5_change = abs(self.joint_offsets_store[arm_side]["joint5"] - prev_j5)
                        j3_change = abs(self.joint_offsets_store[arm_side]["joint3"] - prev_j3)
                        
                        tf_vec_now = self.marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
                        if tf_vec_now is not None and len(tf_vec_now) == 6:
                            now_bracket_pos = np.array(tf_vec_now[:3]) * 1000.0
                            now_bracket_rot = np.array(tf_vec_now[3:])
                        else:
                            now_bracket_pos = prev_bracket_pos
                            now_bracket_rot = prev_bracket_rot
                            
                        pos_change = np.linalg.norm(now_bracket_pos - prev_bracket_pos)
                        
                        # Compute rotation change in degrees
                        R_prev = R_scipy.from_euler('ZYX', [prev_bracket_rot[2], prev_bracket_rot[1], prev_bracket_rot[0]], degrees=True)
                        R_now = R_scipy.from_euler('ZYX', [now_bracket_rot[2], now_bracket_rot[1], now_bracket_rot[0]], degrees=True)
                        rot_change = np.rad2deg(np.linalg.norm((R_now * R_prev.inv()).as_rotvec()))
                        
                        self.log_msg.emit(f"\n[PASS 1 EVALUATION] Staged parameter changes for {arm_side.upper()} Arm:")
                        self.log_msg.emit(f"  * Joint 6 Change      : {j6_change:.4f}°")
                        self.log_msg.emit(f"  * Joint 5 Change      : {j5_change:.4f}°")
                        self.log_msg.emit(f"  * Joint 3 Change      : {j3_change:.4f}°")
                        self.log_msg.emit(f"  * Bracket Pos Change  : {pos_change:.4f} mm")
                        self.log_msg.emit(f"  * Bracket Rot Change  : {rot_change:.4f}°")
                        
                        # Early Exit Thresholds: joints < 0.05°, bracket pos < 0.5 mm, bracket rot < 0.1°
                        if j6_change < 0.05 and j5_change < 0.05 and j3_change < 0.05 and pos_change < 0.5 and rot_change < 0.1:
                            self.log_msg.emit(f"[PASS 1 EVALUATION] All changes are within tolerance thresholds.")
                            self.log_msg.emit(f"[PASS 1 EVALUATION] Skipping Pass 2 (Early Exit) for {arm_side.upper()} Arm.")
                            break
                        else:
                            self.log_msg.emit(f"[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.")
                            # Update prev values for Pass 2 check
                            prev_j6 = self.joint_offsets_store[arm_side]["joint6"]
                            prev_j5 = self.joint_offsets_store[arm_side]["joint5"]
                            prev_j3 = self.joint_offsets_store[arm_side]["joint3"]
                            prev_bracket_pos = now_bracket_pos
                            prev_bracket_rot = now_bracket_rot
                            
                self.log_msg.emit(f"[INFO] {arm_side.upper()} arm sequential calibration completed successfully.")
                if self.stop_event.is_set(): return
                time.sleep(1.0)
                
            self.log_msg.emit("\n" + "="*50)
            self.log_msg.emit("   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!")
            self.log_msg.emit("="*50 + "\n")
            
            # Print Final Calibrated Results Report in the same style as simulated ground truth
            self.log_msg.emit("[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):")
            for arm in ["right", "left"]:
                j_store = self.joint_offsets_store.get(arm, {})
                j6_cal = j_store.get("joint6", 0.0)
                j5_cal = j_store.get("joint5", 0.0)
                j3_cal = j_store.get("joint3", 0.0)
                
                tf_vec = self.marker_calibrator.camera_config.get(f"Tf_to_marker_{arm}")
                if tf_vec is not None and len(tf_vec) == 6:
                    x_cal = tf_vec[0] * 1000.0
                    y_cal = tf_vec[1] * 1000.0
                    z_cal = tf_vec[2] * 1000.0
                    r_cal = tf_vec[3]
                    p_cal = tf_vec[4]
                    y_cal_deg = tf_vec[5]
                else:
                    ver_key = "1.3" if is_v13 else "1.2"
                    nominal_vec = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
                    x_cal = nominal_vec[0] * 1000.0
                    y_cal = nominal_vec[1] * 1000.0
                    z_cal = nominal_vec[2] * 1000.0
                    r_cal = nominal_vec[3]
                    p_cal = nominal_vec[4]
                    y_cal_deg = nominal_vec[5]

                ver_key = "1.3" if is_v13 else "1.2"
                nominal_vec = self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
                x_nom = nominal_vec[0] * 1000.0
                y_nom = nominal_vec[1] * 1000.0
                z_nom = nominal_vec[2] * 1000.0
                r_nom = nominal_vec[3]
                p_nom = nominal_vec[4]
                y_nom_deg = nominal_vec[5]
                
                dx = x_cal - x_nom
                dy = y_cal - y_nom
                dz = z_cal - z_nom
                
                from scipy.spatial.transform import Rotation as R_scipy
                R_ideal = R_scipy.from_euler('ZYX', [y_nom_deg, p_nom, r_nom], degrees=True)
                R_actual = R_scipy.from_euler('ZYX', [y_cal_deg, p_cal, r_cal], degrees=True)
                R_offset = R_actual * R_ideal.inv()
                yaw_off, pitch_off, roll_off = R_offset.as_euler('ZYX', degrees=True)
                
                roll_off = (roll_off + 180) % 360 - 180
                pitch_off = (pitch_off + 180) % 360 - 180
                yaw_off = (yaw_off + 180) % 360 - 180
                
                self.log_msg.emit(f"  --- {arm.upper()} ARM ---")
                self.log_msg.emit(f"  * Bracket Pos: X: {dx:+.1f}, Y: {dy:+.1f}, Z: {dz:+.1f} mm")
                self.log_msg.emit(f"  * Bracket Rot: R: {roll_off:+.2f}, P: {pitch_off:+.2f}, Y: {yaw_off:+.2f} deg")
                self.log_msg.emit(f"  * Joint Offsets: Joint 6: {j6_cal:+.2f}°, Joint 5: {j5_cal:+.2f}°, Joint 3: {j3_cal:+.2f}°")
            self.log_msg.emit("==================================================\n")
        except Exception as e:
            self.error_msg = str(e)
            self.log_msg.emit(f"[ERROR] Full Auto sequential calibration failed: {e}")
            import traceback
            self.log_msg.emit(traceback.format_exc())
        finally:
            self.finished_signal.emit()


# --- Unified Calibration App ---
class UnifiedCalibrationApp(QWidget):
    log_signal_safe = Signal(str)
    update_ui_signal_safe = Signal(str)

    def __init__(self, marker_st, robot, arm_side="right", ui_only=False):
        super().__init__()
        self.log_signal_safe.connect(self._log_msg_slot)
        self.update_ui_signal_safe.connect(self._update_ui_slot)

        self.marker_st = marker_st
        self.robot = robot
        self.arm_side = arm_side
        self.ui_only = ui_only
        
        # Core Calibrator Instances
        self.marker_calibrator = MarkerCalibrator(marker_st, robot)
        self.joint_calibrator = JointCalibrator(marker_st, robot)
        self.robot_version = "1.2"
        
        # Intrinsics Calibrator (Tab 3 용)
        self.intrinsics_calibrator = IntrinsicsCalibrator()
        # Default: 8x5 squares, 36mm x 27mm, DICT_5X5_100
        self.intrinsics_calibrator.set_board(8, 5, IntrinsicsCalibrator.BoardPattern.CHARUCOBOARD, 36.0, 27.0, "DICT_5X5_100")
        
        try:
            
            self.marker_detector = Marker_Detection()
            self.marker_detector.set_marker_type("plate")
        except ImportError:
            self.marker_detector = None
            
        self.monitor_enabled = False
        self.captured_images = []
        self.current_guide_idx = 0
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
        self.head_move_count = 0
        
        self.auto_config = AutoCollectionConfig()
        self.auto_motion_plan = None
        self.auto_base_head_q = None
        self.auto_ready_done = False
        self.auto_motion_running = False
        self.auto_stop_requested = False
        self.auto_motion_thread = None
        
        self.last_result_path = None
        self.last_home_reset_path = None
        self.last_dataset_path = None
        self.dataset_saved_in_session = False
        self.current_session_dataset_path = None
        
        self.check_state_moved = False
        
        if self.robot:
            try:
                self.model = self.robot.model()
                self.dyn_model = self.robot.get_dynamics()
            except Exception:
                self.model = None
                self.dyn_model = None
        else:
            self.model = None
            self.dyn_model = None
        
        self.recommended_joint_offset = None
        
        self.marker_calibrator.joint_offsets = self.joint_offsets
        self.joint_calibrator.joint_offsets = self.joint_offsets
        if self.robot:
            try:
                self.robot.joint_offsets = self.joint_offsets
            except AttributeError:
                pass
        
        self.setWindowTitle("Unified Robot Calibration Suite")
        self.resize(1050, 700)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # 1. 200ms poll timer (탭 1, 2, 4 용)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera_status)
        
        # 2. 33ms video timer (탭 3용)
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        
        # 3. Dedicated Temperature Monitor Timer (runs continuously every 2 seconds)
        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.poll_camera_temperature)
        self.temp_timer.start(2000)
        
        self.init_ui()
        self.load_bracket_design_values()
        self.update_applied_offset_label()
        
        # 초기화 시 탭 상태에 맞춰 타이머 활성화
        self.on_left_tab_changed(self.left_tabs.currentIndex())
        
        self.active_worker = None

    def load_offsets_from_yaml(self):
        self.joint_offsets_store = {
            "left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
            "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0}
        }
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                jo = data.get("joint_offset", {})
                for arm in ["left", "right"]:
                    arm_data = jo.get(arm, {})
                    if isinstance(arm_data, dict):
                        self.joint_offsets_store[arm]["joint3"] = float(arm_data.get("joint3", 0.0))
                        self.joint_offsets_store[arm]["joint5"] = float(arm_data.get("joint5", 0.0))
                        self.joint_offsets_store[arm]["joint6"] = float(arm_data.get("joint6", 0.0))
                self.log_msg(f"[INFO] Loaded joint offsets from setting.yaml: "
                             f"R[J3={self.joint_offsets_store['right']['joint3']:.4f}°, "
                             f"J5={self.joint_offsets_store['right']['joint5']:.4f}°, "
                             f"J6={self.joint_offsets_store['right']['joint6']:.4f}°] "
                             f"L[J3={self.joint_offsets_store['left']['joint3']:.4f}°, "
                             f"J5={self.joint_offsets_store['left']['joint5']:.4f}°, "
                             f"J6={self.joint_offsets_store['left']['joint6']:.4f}°]")
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

    def save_offsets_to_yaml(self):
        config_path = CONFIG_PATHS["setting_yaml"]
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            lines = []
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    lines = f.readlines()
            
            jo_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("joint_offset:"):
                    jo_idx = i
                    break

            if jo_idx == -1:
                # If joint_offset doesn't exist, we append it to the end of the file
                if lines and not lines[-1].endswith("\n"):
                    lines.append("\n")
                lines.append("joint_offset:\n")
                lines.append("  left:\n")
                lines.append(f"    joint3: {self.joint_offsets_store['left']['joint3']}\n")
                lines.append(f"    joint5: {self.joint_offsets_store['left'].get('joint5', 0.0)}\n")
                lines.append(f"    joint6: {self.joint_offsets_store['left'].get('joint6', 0.0)}\n")
                lines.append("  right:\n")
                lines.append(f"    joint3: {self.joint_offsets_store['right']['joint3']}\n")
                lines.append(f"    joint5: {self.joint_offsets_store['right'].get('joint5', 0.0)}\n")
                lines.append(f"    joint6: {self.joint_offsets_store['right'].get('joint6', 0.0)}\n")
            else:
                # joint_offset 블록이 끝나는 지점(들여쓰기가 없는 다음 라인 또는 파일 끝)을 찾습니다.
                block_end = len(lines)
                for i in range(jo_idx + 1, len(lines)):
                    line = lines[i]
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if not line.startswith(" ") and not line.startswith("\t"):
                        block_end = i
                        break
                
                new_jo_lines = [
                    "joint_offset:\n",
                    "  left:\n",
                    f"    joint3: {self.joint_offsets_store['left']['joint3']}\n",
                    f"    joint5: {self.joint_offsets_store['left'].get('joint5', 0.0)}\n",
                    f"    joint6: {self.joint_offsets_store['left'].get('joint6', 0.0)}\n",
                    "  right:\n",
                    f"    joint3: {self.joint_offsets_store['right']['joint3']}\n",
                    f"    joint5: {self.joint_offsets_store['right'].get('joint5', 0.0)}\n",
                    f"    joint6: {self.joint_offsets_store['right'].get('joint6', 0.0)}\n"
                ]
                lines = lines[:jo_idx] + new_jo_lines + lines[block_end:]

            with open(config_path, "w") as f:
                f.writelines(lines)
            self.log_msg(f"[SUCCESS] Saved offsets permanently to setting.yaml!")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save setting.yaml: {e}")

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
        self.ip_input = QLineEdit("192.168.30.1:50051")
        if self.ui_only:
            self.ip_input.setText("127.0.0.1:50051")
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
        self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
        self.btn_connect.clicked.connect(self.connect_robot)
        self.btn_connect.setFixedHeight(28)
        connect_head_row.addWidget(self.btn_connect)
        
        # Head checkbox — controls whether head servos are enabled on connect
        self.chk_servo_head = QCheckBox("Head")
        self.chk_servo_head.setChecked(True)
        self.chk_servo_head.setStyleSheet("color: #cccccc;")
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
        self.btn_stop_motion.setStyleSheet("background-color: #ff1744; color: #ffffff; font-weight: bold;")
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
        self.btn_joint_ready.setStyleSheet("background-color: #6a1b9a; color: white;")
        self.btn_joint_ready.clicked.connect(self.move_to_ready_pose_joint)
        
        self.btn_joint_start = QPushButton("START SWEEP")
        self.btn_joint_start.setStyleSheet("background-color: #1565c0; color: white;")
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
        self.btn_marker_ready.setStyleSheet("background-color: #6a1b9a; color: white;")
        self.btn_marker_ready.clicked.connect(self.move_to_ready_pose_marker)
        
        self.btn_marker_start = QPushButton("START SWEEP")
        self.btn_marker_start.setStyleSheet("background-color: #1565c0; color: white;")
        self.btn_marker_start.clicked.connect(self.start_calibration_marker)
        
        self.btn_marker_result = QPushButton("UNIFIED RESULT")
        self.btn_marker_result.setStyleSheet("background-color: #2e7d32; color: white;")
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
        self.btn_full_auto_ready.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        self.btn_full_auto_ready.clicked.connect(self.move_to_ready_full_auto)
        
        self.btn_full_auto_start = QPushButton("START FULL AUTO")
        self.btn_full_auto_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_full_auto_start.clicked.connect(self.start_full_auto)
        
        self.btn_full_auto_apply = QPushButton("APPLY FULL AUTO RESULTS")
        self.btn_full_auto_apply.setStyleSheet("background-color: #e65100; color: white; font-weight: bold;")
        self.btn_full_auto_apply.clicked.connect(self.apply_full_auto_results)
        self.btn_full_auto_apply.setEnabled(False) # Enabled after full auto finishes
        
        full_auto_sublayout.addWidget(QLabel("Full Auto Sequential Calibration:"))
        full_auto_sublayout.addWidget(self.btn_full_auto_ready)
        full_auto_sublayout.addWidget(self.btn_full_auto_start)
        full_auto_sublayout.addWidget(self.btn_full_auto_apply)
        full_auto_sublayout.addStretch()
        full_auto_subtab.setLayout(full_auto_sublayout)
        
        # Add workflow subtabs in order of: Full Auto, Joint Calib, Marker Calib
        self.workflow_tabs.addTab(full_auto_subtab, "Auto")
        self.workflow_tabs.addTab(joint_subtab, "Joint")
        self.workflow_tabs.addTab(marker_subtab, "Marker")
        
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
        desc_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        home_offset_layout.addWidget(desc_label)
        
        btn_row = QHBoxLayout()
        self.btn_home_reset = QPushButton("Home Offset Reset")
        self.btn_home_reset.setStyleSheet("background-color: #d84315; color: white; font-weight: bold;")
        self.btn_home_reset.clicked.connect(self.home_offset_reset)
        self.btn_home_reset.setFixedHeight(28)
        btn_row.addWidget(self.btn_home_reset)

        self.btn_step2_zero_pose = QPushButton("Zero Pose")
        self.btn_step2_zero_pose.setStyleSheet("background-color: #37474f; color: white; font-weight: bold;")
        self.btn_step2_zero_pose.setFixedHeight(28)
        self.btn_step2_zero_pose.clicked.connect(self.step2_zero_pose_check)
        btn_row.addWidget(self.btn_step2_zero_pose)
        
        home_offset_layout.addLayout(btn_row)
        
        hint_label = QLabel("Tip: Double-click any cell in the table below to manually stage individual offsets.")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #2979ff; font-size: 11px; font-weight: bold;")
        home_offset_layout.addWidget(hint_label)
        
        home_offset_box.setLayout(home_offset_layout)
        self.home_offset_box = home_offset_box

        dash_box = QGroupBox("Calibration Status & Monitoring")
        dash_layout = QVBoxLayout()
        dash_layout.setSpacing(4)
        dash_layout.setContentsMargins(8, 4, 8, 4)
        
        # Monitoring Table
        self.tbl_offset_monitor = QTableWidget(2, 3)
        self.tbl_offset_monitor.setHorizontalHeaderLabels(["Joint 6 (Roll/Yaw 2)", "Joint 5 (Wrist Pitch)", "Joint 3 (Elbow)"])
        self.tbl_offset_monitor.setVerticalHeaderLabels(["Right Arm", "Left Arm"])
        self.tbl_offset_monitor.setFixedHeight(110)
        self.tbl_offset_monitor.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_offset_monitor.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_offset_monitor.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.tbl_offset_monitor.setStyleSheet("""
            QTableWidget {
                background-color: #121212;
                color: #00e5ff;
                gridline-color: #2d2d2d;
                font-weight: bold;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #888888;
                font-weight: bold;
                padding: 2px;
                border: 1px solid #2d2d2d;
            }
        """)
        dash_layout.addWidget(self.tbl_offset_monitor)
        
        # Marker Bracket Design Offset UI GroupBox (Nested inside dash_box)
        bracket_box = QGroupBox("Marker Bracket Design Offset (Tf_to_marker)")
        bracket_layout = QVBoxLayout()
        bracket_layout.setSpacing(4)
        bracket_layout.setContentsMargins(6, 6, 6, 6)
        
        input_style = "background-color: #1c1c1c; color: #00e5ff; border: 1px solid #3d3d3d; border-radius: 3px; padding: 2px;"
        
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
        self.btn_apply_bracket.setStyleSheet("background-color: #2979ff; color: white; font-weight: bold; font-size: 11px;")
        self.btn_apply_bracket.setFixedHeight(24)
        self.btn_apply_bracket.clicked.connect(self.apply_bracket_design_values)
        bracket_layout.addWidget(self.btn_apply_bracket)
        
        bracket_box.setLayout(bracket_layout)
        dash_layout.addWidget(bracket_box)
        
        # Apply & Clear buttons for Joint offsets
        btn_joint_layout = QHBoxLayout()
        btn_joint_layout.setSpacing(6)
        
        self.btn_joint_apply = QPushButton("APPLY OFFSET")
        self.btn_joint_apply.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; font-size: 11px;")
        self.btn_joint_apply.clicked.connect(self.apply_joint_offset)
        self.btn_joint_apply.setFixedHeight(24)
        
        self.btn_joint_clear = QPushButton("CLEAR OFFSET")
        self.btn_joint_clear.setStyleSheet("background-color: #555555; color: white; font-weight: bold; font-size: 11px;")
        self.btn_joint_clear.clicked.connect(self.clear_joint_offset)
        self.btn_joint_clear.setFixedHeight(24)
        
        btn_joint_layout.addWidget(self.btn_joint_apply)
        btn_joint_layout.addWidget(self.btn_joint_clear)
        
        dash_layout.addLayout(btn_joint_layout)
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
        
        self.temp_label = QLabel("Camera Temp: -- °C")
        self.temp_label.setStyleSheet("color: #ff5500; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.temp_label)
        
        btn_layout = QHBoxLayout()
        self.btn_monitor = QPushButton("Marker Monitor: OFF")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self.on_monitor_toggled)
        self.btn_monitor.setFixedHeight(26)
        
        self.btn_camera_feed = QPushButton("Camera Feed")
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
        console_title.setStyleSheet("color: #2979ff; margin-bottom: 2px;")
        
        self.btn_show_plot = QPushButton("Show Calibration Plot")
        self.btn_show_plot.setStyleSheet("background-color: #2979ff; color: white; font-weight: bold; font-size: 11px;")
        self.btn_show_plot.setFixedHeight(24)
        self.btn_show_plot.clicked.connect(self.open_plot_dialog)
        
        log_header.addWidget(console_title)
        log_header.addStretch()
        log_header.addWidget(self.btn_show_plot)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        
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
        self.video_label.setMinimumSize(640, 480)
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
        self.btn_int_capture.setStyleSheet("background-color: #1565c0; color: white; font-size: 13px;")
        self.btn_int_capture.clicked.connect(self.capture_intrinsics_frame)
        controls_layout.addWidget(self.btn_int_capture)
        
        self.btn_int_calibrate = QPushButton("RUN CALIBRATION")
        self.btn_int_calibrate.setMinimumHeight(45)
        self.btn_int_calibrate.setStyleSheet("background-color: #2e7d32; color: white; font-size: 13px;")
        self.btn_int_calibrate.clicked.connect(self.run_intrinsics_calibration)
        controls_layout.addWidget(self.btn_int_calibrate)
        
        self.btn_int_save = QPushButton("SAVE PARAMETERS")
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #e65100; color: white; font-size: 13px;")
        self.btn_int_save.clicked.connect(self.save_intrinsics_calibration)
        self.btn_int_save.setEnabled(False)
        controls_layout.addWidget(self.btn_int_save)
        
        self.btn_int_reset = QPushButton("RESET CAPTURES")
        self.btn_int_reset.setMinimumHeight(30)
        self.btn_int_reset.setStyleSheet("background-color: #37474f; color: white;")
        self.btn_int_reset.clicked.connect(self.reset_intrinsics_captures)
        controls_layout.addWidget(self.btn_int_reset)
        
        controls_box.setLayout(controls_layout)
        
        int_right = QVBoxLayout()
        
        stats_box2 = QGroupBox("Capture Stats")
        stats_layout2 = QHBoxLayout()
        self.lbl_captured = QLabel("Captured Frames: 0")
        self.lbl_captured.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #2979ff;")
        
        self.lbl_temp = QLabel("Camera Temp: -- °C")
        self.lbl_temp.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #ff5500;")
        
        stats_layout2.addWidget(self.lbl_captured)
        stats_layout2.addStretch()
        stats_layout2.addWidget(self.lbl_temp)
        stats_box2.setLayout(stats_layout2)
        
        int_right.addWidget(stats_box2)
        int_right.addWidget(controls_box) # Placed below stats box!
        int_right.addStretch()
        
        camera_tab_layout.addLayout(int_left, 2)
        camera_tab_layout.addLayout(int_right, 1)
        camera_tab.setLayout(camera_tab_layout)
        
        # ==========================================
        # Overview Tab
        # ==========================================
        overview_tab = QWidget()
        overview_layout = QVBoxLayout()
        overview_layout.setContentsMargins(20, 20, 20, 20)
        
        overview_title = QLabel("Calibration Process Overview")
        overview_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        overview_title.setAlignment(Qt.AlignCenter)
        overview_layout.addWidget(overview_title)
        
        self.overview_img = QLabel()
        process_pix = QPixmap("img/process.png")
        if not process_pix.isNull():
            self.overview_img.setPixmap(process_pix.scaled(1000, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.overview_img.setText("[img/process.png not found]")
        self.overview_img.setAlignment(Qt.AlignCenter)
        self.overview_img.setStyleSheet("border: 2px solid #555; background-color: #222; border-radius: 8px;")
        overview_layout.addWidget(self.overview_img, stretch=1)
        
        self.btn_start_wizard = QPushButton("Start Wizard")
        self.btn_start_wizard.setStyleSheet("background-color: #d84315; color: white; font-weight: bold; font-size: 18px; padding: 10px;")
        self.btn_start_wizard.setFixedWidth(400)
        self.btn_start_wizard.clicked.connect(self.show_wizard_ui)
        overview_layout.addWidget(self.btn_start_wizard, alignment=Qt.AlignCenter)
        
        self.wizard_widget = CalibrationWizardWidget(self)
        self.wizard_widget.setVisible(False)
        overview_layout.addWidget(self.wizard_widget, stretch=1)
        
        overview_tab.setLayout(overview_layout)

        # ==========================================
        # Step 1 Tab: Contains Main + Camera as sub-tabs
        # ==========================================
        step1_tab = QWidget()
        step1_layout = QVBoxLayout()
        step1_layout.setContentsMargins(0, 0, 0, 0)
        
        self.step1_tabs = QTabWidget()
        self.step1_tabs.currentChanged.connect(self._on_step1_subtab_changed)
        self.step1_tabs.addTab(main_tab, "Main")
        self.step1_tabs.addTab(camera_tab, "Camera")
        
        step1_layout.addWidget(self.step1_tabs)
        step1_tab.setLayout(step1_layout)
        
        self.left_tabs.addTab(overview_tab, "Overview")
        self.left_tabs.addTab(step1_tab, "Step 1")
        
        # ==========================================
        # Step 2 Tab: Shared widgets + empty Box1
        # ==========================================
        step2_tab = QWidget()
        step2_layout = QVBoxLayout()
        step2_layout.setContentsMargins(5, 5, 5, 5)
        
        # Step 2 columns: Left (conn_head + home_offset + Box1), Right (status + log)
        step2_columns = QHBoxLayout()
        
        # Step 2 Left Column
        self.step2_left_col = QVBoxLayout()
        # Placeholders for reparented widgets — they will be moved here on tab switch
        # conn_head_box and home_offset_box go here
        
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
        self.step2_mode_sel.currentTextChanged.connect(self.update_step2_est_samples)
        QTimer.singleShot(100, self.update_step2_est_samples)
        
        # Actions Box (mirrors calibration_ui Actions section)
        actions_box = QGroupBox("Actions")
        actions_box.setStyleSheet("QGroupBox { border: 1px solid #555; border-radius: 4px; } QGroupBox::title { color: #ff9800; font-weight: bold; }")
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(4)
        actions_layout.setContentsMargins(6, 6, 6, 6)
        
        # Top row: Zero Pose Check, Stop (Home Offset Reset and Camera Feed excluded — already in Step 1)
        top_action_row = QHBoxLayout()
        self.btn_step2_zero_pose = QPushButton("Zero Pose Check")
        self.btn_step2_zero_pose.setStyleSheet("background-color: #37474f; color: white; font-weight: bold;")
        self.btn_step2_zero_pose.setFixedHeight(28)
        self.btn_step2_zero_pose.clicked.connect(self.step2_zero_pose_check)
        top_action_row.addWidget(self.btn_step2_zero_pose)
        
        self.btn_step2_stop = QPushButton("Stop")
        self.btn_step2_stop.setStyleSheet("background-color: #ff1744; color: white; font-weight: bold;")
        self.btn_step2_stop.setFixedHeight(28)
        self.btn_step2_stop.clicked.connect(self.stop_motion)
        top_action_row.addWidget(self.btn_step2_stop)
        actions_layout.addLayout(top_action_row)
        
        # Numbered actions row 1
        act_row1 = QHBoxLayout()
        self.btn_step2_init_pose = QPushButton("1) Init Pose")
        self.btn_step2_init_pose.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        self.btn_step2_init_pose.setFixedHeight(28)
        self.btn_step2_init_pose.clicked.connect(self.step2_init_pose)
        act_row1.addWidget(self.btn_step2_init_pose)
        
        self.btn_step2_auto_motion = QPushButton("2) Auto Motion")
        self.btn_step2_auto_motion.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;")
        self.btn_step2_auto_motion.setFixedHeight(28)
        self.btn_step2_auto_motion.clicked.connect(self.step2_auto_motion)
        act_row1.addWidget(self.btn_step2_auto_motion)
        actions_layout.addLayout(act_row1)
        
        # Numbered actions row 2
        act_row2 = QHBoxLayout()
        self.btn_step2_calculate = QPushButton("3) Calculate")
        self.btn_step2_calculate.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_step2_calculate.setFixedHeight(28)
        self.btn_step2_calculate.clicked.connect(self.step2_calculate)
        act_row2.addWidget(self.btn_step2_calculate)
        
        self.btn_step2_clear = QPushButton("4) Clear Samples")
        self.btn_step2_clear.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        self.btn_step2_clear.setFixedHeight(28)
        self.btn_step2_clear.clicked.connect(self.step2_clear_samples)
        act_row2.addWidget(self.btn_step2_clear)
        actions_layout.addLayout(act_row2)
        
        # Numbered actions row 3
        act_row3 = QHBoxLayout()
        self.btn_step2_apply_home = QPushButton("5) Apply Home Offset")
        self.btn_step2_apply_home.setStyleSheet("background-color: #e65100; color: white; font-weight: bold;")
        self.btn_step2_apply_home.setFixedHeight(28)
        self.btn_step2_apply_home.clicked.connect(self.step2_apply_home_offset)
        act_row3.addWidget(self.btn_step2_apply_home)
        
        self.btn_step2_check_state = QPushButton("6) Check Calibration State")
        self.btn_step2_check_state.setStyleSheet("background-color: #00838f; color: white; font-weight: bold;")
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
        
        self.left_tabs.addTab(step2_tab, "Step 2")
        
        # Keep references for reparenting logic
        # Step 1 Main tab column layout indices for reinserting shared widgets
        self._step1_col1 = col1_layout
        self._step1_col2 = col2_layout
        self._step1_col3 = col3_layout
        
        # Assemble full-width tabs
        main_layout.addWidget(self.left_tabs)
        
        # Create outer vertical layout to place QUIT button at the very bottom
        outer_layout = QVBoxLayout()
        outer_layout.addLayout(main_layout)
        
        self.btn_quit_global = QPushButton("QUIT")
        self.btn_quit_global.setMinimumHeight(40)
        self.btn_quit_global.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold; font-size: 14px;")
        self.btn_quit_global.clicked.connect(self.close)
        outer_layout.addWidget(self.btn_quit_global)
        
        self.setLayout(outer_layout)
        
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

    @property
    def is_mock(self) -> bool:
        return self.robot is None

    # --- Common Helper Functions ---
    def _log_msg_slot(self, msg):
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

    def log_msg(self, msg):
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        if QThread.currentThread() == QApplication.instance().thread():
            self._log_msg_slot(msg)
        else:
            self.log_signal_safe.emit(msg)

    def _write_step2_log(self, msg):
        import os
        from core.paths import CONFIG_PATHS
        log_file = os.path.join(CONFIG_PATHS["txt_dir"], "step2_capture_log.txt")
        try:
            os.makedirs(CONFIG_PATHS["txt_dir"], exist_ok=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            self.log_msg(f"Failed to write step2 log: {e}")

    def safe_cancel_control(self):
        # 동일 gRPC 커넥션에 대한 동시 gRPC 호출로 인한 C++ SDK Segfault 방지
        if self.robot is None:
            return
        if self.is_mock:
            self.log_msg("[STOP] (Mock Mode) Cancel control called.")
            return

        try:
            import rby1_sdk as rby
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip().lower()
            temp_robot = rby.create_robot(addr, model)
            if temp_robot.connect():
                temp_robot.cancel_control()
                temp_robot.disconnect()
                self.log_msg("[INFO] Control cancelled safely via temporary connection.")
            else:
                self.robot.cancel_control()
        except Exception as e:
            try:
                self.robot.cancel_control()
            except Exception:
                pass

    def connect_robot(self):
        from core.calibration.CalibratorBase import BaseCalibrator

        if self.robot:
            self.log_msg("[INFO] Disconnecting from robot...")
            if not self.is_mock:
                MarkerCalibrator.terminate_robot(self.robot)
            self.robot = None
            self.model = None
            self.dyn_model = None
            self.robot_version = "1.2"
            self.marker_calibrator.robot = None
            self.marker_calibrator.robot_version = "1.2"
            self.joint_calibrator.robot = None
            self.joint_calibrator.robot_version = "1.2"
            self.update_joint_modes()
            self.load_offsets_from_yaml()
            self.update_applied_offset_label()
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            if hasattr(self, 'chk_servo_head'):
                self.chk_servo_head.setEnabled(True)
            self.log_msg("[INFO] Robot disconnected.")
            return

        try:
            addr = self.ip_input.text().strip()
            model = self.model_input.currentText().strip()
            
            # Read head checkbox state (like calibration_ui's servo_head)
            head_enabled = self.chk_servo_head.isChecked() if hasattr(self, 'chk_servo_head') else True
            self.include_head_motion = head_enabled
            
            # Update connection button to loading state
            self.btn_connect.setText("CONNECTING...")
            self.btn_connect.setStyleSheet("background-color: #ffb74d; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(False)
            self.log_msg(f"[INFO] 로봇 연결 시도 중... (IP: {addr})")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # 1. Create and connect robot
            robot = rby.create_robot(addr, model)
            if not robot.connect():
                raise ConnectionError(f"Failed to connect robot at {addr}")
            time.sleep(1)

            # 2. Safety check: Verify actual connected robot model matches expected model
            try:
                robot_info = robot.get_robot_info()
                actual_model = robot_info.robot_model_name.lower()
                expected_model = model.lower()
                if actual_model != expected_model:
                    self.log_msg(f"[WARNING] Model mismatch! UI selected model: {model}, but actual robot model is: {robot_info.robot_model_name}. Auto-reconnecting with actual model...")
                    robot.disconnect()
                    robot = rby.create_robot(addr, robot_info.robot_model_name.lower())
                    if not robot.connect():
                        raise ConnectionError(f"Failed to connect robot {addr} with actual model {robot_info.robot_model_name}")
                    time.sleep(1)
            except Exception as e:
                self.log_msg(f"[ERROR] Safety check failed: {e}")

            # 3. Turn on power if not already ON
            try:
                if not robot.is_power_on(".*"):
                    self.log_msg("[INFO] Power is not ON. Turning overall power on...")
                    if not robot.power_on(".*"):
                        raise RuntimeError("Failed to turn power on.")
                    time.sleep(1)
                else:
                    self.log_msg("[INFO] Power is already ON.")
            except Exception as e:
                self.log_msg(f"[ERROR] Power configuration failed: {e}")

            # 4. Turn on servos if not already ON
            try:
                if not robot.is_servo_on(".*"):
                    self.log_msg("[INFO] Turning servos on...")
                    if not robot.servo_on(".*"):
                        raise RuntimeError("Failed to turn servos on.")
                    time.sleep(1)
                else:
                    self.log_msg("[INFO] Servos are ON.")
            except Exception as e:
                self.log_msg(f"[ERROR] Servo configuration failed: {e}")

            # 5. Enable control manager with False (standard mode, unlimited mode disabled)
            try:
                cm_state = robot.get_control_manager_state()
                if cm_state.state in [
                    rby.ControlManagerState.State.MajorFault,
                    rby.ControlManagerState.State.MinorFault,
                ]:
                    self.log_msg("[WARNING] Control manager is in fault state. Resetting...")
                    robot.reset_fault_control_manager()
                    time.sleep(0.5)

                cm_state = robot.get_control_manager_state()
                if cm_state.state == rby.ControlManagerState.State.Enabled:
                    self.log_msg("[INFO] Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...")
                    robot.disable_control_manager()
                    time.sleep(0.5)

                self.log_msg("[INFO] Enabling control manager with unlimited_mode_enabled=True...")
                if not robot.enable_control_manager(unlimited_mode_enabled=True):
                    raise RuntimeError("Failed to enable control manager.")
                time.sleep(1)
            except Exception as e:
                self.log_msg(f"[ERROR] Control manager configuration failed: {e}")

            self.robot = robot
                
            if self.robot:
                self.model = self.robot.model()
                self.dyn_model = self.robot.get_dynamics()
                self.marker_calibrator.robot = self.robot
                self.joint_calibrator.robot = self.robot
                
                # Check for head presence
                if len(self.model.head_idx) == 0:
                    if hasattr(self, 'chk_servo_head'):
                        self.chk_servo_head.setChecked(False)
                        self.chk_servo_head.setEnabled(False)
                    self.include_head_motion = False
                    self.log_msg("[INFO] No head joints detected. Head motion disabled (Torso base).")
                else:
                    if hasattr(self, 'chk_servo_head'):
                        self.chk_servo_head.setEnabled(True)
                
                # Determine version classification automatically
                detected_version = "1.2"
                if not self.is_mock:
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
                else:
                    detected_version = "1.3" if model == "m" else "1.2"
                    self.log_msg(f"[INFO] Connected to mock robot. Using default version: {detected_version}")
                    print(f"[INFO] Connected to mock robot. Using default version: {detected_version}")
                
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

                # Update UI modes and offsets based on version classification
                self.update_joint_modes()
                self.load_offsets_from_yaml()
                self.update_applied_offset_label()
                self.load_bracket_design_values()

                # Setup SimulatedMarkerTransform if simulator is connected in UI Mode
                if self.ui_only and not self.is_mock:
                    self.marker_st = SimulatedMarkerTransform(self.robot, self.marker_calibrator.camera_config, self.robot_version)
                    self.marker_calibrator.marker_st = self.marker_st
                    self.joint_calibrator.marker_st = self.marker_st
                    self.log_msg("[INFO] Configured SimulatedMarkerTransform for simulation motion.")
                    
                if self.ui_only:
                    self.step2_mode_sel.setCurrentText("sim")
                    self.log_msg("[INFO] Automatically switched Step 2 Mode to 'sim' because camera is not connected.")

                self.log_msg(f"[INFO] Robot successfully connected and initialized (Classified Version: {detected_version}).")
                self.btn_connect.setText("CONNECTED")
                self.btn_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 4px 8px; font-size: 11px;")
                self.btn_connect.setEnabled(True)
            else:
                self.log_msg("[ERROR] Robot initialization failed. Check IP.")
                self.btn_connect.setText("CONNECT")
                self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
                self.btn_connect.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_msg(f"[ERROR] Connection failure: {e}")
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 4px 8px; font-size: 11px;")
            self.btn_connect.setEnabled(True)

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
        self.overview_img.setVisible(False)
        self.btn_start_wizard.setVisible(False)
        self.wizard_widget.setVisible(True)
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
        # Camera Tab이 켜져있을 때는 poll_camera_status 생략 (update_video_frame이 처리함)
        if self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1:
            return
            
        try:
            # 좌/우 중 하나라도 인식되었는지 확인하기 위해 "all"로 검출 수행
            res_all = self.marker_st.get_marker_transform(sampling_time=0, side="all")
            detected = bool(res_all and len(res_all) > 0)
            self.update_marker_indicator(detected)
            
            if not self.btn_monitor.isChecked():
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText("Position: Monitor Off")
                return
                
            # 모니터가 켜진 경우, 현재 active arm side의 개별 마커 좌표 조회 및 표시
            res = self.marker_st.get_marker_transform(sampling_time=0, side=self.arm_side)
            if res and len(res) > 0:
                pose = np.array(res[0]).reshape(4, 4) if isinstance(res, list) else np.array(list(res.values())[0]).reshape(4, 4)
                x, y, z = pose[:3, 3] * 1000.0
                
                self.log_msg(f"[LIVE] Marker ({self.arm_side}) X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm")
                
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText(f"Position ({self.arm_side}): X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f} mm")
            else:
                if hasattr(self, 'lbl_marker_pos'):
                    self.lbl_marker_pos.setText(f"Position ({self.arm_side}): Marker Not Detected")
        except Exception:
            pass

    def poll_camera_temperature(self):
        if self.ui_only or self.marker_st is None:
            return
        try:
            if hasattr(self.marker_st, 'camera') and self.marker_st.camera is not None:
                temp = self.marker_st.camera.get_camera_temperature()
                if temp is not None:
                    text = f"Camera Temp: {temp:.1f} °C"
                    if hasattr(self, 'temp_label') and self.temp_label is not None:
                        self.temp_label.setText(text)
                    if hasattr(self, 'lbl_temp') and self.lbl_temp is not None:
                        self.lbl_temp.setText(text)
        except Exception:
            pass

    def _on_step1_subtab_changed(self, index):
        """Handle sub-tab switching within Step 1 (Main=0, Camera=1)."""
        if not hasattr(self, 'poll_timer') or not hasattr(self, 'video_timer'):
            return
        # Only act if Step 1 is the active top-level tab
        if self.left_tabs.currentIndex() != 1:
            return
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        if index == 1 or dialog_visible:  # Camera sub-tab
            if self.poll_timer.isActive():
                self.poll_timer.stop()
            self.video_timer.start(50)
        else:  # Main sub-tab
            if self.video_timer.isActive():
                self.video_timer.stop()
            if not self.ui_only and self.marker_st is not None:
                self.poll_timer.start(200)

    def on_left_tab_changed(self, index):
        # 방어적 코드: 타이머 객체가 아직 미생성된 상태이면 처리를 생략
        if not hasattr(self, 'poll_timer') or not hasattr(self, 'video_timer'):
            return

        # Reparent shared widgets between Step 1 and Step 2
        self._reparent_shared_widgets(index)

        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()

        if index == 1:  # Step 1 tab
            # Delegate to sub-tab handler
            self._on_step1_subtab_changed(self.step1_tabs.currentIndex())
        elif index == 2:  # Step 2 tab
            # Step 2 has no camera feed — stop video, start poll
            if self.video_timer.isActive():
                self.video_timer.stop()
            if dialog_visible:
                self.video_timer.start(50)
            elif not self.ui_only and self.marker_st is not None:
                self.poll_timer.start(200)
        else:
            # Check if wizard is running and on slide 1, or dialog visible
            wizard_slide1_active = (hasattr(self, "wizard_widget") and self.wizard_widget.isVisible() and self.wizard_widget.stacked_widget.currentIndex() == 0)
            if wizard_slide1_active or dialog_visible:
                if self.poll_timer.isActive():
                    self.poll_timer.stop()
                self.video_timer.start(50)
            else:
                if self.video_timer.isActive():
                    self.video_timer.stop()
                if not self.ui_only and self.marker_st is not None:
                    self.poll_timer.start(200)

    def _reparent_shared_widgets(self, top_tab_index):
        """Move shared GroupBoxes between Step 1 Main and Step 2 layouts."""
        if not hasattr(self, 'conn_head_box') or self.conn_head_box is None:
            return
        if not hasattr(self, 'step2_left_col'):
            return

        if top_tab_index == 2:  # Switching TO Step 2
            # Move shared widgets into Step 2 columns
            # Left column: conn_head_box, home_offset_box at top, then box1 fills rest
            self.step2_left_col.insertWidget(0, self.conn_head_box)
            self.step2_left_col.insertWidget(1, self.home_offset_box)
            # box1 stays at position 2 (already there)

            # Right column: status_box, log_box
            self.step2_right_col.insertWidget(0, self.status_box)
            self.step2_right_col.insertWidget(1, self.log_box)
            # Set stretch for log_box in step2
            self.step2_right_col.setStretchFactor(self.log_box, 1)

            # Hide "Show Calibration Plot" button in Step 2
            if hasattr(self, 'btn_show_plot'):
                self.btn_show_plot.hide()

        else:  # Switching TO Step 1 (or any other tab)
            # Move shared widgets back into Step 1 Main columns
            self._step1_col1.insertWidget(0, self.conn_head_box)

            self._step1_col2.insertWidget(0, self.home_offset_box)

            self._step1_col3.insertWidget(0, self.status_box)
            self._step1_col3.insertWidget(1, self.log_box)
            self._step1_col3.setStretchFactor(self.log_box, 1)

            # Show "Show Calibration Plot" button back in Step 1
            if hasattr(self, 'btn_show_plot'):
                self.btn_show_plot.show()

    # =============================================
    # Step 2 Action Handlers
    # =============================================

    def update_step2_est_samples(self, *args):
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
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

                    active_arms = ["right", "left"]
                    temp_plan = build_incremental_motion_plan(
                        self.robot, self.dyn_model, self.auto_config, active_arms
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
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.ensure_result_dir()
        dataset_path = result_dir / f"dataset_{timestamp}.npz"
        result_path = result_dir / f"result_{timestamp}.json"
        return dataset_path, result_path

    def get_latest_result_path(self):
        if self.last_result_path is not None and Path(self.last_result_path).exists():
            return Path(self.last_result_path)

        result_dir = self.ensure_result_dir()
        result_files = sorted(
            result_dir.glob("result_*.json"),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )
        if not result_files:
            raise RuntimeError(f"No calibration result JSON found in {result_dir}")

        self.last_result_path = result_files[0]
        return self.last_result_path

    def get_home_reset_path_for_result(self, result_path):
        path = Path(current_dir) / "config" / "home_reset_baseline.json"
        if path.exists():
            self.last_home_reset_path = path
            return path
        return None

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
                import json
                with open(path, "r") as f:
                    data = json.load(f)
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

    def move_home_offset_candidate_path(self, json_path, label, arm, include_head):
        self.ensure_home_offset_robot()
        result = move_to_offset_candidate_from_json(
            robot=self.robot,
            model=self.model,
            arm=arm,
            json_path=str(json_path),
            include_head=include_head,
            minimum_time=10,
            move_zero_first=True,
        )

        self.log_msg(f"\n===== HOME OFFSET PREVIEW: {label} =====")
        self.log_msg(f"JSON: {json_path}")
        self.log_msg(f"Arm: {result['arm']}")
        if result.get("right_offset_deg") is not None:
            self.log_msg(f"Right move offset (deg): {result['right_offset_deg']}")
        if result.get("left_offset_deg") is not None:
            self.log_msg(f"Left move offset (deg): {result['left_offset_deg']}")
        if result.get("head_offset_deg") is not None:
            self.log_msg(f"Head move offset (deg): {result['head_offset_deg']}")
        self.log_msg("Preview move complete. Inspect the robot pose before applying.")
        return result

    def move_to_check_position_candidate_path(self, json_path, label, arm, include_head):
        self.ensure_home_offset_robot()
        
        # Load offsets from json
        from homeoffset_core import load_offset_from_json, _split_arm_offset, _normalize_head_offset, movej
        offset_rad, head_offset_rad = load_offset_from_json(str(json_path))
        
        apply_mode, right_offset_rad, left_offset_rad = _split_arm_offset(
            self.model,
            arm,
            offset_rad,
        )
        
        right_offset_to_apply = -right_offset_rad
        left_offset_to_apply = -left_offset_rad
        head_offset_full, head_offset_size = _normalize_head_offset(
            self.model,
            head_offset_rad,
            include_head,
        )
        head_offset_to_apply = None if head_offset_full is None else -head_offset_full
        
        # Determine version key
        version_key = "v1.2"
        if self.robot is not None:
            try:
                robot_info = self.robot.get_robot_info()
                raw_version = robot_info.robot_model_version
                if "1.3" in raw_version:
                    version_key = "v1.3"
            except Exception:
                pass
                
        # Load check_calib ready poses
        ready_poses_path = Path(current_dir) / "config" / "ready_poses.yaml"
        check_calib_joints = {
            "right_arm": [-90.0, -45.0, 73.0, -107.0, 90.0, 90.0, 0.0],
            "left_arm": [-90.0, 45.0, -73.0, -107.0, -90.0, 90.0, 0.0]
        }
        if ready_poses_path.exists():
            try:
                import yaml
                with open(ready_poses_path, "r") as f:
                    ready_poses = yaml.safe_load(f) or {}
                if version_key in ready_poses and "check_calib" in ready_poses[version_key]:
                    check_calib_joints = ready_poses[version_key]["check_calib"]
            except Exception as e:
                self.log_msg(f"[WARNING] Failed to parse ready_poses.yaml: {e}")
                
        # 1. Move to 1st ready pose (like check_calibration_state)
        active_arms = []
        if arm in ("right", "both"):
            active_arms.append("right")
        if arm in ("left", "both"):
            active_arms.append("left")
            
        self.log_msg(f"\n[Check Position] Step 1: Moving to Joint Ready Pose...")
        ok = movej(
            self.robot,
            torso=np.deg2rad([0, 30, -60, 30, 0, 0]),
            right_arm=np.deg2rad([-45, -30, 0, -90, 0, 45, 0]) if "right" in active_arms else np.deg2rad([0, 0, 0, -90, 0, 0, 0]),
            left_arm=np.deg2rad([-45, 30, 0, -90, 0, 45, 0]) if "left" in active_arms else np.deg2rad([0, 0, 0, -90, 0, 0, 0]),
            minimum_time=5
        )
        if not ok:
            raise RuntimeError("Failed to move robot to Step 1 Ready Pose")
        time.sleep(2.0)
        
        # 2. Move to Check Position with offsets added
        self.log_msg(f"[Check Position] Step 2: Moving to Check Pose with Offsets...")
        
        right_target = np.deg2rad(check_calib_joints["right_arm"]) + right_offset_to_apply
        left_target = np.deg2rad(check_calib_joints["left_arm"]) + left_offset_to_apply
        
        head_zero_pose = np.zeros(len(self.model.head_idx)) if include_head else None
        head_target_pose = None
        if include_head:
            head_target_pose = (
                head_zero_pose
                if head_offset_to_apply is None
                else head_zero_pose + head_offset_to_apply
            )
            
        ok = movej(
            self.robot,
            torso=np.deg2rad([0, 30, -60, 30, 0, 0]),
            right_arm=right_target,
            left_arm=left_target,
            head=head_target_pose if include_head else None,
            minimum_time=10
        )
        if not ok:
            raise RuntimeError("Failed to move robot to Step 2 Check Pose")
        time.sleep(2.0)
        
        offset_to_apply = np.concatenate([right_offset_to_apply, left_offset_to_apply])
        head_offset_deg = None
        if head_offset_to_apply is not None:
            head_offset_deg = np.rad2deg(head_offset_to_apply[:head_offset_size]).tolist()
            
        result = {
            "status": "success",
            "arm": apply_mode,
            "offset_deg": np.rad2deg(offset_to_apply).tolist(),
            "right_offset_deg": np.rad2deg(right_offset_to_apply).tolist(),
            "left_offset_deg": np.rad2deg(left_offset_to_apply).tolist(),
            "head_offset_deg": head_offset_deg,
        }
        
        self.log_msg(f"\n===== HOME OFFSET PREVIEW: {label} =====")
        self.log_msg(f"JSON: {json_path}")
        self.log_msg(f"Arm: {result['arm']}")
        if result.get("right_offset_deg") is not None:
            self.log_msg(f"Right move offset (deg): {result['right_offset_deg']}")
        if result.get("left_offset_deg") is not None:
            self.log_msg(f"Left move offset (deg): {result['left_offset_deg']}")
        if result.get("head_offset_deg") is not None:
            self.log_msg(f"Head move offset (deg): {result['head_offset_deg']}")
        self.log_msg("Preview move complete. Inspect the robot pose before applying.")
        return result

    def apply_current_pose_home_offset(self, arm, include_head):
        self.ensure_home_offset_robot()
        result = reset_current_pose_home_offsets(
            self.robot,
            self.model,
            arm=arm,
            include_head=include_head,
        )

        # Robot reconnection should be done in the main thread to avoid GUI thread safety issues.
        # We will signal the caller to handle the reconnection.
        result['needs_reconnect'] = True
        return result
        return result

    def run_auto_motion_step_blocking(self):
        if self.robot is None or self.model is None:
            raise RuntimeError("Robot is not connected.")

        pose_target = self.get_auto_pose_target_count()
        if self.head_move_count >= pose_target:
            self.log_msg("Auto motions have already been executed.")
            return True

        if not self.auto_ready_done:
            raise RuntimeError("Please move to Init Pose first.")

        active_arms = ["right", "left"]

        # Re-build incremental motion plan based on the CURRENT (possibly teached) pose
        if self.auto_motion_plan is None or self.head_move_count == 0:
            self.log_msg(f"Building motion plan based on current pose... (Angle={self.auto_config.angle_step_deg}deg, Pos={self.auto_config.position_step_m}m, StepX={self.auto_config.step_x_m}m, MaxX={self.auto_config.max_x}m)")
            self.auto_motion_plan = build_incremental_motion_plan(
                self.robot, self.dyn_model, self.auto_config, active_arms
            )
            self.update_head_pose_status()
            self.update_step2_est_samples()

        if self.include_head_motion and self.auto_base_head_q is None:
            head_cfg = get_head_config(self.model)
            if head_cfg["head_idx"] is not None:
                self.auto_base_head_q = self.robot.get_state().position[head_cfg["head_idx"]].copy()
                self.log_msg(f"Auto base head pose (deg): {np.round(np.rad2deg(self.auto_base_head_q), 3)}")
            else:
                self.auto_base_head_q = None
                self.include_head_motion = False

        if self.auto_stop_requested:
            self.log_msg("Auto Motion stopped by user.")
            return False

        motion_plan_step = self.auto_motion_plan[self.head_move_count]
        
        motion_info = execute_auto_motion_step(
            robot=self.robot,
            config=self.auto_config,
            motion_plan_step=motion_plan_step,
            active_arms=active_arms,
            include_head_motion=self.include_head_motion,
        )
        self.log_msg(f"Auto motion done: {motion_plan_step['desc']}")

        if self.auto_stop_requested:
            self.log_msg("Auto Motion stopped by user.")
            return False

        q_arm, q_head, T_meas = self.capture_one_sample()
        if q_arm is None:
            self.head_move_count += 1
            self.update_head_pose_status()
            self.log_msg("Capture failed after motion. This pose is skipped.")
            return False

        self.shared_arm_q_list.append(q_arm)
        if q_head is not None:
            self.shared_head_q_list.append(q_head)
        self.shared_T_list.append(T_meas)
        self.head_move_count += 1
        self.update_sample_counts()
        self.update_head_pose_status()
        return True

    def move_to_all_auto_motions(self):
        if not self.auto_ready_done:
            raise RuntimeError("Please move to Init Pose first.")

        try:
            self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
            self.auto_config.position_step_m = float(self.step2_pos_step.text())
            self.auto_config.step_x_m = float(self.step2_step_x.text())
            self.auto_config.max_x = float(self.step2_max_x.text())
        except Exception as e:
            self.log_msg(f"Failed to read auto config: {e}. Using current values.")

        if self.auto_motion_plan is None or len(self.auto_motion_plan) == 0:
            self.log_msg("Motion plan is missing or empty. Re-building...")
            active_arms = ["right", "left"]
            self.auto_motion_plan = build_incremental_motion_plan(
                self.robot, self.dyn_model, self.auto_config, active_arms
            )

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

        self.auto_motion_thread = Step2AutoMotionWorker(self, parent=self)
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
        
        # Auto-slice for single-arm mode if data has both
        active_arms = ["right", "left"]
        optimize_head = self.include_head_motion
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

    def update_sample_counts(self):
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
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
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
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

    def capture_one_sample(self):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")

        cfg = get_both_arm_config(self.model, version=self.get_robot_version())
        head_idx = self.get_capture_head_idx()

        # In sim mode bypass camera and return dummy marker data
        if self.step2_mode_sel.currentText() == "sim":
            state = self.robot.get_state()
            q_full = state.position.copy()
            q_arm = q_full[cfg["arm_idx"]].copy()
            q_head = q_full[head_idx].copy() if head_idx is not None else None
            # MOCK_GT_OFFSETS 및 기구학 모델(FK)을 반영한 현실적인 가상 마커 포즈 생성
            T_meas_right = self.marker_calibrator.get_simulated_marker_pose("right", q_actual=q_full)
            T_meas_left = self.marker_calibrator.get_simulated_marker_pose("left", q_actual=q_full)
            T_meas = np.stack([T_meas_right, T_meas_left], axis=0)
            self.log_msg("Sim/Test mode: Using simulated marker data based on MOCK_GT_OFFSETS and FK.")
        else:
            q_arm, q_head, T_meas = capture_robot_sample(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                marker_transform=self.marker_st,
                head_idx=head_idx,
                side="all",
            )
        if T_meas is None:
            self.log_msg("Marker not detected.")
            return None, None, None

        status_lines = []
        status_lines.append(f"q_arm = {np.round(q_arm, 3)}")
        if q_head is not None:
            status_lines.append(f"q_head = {np.round(q_head, 3)}")
        else:
            status_lines.append("q_head = None")
        status_lines.append(f"marker_right =\n{np.round(T_meas[0], 3)}")
        status_lines.append(f"marker_left =\n{np.round(T_meas[1], 3)}")
        
        status_str = "\n".join(status_lines)
        self._write_step2_log("--- Captured Sample ---\n" + status_str + "\n")
        
        return q_arm, q_head, T_meas

    def run_optimizer(
        self,
        active_arms,
        optimize_head,
        optimize_camera,
        q_arm_list,
        q_head_list,
        T_meas_list,
        result_path,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1.0,
        solver_type="QP Solver",
        use_sag=False,
    ):
        if self.model is None:
            raise RuntimeError("Robot is not connected.")

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

        head_cfg = get_head_config(self.model)

        apply_limits = getattr(self, "apply_joint_offset_flag", False)
        joint_offsets = None
        if apply_limits:
            joint_offsets = {
                "right": {
                    "joint3": self.joint_offsets_store["right"].get("joint3", 0.0),
                    "joint5": self.joint_offsets_store["right"].get("joint5", 0.0),
                    "joint6": self.joint_offsets_store["right"].get("joint6", 0.0),
                },
                "left": {
                    "joint3": self.joint_offsets_store["left"].get("joint3", 0.0),
                    "joint5": self.joint_offsets_store["left"].get("joint5", 0.0),
                    "joint6": self.joint_offsets_store["left"].get("joint6", 0.0),
                }
            }
            self.log_msg(f"[INFO] Applying joint offset bounds: {joint_offsets}")

        if solver_type == "QP Solver":
            if optimize_head and optimize_camera:
                self.log_msg("\n[INFO] === 1-PASS QP: Optimizing Arm + Head + Camera Trans (Camera Rot locked) ===")
                actual_lambda_cam_rot = 1e6
            else:
                actual_lambda_cam_rot = lambda_cam_rot

            optimizer = QPCalibrationOptimizer(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                ee_links=ee_links,
                mount_to_cam_nom=cfg["mount_to_cam_nom"],
                head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom,
                head_idx=head_cfg["head_idx"],
                lambda_cam_pos=lambda_cam_pos,
                lambda_cam_rot=actual_lambda_cam_rot,
                use_sag=use_sag,
                optimize_head=optimize_head,
                optimize_camera=optimize_camera,
                active_arms=active_arms,
                estimate_measurement_noise=True,
                apply_joint_offset_limits=apply_limits,
                joint_offsets_to_apply=joint_offsets,
            )
            q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
                q_arm_list,
                q_head_list,
                T_meas_list,
            )
        else:
            optimizer = CalibrationOptimizer(
                robot=self.robot,
                arm_idx=cfg["arm_idx"],
                ee_links=ee_links,
                mount_to_cam_nom=cfg["mount_to_cam_nom"],
                head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom,
                active_arms=active_arms,
                optimize_arm=True,
                optimize_head=optimize_head,
                optimize_camera=optimize_camera,
                head_idx=head_cfg["head_idx"],
                use_head_kinematics=optimize_head,
                lambda_cam_pos=lambda_cam_pos,
                lambda_cam_rot=lambda_cam_rot,
                use_sag=use_sag,
                estimate_measurement_noise=True,
            )

            q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
                q_arm_list,
                q_head_list,
                T_meas_list,
            )
        
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
        
        if optimize_head:
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
        }

        if self.last_home_reset_path is not None and Path(self.last_home_reset_path).exists():
            result_dict["home_reset_baseline_path"] = str(self.last_home_reset_path)

        if optimize_head:
            result_dict["xi_mount_cam"] = result_dict["xi_cam"]
        else:
            result_dict["xi_head_base_cam"] = result_dict["xi_cam"]

        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=4)
            
        history_path = os.path.join(os.path.dirname(result_path), "calibration_history.txt")
        try:
            with open(history_path, "a") as f:
                import datetime
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

    def step2_zero_pose_check(self):
        self.log_msg("[Step2] Zero Pose Check requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        
        arm = "both"
        
        self.zero_pose_worker = Step2ZeroPoseCheckWorker(
            self.robot,
            self.model,
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

    def step2_stop_auto_motion(self):
        self.log_msg("[Step2] Stop Auto Motion requested.")
        self.request_stop_all_auto_motion()
        self.auto_save_current_dataset()

    def step2_init_pose(self):
        self.log_msg("[Step2] Init Pose requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        if self.auto_motion_running or self.auto_motion_thread is not None:
            QMessageBox.critical(self, "Execution Error", "Another robot operation is currently running.")
            return
            
        self.auto_motion_running = True
        active_arms = ["right", "left"]
        self.auto_motion_thread = Step2InitPoseWorker(
            self.robot,
            active_arms,
            self.auto_config.priority,
            parent=self
        )
        self.auto_motion_thread.log_signal.connect(self.log_msg)
        def on_finished(success, error_msg):
            self.auto_motion_running = False
            self.auto_motion_thread = None
            if success:
                self.auto_ready_done = True
                self.auto_base_head_q = None
                try:
                    self.auto_config.angle_step_deg = float(self.step2_angle_step.text())
                    self.auto_config.position_step_m = float(self.step2_pos_step.text())
                    self.auto_config.step_x_m = float(self.step2_step_x.text())
                    self.auto_config.max_x = float(self.step2_max_x.text())
                except Exception as e:
                    self.log_msg(f"Failed to read auto config: {e}. Using default values.")

                self.auto_motion_plan = None
                if self.include_head_motion:
                    head_cfg = get_head_config(self.model)
                    if head_cfg["head_idx"] is not None:
                        self.auto_base_head_q = self.robot.get_state().position[head_cfg["head_idx"]].copy()
                        self.log_msg(f"Auto base head pose (deg): {np.round(np.rad2deg(self.auto_base_head_q), 3)}")
                    else:
                        self.auto_base_head_q = None
                        self.include_head_motion = False

                self.head_move_count = 0
                self.update_head_pose_status()
                self.update_step2_est_samples()
                
                QMessageBox.information(
                    self,
                    "Teaching Required",
                    "Robot has moved to the initial pose.\n\n"
                    "Please adjust the robot's pose so that the marker is clearly visible to the camera.\n"
                    "Once adjusted, press '2) Auto Motion' to start the sequence."
                )
            else:
                self.log_msg(f"Init pose failed: {error_msg}")
                if not self.auto_stop_requested:
                    QMessageBox.critical(self, "Init Error", error_msg)
        
        self.auto_motion_thread.finished_signal.connect(on_finished)
        self.auto_motion_thread.start()

    def step2_auto_motion(self):
        self.log_msg("[Step2] Auto Motion requested.")
        mode = self.step2_mode_sel.currentText()
        if mode not in ["live", "sim"]:
            self.log_msg("[Step2] Auto motion is only available in live or sim mode.")
            return
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        try:
            self.move_to_all_auto_motions()
        except Exception as e:
            QMessageBox.critical(self, "Auto Motion Error", str(e))
            self.log_msg(f"Auto motion failed: {e}")

    def step2_calculate(self):
        self.log_msg("[Step2] Calculate requested.")
        if not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        try:
            mode = self.step2_mode_sel.currentText()
            active_arms = ["right", "left"]
            optimize_head = self.include_head_motion
            optimize_camera = True
            
            lambda_cam_pos = 1.0
            lambda_cam_rot = 1.0
            
            if not self.include_head_motion and optimize_head:
                optimize_head = False
                self.log_msg("Headless mode selected; optimize_head changed to False.")

            if len(active_arms) == 1:
                cfg = get_arm_config(self.model, active_arms[0], version=self.get_robot_version())
                ee_links = {active_arms[0]: cfg["ee_link"]}
                ee_to_marker_nom = {active_arms[0]: cfg["ee_to_marker_nom"]}
            else:
                cfg = get_both_arm_config(self.model, version=self.get_robot_version())
                ee_links = cfg["ee_links"]
                ee_to_marker_nom = cfg["ee_to_marker_nom"]
                
            head_cfg = get_head_config(self.model)

            if mode in ["live", "sim"]:
                if len(self.shared_arm_q_list) == 0:
                    QMessageBox.warning(self, "Warning", "No recorded samples.")
                    return

                q_arm_list = np.array(self.shared_arm_q_list)
                q_head_list = np.array(self.shared_head_q_list) if self.shared_head_q_list else None
                T_meas_list = np.array(self.shared_T_list)
                
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
                npz_path = self.resolve_input_path(self.step2_path_input.text().strip())
                q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)
                
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
            
            self.calc_worker = Step2CalculateWorker(
                self,
                active_arms,
                optimize_head,
                optimize_camera,
                q_arm_list,
                q_head_list,
                T_meas_list,
                result_path,
                lambda_cam_pos,
                lambda_cam_rot
            )
            self.calc_worker.log_signal.connect(self.log_msg)
            def on_finished(success, error_msg):
                if success:
                    self.log_msg("Optimization finished successfully.")
                    QMessageBox.information(self, "Step 2 Calculation", "Optimization finished successfully!\nCheck the logs and Result Output for details.")
                else:
                    self.log_msg(f"[Error] Optimization failed: {error_msg}")
                    QMessageBox.warning(self, "Step 2 Calculation Failed", f"Optimization failed:\n{error_msg}")
            self.calc_worker.finished_signal.connect(on_finished)
            self.calc_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Calculate Error", str(e))
            self.log_msg(f"Calculate failed: {e}")

    def step2_clear_samples(self):
        self.log_msg("[Step2] Clear Samples requested.")
        self.stop_all_auto_motion_internal(cancel_robot=True)
        reset_motion_state()
        self.shared_arm_q_list.clear()
        self.shared_head_q_list.clear()
        self.shared_T_list.clear()
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
            
            dialog = ApplyHomeOffsetDialog(
                parent=self,
                result_path=result_path,
                baseline_path=baseline_path,
                arm=arm,
                include_head=self.include_head_motion,
                compare_summary=compare_summary
            )
            dialog.exec()
            
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
                
            self.joint_calibrator.joint_offsets = self.joint_offsets
            self.marker_calibrator.joint_offsets = self.joint_offsets
            
            self.save_offsets_to_yaml()
            self.update_applied_offset_label()
            
            self.log_msg("[CLEAR] Staged and saved offsets cleared to 0.0 for BOTH Arms.")

    def load_bracket_design_values(self):
        config_path = CONFIG_PATHS["setting_yaml"]
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                    is_v13 = self.get_robot_version() == "1.3"
                    # Load Left Arm values
                    val_left = data["camera"].get("Tf_to_marker_left", None)
                    if val_left and len(val_left) == 6:
                        if is_v13 and abs(val_left[0]) < 0.05:
                            val_left = data["camera"].get("Tf_to_marker_left_v13", self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.3"]["left"])
                        elif not is_v13 and abs(val_left[0]) > 0.05:
                            val_left = data["camera"].get("Tf_to_marker_left_v12", self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.2"]["left"])
                        self.txt_bracket_l_x.setText(f"{val_left[0]:.4f}")
                        self.txt_bracket_l_y.setText(f"{val_left[1]:.4f}")
                        self.txt_bracket_l_z.setText(f"{val_left[2]:.4f}")
                        self.txt_bracket_l_roll.setText(f"{val_left[3]:.2f}")
                        self.txt_bracket_l_pitch.setText(f"{val_left[4]:.2f}")
                        self.txt_bracket_l_yaw.setText(f"{val_left[5]:.2f}")
                        # Sync back to memory configs
                        self.marker_calibrator.camera_config["Tf_to_marker_left"] = val_left
                        self.joint_calibrator.camera_config["Tf_to_marker_left"] = val_left
                    
                    # Load Right Arm values
                    val_right = data["camera"].get("Tf_to_marker_right", None)
                    if val_right and len(val_right) == 6:
                        if is_v13 and abs(val_right[0]) < 0.05:
                            val_right = data["camera"].get("Tf_to_marker_right_v13", self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.3"]["right"])
                        elif not is_v13 and abs(val_right[0]) > 0.05:
                            val_right = data["camera"].get("Tf_to_marker_right_v12", self.joint_calibrator.NOMINAL_BRACKET_TEMPLATES["1.2"]["right"])
                        self.txt_bracket_r_x.setText(f"{val_right[0]:.4f}")
                        self.txt_bracket_r_y.setText(f"{val_right[1]:.4f}")
                        self.txt_bracket_r_z.setText(f"{val_right[2]:.4f}")
                        self.txt_bracket_r_roll.setText(f"{val_right[3]:.2f}")
                        self.txt_bracket_r_pitch.setText(f"{val_right[4]:.2f}")
                        self.txt_bracket_r_yaw.setText(f"{val_right[5]:.2f}")
                        # Sync back to memory configs
                        self.marker_calibrator.camera_config["Tf_to_marker_right"] = val_right
                        self.joint_calibrator.camera_config["Tf_to_marker_right"] = val_right
                    
                    self.log_msg(f"[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory")
                    return
            self.log_msg("[WARNING] Could not load bracket design values from setting.yaml.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to load setting.yaml: {e}")

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
            
            lines = []
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    lines = f.readlines()
            
            def update_key_in_lines(lines_list, key_str, new_vals_list):
                camera_idx = -1
                for idx, line in enumerate(lines_list):
                    if line.strip().startswith("camera:"):
                        camera_idx = idx
                        break
                
                new_val_str = f"[{new_vals_list[0]:.5f}, {new_vals_list[1]:.5f}, {new_vals_list[2]:.5f}, {new_vals_list[3]:.2f}, {new_vals_list[4]:.2f}, {new_vals_list[5]:.2f}]"
                key_found = False
                if camera_idx != -1:
                    i = camera_idx + 1
                    while i < len(lines_list):
                        line = lines_list[i]
                        stripped = line.strip()
                        if not stripped:
                            i += 1
                            continue
                        if not line.startswith(" ") and not line.startswith("\t") and stripped.endswith(":"):
                            break
                        
                        if stripped.startswith(f"{key_str}:"):
                            comment = ""
                            if "#" in line:
                                comment_idx = line.find("#")
                                comment = " " + line[comment_idx:].rstrip()
                            
                            indent = len(line) - len(line.lstrip())
                            lines_list[i] = " " * indent + f"{key_str}: {new_val_str}{comment}\n"
                            key_found = True
                            break
                        i += 1
                
                if not key_found:
                    if camera_idx == -1:
                        lines_list.append("camera:\n")
                        lines_list.append(f"  {key_str}: {new_val_str}\n")
                    else:
                        lines_list.insert(camera_idx + 1, f"  {key_str}: {new_val_str}\n")
            
            update_key_in_lines(lines, "Tf_to_marker_left", [l_x, l_y, l_z, l_roll, l_pitch, l_yaw])
            update_key_in_lines(lines, "Tf_to_marker_right", [r_x, r_y, r_z, r_roll, r_pitch, r_yaw])
            
            with open(config_path, "w") as f:
                f.writelines(lines)
                
            self.log_msg(f"[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml")
            if not silent:
                QMessageBox.information(self, "Success", "Bracket design values saved for both arms!")
            
            if not self.ui_only and self.marker_st is not None:
                detector = self.marker_st.marker_detection
                if hasattr(detector, 'camera_config'):
                    detector.camera_config["Tf_to_marker_left"] = [l_x, l_y, l_z, l_roll, l_pitch, l_yaw]
                    detector.camera_config["Tf_to_marker_right"] = [r_x, r_y, r_z, r_roll, r_pitch, r_yaw]
                    detector.Tf_to_marker_tf_left = detector.make_transform(detector.camera_config["Tf_to_marker_left"])
                    detector.Tf_to_marker_tf_right = detector.make_transform(detector.camera_config["Tf_to_marker_right"])
                    self.log_msg("[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.")
        except Exception as e:
            self.log_msg(f"[ERROR] Failed to save bracket values: {e}")
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to save bracket values: {e}")

    def set_controls_enabled(self, enabled):
        if hasattr(self, 'btn_full_auto_start'):
            self.btn_full_auto_start.setEnabled(enabled)
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
        
        self.btn_int_capture.setEnabled(enabled)
        self.btn_int_calibrate.setEnabled(enabled)
        self.btn_int_reset.setEnabled(enabled)
        
        if hasattr(self, 'chk_servo_head'):
            self.chk_servo_head.setEnabled(enabled)
            
        self.btn_connect.setEnabled(enabled)
        self.model_input.setEnabled(enabled)
        self.workflow_tabs.setEnabled(enabled)
        self.arm_sel.setEnabled(enabled)
        self.joint_mode_sel.setEnabled(enabled)
        if hasattr(self, 'marker_axis_sel'):
            self.marker_axis_sel.setEnabled(enabled)
        if hasattr(self, 'btn_camera_feed'):
            self.btn_camera_feed.setEnabled(True) # Keep camera feed button enabled always!

    def on_action_finished(self):
        self.set_controls_enabled(True)

    def on_move_ready_joint_finished(self):
        self.ready_done_joint = True
        self.on_action_finished()

    def on_move_ready_marker_finished(self):
        self.ready_done_marker = True
        self.on_action_finished()

    def get_robot_version(self) -> str:
        return str(getattr(self, "robot_version", "1.2"))

    def move_to_ready_full_auto(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
        self.set_controls_enabled(False)
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.active_worker = FullAutoReadyWorker(
            self.joint_calibrator,
            self.marker_calibrator,
            ui_only=self.ui_only
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_full_auto_ready_finished)
        self.active_worker.start()

    def on_full_auto_ready_finished(self):
        self.set_controls_enabled(True)
        error_msg = getattr(self.active_worker, 'error_msg', None) if self.active_worker else None
        self.active_worker = None
        
        # Restart poll_timer if appropriate
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        camera_subtab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
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
        return result_files[0]

    def get_latest_home_reset_path(self, required=True):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = Path(os.path.abspath(os.path.join(current_dir, "config", "home_reset_baseline.json")))
        if path.exists():
            return path
        if required:
            raise RuntimeError(f"No home reset baseline JSON found at {path}")
        return None

    def get_home_reset_path_for_result(self, result_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = Path(os.path.abspath(os.path.join(current_dir, "config", "home_reset_baseline.json")))
        if path.exists():
            return path
        return self.get_latest_home_reset_path(required=False)

    def apply_home_offset(self):
        try:
            result_path = self.get_latest_result_path()
            baseline_path = self.get_home_reset_path_for_result(result_path)
            
            # Open Apply Home Offset Dialog
            dialog = ApplyHomeOffsetDialog(
                self,
                result_path,
                baseline_path,
                self.arm_side,
                include_head=True
            )
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Apply Home Offset Error", str(e))
            self.log_msg(f"[ERROR] Apply home offset failed: {e}")

    def home_offset_reset(self):
        if not self.ui_only and not self.robot:
            QMessageBox.critical(self, "Error", "Robot is not connected.")
            return

        msg = (
            "Warning: Home Offset Reset will physically redefine the zero offset positions of your robot joints.\n\n"
            "Steps:\n"
            "1. Manually teach/move BOTH arms close to their home pose using direct teaching.\n"
            "2. Ensure the head is also centered/aligned if you want to reset head offsets.\n"
            "3. Click OK to start the process.\n\n"
            "During this, the control manager will disable, 48v power will cycle, and the robot connection will automatically restart."
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Home Offset Reset")
        dialog.setStyleSheet(DARK_STYLESHEET)
        layout = QVBoxLayout(dialog)

        # Image
        img_label = QLabel()
        pixmap = QPixmap("img/home_offset_position.png")
        if not pixmap.isNull():
            img_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_label.setText("[img/home_offset_position.png not found]")
        img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)

        # Text
        msg_label = QLabel(msg)
        msg_label.setStyleSheet("font-size: 14px; color: white;")
        layout.addWidget(msg_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background-color: #555; color: white; padding: 5px;")
        
        btn_ok.clicked.connect(dialog.accept)
        btn_cancel.clicked.connect(dialog.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        if dialog.exec() != QDialog.Accepted:
            return

        self.set_controls_enabled(False)
        self.btn_home_reset.setEnabled(False)
        
        # Start worker thread
        self.active_worker = HomeOffsetResetWorker(
            self.robot,
            self.robot.model() if (self.robot and not self.is_mock) else None,
            self.model_input.currentText().strip() if hasattr(self, 'model_input') else "a",
            include_head=True
        )
        self.active_worker.log_signal.connect(self.log_msg)
        self.active_worker.finished_signal.connect(self.on_home_offset_reset_finished)
        self.active_worker.start()

    def on_home_offset_reset_finished(self, result):
        self.set_controls_enabled(True)
        self.btn_home_reset.setEnabled(True)
        self.active_worker = None
        
        if result.get("success", False):
            self.log_msg("Re-connecting and initializing robot...")
            if self.robot:
                self.connect_robot() # Disconnects first
                QApplication.processEvents()
            self.connect_robot() # Connects again
            self.log_msg("Home Offset Reset complete!")
            QMessageBox.information(self, "Success", "Home Offset Reset, Power, and Servo Initialization completed successfully!")
        else:
            QMessageBox.warning(self, "Warning", f"Home Offset Reset finished, but some joints failed to reset: {result.get('error', '')}")

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
                        os.remove(os.path.join(plot_dir, f_name))
                    except Exception:
                        pass
        if plot_dir:
            txt_dir = os.path.abspath(os.path.join(os.path.dirname(plot_dir), "result_txt"))
            if os.path.exists(txt_dir):
                for f_name in os.listdir(txt_dir):
                    if f_name.endswith(".txt"):
                        try:
                            os.remove(os.path.join(txt_dir, f_name))
                        except Exception:
                            pass
    def apply_full_auto_results(self):
        reply = QMessageBox.question(self, 'Apply Full Auto Results', 
                                     "Do you want to apply all calibrated Joint Offsets and Marker Brackets to setting.yaml?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.apply_joint_offset()
            self.apply_bracket_design_values(silent=True)
            self.log_msg("[APPLY] Full auto results (Joints & Brackets) applied successfully.")
            QMessageBox.information(self, "Apply Complete", "All full auto calibration results have been applied successfully.")
            

    def start_full_auto(self):
        if not self.ui_only and not self.robot:
            self.log_msg("[ERROR] Robot is not connected!")
            return
            
        self.clear_old_plots()
            
        self.log_text.clear()
        self.log_msg("[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...")
        if self.ui_only:
            self.log_msg("[MOCK GT] Simulated Ground-Truth Offsets:")
            is_v13 = self.get_robot_version() == "1.3"
            for arm in ["right", "left"]:
                mock_gt = self.joint_calibrator.MOCK_GT_OFFSETS[arm]
                j6_gt = mock_gt["joint6"]
                j5_gt = mock_gt["joint5_v13"] if is_v13 else mock_gt["joint5_v12"]
                j3_gt = mock_gt["joint3"]
                pos_gt = [x * 1000.0 for x in mock_gt["bracket_pos"]] # convert from m to mm
                rpy_gt = mock_gt["bracket_rpy"]
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
        
        
        self.full_auto_stop_event = threading.Event()
        
        # update mock robot version on calibrators just in case
        v = self.get_robot_version()
        self.marker_calibrator.robot_version = v
        self.joint_calibrator.robot_version = v
        
        self.active_worker = FullAutoWorker(
            self.joint_calibrator,
            self.marker_calibrator,
            ui_only=self.ui_only,
            stop_event=self.full_auto_stop_event,
            joint_offsets_store=self.joint_offsets_store,
            save_debug=self.chk_save_debug.isChecked()
        )
        
        self.active_worker.log_msg.connect(self.log_msg)
        self.active_worker.status_signal.connect(self.update_marker_indicator)
        self.active_worker.bracket_finished_signal.connect(self.handle_full_auto_bracket_finished)
        self.active_worker.joint_finished_signal.connect(self.handle_full_auto_joint_finished)
        self.active_worker.finished_signal.connect(self.on_full_auto_finished)
        self.active_worker.start()

    def stop_full_auto(self):
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
            self.btn_full_auto_apply.setEnabled(True)
        
        was_stopped = False
        if hasattr(self, 'full_auto_stop_event') and self.full_auto_stop_event is not None:
            was_stopped = self.full_auto_stop_event.is_set()
            
        error_msg = getattr(self.active_worker, 'error_msg', None) if self.active_worker else None
        self.active_worker = None
        self.log_msg("[INFO] Full Auto sequential calibration ended.")
        
        # Restart poll_timer if appropriate (not tab 2 and feed dialog closed)
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        camera_subtab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
        if not camera_subtab_active and not dialog_visible:
            if not self.poll_timer.isActive():
                self.poll_timer.start(200)
                
        if error_msg:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Calibration Failed")
            msg_box.setText(f"Full Auto Sequential Calibration failed!\n\nReason:\n{error_msg}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            msg_box.exec()
            
            # Automatically show plot if generated
            if self.generated_plots:
                self.open_plot_dialog()
        elif not was_stopped:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Calibration Complete")
            msg_box.setText("Full Auto Sequential Calibration has completed successfully!\n\n"
                            "Please review the calibrated offsets in the table and click 'APPLY BRACKETS' / 'APPLY OFFSET' to save them.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            msg_box.exec()
            
            # Automatically show plot if generated
            if self.generated_plots:
                self.open_plot_dialog()

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
            recommended = np.clip(recommended, -3.0, 0.0)
            
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
    def move_to_ready_pose_joint(self):
        if not self.robot and not self.ui_only:
            self.log_msg("[ERROR] Robot is not connected!")
            return


        mode = self.get_selected_joint_mode()
        
        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.ready_worker = MoveToReadyWorker(self.joint_calibrator, self.arm_side, mode)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_move_ready_joint_finished)
        self.ready_worker.start()

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
        if self.ui_only:
            mock_gt = self.joint_calibrator.MOCK_GT_OFFSETS[self.arm_side]
            is_v13 = self.get_robot_version() == "1.3"
            j_gt = {
                "wrist_roll_v13": mock_gt["joint6"],
                "wrist_yaw2": mock_gt["joint6"],
                "wrist_pitch_v13": mock_gt["joint5_v13"],
                "wrist_pitch": mock_gt["joint5_v12"],
                "elbow": mock_gt["joint3"]
            }
            gt_val = j_gt.get(mode, 0.0)
            self.log_msg(f"[MOCK GT] Simulated Target Joint Offset: {gt_val:+.2f}°")
        
        curr_offset = self.joint_offsets[self.arm_side].get(offset_key, 0.0)
        sweep_time = 20.0 if mode == "elbow" else 15.0
        self.active_worker = JointCalibrationWorker(
            self.joint_calibrator, self.arm_side, mode, 
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
    def move_to_ready_pose_marker(self):
        if not self.robot and not self.ui_only:
            self.log_msg("[ERROR] Robot is not connected!")
            return


        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.ready_worker = MoveToReadyWorker(self.marker_calibrator, self.arm_side)
        self.ready_worker.log_signal.connect(self.log_msg)
        self.ready_worker.finished_signal.connect(self.on_move_ready_marker_finished)
        self.ready_worker.start()

    # Move to Center is removed as it is no longer needed

    def start_calibration_marker(self):
        # 1. Prerequisite Check: Joint 6 (Wrist Roll / Wrist Yaw 2) must be calibrated first
        if not self.wrist_roll_calibrated.get(self.arm_side, False):
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Prerequisite Check")
            msg_box.setText(
                "Marker Bracket Calibration requires Joint 6 (Wrist Roll / Wrist Yaw 2) to be calibrated first.\n\n"
                "Joint 6 has not been calibrated yet. Please go to the Joint Calibration tab, select Joint 6, and perform calibration."
            )
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            return

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
        if self.ui_only:
            mock_gt = self.joint_calibrator.MOCK_GT_OFFSETS[self.arm_side]
            pos_gt = [x * 1000.0 for x in mock_gt["bracket_pos"]] # convert from m to mm
            rpy_gt = mock_gt["bracket_rpy"]
            self.log_msg(f"[MOCK GT] Simulated Bracket Offset (Tf_to_marker):")
            self.log_msg(f"  * Pos: X: {pos_gt[0]:+.1f}, Y: {pos_gt[1]:+.1f}, Z: {pos_gt[2]:+.1f} mm")
            self.log_msg(f"  * Rot: R: {rpy_gt[0]:+.2f}, P: {rpy_gt[1]:+.2f}, Y: {rpy_gt[2]:+.2f} deg")

        try:
            tolerance = float(self.tolerance_input.text())
        except ValueError:
            tolerance = 0.5
        self.active_worker = MarkerCalibrationWorker(
            self.marker_calibrator, self.arm_side, 
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
            self.log_msg(f"    - Opt Delta 5 (5축 오프셋): {res['opt_delta_5']:.3f} deg")
            self.log_msg(f"    - Opt Delta 6 (6축 오프셋): {res['opt_delta_6']:.3f} deg")
            self.log_msg(f"    - Min Circle Fitting Radius (최소 원 피팅 반지름): {res['min_radius']:.2f} mm")
            
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
                self.log_msg(f"    - Opt Delta 5 (5축 오프셋): {res['opt_delta_5']:.3f} deg")
                self.log_msg(f"    - Opt Delta 6 (6축 오프셋): {res['opt_delta_6']:.3f} deg")
                self.log_msg(f"    - Min Circle Fitting Radius (최소 원 피팅 반지름): {res['min_radius']:.2f} mm")
                
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
        if self.ui_only:
            self.log_msg("[MOCK] Moved head to target angles.")
            return
            
        self.set_controls_enabled(False)
        if self.poll_timer.isActive(): self.poll_timer.stop()
        self.joint_calibrator.stop_requested = False
        self.marker_calibrator.stop_requested = False
        self.head_worker = ManualHeadWorker(self.joint_calibrator, np.radians(yaw), np.radians(pitch))
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
        # Camera 서브탭(Step1 > Camera)이 활성화되어 있거나, Camera Feed 대화상자가 열려있을 때 업데이트
        dialog_visible = hasattr(self, 'feed_dialog') and self.feed_dialog is not None and self.feed_dialog.isVisible()
        camera_tab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
        wizard_slide1_active = (hasattr(self, 'wizard_widget') and self.wizard_widget.isVisible() and self.wizard_widget.stacked_widget.currentIndex() == 0)
        if not camera_tab_active and not dialog_visible and not wizard_slide1_active:
            return

        if not self.ui_only and self.marker_st is not None:
            self.marker_st.camera.capture_image()
            img = self.marker_st.camera.get_color_image()
            
            # 백그라운드 마커 검출 및 상태 표시 업데이트
            try:
                res_all = self.marker_st.get_marker_transform(sampling_time=0, side="all")
                detected = bool(res_all and len(res_all) > 0)
                self.update_marker_indicator(detected)
            except Exception:
                pass
                
            if img is None:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(img, "No Camera Detected", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
        else:
            # Mock image
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "UI-ONLY MODE", (440, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 100, 100), 3)

        if img is None:
            return
            
        self.current_frame = img.copy()
        display_img = img.copy()
        
        guide_checked = (hasattr(self, 'chk_int_guide') and self.chk_int_guide.isChecked()) or (hasattr(self, 'wizard_widget') and self.wizard_widget.chk_int_guide.isChecked())
        if guide_checked and (camera_tab_active or wizard_slide1_active):
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
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if wizard_slide1_active and hasattr(self.wizard_widget, 'wizard_video_label'):
            self.wizard_widget.wizard_video_label.setPixmap(pixmap.scaled(self.wizard_widget.wizard_video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if dialog_visible:
            w_lbl = max(20, self.feed_dialog.lbl_feed.width())
            h_lbl = max(20, self.feed_dialog.lbl_feed.height())
            self.feed_dialog.lbl_feed.setPixmap(pixmap.scaled(w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.FastTransformation))

    def keyPressEvent(self, event):
        camera_tab_active = (self.left_tabs.currentIndex() == 1 and hasattr(self, 'step1_tabs') and self.step1_tabs.currentIndex() == 1)
        wizard_slide1_active = (hasattr(self, 'wizard_widget') and self.wizard_widget.isVisible() and self.wizard_widget.stacked_widget.currentIndex() == 0)
        if event.key() == Qt.Key_C and (camera_tab_active or wizard_slide1_active):
            self.capture_intrinsics_frame()
        super().keyPressEvent(event)

    def capture_intrinsics_frame(self):
        if hasattr(self, 'current_frame'):
            self.captured_images.append(self.current_frame.copy())
            self.lbl_captured.setText(f"Captured Frames: {len(self.captured_images)}")
            self.log_msg(f"[INTRINSICS] Frame {len(self.captured_images)} captured.")
            
            num_steps = len(IntrinsicsCalibrator.CALIB_GUIDELINES)
            if hasattr(self, 'chk_int_guide') and self.chk_int_guide.isChecked() and self.current_guide_idx < num_steps:
                self.current_guide_idx += 1
                if self.current_guide_idx == num_steps:
                    self.log_msg(f"[INTRINSICS] All {num_steps} guided frames captured! You can now run calibration.")

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

    def run_intrinsics_calibration(self):
        if len(self.captured_images) < 5:
            self.log_msg("[ERROR] Need at least 5 frames to run calibration!")
            return
            
        self.log_msg(f"\n[INTRINSICS] Running calibration on {len(self.captured_images)} images. Please wait...")
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        success = self.intrinsics_calibrator.run_calibration_with_images(self.captured_images, None)
        
        QApplication.restoreOverrideCursor()
        
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
        else:
            self.log_msg("[ERROR] Calibration failed. Check images and board settings.")

    def save_intrinsics_calibration(self):
        if self.intrinsics_calibrator.cameraMatrix is None:
            self.log_msg("[ERROR] No calibration data to save!")
            return
        try:
            data = {
                "camera_matrix": self.intrinsics_calibrator.cameraMatrix.tolist(),
                "dist_coeffs": self.intrinsics_calibrator.distCoeffs.flatten().tolist(),
                "rms_error": float(self.intrinsics_calibrator.rms_error),
                "width": int(self.captured_images[0].shape[1]),
                "height": int(self.captured_images[0].shape[0])
            }
            os.makedirs(os.path.dirname(self.output_yaml), exist_ok=True)
            with open(self.output_yaml, "w") as f:
                yaml.dump(data, f)
            self.log_msg(f"[SUCCESS] Intrinsic parameters saved to: {self.output_yaml}")
            
            # Sync with the local marker detector instances
            if self.marker_detector is not None:
                self.marker_detector.fx = self.intrinsics_calibrator.cameraMatrix[0, 0]
                self.marker_detector.fy = self.intrinsics_calibrator.cameraMatrix[1, 1]
                self.marker_detector.principal_point = [self.intrinsics_calibrator.cameraMatrix[0, 2], self.intrinsics_calibrator.cameraMatrix[1, 2]]
                self.marker_detector.dist_coeffs = self.intrinsics_calibrator.distCoeffs
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
            
        if not self.ui_only and self.marker_st is not None:
            try:
                self.marker_st.camera.stream_off()
                print("Camera stream closed.")
            except Exception:
                pass
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Unified Robot Calibration Suite GUI")
    parser.add_argument("--ui", action="store_true", help="Start only UI for debugging/simulation")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    robot = None
    marker_st = None

    if not args.ui:
        print("[INFO] Initializing Camera Marker Transform System...")
        try:
            marker_st = Marker_Transform()
            marker_st.marker_detection.set_marker_type("plate")
        except Exception as e:
            print(f"[ERROR] Failed to load camera marker system: {e}")
            print("[INFO] Fallback to UI-only mode.")
            args.ui = True
    else:
        print("[INFO] Starting in simulation (UI-only) mode.")

    gui = UnifiedCalibrationApp(marker_st, robot, "right", ui_only=args.ui)
    gui.show()
    
    try:
        sys.exit(app.exec())
    finally:
        if marker_st:
            try:
                marker_st.camera.stream_off()
                print("Camera resource released.")
            except Exception:
                pass

if __name__ == "__main__":
    main()
