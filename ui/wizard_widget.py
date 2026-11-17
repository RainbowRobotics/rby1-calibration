from ui.core_bridge import idle_core_action
from core.storage import ResultStorage
from core.storage import StoragePaths
import os
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QGroupBox, QCheckBox, QLineEdit, QMessageBox, QDialog,
    QRadioButton, QButtonGroup, QSpinBox, QSlider, QScrollArea, QFrame, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, QObject, QEvent, QUrl
from PySide6.QtGui import QFont, QPixmap, QDesktopServices
from core.language import LanguageManager, tr
from ui.marker_monitor_panel import MarkerMonitorPanel
from ui.busy_indicator import BusyIndicator
from ui.step_progress import StepProgress
from core.robot.home_offset import home_reset_pending

def get_asset_path(relative_path):
    return StoragePaths.asset(relative_path)


class WrappedLabelHeight(QObject):
    """Keeps word-wrapped labels at least as tall as their wrapped text.

    In centred or horizontal layouts Qt sized these labels by their one-line hint, so text that
    wrapped (after the bundled Korean font, 2026-11) was cut off at the top and bottom. The label is
    re-measured when it is resized and when its parent re-lays out (setText triggers that)."""

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.by_parent = {}
        for label in labels:
            label.setProperty("wrap_base_min_height", label.minimumHeight())
            label.installEventFilter(self)
            self.by_parent.setdefault(label.parentWidget(), []).append(label)
        for widget in self.by_parent:
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize and isinstance(obj, QLabel):
            self.reserve(obj)
        elif event.type() == QEvent.LayoutRequest:
            for label in self.by_parent.get(obj, ()):
                self.reserve(label)
        return False

    @staticmethod
    def reserve(label):
        need = max(label.property("wrap_base_min_height") or 0, label.heightForWidth(label.width()))
        if label.minimumHeight() != need:
            label.setMinimumHeight(need)

class HowToMoveArmsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.how_to_move_arms.title"))
        self.resize(750, 520)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #ffffff; font-size: 14px; }
            QGroupBox { border: 1px solid #333333; border-radius: 8px; margin-top: 15px; font-weight: bold; font-size: 15px; color: #e0e0e0; padding: 10px; background-color: #1a1a1a; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 15px; padding: 0 6px; background-color: #1e1e1e; color: #90caf9; }
            QPushButton { background-color: #1e88e5; color: white; font-weight: bold; font-size: 14px; padding: 8px 16px; border-radius: 6px; }
            QPushButton:hover { background-color: #2196f3; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl_title = QLabel(tr("dialogs.how_to_move_arms.lbl_title"))
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffd700;")
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        img_lbl = QLabel()
        pix = QPixmap(get_asset_path("img/teaching_button.png"))
        if not pix.isNull():
            img_lbl.setPixmap(pix.scaled(550, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_lbl.setText("[img/teaching_button.png]")
        img_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_lbl)

        box = QGroupBox(tr("dialogs.how_to_move_arms.box_title"))
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)

        insts = [
            tr("dialogs.how_to_move_arms.step1"),
            tr("dialogs.how_to_move_arms.step2")
        ]
        for txt in insts:
            lbl = QLabel(txt)
            lbl.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
            lbl.setWordWrap(True)
            box_layout.addWidget(lbl)

        warn_lbl = QLabel(tr("dialogs.how_to_move_arms.warn_lbl"))
        warn_lbl.setStyleSheet("font-size: 16px; color: #ff5252; font-weight: bold;")
        warn_lbl.setWordWrap(True)
        warn_lbl.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(warn_lbl)

        layout.addWidget(box)

        btn_close = QPushButton(tr("dialogs.how_to_move_arms.btn_close"))
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignCenter)


ROBOT_PORT = 50051
UPC_ROBOT_ADDRESS = f"192.168.30.1:{ROBOT_PORT}"
HOTSPOT_ROBOT_ADDRESS = f"192.168.12.1:{ROBOT_PORT}"


def normalize_robot_address(text):
    """As read off the robot's OLED ("192.168.8.130/23"): drop the network suffix, add the port when missing."""
    address = text.strip().split("/")[0].strip()
    return address if not address or ":" in address else f"{address}:{ROBOT_PORT}"


class RobotIpDialog(QDialog):
    """External PC: where to read the robot's address (the OLED display on its back), then enter it."""

    def __init__(self, current_address="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.robot_ip.title"))
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #ffffff; font-size: 14px; }
            QLineEdit { background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px; font-size: 15px; }
            QPushButton { background-color: #1e88e5; color: white; font-weight: bold; font-size: 14px; padding: 8px 16px; border-radius: 6px; }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        img_lbl = QLabel()
        pix = QPixmap(get_asset_path("img/robot_oled_ip.jpg"))
        if not pix.isNull():
            img_lbl.setPixmap(pix.scaled(480, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_lbl.setText("[img/robot_oled_ip.jpg]")
        img_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_lbl)
        self.lbl_guide = QLabel(tr("dialogs.robot_ip.guide"))
        self.lbl_guide.setWordWrap(True)
        self.lbl_guide.setFixedWidth(520)
        layout.addWidget(self.lbl_guide, alignment=Qt.AlignCenter)
        self.lbl_hotspot = QLabel(tr("dialogs.robot_ip.hotspot", address=HOTSPOT_ROBOT_ADDRESS))
        self.lbl_hotspot.setStyleSheet("color: #ffd700; font-weight: bold;")
        self.lbl_hotspot.setWordWrap(True)
        self.lbl_hotspot.setFixedWidth(520)
        layout.addWidget(self.lbl_hotspot, alignment=Qt.AlignCenter)
        row = QHBoxLayout()
        lbl = QLabel(tr("dialogs.robot_ip.label"))
        lbl.setStyleSheet("font-weight: bold;")
        row.addWidget(lbl)
        self.address_input = QLineEdit(current_address)
        row.addWidget(self.address_input, 1)
        layout.addLayout(row)
        buttons = QHBoxLayout()
        self.btn_ok = QPushButton(tr("dialogs.robot_ip.btn_ok"))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton(tr("dialogs.robot_ip.btn_cancel"))
        self.btn_cancel.setStyleSheet("background-color: #546e7a;")
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_ok, 2)
        buttons.addWidget(self.btn_cancel, 1)
        layout.addLayout(buttons)

    def address(self):
        return normalize_robot_address(self.address_input.text())


class CalibrationWizardWidget(QWidget):
    # Slide order. Everything that depends on a slide position uses these names (main_ui too),
    # so slides can be added or moved by editing this list alone.
    # 2026-09-30: the gripper-removal and marker-bracket slides are one slide, and the separate
    # intrinsics-check slide is gone: the camera slide warns when the connected unit (S/N) has no
    # intrinsics of its own and opens the optional calibration slide from a button.
    # 2026-10-01: the bracket slide comes after the robot connection, because its photo depends on
    # the robot version, which is only known once the robot is connected. The exposure slide comes
    # before the zero pose: its buttons put one arm at a time in the J5 ready pose to show the marker
    # to the camera, and the zero pose and home offset reset follow it.
    # 2026-11-12 (user): robot connection first, then a gripper-change pose (flange up, tool power off),
    # then the camera and the gripper/marker slides. Gripper removal is v1.2 only (v1.3 brackets go on
    # the gripper); the home offset reset is passed over while an unapplied reset is on record.
    # 2026-11-12 (user): the camera's distortion (intrinsics) calibration comes first, before the robot:
    # a board-printing slide, then the calibration itself. Both are passed over for a camera whose serial
    # number already has its own intrinsics file. They are numbered 0 so steps 1-9 read the same either way.
    SLIDE_INTRINSICS_PREP = 0    # 0 print the ChArUco board, flat surface (accuracy notice)
    SLIDE_INTRINSICS_CALIB = 1   # 0-1 intrinsics calibration
    SLIDE_ROBOT_CONNECT = 2      # 1 robot connection
    SLIDE_GRIPPER_POSE = 3       # 2 gripper-change pose + tool flange power off (v1.2)
    SLIDE_CAMERA_MOUNT = 4       # 3 camera bracket assembly & mounting (+ intrinsics warning)
    SLIDE_GRIPPER_REMOVAL = 5    # 4 gripper removal (v1.2 robots only)
    SLIDE_MARKER_BRACKET = 6     # 5 marker bracket attachment (photo by robot version)
    SLIDE_EXPOSURE = 7           # 6 exposure & marker recognition (right arm test pose, auto exposure, zero pose)
    SLIDE_HOME_OFFSET = 8        # 7 home offset reset
    SLIDE_CALIBRATION = 9        # 8 calibration start
    SLIDE_APPLY = 10             # 9 apply home offset
    SLIDE_COUNT = 11
    CHARUCO_BOARD_PDF = "calib.io_charuco_297x210_5x8_36_27_DICT_5X5.pdf"
    # 2026-11-12 (user): the separate zero-pose slide is gone; the exposure slide's zero button does
    # the same move and its Next waits for it, so the reset still starts from the zero pose.
    # Marker bracket photo per robot version (taken from the connected robot).
    BRACKET_ASSEMBLE_IMAGES = {"1.2": "img/marker_bracket_assemble_1_2.png",
                               "1.3": "img/marker_bracket_assemble_1_3.png"}
    TITLE_KEYS = {
        SLIDE_INTRINSICS_PREP: "wizard.slides.slide_intrinsics_prep.title",
        SLIDE_GRIPPER_POSE: "wizard.slides.slide_gripper_pose.title",
        SLIDE_CAMERA_MOUNT: "wizard.slides.slide_0.title",
        SLIDE_INTRINSICS_CALIB: "wizard.slides.slide_3.title",
        SLIDE_GRIPPER_REMOVAL: "wizard.slides.slide_1.title",
        SLIDE_MARKER_BRACKET: "wizard.slides.slide_marker_bracket.title",
        SLIDE_ROBOT_CONNECT: "wizard.slides.slide_4.title",
        SLIDE_EXPOSURE: "wizard.slides.slide_exposure.title",
        SLIDE_HOME_OFFSET: "wizard.slides.slide_6.title",
        SLIDE_CALIBRATION: "wizard.slides.slide_7.title",
        SLIDE_APPLY: "wizard.slides.slide_8.title",
    }
    # Progress line on top (2026-11-12, user): four stages, each a run of slides.
    STAGE_KEYS = ("prep", "mount", "calib", "result")
    SLIDE_STAGE = {SLIDE_INTRINSICS_PREP: 0, SLIDE_INTRINSICS_CALIB: 0, SLIDE_ROBOT_CONNECT: 0,
                   SLIDE_GRIPPER_POSE: 1, SLIDE_CAMERA_MOUNT: 1, SLIDE_GRIPPER_REMOVAL: 1, SLIDE_MARKER_BRACKET: 1,
                   SLIDE_EXPOSURE: 2, SLIDE_HOME_OFFSET: 2, SLIDE_CALIBRATION: 2, SLIDE_APPLY: 3}
    # The smaller line on the calibration slide follows the core's own progress events of the Full Auto run.
    RUN_KEYS = ("run_wrist", "run_head", "run_collect", "run_solve")
    # Slides that show the live camera feed (main_ui.update_video_frame).
    VIDEO_SLIDES = (SLIDE_CAMERA_MOUNT, SLIDE_INTRINSICS_CALIB, SLIDE_EXPOSURE)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 6, 10, 6)
        # Tight gaps: the progress line on top must still leave every slide inside a maximized
        # 1536x889 window (1920x1200 at 125 %).
        self.layout.setSpacing(4)

        self.stage_progress = StepProgress(len(self.STAGE_KEYS), dot=20, font_px=13)
        self.stage_progress.layout().setContentsMargins(80, 0, 80, 0)
        self.layout.addWidget(self.stage_progress)

        self.lbl_wizard_title = QLabel()
        self.lbl_wizard_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffd700;")
        self.lbl_wizard_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_wizard_title)

        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget, stretch=1)

        # Navigation Layout
        self.nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton(tr("wizard.btn_prev"))
        self.btn_skip = QPushButton(tr("wizard.btn_skip"))
        self.btn_next = QPushButton(tr("wizard.btn_next"))

        # Make all navigation buttons identical in size and enlarged
        for btn in (self.btn_prev, self.btn_skip, self.btn_next):
            btn.setFixedSize(140, 40)

        self.btn_prev.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_skip.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_next.setStyleSheet("background-color: #1e88e5; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")

        self.btn_prev.clicked.connect(self.go_prev)
        self.btn_skip.clicked.connect(self.go_next)
        self.btn_next.clicked.connect(self.go_next)

        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_skip)
        self.nav_layout.addWidget(self.btn_next)

        self.layout.addLayout(self.nav_layout)

        # State tracking for each step to enable Next
        self.step_completed = [False] * self.SLIDE_COUNT
        self.step_completed[self.SLIDE_INTRINSICS_PREP] = True
        self.step_completed[self.SLIDE_GRIPPER_POSE] = False      # Move (+ power off) or Skip
        self.step_completed[self.SLIDE_CAMERA_MOUNT] = True
        self.step_completed[self.SLIDE_GRIPPER_REMOVAL] = True
        self.step_completed[self.SLIDE_MARKER_BRACKET] = True
        self.step_completed[self.SLIDE_INTRINSICS_CALIB] = False  # Optional (Skip)
        self.step_completed[self.SLIDE_ROBOT_CONNECT] = False
        self.step_completed[self.SLIDE_HOME_OFFSET] = False       # Reset or Skip
        self.step_completed[self.SLIDE_EXPOSURE] = False          # Must confirm the exposure and reach the zero pose
        self.exposure_zero_done = False
        self.step_completed[self.SLIDE_CALIBRATION] = False
        self.step_completed[self.SLIDE_APPLY] = True

        self.check_pose_init_done = False

        # Unified Timer for Step 1 + Step 2 Calibration
        self.unified_timer = QTimer(self)
        self.unified_timer.timeout.connect(self.update_unified_time)
        self.unified_elapsed = 0
        self.run_phase, self.run_detail, self.run_skipped, self.run_outcome = -1, None, set(), None

        # Connect language changed signal
        LanguageManager.instance().language_changed.connect(self.on_language_changed)

        self.status_texts = {}     # label -> (i18n key, kwargs) of the status it shows
        self.setup_slides()
        self.wrapped_label_height = WrappedLabelHeight(
            [label for label in self.findChildren(QLabel) if label.wordWrap()], self)
        self.stacked_widget.currentChanged.connect(self.update_navigation)
        bridge = getattr(parent, "core_bridge", None)
        if bridge is not None:
            bridge.progress.connect(self.on_core_progress)
        self.refresh_run_progress()
        self.update_navigation(0)

    @staticmethod
    def status_text(key, kwargs):
        """kwargs values may be (key, kwargs) pairs themselves, translated too."""
        return tr(key, **{k: tr(v[0], **v[1]) if isinstance(v, tuple) else v for k, v in kwargs.items()})

    def set_status(self, label, key, style=None, **kwargs):
        """Show a translated status line; a language switch shows it again in the new language."""
        self.status_texts[label] = (key, kwargs)
        label.setText(self.status_text(key, kwargs))
        if style:
            label.setStyleSheet(style)

    def on_language_changed(self, lang):
        self.update_navigation(self.stacked_widget.currentIndex())
        self.refresh_run_progress()
        for label, (key, kwargs) in self.status_texts.items():
            label.setText(self.status_text(key, kwargs))

        # Slide 0
        if hasattr(self, 't0'): self.t0.setText(tr("wizard.slides.slide_0.title"))
        if hasattr(self, 'd0_box'): self.d0_box.setTitle(tr("wizard.slides.slide_0.box_title"))
        if hasattr(self, 'lbl_inst0_1'): self.lbl_inst0_1.setText(tr("wizard.slides.slide_0.inst1"))
        if hasattr(self, 'lbl_inst0_2'): self.lbl_inst0_2.setText(tr("wizard.slides.slide_0.inst2"))
        if hasattr(self, 'lbl_cam_bracket_head'): self.lbl_cam_bracket_head.setText(tr("wizard.slides.slide_0.cap_head"))
        if hasattr(self, 'lbl_cam_bracket_nohead'): self.lbl_cam_bracket_nohead.setText(tr("wizard.slides.slide_0.cap_nohead"))
        if hasattr(self, 'lbl_prep_notice'): self.lbl_prep_notice.setText(tr("wizard.slides.slide_intrinsics_prep.notice"))
        if hasattr(self, 'lbl_prep_steps'): self.lbl_prep_steps.setText(self.intrinsics_prep_steps())
        if hasattr(self, 'btn_open_board'): self.btn_open_board.setText(tr("wizard.slides.slide_intrinsics_prep.btn_open"))

        # Gripper-change pose
        if hasattr(self, 'lbl_gp_desc'): self.refresh_gripper_pose_desc()
        if hasattr(self, 'btn_gripper_pose'): self.btn_gripper_pose.setText(tr("wizard.slides.slide_gripper_pose.btn_move"))
        if hasattr(self, 'btn_gp_stop'): self.btn_gp_stop.setText(tr("wizard.btn_stop_motion"))
        if hasattr(self, 'lbl_gripper_pose_status'): self.set_gripper_pose_status(*self.gripper_pose_status)

        # Gripper removal & marker bracket attachment
        if hasattr(self, 'd1_2_box'): self.d1_2_box.setTitle(tr("wizard.slides.slide_1.box_title"))
        if hasattr(self, 'lbl_m1'): self.lbl_m1.setText(tr("wizard.slides.slide_1.inst1"))
        if hasattr(self, 'mb_box'): self.mb_box.setTitle(tr("wizard.slides.slide_marker_bracket.box_title"))
        if hasattr(self, 'lbl_mb_cap_assemble'): self.lbl_mb_cap_assemble.setText(tr("wizard.slides.slide_marker_bracket.cap_assemble"))
        if hasattr(self, 'lbl_mb_cap_overview'): self.lbl_mb_cap_overview.setText(tr("wizard.slides.slide_marker_bracket.cap_overview"))
        for i, name in enumerate(("lbl_mb1", "lbl_mb2", "lbl_mb3"), start=1):
            if hasattr(self, name): getattr(self, name).setText(tr(f"wizard.slides.slide_marker_bracket.inst{i}"))

        # Exposure & marker recognition
        if hasattr(self, 't_exp'): self.t_exp.setText(tr("wizard.slides.slide_exposure.title"))
        if hasattr(self, 'd_exp'): self.d_exp.setText(self.exposure_instructions())
        if hasattr(self, 'marker_monitor_panel'): self.marker_monitor_panel.retranslate()
        if hasattr(self, 'exp_motion_box'): self.exp_motion_box.setTitle(tr("wizard.slides.slide_exposure.motion_title"))
        if hasattr(self, 'lbl_exp_motion_note'): self.lbl_exp_motion_note.setText(tr("wizard.slides.slide_exposure.motion_note"))
        if hasattr(self, 'btn_exp_auto'): self.btn_exp_auto.setText(tr("wizard.slides.slide_exposure.btn_auto"))
        if hasattr(self, 'btn_exp_zero'): self.btn_exp_zero.setText(tr("wizard.slides.slide_exposure.btn_zero"))
        if hasattr(self, 'btn_exp_stop'): self.btn_exp_stop.setText(tr("wizard.btn_stop_motion"))
        if hasattr(self, 'lbl_exp_motion_status'): self.set_exposure_motion_status(*self.exposure_motion_status)
        if hasattr(self, 'lbl_wiz_exp_not_saved'): self.lbl_wiz_exp_not_saved.setText(tr("wizard.slides.slide_exposure.not_saved_warn"))
        if hasattr(self, 'chk_wiz_auto_exp'): self.chk_wiz_auto_exp.setText(tr("wizard.slides.slide_exposure.auto_exposure"))
        if hasattr(self, 'lbl_wiz_exp_text'): self.lbl_wiz_exp_text.setText(tr("wizard.slides.slide_exposure.exposure_label"))
        if hasattr(self, 'btn_wiz_apply_exp'): self.btn_wiz_apply_exp.setText(tr("wizard.slides.slide_exposure.btn_apply"))
        if hasattr(self, 'btn_wiz_cancel_exp'): self.btn_wiz_cancel_exp.setText(tr("wizard.slides.slide_exposure.btn_cancel"))
        if hasattr(self, 'chk_wiz_exp_confirmed'): self.chk_wiz_exp_confirmed.setText(tr("wizard.slides.slide_exposure.chk_confirmed"))
        if hasattr(self, 'exp_ctrl_box'): self.exp_ctrl_box.setTitle(tr("wizard.slides.slide_exposure.manual_title"))
        if hasattr(self, 'btn_exp_accept'): self.btn_exp_accept.setText(tr("wizard.slides.slide_exposure.btn_accept"))
        if hasattr(self, 'chk_exp_manual'): self.chk_exp_manual.setText(tr("wizard.slides.slide_exposure.chk_manual"))

        # Intrinsics calibration (optional)
        if hasattr(self, 't1'): self.t1.setText(tr("wizard.slides.slide_3.title"))
        if hasattr(self, 'lbl_skip_hint1'): self.lbl_skip_hint1.setText(tr("wizard.slides.slide_3.skip_hint"))
        if hasattr(self, 'instr_box'): self.instr_box.setTitle(tr("wizard.slides.slide_3.box_guidelines"))
        if hasattr(self, 'lbl_inst_3_1'): self.lbl_inst_3_1.setText(tr("wizard.slides.slide_3.inst1"))
        if hasattr(self, 'lbl_inst_3_2'): self.lbl_inst_3_2.setText(tr("wizard.slides.slide_3.inst2"))
        if hasattr(self, 'lbl_inst_3_3'): self.lbl_inst_3_3.setText(tr("wizard.slides.slide_3.inst3"))
        if hasattr(self, 'lbl_inst_3_4'): self.lbl_inst_3_4.setText(tr("wizard.slides.slide_3.inst4"))
        if hasattr(self, 'controls_box'): self.controls_box.setTitle(tr("wizard.slides.slide_3.box_controls"))
        if hasattr(self, 'chk_int_guide'): self.chk_int_guide.setText(tr("wizard.slides.slide_3.guide_overlay"))
        if hasattr(self, 'btn_int_capture'): self.btn_int_capture.setText(tr("wizard.slides.slide_3.btn_capture"))
        if hasattr(self, 'btn_int_calibrate'): self.btn_int_calibrate.setText(tr("wizard.slides.slide_3.btn_calibrate"))
        if hasattr(self, 'btn_int_save'): self.btn_int_save.setText(tr("wizard.slides.slide_3.btn_save"))
        if hasattr(self, 'btn_int_reset'): self.btn_int_reset.setText(tr("wizard.slides.slide_3.btn_reset"))
        if hasattr(self, 'stats_box2'): self.stats_box2.setTitle(tr("wizard.slides.slide_3.box_stats"))
        if hasattr(self, 'int_busy'): self.int_busy.retranslate()
        self.refresh_intrinsics_serial_status()

        # Slide 5 (Robot Connect)
        if hasattr(self, 't2'): self.t2.setText(tr("wizard.slides.slide_4.title"))
        if hasattr(self, 'd2'): self.d2.setText(tr("wizard.slides.slide_4.inst1"))
        if hasattr(self, 'head_desc'): self.head_desc.setText(tr("wizard.slides.slide_4.head_note"))
        if hasattr(self, 'lbl_bracket_query'): self.lbl_bracket_query.setText(tr("wizard.slides.slide_4.additional_bracket_query"))
        if hasattr(self, 'rdo_bracket_yes'): self.rdo_bracket_yes.setText(tr("wizard.slides.slide_4.yes"))
        if hasattr(self, 'rdo_bracket_no'): self.rdo_bracket_no.setText(tr("wizard.slides.slide_4.no"))
        if hasattr(self, 'conn_box'): self.conn_box.setTitle(tr("wizard.slides.slide_4.box_title"))
        if hasattr(self, 'lbl_run_where'): self.lbl_run_where.setText(tr("wizard.slides.slide_4.run_where"))
        if hasattr(self, 'btn_ip_upc'): self.btn_ip_upc.setText(tr("wizard.slides.slide_4.btn_upc"))
        if hasattr(self, 'btn_ip_external'): self.btn_ip_external.setText(tr("wizard.slides.slide_4.btn_external"))
        if hasattr(self, 'btn_wizard_connect'):
            self.btn_wizard_connect.setText(tr("wizard.btn_connected" if self.parent_app.robot is not None else "wizard.btn_connect"))

        # Slide 7 (Home Offset Pose)
        if hasattr(self, 't3_2'): self.t3_2.setText(tr("wizard.slides.slide_6.title"))
        if hasattr(self, 'teach_box'): self.teach_box.setTitle(tr("wizard.slides.slide_6.teach_title"))
        if hasattr(self, 'lbl_teach'): self.lbl_teach.setText(tr("wizard.slides.slide_6.teach_desc"))
        if hasattr(self, 'btn_how_to_move'): self.btn_how_to_move.setText(tr("wizard.slides.slide_6.btn_how_to_move"))
        if hasattr(self, 'lbl_p1'): self.lbl_p1.setText(tr("wizard.slides.slide_6.inst1"))
        if hasattr(self, 'lbl_p1_warn'): self.lbl_p1_warn.setText(tr("wizard.slides.slide_6.shoulder_warn"))
        if hasattr(self, 'lbl_p2'): self.lbl_p2.setText(tr("wizard.slides.slide_6.inst2"))
        if hasattr(self, 'lbl_p3'): self.lbl_p3.setText(tr("wizard.slides.slide_6.inst3"))
        if hasattr(self, 'btn_step3_reset'): self.btn_step3_reset.setText(tr("wizard.slides.slide_6.btn_reset"))

        # Slide 8 (Calibration Pipeline)
        if hasattr(self, 't4'): self.t4.setText(tr("wizard.slides.slide_7.title"))
        if hasattr(self, 'd4_step1'): self.d4_step1.setText(tr("wizard.slides.slide_7.desc"))
        if hasattr(self, 'btn_start_unified'): self.btn_start_unified.setText(tr("wizard.btn_start_calibration"))
        if hasattr(self, 'aux_box4'): self.aux_box4.setTitle(tr("wizard.safety_title"))
        if hasattr(self, 'feed_desc'): self.feed_desc.setText(tr("wizard.slides.slide_7.feed_desc"))
        if hasattr(self, 'btn_feed4'): self.btn_feed4.setText(tr("wizard.btn_open_feed"))
        if hasattr(self, 'stop_desc'): self.stop_desc.setText(tr("wizard.slides.slide_7.stop_desc"))
        if hasattr(self, 'btn_stop4'): self.btn_stop4.setText(tr("wizard.btn_stop_motion"))

        # Slide 9 (Apply Offset)
        if hasattr(self, 't6'): self.t6.setText(tr("wizard.slides.slide_8.title"))
        if hasattr(self, 'd6'): self.d6.setText(tr("wizard.slides.slide_8.desc"))
        if hasattr(self, 'apply_instructions_box'): self.apply_instructions_box.setTitle(tr("wizard.slides.slide_8.box_title"))
        if hasattr(self, 'lbl_apply1'): self.lbl_apply1.setText(tr("wizard.slides.slide_8.inst1"))
        if hasattr(self, 'lbl_apply2'): self.lbl_apply2.setText(tr("wizard.slides.slide_8.inst2"))
        if hasattr(self, 'lbl_apply3'): self.lbl_apply3.setText(tr("wizard.slides.slide_8.inst3"))
        if hasattr(self, 'lbl_apply4'): self.lbl_apply4.setText(tr("wizard.slides.slide_8.inst4"))
        if hasattr(self, 'btn_rollback_zero'):
            self.btn_rollback_zero.setText(tr("wizard.slides.slide_8.btn_rollback_zero"))
        if hasattr(self, 'btn_new_offset_zero'):
            self.btn_new_offset_zero.setText(tr("wizard.slides.slide_8.btn_new_offset_zero"))
        if hasattr(self, 'btn_rollback_preview'):
            self.btn_rollback_preview.setText(tr("wizard.slides.slide_8.btn_rollback_preview"))
        if hasattr(self, 'btn_new_offset_preview'):
            self.btn_new_offset_preview.setText(tr("wizard.slides.slide_8.btn_new_offset_preview"))
        if hasattr(self, 'btn_rollback_joint'):
            self.btn_rollback_joint.setText(tr("wizard.slides.slide_8.btn_rollback_joint"))
        if hasattr(self, 'btn_apply_new_offset'):
            self.btn_apply_new_offset.setText(tr("wizard.slides.slide_8.btn_apply_new_offset"))
        if hasattr(self, 'lbl_wiz_opt_locked'):
            self.lbl_wiz_opt_locked.setText(tr("dialogs.apply_home_offset.opt_locked"))

    @staticmethod
    def image_label(path, width, height):
        lbl = QLabel()
        pix = QPixmap(get_asset_path(path))
        if not pix.isNull():
            lbl.setPixmap(pix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl.setText(f"[{path} not found]")
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    @staticmethod
    def exposure_instructions():
        # Brightness first: operators did not know what a good exposure looks like (2026-11 feedback).
        return f"{tr('wizard.slides.slide_exposure.brightness_guide')}\n{tr('wizard.slides.slide_exposure.inst')}"

    @staticmethod
    def caption_label(text, style="font-size: 14px; color: #ffd700; font-weight: bold;"):
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    def setup_slides(self):
        # Built in the order below for readability, added to the stack in SLIDE_* order at the end.
        slides = {}

        # -----------------------------------------
        # 1-1. Camera Bracket Assembly & Mounting
        # -----------------------------------------
        slide0 = QWidget()
        l0 = QVBoxLayout(slide0)
        l0.setSpacing(14)
        l0.setAlignment(Qt.AlignCenter)

        self.t0 = QLabel(tr("wizard.slides.slide_0.title"))
        self.t0.setVisible(False)

        # Robot with a head (camera on the head bracket) | without a head (camera on the body).
        cam_bracket_row = QHBoxLayout()
        cam_bracket_row.setSpacing(24)
        cam_bracket_row.setAlignment(Qt.AlignCenter)
        for image, caption_key, attr in (("img/camera_bracket_head.png", "wizard.slides.slide_0.cap_head", "lbl_cam_bracket_head"),
                                         ("img/camera_bracket_nohead.png", "wizard.slides.slide_0.cap_nohead", "lbl_cam_bracket_nohead")):
            col = QVBoxLayout()
            col.setSpacing(6)
            col.addWidget(self.image_label(image, 380, 200))
            caption = self.caption_label(tr(caption_key))
            setattr(self, attr, caption)
            col.addWidget(caption)
            cam_bracket_row.addLayout(col)
        l0.addLayout(cam_bracket_row)
        # The recognised-camera box was dropped (2026-11-12, user): the camera is checked on the
        # distortion calibration slides before the robot connection.

        self.d0_box = QGroupBox(tr("wizard.slides.slide_0.box_title"))
        self.d0_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.d0_box.setFixedWidth(750)
        d0_layout = QVBoxLayout(self.d0_box)
        d0_layout.setSpacing(8)

        self.lbl_inst0_1 = QLabel(tr("wizard.slides.slide_0.inst1"))
        self.lbl_inst0_1.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_inst0_1.setWordWrap(True)
        d0_layout.addWidget(self.lbl_inst0_1)

        self.lbl_inst0_2 = QLabel(tr("wizard.slides.slide_0.inst2"))
        self.lbl_inst0_2.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_inst0_2.setWordWrap(True)
        d0_layout.addWidget(self.lbl_inst0_2)

        l0.addWidget(self.d0_box, alignment=Qt.AlignCenter)
        slides[self.SLIDE_CAMERA_MOUNT] = slide0

        # 2. Gripper-change pose: arms to the pose of the robot version, then the tool flange power off.
        slide_gp = QWidget()
        l_gp = QVBoxLayout(slide_gp)
        l_gp.setSpacing(14)
        l_gp.setAlignment(Qt.AlignCenter)
        self.lbl_gp_desc = QLabel(tr("wizard.slides.slide_gripper_pose.desc"))
        self.lbl_gp_desc.setStyleSheet("font-size: 16px; color: #dddddd; font-weight: bold;")
        self.lbl_gp_desc.setWordWrap(True)
        self.lbl_gp_desc.setAlignment(Qt.AlignCenter)
        self.lbl_gp_desc.setFixedWidth(900)
        l_gp.addWidget(self.lbl_gp_desc, alignment=Qt.AlignCenter)
        gp_row = QHBoxLayout()
        gp_row.setSpacing(12)
        gp_row.setAlignment(Qt.AlignCenter)
        self.btn_gripper_pose = QPushButton(tr("wizard.slides.slide_gripper_pose.btn_move"))
        self.btn_gripper_pose.setMinimumSize(360, 52)
        self.btn_gripper_pose.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 20px;")
        self.btn_gripper_pose.clicked.connect(self.move_to_gripper_pose)
        gp_row.addWidget(self.btn_gripper_pose)
        self.btn_gp_stop = QPushButton(tr("wizard.btn_stop_motion"))
        self.btn_gp_stop.setMinimumSize(160, 52)
        self.btn_gp_stop.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_gp_stop.clicked.connect(self.exposure_stop_motion)
        gp_row.addWidget(self.btn_gp_stop)
        l_gp.addLayout(gp_row)
        self.lbl_gripper_pose_status = QLabel()
        self.lbl_gripper_pose_status.setAlignment(Qt.AlignCenter)
        self.lbl_gripper_pose_status.setWordWrap(True)
        l_gp.addWidget(self.lbl_gripper_pose_status)
        self.set_gripper_pose_status("idle")
        slides[self.SLIDE_GRIPPER_POSE] = slide_gp

        # 4. Gripper removal (v1.2 only; a v1.3 bracket goes on the gripper).
        slide_gr = QWidget()
        l_gr = QVBoxLayout(slide_gr)
        l_gr.setSpacing(10)
        l_gr.setAlignment(Qt.AlignCenter)
        self.d1_2_box = QGroupBox(tr("wizard.slides.slide_1.box_title"))
        self.d1_2_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.d1_2_box.setFixedWidth(900)
        d1_2_layout = QVBoxLayout(self.d1_2_box)
        d1_2_layout.setSpacing(8)
        self.lbl_m1 = QLabel(tr("wizard.slides.slide_1.inst1"))
        self.lbl_m1.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_m1.setWordWrap(True)
        self.lbl_m1.setOpenExternalLinks(True)
        d1_2_layout.addWidget(self.lbl_m1)
        l_gr.addWidget(self.d1_2_box, alignment=Qt.AlignCenter)
        slides[self.SLIDE_GRIPPER_REMOVAL] = slide_gr

        # 5. Marker bracket attachment.
        slide_mb = QWidget()
        l_mb = QVBoxLayout(slide_mb)
        l_mb.setSpacing(10)
        l_mb.setAlignment(Qt.AlignCenter)

        # How the bracket bolts to the flange (photo of the connected robot's version: this slide
        # comes after the robot connection) | which marker goes on which arm.
        mb_row = QHBoxLayout()
        mb_row.setSpacing(24)
        mb_row.setAlignment(Qt.AlignCenter)
        col = QVBoxLayout()
        col.setSpacing(6)
        self.img_mb_assemble = self.image_label(self.BRACKET_ASSEMBLE_IMAGES["1.2"], 400, 180)
        col.addWidget(self.img_mb_assemble)
        self.lbl_mb_cap_assemble = self.caption_label(tr("wizard.slides.slide_marker_bracket.cap_assemble"))
        col.addWidget(self.lbl_mb_cap_assemble)
        mb_row.addLayout(col)
        col = QVBoxLayout()
        col.setSpacing(6)
        col.addWidget(self.image_label("img/marker_bracket_overview.png", 400, 180))
        self.lbl_mb_cap_overview = self.caption_label(tr("wizard.slides.slide_marker_bracket.cap_overview"))
        col.addWidget(self.lbl_mb_cap_overview)
        mb_row.addLayout(col)
        l_mb.addLayout(mb_row)
        self.bracket_version = "1.2"

        self.mb_box = QGroupBox(tr("wizard.slides.slide_marker_bracket.box_title"))
        self.mb_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.mb_box.setFixedWidth(900)
        mb_layout = QVBoxLayout(self.mb_box)
        mb_layout.setSpacing(8)
        for i in (1, 2, 3):
            lbl = QLabel(tr(f"wizard.slides.slide_marker_bracket.inst{i}"))
            lbl.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
            lbl.setWordWrap(True)
            setattr(self, f"lbl_mb{i}", lbl)
            mb_layout.addWidget(lbl)
        l_mb.addWidget(self.mb_box, alignment=Qt.AlignCenter)
        slides[self.SLIDE_MARKER_BRACKET] = slide_mb

        # -----------------------------------------
        # 3-1. Camera Exposure & Marker Recognition (before the zero pose: the buttons put one arm at
        # a time in the J5 ready pose so its marker faces the camera, then send the robot to zero)
        # -----------------------------------------
        slide_exp = QWidget()
        slide_exp_layout = QVBoxLayout(slide_exp)
        slide_exp_layout.setSpacing(10)

        self.t_exp = QLabel(tr("wizard.slides.slide_exposure.title"))
        self.t_exp.setVisible(False)

        self.d_exp = QLabel(self.exposure_instructions())
        self.d_exp.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.d_exp.setWordWrap(True)
        self.d_exp.setAlignment(Qt.AlignCenter)
        slide_exp_layout.addWidget(self.d_exp)

        content_exp_layout = QHBoxLayout()

        # Left: Live Camera Feed
        exp_left = QVBoxLayout()
        self.wizard_exposure_video_label = QLabel()
        self.set_status(self.wizard_exposure_video_label, "wizard.st.feed_loading")
        self.wizard_exposure_video_label.setAlignment(Qt.AlignCenter)
        self.wizard_exposure_video_label.setMinimumSize(480, 290)
        # Each frame is scaled to the label's size; letting the pixmap set the size hint grew the label
        # frame after frame until the slide scrolled (2026-11-12, 1536x889 window).
        self.wizard_exposure_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.wizard_exposure_video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        exp_left.addWidget(self.wizard_exposure_video_label, 1)

        # Live recognition and jitter of both markers, measured by the core (MarkerMonitor). The same
        # panel is on the Camera > Brightness tab.
        self.marker_monitor_panel = MarkerMonitorPanel(sides=("right",))
        self.marker_monitor_panel.restart_requested.connect(self.sync_marker_monitor)
        exp_left.addWidget(self.marker_monitor_panel)
        content_exp_layout.addLayout(exp_left, 3)

        # Right: robot poses that show a marker to the camera, then the exposure controls
        exp_right = QVBoxLayout()
        self.exp_motion_box = QGroupBox(tr("wizard.slides.slide_exposure.motion_title"))
        self.exp_motion_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 15px;}")
        motion_layout = QVBoxLayout(self.exp_motion_box)
        motion_layout.setSpacing(6)
        self.lbl_exp_motion_note = QLabel(tr("wizard.slides.slide_exposure.motion_note"))
        self.lbl_exp_motion_note.setStyleSheet("color: #dddddd; font-size: 13px; font-weight: bold;")
        self.lbl_exp_motion_note.setWordWrap(True)
        motion_layout.addWidget(self.lbl_exp_motion_note)
        # Steps 1) to 4) run top to bottom (2026-11-12, user: the loose buttons were confusing). 1) is one
        # test pose (right arm J5 ready pose) and the auto exposure on the right marker; Stop sits beside it.
        step1_row = QHBoxLayout()
        self.btn_exp_auto = QPushButton(tr("wizard.slides.slide_exposure.btn_auto"))
        self.btn_exp_auto.setMinimumHeight(40)
        self.btn_exp_auto.setStyleSheet("background-color: #448aff; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_exp_auto.clicked.connect(self.exposure_run_auto)
        step1_row.addWidget(self.btn_exp_auto, 3)
        self.btn_exp_stop = QPushButton(tr("wizard.btn_stop_motion"))
        self.btn_exp_stop.setMinimumHeight(40)
        self.btn_exp_stop.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 14px; border-radius: 6px;")
        self.btn_exp_stop.clicked.connect(self.exposure_stop_motion)
        step1_row.addWidget(self.btn_exp_stop, 1)
        motion_layout.addLayout(step1_row)
        self.btn_exp_zero = QPushButton(tr("wizard.slides.slide_exposure.btn_zero"))
        self.btn_exp_zero.setMinimumHeight(36)
        self.btn_exp_zero.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 14px; border-radius: 6px;")
        self.btn_exp_zero.clicked.connect(self.exposure_move_zero)
        self.lbl_exp_motion_status = QLabel()
        self.lbl_exp_motion_status.setWordWrap(True)
        self.lbl_exp_motion_status.setAlignment(Qt.AlignCenter)
        motion_layout.addWidget(self.lbl_exp_motion_status)
        self.set_exposure_motion_status("idle")
        exp_right.addWidget(self.exp_motion_box)

        self.exp_ctrl_box = exp_ctrl_box = QGroupBox(tr("wizard.slides.slide_exposure.manual_title"))   # 2)
        exp_ctrl_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 15px;}")
        exp_ctrl_layout = QVBoxLayout()
        exp_ctrl_layout.setSpacing(6)

        # 2026-11-12 (user): after 1) the operator answers "use this value?" -- Yes ticks 3) and goes on to
        # 4) the zero pose; the exposure controls and the 3) check show only with 'Adjust by hand'.
        self.lbl_exp_choice = QLabel()
        self.lbl_exp_choice.setWordWrap(True)
        self.lbl_exp_choice.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        exp_ctrl_layout.addWidget(self.lbl_exp_choice)
        choice_row = QHBoxLayout()
        self.btn_exp_accept = QPushButton(tr("wizard.slides.slide_exposure.btn_accept"))
        self.btn_exp_accept.setMinimumHeight(40)
        self.btn_exp_accept.setStyleSheet(
            "QPushButton { background-color: #1e88e5; color: white; font-weight: bold; font-size: 14px; border-radius: 6px; padding: 0 12px; }"
            "QPushButton:disabled { background-color: #262626; color: #616161; }")
        self.btn_exp_accept.clicked.connect(self.accept_auto_exposure)
        choice_row.addWidget(self.btn_exp_accept, 1)
        # Hand-path messages (applied, cancelled, confirmed) take Yes's place while 'Adjust by hand' is on.
        self.lbl_wiz_exp_status = QLabel("")
        self.lbl_wiz_exp_status.setStyleSheet("color: #ff9800; font-size: 13px; font-weight: bold;")
        self.lbl_wiz_exp_status.setWordWrap(True)
        choice_row.addWidget(self.lbl_wiz_exp_status, 1)
        self.chk_exp_manual = QCheckBox(tr("wizard.slides.slide_exposure.chk_manual"))
        self.chk_exp_manual.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px;")
        self.chk_exp_manual.toggled.connect(self.show_manual_exposure)
        choice_row.addWidget(self.chk_exp_manual)
        exp_ctrl_layout.addLayout(choice_row)

        self.exp_manual_panel = QWidget()
        manual_layout = QVBoxLayout(self.exp_manual_panel)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(6)
        exp_ctrl_layout.addWidget(self.exp_manual_panel)

        self.chk_wiz_auto_exp = QCheckBox(tr("wizard.slides.slide_exposure.auto_exposure"))
        self.chk_wiz_auto_exp.setChecked(True)
        self.chk_wiz_auto_exp.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px;")
        self.chk_wiz_auto_exp.toggled.connect(self.on_wiz_auto_exp_toggled)
        # The exposure is kept in memory only (setting.yaml is not written). One line beside the
        # checkbox: wrapped labels in this box squeezed the confirmation checkbox out of shape.
        self.lbl_wiz_exp_not_saved = QLabel(tr("wizard.slides.slide_exposure.not_saved_warn"))
        self.lbl_wiz_exp_not_saved.setStyleSheet("color: #ffb74d; font-size: 12px; font-weight: bold;")
        auto_row = QHBoxLayout()
        auto_row.addWidget(self.chk_wiz_auto_exp)
        auto_row.addStretch()
        auto_row.addWidget(self.lbl_wiz_exp_not_saved)
        manual_layout.addLayout(auto_row)

        spin_row = QHBoxLayout()
        self.lbl_wiz_exp_text = QLabel(tr("wizard.slides.slide_exposure.exposure_label"))
        self.lbl_wiz_exp_text.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        spin_row.addWidget(self.lbl_wiz_exp_text)

        self.spin_wiz_exp = QSpinBox()
        self.spin_wiz_exp.setRange(1, 200000)  # narrowed to the camera's own range once connected
        self.spin_wiz_exp.setSingleStep(500)
        self.spin_wiz_exp.setValue(6000)
        self.spin_wiz_exp.setEnabled(False)
        self.spin_wiz_exp.setStyleSheet("background-color: #1e1e1e; color: #00e5ff; font-weight: bold; font-size: 14px; padding: 4px;")
        self.spin_wiz_exp.valueChanged.connect(self.on_wiz_exposure_changed)
        spin_row.addWidget(self.spin_wiz_exp)

        self.lbl_wiz_exp_ms = QLabel("6.0 ms")
        self.lbl_wiz_exp_ms.setStyleSheet("color: #ffd700; font-weight: bold; font-size: 14px; min-width: 55px;")
        spin_row.addWidget(self.lbl_wiz_exp_ms)
        manual_layout.addLayout(spin_row)

        self.slider_wiz_exp = QSlider(Qt.Horizontal)
        self.slider_wiz_exp.setRange(1, 200000)
        self.slider_wiz_exp.setSingleStep(500)
        self.slider_wiz_exp.setPageStep(5000)
        self.slider_wiz_exp.setValue(6000)
        self.slider_wiz_exp.setEnabled(False)
        self.slider_wiz_exp.valueChanged.connect(self.on_wiz_exposure_changed)
        manual_layout.addWidget(self.slider_wiz_exp)

        # Filled in by the main window from the connected camera (its range and unit). Beside the value,
        # not on a row of its own, so the hand controls fit a maximized laptop window.
        self.lbl_wiz_exp_range = QLabel("")
        self.lbl_wiz_exp_range.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        spin_row.addWidget(self.lbl_wiz_exp_range)

        # Action Buttons Row (APPLY, CANCEL)
        wiz_btn_row = QHBoxLayout()
        self.btn_wiz_apply_exp = QPushButton(tr("wizard.slides.slide_exposure.btn_apply"))
        self.btn_wiz_apply_exp.setMinimumHeight(34)
        self.btn_wiz_apply_exp.setStyleSheet("background-color: #388e3c; color: white; font-weight: bold; font-size: 14px; border-radius: 6px;")
        self.btn_wiz_apply_exp.clicked.connect(self.on_wiz_apply_exp_clicked)
        wiz_btn_row.addWidget(self.btn_wiz_apply_exp)

        self.btn_wiz_cancel_exp = QPushButton(tr("wizard.slides.slide_exposure.btn_cancel"))
        self.btn_wiz_cancel_exp.setMinimumHeight(34)
        self.btn_wiz_cancel_exp.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 14px; border-radius: 6px;")
        self.btn_wiz_cancel_exp.clicked.connect(self.on_wiz_cancel_exp_clicked)
        wiz_btn_row.addWidget(self.btn_wiz_cancel_exp)
        manual_layout.addLayout(wiz_btn_row)

        exp_ctrl_layout.addStretch()

        # Confirmation Checkbox to Enable Next Button
        self.chk_wiz_exp_confirmed = QCheckBox(tr("wizard.slides.slide_exposure.chk_confirmed"))
        self.chk_wiz_exp_confirmed.setChecked(False)
        self.chk_wiz_exp_confirmed.setStyleSheet("""
            QCheckBox {
                color: #80d8ff;
                font-size: 14px;
                font-weight: bold;
                padding: 6px 10px;
                border: 2px solid #0091ea;
                border-radius: 6px;
                background-color: #0d2744;
            }
            QCheckBox:hover {
                border: 2px solid #00e5ff;
            }
            QCheckBox::indicator {
                width: 22px;
                height: 22px;
                border: 2px solid #616161;
                border-radius: 4px;
                background-color: #212121;
            }
            QCheckBox::indicator:checked {
                background-color: #00b0ff;
                border: 2px solid #80d8ff;
            }
        """)
        self.chk_wiz_exp_confirmed.toggled.connect(self.on_wiz_exp_confirmed_toggled)
        # 2026-11-17 (user): Yes already runs 4) the zero pose, so 3) and 4) live in the hand panel, on one
        # row so the hand path does not scroll.
        finish_row = QHBoxLayout()
        finish_row.addWidget(self.chk_wiz_exp_confirmed, 3)
        finish_row.addWidget(self.btn_exp_zero, 2)
        manual_layout.addLayout(finish_row)

        exp_ctrl_box.setLayout(exp_ctrl_layout)
        exp_right.addWidget(exp_ctrl_box)
        self.set_exposure_choice("choice_idle")
        self.show_manual_exposure(False)
        content_exp_layout.addLayout(exp_right, 2)

        slide_exp_layout.addLayout(content_exp_layout)
        slides[self.SLIDE_EXPOSURE] = slide_exp

        # 0. Before the intrinsics: why it matters, then print the board and lay it flat.
        slide_prep = QWidget()
        l_prep = QVBoxLayout(slide_prep)
        l_prep.setSpacing(16)
        l_prep.setAlignment(Qt.AlignCenter)
        self.lbl_prep_notice = QLabel(tr("wizard.slides.slide_intrinsics_prep.notice"))
        self.lbl_prep_notice.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 20px;")
        self.lbl_prep_notice.setWordWrap(True)
        self.lbl_prep_notice.setAlignment(Qt.AlignCenter)
        self.lbl_prep_notice.setFixedWidth(900)
        l_prep.addWidget(self.lbl_prep_notice, alignment=Qt.AlignCenter)
        self.lbl_prep_steps = QLabel(self.intrinsics_prep_steps())
        self.lbl_prep_steps.setStyleSheet("font-size: 16px; color: #dddddd; font-weight: bold;")
        self.lbl_prep_steps.setWordWrap(True)
        self.lbl_prep_steps.setFixedWidth(900)
        l_prep.addWidget(self.lbl_prep_steps, alignment=Qt.AlignCenter)
        self.btn_open_board = QPushButton(tr("wizard.slides.slide_intrinsics_prep.btn_open"))
        self.btn_open_board.setMinimumSize(360, 50)
        self.btn_open_board.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 20px;")
        self.btn_open_board.clicked.connect(self.open_charuco_board)
        self.btn_open_board.setEnabled(os.path.isfile(get_asset_path(self.CHARUCO_BOARD_PDF)))
        l_prep.addWidget(self.btn_open_board, alignment=Qt.AlignCenter)
        slides[self.SLIDE_INTRINSICS_PREP] = slide_prep

        # -----------------------------------------
        # Slide 3: Camera Intrinsics Calibration (Optional)
        # -----------------------------------------
        slide1 = QWidget()
        slide1_layout = QVBoxLayout(slide1)

        header1 = QVBoxLayout()
        self.t1 = QLabel(tr("wizard.slides.slide_3.title"))
        self.t1.setVisible(False)

        self.lbl_skip_hint1 = QLabel(tr("wizard.slides.slide_3.skip_hint"))
        self.lbl_skip_hint1.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 20px;")
        self.lbl_skip_hint1.setWordWrap(True)
        self.lbl_skip_hint1.setAlignment(Qt.AlignCenter)
        header1.addWidget(self.lbl_skip_hint1)

        self.lbl_int_serial_status = QLabel("")
        self.lbl_int_serial_status.setWordWrap(True)
        self.lbl_int_serial_status.setAlignment(Qt.AlignCenter)
        header1.addWidget(self.lbl_int_serial_status)
        slide1_layout.addLayout(header1)

        content1_layout = QHBoxLayout()

        int_left = QVBoxLayout()
        self.wizard_video_label = QLabel()
        self.set_status(self.wizard_video_label, "wizard.st.feed_loading")
        self.wizard_video_label.setAlignment(Qt.AlignCenter)
        self.wizard_video_label.setMinimumSize(480, 300)
        # Same growth loop as the exposure feed (frames scaled to the label's size, 2026-11-12).
        self.wizard_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.wizard_video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        int_left.addWidget(self.wizard_video_label, 3)

        self.instr_box = QGroupBox(tr("wizard.slides.slide_3.box_guidelines"))
        self.instr_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        instr_layout = QVBoxLayout()
        self.lbl_inst_3_1 = QLabel(tr("wizard.slides.slide_3.inst1"))
        self.lbl_inst_3_1.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_1.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_1)

        self.lbl_inst_3_2 = QLabel(tr("wizard.slides.slide_3.inst2"))
        self.lbl_inst_3_2.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_2.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_2)

        self.lbl_inst_3_3 = QLabel(tr("wizard.slides.slide_3.inst3"))
        self.lbl_inst_3_3.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_3.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_3)

        self.lbl_inst_3_4 = QLabel(tr("wizard.slides.slide_3.inst4"))
        self.lbl_inst_3_4.setStyleSheet("color: #dddddd; font-size: 14px; font-weight: bold;")
        self.lbl_inst_3_4.setWordWrap(True)
        instr_layout.addWidget(self.lbl_inst_3_4)

        self.instr_box.setLayout(instr_layout)
        int_left.addWidget(self.instr_box, 1)

        self.controls_box = QGroupBox(tr("wizard.slides.slide_3.box_controls"))
        self.controls_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        controls_layout = QVBoxLayout()

        self.chk_int_guide = QCheckBox(tr("wizard.slides.slide_3.guide_overlay"))
        self.chk_int_guide.setChecked(True)
        self.chk_int_guide.setStyleSheet("color: #00e5ff; font-size: 15px; font-weight: bold;")
        self.chk_int_guide.stateChanged.connect(self.parent_app.on_guide_changed)
        controls_layout.addWidget(self.chk_int_guide)

        self.btn_int_capture = QPushButton(tr("wizard.slides.slide_3.btn_capture"))
        self.btn_int_capture.setMinimumHeight(45)
        self.btn_int_capture.setStyleSheet("background-color: #1e88e5; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_capture.clicked.connect(self.step1_capture)
        controls_layout.addWidget(self.btn_int_capture)

        self.btn_int_calibrate = QPushButton(tr("wizard.slides.slide_3.btn_calibrate"))
        self.btn_int_calibrate.setMinimumHeight(45)
        self.btn_int_calibrate.setStyleSheet("background-color: #43a047; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_calibrate.clicked.connect(self.step1_run)
        controls_layout.addWidget(self.btn_int_calibrate)

        self.btn_int_save = QPushButton(tr("wizard.slides.slide_3.btn_save"))
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #fb8c00; color: #000000; font-size: 14px; font-weight: bold;")
        self.btn_int_save.clicked.connect(self.step1_save)
        controls_layout.addWidget(self.btn_int_save)

        self.btn_int_reset = QPushButton(tr("wizard.slides.slide_3.btn_reset"))
        self.btn_int_reset.setMinimumHeight(35)
        self.btn_int_reset.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 13px;")
        self.btn_int_reset.clicked.connect(self.parent_app.reset_intrinsics_captures)
        controls_layout.addWidget(self.btn_int_reset)

        self.controls_box.setLayout(controls_layout)

        int_right = QVBoxLayout()

        self.stats_box2 = QGroupBox(tr("wizard.slides.slide_3.box_stats"))
        self.stats_box2.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        stats_layout2 = QHBoxLayout()
        self.lbl_captured = QLabel()
        self.set_status(self.lbl_captured, "wizard.st.int_captured", frames=0)
        self.lbl_captured.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #448aff;")

        self.lbl_temp = QLabel(self.parent_app.get_temp_label_text())
        self.lbl_temp.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #fb8c00;")

        stats_layout2.addWidget(self.lbl_captured)
        stats_layout2.addStretch()
        stats_layout2.addWidget(self.lbl_temp)
        self.stats_box2.setLayout(stats_layout2)

        self.lbl_step1_status = QLabel()
        self.set_status(self.lbl_step1_status, "wizard.st.int_waiting")
        self.lbl_step1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step1_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")

        int_right.addWidget(self.stats_box2)
        int_right.addWidget(self.controls_box)
        int_right.addStretch()
        int_right.addWidget(self.lbl_step1_status)
        # Takes the status line's place while the solve runs (UnifiedCalibrationApp.set_intrinsics_busy).
        self.int_busy = BusyIndicator("main_tabs.intrinsics_calculating")
        int_right.addWidget(self.int_busy)
        int_right.addStretch()

        content1_layout.addLayout(int_left, 2)
        content1_layout.addLayout(int_right, 1)
        slide1_layout.addLayout(content1_layout, 1)
        slides[self.SLIDE_INTRINSICS_CALIB] = slide1

        # -----------------------------------------
        # Slide 4: Robot Connection
        # -----------------------------------------
        slide2 = QWidget()
        l2 = QVBoxLayout(slide2)
        l2.setSpacing(12)
        l2.setAlignment(Qt.AlignCenter)

        self.t2 = QLabel(tr("wizard.slides.slide_4.title"))
        self.t2.setVisible(False)

        self.d2 = QLabel(tr("wizard.slides.slide_4.inst1"))
        self.d2.setStyleSheet("font-size: 16px; color: #dddddd;")
        self.d2.setWordWrap(True)
        self.d2.setAlignment(Qt.AlignCenter)
        l2.addWidget(self.d2)

        self.lbl_step2_status = QLabel()
        self.set_status(self.lbl_step2_status, "wizard.st.waiting")
        self.lbl_step2_status.setAlignment(Qt.AlignCenter)
        self.lbl_step2_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l2.addWidget(self.lbl_step2_status)

        # Question Label (Centered, Yellow, Bold)
        self.lbl_bracket_query = QLabel(tr("wizard.slides.slide_4.additional_bracket_query"))
        self.lbl_bracket_query.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffd700; margin-top: 10px; margin-bottom: 5px;")
        self.lbl_bracket_query.setAlignment(Qt.AlignCenter)
        # 2026-11-17 (user): bracket question on the left, robot connection on the right (no scrolling).
        bracket_col = QVBoxLayout()
        bracket_col.addWidget(self.lbl_bracket_query)

        # 2-Column Image + Radio Button Layout
        bracket_layout = QHBoxLayout()
        bracket_layout.setAlignment(Qt.AlignCenter)
        bracket_layout.setSpacing(40)  # Generous spacing between columns

        rdo_style = """
            QRadioButton {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 2px solid #444444;
                border-radius: 6px;
                padding: 8px 30px;
                font-size: 15px;
                font-weight: bold;
            }
            QRadioButton::indicator {
                width: 0px;
                height: 0px;
            }
            QRadioButton:checked {
                background-color: #1e88e5;
                color: #ffffff;
                border: 2px solid #1e88e5;
            }
            QRadioButton:hover {
                border: 2px solid #448aff;
            }
        """

        # Left Column: Yes (Additional Bracket)
        col_yes = QVBoxLayout()
        col_yes.setAlignment(Qt.AlignCenter)
        col_yes.setSpacing(10)

        img_yes = QLabel()
        pix_yes = QPixmap(get_asset_path("img/additional_head_bracket.png"))
        if not pix_yes.isNull():
            img_yes.setPixmap(pix_yes.scaled(350, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_yes.setText("[additional_head_bracket.png not found]")
        img_yes.setAlignment(Qt.AlignCenter)
        img_yes.setStyleSheet("border: 1px solid #444444; border-radius: 4px; background-color: #1e1e1e;")
        col_yes.addWidget(img_yes)

        self.rdo_bracket_yes = QRadioButton(tr("wizard.slides.slide_4.yes"))
        self.rdo_bracket_yes.setStyleSheet(rdo_style)
        col_yes.addWidget(self.rdo_bracket_yes, alignment=Qt.AlignCenter)

        # Right Column: No (Standard/Direct Mount)
        col_no = QVBoxLayout()
        col_no.setAlignment(Qt.AlignCenter)
        col_no.setSpacing(10)

        img_no = QLabel()
        pix_no = QPixmap(get_asset_path("img/standard_head_bracket.png"))
        if not pix_no.isNull():
            img_no.setPixmap(pix_no.scaled(350, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_no.setText("[standard_head_bracket.png not found]")
        img_no.setAlignment(Qt.AlignCenter)
        img_no.setStyleSheet("border: 1px solid #444444; border-radius: 4px; background-color: #1e1e1e;")
        col_no.addWidget(img_no)

        self.rdo_bracket_no = QRadioButton(tr("wizard.slides.slide_4.no"))
        self.rdo_bracket_no.setStyleSheet(rdo_style)
        col_no.addWidget(self.rdo_bracket_no, alignment=Qt.AlignCenter)

        bracket_layout.addLayout(col_yes)
        bracket_layout.addLayout(col_no)
        bracket_col.addLayout(bracket_layout)

        # Group them
        self.bracket_btn_group = QButtonGroup(self)
        self.bracket_btn_group.addButton(self.rdo_bracket_yes)
        self.bracket_btn_group.addButton(self.rdo_bracket_no)

        self.rdo_bracket_no.setChecked(True)
        self.rdo_bracket_yes.toggled.connect(self.on_bracket_radio_changed)
        self.rdo_bracket_no.toggled.connect(self.on_bracket_radio_changed)

        self.conn_box = QGroupBox(tr("wizard.slides.slide_4.box_title"))
        self.conn_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        self.conn_box.setFixedWidth(560)
        conn_layout = QVBoxLayout()
        conn_layout.setSpacing(10)

        # 2026-11-17 (user): operators did not know which address to type. UPC fills it in; an external PC
        # gets the OLED-display guide. Without a camera the field keeps the simulator address below.
        # The question has a line of its own: beside the buttons the English text was cut off.
        self.lbl_run_where = QLabel(tr("wizard.slides.slide_4.run_where"))
        self.lbl_run_where.setStyleSheet("font-size: 14px; font-weight: bold;")
        conn_layout.addWidget(self.lbl_run_where)
        where_row = QHBoxLayout()
        self.btn_ip_upc = QPushButton(tr("wizard.slides.slide_4.btn_upc"))
        self.btn_ip_external = QPushButton(tr("wizard.slides.slide_4.btn_external"))
        for btn in (self.btn_ip_upc, self.btn_ip_external):
            btn.setStyleSheet("background-color: #2b5278; color: white; font-weight: bold; font-size: 14px; padding: 6px 12px; border-radius: 4px;")
            where_row.addWidget(btn)
        self.btn_ip_upc.clicked.connect(lambda: self.wizard_ip_input.setText(UPC_ROBOT_ADDRESS))
        self.btn_ip_external.clicked.connect(self.ask_external_robot_ip)
        conn_layout.addLayout(where_row)

        ip_row = QHBoxLayout()
        lbl_ip = QLabel("IP/Port:")
        lbl_ip.setStyleSheet("font-size: 15px; font-weight: bold;")
        ip_row.addWidget(lbl_ip)

        is_sim_or_ui = self.parent_app.ui_only or getattr(self.parent_app.core.observer, 'sim', False)
        default_ip = f"127.0.0.1:{ROBOT_PORT}" if is_sim_or_ui else UPC_ROBOT_ADDRESS
        self.wizard_ip_input = QLineEdit(default_ip)
        self.wizard_ip_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px; font-size: 15px;")
        ip_row.addWidget(self.wizard_ip_input)
        conn_layout.addLayout(ip_row)

        connect_row = QHBoxLayout()
        self.btn_wizard_connect = QPushButton(tr("wizard.btn_connect"))
        self.btn_wizard_connect.setMinimumWidth(160)
        self.btn_wizard_connect.setStyleSheet("background-color: #1e88e5; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px;")
        self.btn_wizard_connect.clicked.connect(self.step2_connect)
        connect_row.addWidget(self.btn_wizard_connect)

        self.wizard_chk_head = QCheckBox("Head")
        self.wizard_chk_head.setChecked(True)
        self.wizard_chk_head.setVisible(False)
        self.wizard_chk_head.toggled.connect(self.sync_bracket_radio)
        connect_row.addWidget(self.wizard_chk_head)
        conn_layout.addLayout(connect_row)

        self.conn_box.setLayout(conn_layout)
        body2 = QHBoxLayout()
        body2.setSpacing(30)
        body2.addStretch(1)
        body2.addLayout(bracket_col)
        body2.addWidget(self.conn_box, alignment=Qt.AlignVCenter)
        body2.addStretch(1)
        l2.addLayout(body2)

        slides[self.SLIDE_ROBOT_CONNECT] = slide2

        # -----------------------------------------
        # 3-3. Home Offset Position Setup
        # -----------------------------------------
        slide3_2 = QWidget()
        l3_2 = QVBoxLayout(slide3_2)
        l3_2.setSpacing(10)

        self.t3_2 = QLabel(tr("wizard.slides.slide_6.title"))
        self.t3_2.setVisible(False)

        self.lbl_step7_status = QLabel()
        self.set_status(self.lbl_step7_status, "wizard.st.waiting")
        self.lbl_step7_status.setAlignment(Qt.AlignCenter)
        self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l3_2.addWidget(self.lbl_step7_status)

        # Poses 1 and 2 are made by hand, so the teaching button is explained above the steps.
        self.teach_box = QGroupBox(tr("wizard.slides.slide_6.teach_title"))
        self.teach_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        teach_layout = QHBoxLayout(self.teach_box)
        teach_layout.setSpacing(15)
        self.lbl_teach = QLabel(tr("wizard.slides.slide_6.teach_desc"))
        self.lbl_teach.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_teach.setWordWrap(True)
        teach_layout.addWidget(self.lbl_teach, 1)
        self.btn_how_to_move = QPushButton(tr("wizard.slides.slide_6.btn_how_to_move"))
        self.btn_how_to_move.setMinimumSize(320, 56)
        self.btn_how_to_move.setStyleSheet("background-color: #448aff; color: white; font-weight: bold; font-size: 17px; border-radius: 6px; padding: 0 15px;")
        self.btn_how_to_move.clicked.connect(self.show_how_to_move_arms_dialog)
        teach_layout.addWidget(self.btn_how_to_move)
        l3_2.addWidget(self.teach_box)

        # One column per step: a big number and what to do on one row (same height for all three),
        # then its photo or button below.
        steps = QGridLayout()
        steps.setHorizontalSpacing(15)
        steps.setVerticalSpacing(8)
        text_style = "font-size: 15px; color: #ffffff; font-weight: bold; padding-top: 12px;"
        self.lbl_p1 = QLabel(tr("wizard.slides.slide_6.inst1"))
        self.lbl_p2 = QLabel(tr("wizard.slides.slide_6.inst2"))
        self.lbl_p3 = QLabel(tr("wizard.slides.slide_6.inst3"))
        for column, lbl in enumerate((self.lbl_p1, self.lbl_p2, self.lbl_p3)):
            lbl.setStyleSheet(text_style)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            header = QHBoxLayout()
            header.setSpacing(10)
            number = QLabel(str(column + 1))
            number.setStyleSheet("font-size: 34px; color: #ffd700; font-weight: bold;")
            number.setAlignment(Qt.AlignTop)
            header.addWidget(number)
            header.addWidget(lbl, 1)
            steps.addLayout(header, 0, column, Qt.AlignTop)
            steps.setColumnStretch(column, 1)

        shoulder_col = QVBoxLayout()
        shoulder_col.setSpacing(6)
        shoulder_col.addWidget(self.image_label("img/offset_reset_pose_shoulder_roll.png", 330, 250))
        self.lbl_p1_warn = self.caption_label(tr("wizard.slides.slide_6.shoulder_warn"),
                                              "font-size: 14px; color: #ff5252; font-weight: bold;")
        shoulder_col.addWidget(self.lbl_p1_warn)
        steps.addLayout(shoulder_col, 1, 0, Qt.AlignTop)
        steps.addWidget(self.image_label("img/offset_reset_pose_elbow.png", 330, 250), 1, 1, Qt.AlignTop)

        self.btn_step3_reset = QPushButton(tr("wizard.slides.slide_6.btn_reset"))
        self.btn_step3_reset.setMinimumHeight(56)
        self.btn_step3_reset.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 17px; border-radius: 6px; padding: 0 15px;")
        self.btn_step3_reset.clicked.connect(self.step3_reset)
        steps.addWidget(self.btn_step3_reset, 1, 2, Qt.AlignTop)
        l3_2.addLayout(steps)
        l3_2.addStretch()

        slides[self.SLIDE_HOME_OFFSET] = slide3_2

        # -----------------------------------------
        # Slide 7: 4. Calibration Start (Unified Step 1 + Step 2)
        # -----------------------------------------
        slide4 = QWidget()
        l4 = QVBoxLayout(slide4)
        l4.setAlignment(Qt.AlignCenter)
        l4.setSpacing(16)

        self.t4 = QLabel(tr("wizard.slides.slide_7.title"))
        self.t4.setVisible(False)

        d4_layout = QVBoxLayout()
        self.d4_step1 = QLabel(tr("wizard.slides.slide_7.desc"))
        self.d4_step1.setStyleSheet("font-size: 16px; color: #ffffff; font-weight: bold;")
        self.d4_step1.setWordWrap(True)
        self.d4_step1.setAlignment(Qt.AlignCenter)
        d4_layout.addWidget(self.d4_step1)
        l4.addLayout(d4_layout)

        self.run_progress = StepProgress(len(self.RUN_KEYS), dot=16, font_px=13)
        self.run_progress.layout().setContentsMargins(120, 0, 120, 0)
        l4.addWidget(self.run_progress)

        self.lbl_step4_status = QLabel()
        self.set_status(self.lbl_step4_status, "wizard.st.waiting")
        self.lbl_step4_status.setAlignment(Qt.AlignCenter)
        self.lbl_step4_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l4.addWidget(self.lbl_step4_status)

        action_row4 = QHBoxLayout()
        self.btn_start_unified = QPushButton(tr("wizard.btn_start_calibration"))
        self.btn_start_unified.setMinimumHeight(50)
        self.btn_start_unified.setMinimumWidth(260)
        self.btn_start_unified.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 18px; border-radius: 6px; padding: 0 15px;")
        self.btn_start_unified.clicked.connect(self.start_unified_calibration)

        action_row4.addStretch()
        action_row4.addWidget(self.btn_start_unified)
        action_row4.addStretch()
        l4.addLayout(action_row4)

        self.aux_box4 = QGroupBox(tr("wizard.safety_title"))
        self.aux_box4.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 15px;}")
        self.aux_box4.setFixedWidth(640)
        aux_layout4 = QVBoxLayout()
        aux_layout4.setSpacing(10)

        feed_row = QHBoxLayout()
        feed_row.addStretch()
        self.feed_desc = QLabel(tr("wizard.slides.slide_7.feed_desc"))
        self.feed_desc.setStyleSheet("font-size: 14px; color: #ffd700; font-weight: bold;")
        feed_row.addWidget(self.feed_desc)
        self.btn_feed4 = QPushButton(tr("wizard.btn_open_feed"))
        self.btn_feed4.setMinimumWidth(160)
        self.btn_feed4.setStyleSheet("background-color: #fb8c00; color: #000000; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        self.btn_feed4.clicked.connect(self.parent_app.toggle_camera_feed_dialog)
        feed_row.addWidget(self.btn_feed4)
        aux_layout4.addLayout(feed_row)

        stop_row = QHBoxLayout()
        stop_row.addStretch()
        self.stop_desc = QLabel(tr("wizard.slides.slide_7.stop_desc"))
        self.stop_desc.setStyleSheet("font-size: 14px; color: #ff5252; font-weight: bold;")
        stop_row.addWidget(self.stop_desc)
        self.btn_stop4 = QPushButton(tr("wizard.btn_stop_motion"))
        self.btn_stop4.setMinimumWidth(160)
        self.btn_stop4.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        self.btn_stop4.clicked.connect(self.stop_unified_calibration)
        stop_row.addWidget(self.btn_stop4)
        aux_layout4.addLayout(stop_row)

        self.aux_box4.setLayout(aux_layout4)
        l4.addWidget(self.aux_box4, alignment=Qt.AlignCenter)

        slides[self.SLIDE_CALIBRATION] = slide4

        # -----------------------------------------
        # Slide 8: 5. Apply Home Offset
        # -----------------------------------------
        slide6 = QWidget()
        l6 = QVBoxLayout(slide6)
        l6.setAlignment(Qt.AlignCenter)
        l6.setSpacing(12)

        self.t6 = QLabel(tr("wizard.slides.slide_8.title"))
        self.t6.setVisible(False)

        self.d6 = QLabel(tr("wizard.slides.slide_8.desc"))
        self.d6.setStyleSheet("font-size: 16px; color: #dddddd;")
        self.d6.setWordWrap(True)
        self.d6.setAlignment(Qt.AlignCenter)
        l6.addWidget(self.d6)

        self.lbl_step6_status = QLabel()
        self.set_status(self.lbl_step6_status, "wizard.st.waiting")
        self.lbl_step6_status.setAlignment(Qt.AlignCenter)
        self.lbl_step6_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l6.addWidget(self.lbl_step6_status)

        apply_row = QHBoxLayout()
        apply_row.setSpacing(15)

        img_apply = QLabel()
        pix_apply = QPixmap(get_asset_path("img/apply_offset.png"))
        if not pix_apply.isNull():
            img_apply.setPixmap(pix_apply.scaled(520, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_apply.setText("[img/apply_offset.png not found]")
        img_apply.setAlignment(Qt.AlignCenter)
        apply_row.addWidget(img_apply)

        self.apply_instructions_box = QGroupBox(tr("wizard.slides.slide_8.box_title"))
        self.apply_instructions_box.setStyleSheet("QGroupBox::title { color: #448aff; font-weight: bold; font-size: 16px;}")
        # Wider lines: at its own width the four steps wrapped to 3-4 lines and the slide no longer fit a
        # maximized 1920x1080 window (2026-11-12).
        self.apply_instructions_box.setMinimumWidth(640)
        apply_instr_layout = QVBoxLayout()
        apply_instr_layout.setSpacing(8)

        self.lbl_apply1 = QLabel(tr("wizard.slides.slide_8.inst1"))
        self.lbl_apply1.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply1.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply1)

        self.lbl_apply2 = QLabel(tr("wizard.slides.slide_8.inst2"))
        self.lbl_apply2.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply2.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply2)

        self.lbl_apply3 = QLabel(tr("wizard.slides.slide_8.inst3"))
        self.lbl_apply3.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply3.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply3)

        self.lbl_apply4 = QLabel(tr("wizard.slides.slide_8.inst4"))
        self.lbl_apply4.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
        self.lbl_apply4.setWordWrap(True)
        apply_instr_layout.addWidget(self.lbl_apply4)

        self.apply_instructions_box.setLayout(apply_instr_layout)
        apply_row.addWidget(self.apply_instructions_box)

        l6.addLayout(apply_row)

        # New buttons layout
        btn_container = QVBoxLayout()
        btn_container.setSpacing(10)

        # Row 0: Rollback / New Offset Zero Pose (same moves as the Step 2 apply dialog's "Move to Zero")
        row0_layout = QHBoxLayout()
        row0_layout.setSpacing(15)

        self.btn_rollback_zero = QPushButton(tr("wizard.slides.slide_8.btn_rollback_zero"))
        self.btn_rollback_zero.setMinimumHeight(40)
        self.btn_rollback_zero.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_rollback_zero.clicked.connect(lambda: self.wizard_move_zero("baseline"))
        row0_layout.addWidget(self.btn_rollback_zero)

        self.btn_new_offset_zero = QPushButton(tr("wizard.slides.slide_8.btn_new_offset_zero"))
        self.btn_new_offset_zero.setMinimumHeight(40)
        self.btn_new_offset_zero.setStyleSheet("background-color: #fb8c00; color: #000000; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_new_offset_zero.clicked.connect(lambda: self.wizard_move_zero("optimized"))
        row0_layout.addWidget(self.btn_new_offset_zero)
        btn_container.addLayout(row0_layout)

        # Row 1: Rollback / New Offset Preview Buttons
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(15)

        self.btn_rollback_preview = QPushButton(tr("wizard.slides.slide_8.btn_rollback_preview"))
        self.btn_rollback_preview.setMinimumHeight(40)
        self.btn_rollback_preview.setStyleSheet("background-color: #546e7a; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_rollback_preview.clicked.connect(lambda: self.wizard_move_check("baseline"))
        row1_layout.addWidget(self.btn_rollback_preview)

        self.btn_new_offset_preview = QPushButton(tr("wizard.slides.slide_8.btn_new_offset_preview"))
        self.btn_new_offset_preview.setMinimumHeight(40)
        self.btn_new_offset_preview.setStyleSheet("background-color: #fb8c00; color: #000000; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_new_offset_preview.clicked.connect(lambda: self.wizard_move_check("optimized"))
        row1_layout.addWidget(self.btn_new_offset_preview)
        btn_container.addLayout(row1_layout)

        # Row 2: Rollback Joint / Apply New Offset Buttons
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(15)

        self.btn_rollback_joint = QPushButton(tr("wizard.slides.slide_8.btn_rollback_joint"))
        self.btn_rollback_joint.setMinimumHeight(45)
        self.btn_rollback_joint.setStyleSheet("background-color: #e53935; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_rollback_joint.clicked.connect(lambda: self.wizard_apply_offset("baseline"))
        row2_layout.addWidget(self.btn_rollback_joint)

        self.btn_apply_new_offset = QPushButton(tr("wizard.slides.slide_8.btn_apply_new_offset"))
        self.btn_apply_new_offset.setMinimumHeight(45)
        self.btn_apply_new_offset.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_apply_new_offset.clicked.connect(lambda: self.wizard_apply_offset("optimized"))
        row2_layout.addWidget(self.btn_apply_new_offset)
        btn_container.addLayout(row2_layout)

        # Why applying the optimized result is disabled (not solved at the current zero in this run,
        # or already applied); previewing it and rollback stay available.
        self.lbl_wiz_opt_locked = QLabel(tr("dialogs.apply_home_offset.opt_locked"))
        self.lbl_wiz_opt_locked.setWordWrap(True)
        self.lbl_wiz_opt_locked.setAlignment(Qt.AlignCenter)
        self.lbl_wiz_opt_locked.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 14px;")
        btn_container.addWidget(self.lbl_wiz_opt_locked)

        l6.addLayout(btn_container)
        self.refresh_apply_gate()

        slides[self.SLIDE_APPLY] = slide6

        if sorted(slides) != list(range(self.SLIDE_COUNT)):
            raise RuntimeError(f"Wizard slides do not match SLIDE_* order: built {sorted(slides)}")
        for index in range(self.SLIDE_COUNT):
            # A slide taller than the window scrolls instead of squeezing (and cutting) its text,
            # and its full height no longer sets the window's minimum size.
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(slides[index])
            self.stacked_widget.addWidget(scroll)

    def set_bracket_version(self, version):
        """Show the marker bracket photo of robot version `version` ("1.2" / "1.3")."""
        version = str(version).removeprefix("v")
        if version not in self.BRACKET_ASSEMBLE_IMAGES:
            return
        self.bracket_version = version
        pix = QPixmap(get_asset_path(self.BRACKET_ASSEMBLE_IMAGES[version]))
        if pix.isNull():
            self.img_mb_assemble.setPixmap(QPixmap())
            self.img_mb_assemble.setText(f"[{self.BRACKET_ASSEMBLE_IMAGES[version]} not found]")
        else:
            self.img_mb_assemble.setPixmap(pix.scaled(400, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def sync_bracket_version_from_robot(self):
        """Show the bracket photo of the connected robot's version (called when the slide opens)."""
        if hasattr(self.parent_app, "get_robot_version"):
            self.set_bracket_version(self.parent_app.get_robot_version())

    def camera_info(self):
        core = getattr(self.parent_app, "core", None)
        observer = getattr(core, "observer", None)
        if observer is None or not hasattr(observer, "get_camera_info"):
            return None
        try:
            return observer.get_camera_info()
        except Exception:
            return None

    def refresh_intrinsics_serial_status(self):
        """Intrinsics slide: the same registered / not registered line as the Camera tab header."""
        if not hasattr(self, "lbl_int_serial_status"):
            return
        info = self.camera_info()
        text, color = self.parent_app.camera_serial_status(info) if info else ("", "#9e9e9e")
        self.lbl_int_serial_status.setText(text)
        self.lbl_int_serial_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 16px;")

    def show_how_to_move_arms_dialog(self):
        dlg = HowToMoveArmsDialog(self)
        dlg.exec()

    def on_wiz_auto_exp_toggled(self, checked):
        self.spin_wiz_exp.setEnabled(not checked)
        self.slider_wiz_exp.setEnabled(not checked)
        if hasattr(self.parent_app, 'chk_auto_exposure'):
            self.parent_app.chk_auto_exposure.blockSignals(True)
            self.parent_app.chk_auto_exposure.setChecked(checked)
            self.parent_app.chk_auto_exposure.blockSignals(False)
        if checked:
            self.parent_app.set_camera_auto_mode()
            self.set_status(self.lbl_wiz_exp_status, "wizard.st.exp_auto_switched",
                            "color: #00e5ff; font-size: 13px; font-weight: bold;")

    def exposure_ms(self, value):
        helper = getattr(self.parent_app, 'exposure_ms', None)
        return helper(value) if helper else float(value) / 1000.0

    def on_wiz_exposure_changed(self, value):
        self.lbl_wiz_exp_ms.setText(f"{self.exposure_ms(value):.1f} ms")
        if self.slider_wiz_exp.value() != value:
            self.slider_wiz_exp.blockSignals(True)
            self.slider_wiz_exp.setValue(value)
            self.slider_wiz_exp.blockSignals(False)
        if self.spin_wiz_exp.value() != value:
            self.spin_wiz_exp.blockSignals(True)
            self.spin_wiz_exp.setValue(value)
            self.spin_wiz_exp.blockSignals(False)

        if hasattr(self.parent_app, 'spin_exposure'):
            self.parent_app.spin_exposure.blockSignals(True)
            self.parent_app.spin_exposure.setValue(value)
            self.parent_app.spin_exposure.blockSignals(False)
        if hasattr(self.parent_app, 'slider_exposure'):
            self.parent_app.slider_exposure.blockSignals(True)
            self.parent_app.slider_exposure.setValue(value)
            self.parent_app.slider_exposure.blockSignals(False)
        if hasattr(self.parent_app, 'lbl_exposure_ms'):
            self.parent_app.lbl_exposure_ms.setText(f"{self.exposure_ms(value):.1f} ms")

    def on_wiz_apply_exp_clicked(self):
        auto_mode = self.chk_wiz_auto_exp.isChecked()
        exp_val = self.spin_wiz_exp.value()
        if self.parent_app.core.observer is not None and hasattr(self.parent_app.core.observer, 'set_camera_exposure'):
            self.parent_app.core.observer.set_camera_exposure(exp_val, auto_exposure=auto_mode)
        self.parent_app.reset_marker_monitor_confirmation()
        if auto_mode:
            self.set_status(self.lbl_wiz_exp_status, "wizard.st.exp_auto_applied")
        else:
            self.set_status(self.lbl_wiz_exp_status, "wizard.st.exp_manual_applied", value=exp_val,
                            ms=f"{self.exposure_ms(exp_val):.1f}")
        self.lbl_wiz_exp_status.setStyleSheet("color: #4caf50; font-size: 13px; font-weight: bold;")
        self.parent_app.log_msg(f"[Camera] Wizard applied exposure (auto={auto_mode}, exposure={exp_val} = {self.exposure_ms(exp_val):.1f} ms)")

    def on_wiz_cancel_exp_clicked(self):
        self.parent_app.cancel_camera_exposure()
        is_auto = self.parent_app.saved_camera_auto_exposure
        val = self.parent_app.saved_camera_exposure_value

        self.chk_wiz_auto_exp.blockSignals(True)
        self.chk_wiz_auto_exp.setChecked(is_auto)
        self.chk_wiz_auto_exp.blockSignals(False)

        self.spin_wiz_exp.blockSignals(True)
        self.spin_wiz_exp.setValue(val)
        self.spin_wiz_exp.setEnabled(not is_auto)
        self.spin_wiz_exp.blockSignals(False)

        self.slider_wiz_exp.blockSignals(True)
        self.slider_wiz_exp.setValue(val)
        self.slider_wiz_exp.setEnabled(not is_auto)
        self.slider_wiz_exp.blockSignals(False)

        self.lbl_wiz_exp_ms.setText(f"{self.exposure_ms(val):.1f} ms")
        self.set_status(self.lbl_wiz_exp_status, "wizard.st.exp_cancelled")
        self.lbl_wiz_exp_status.setStyleSheet("color: #ff9800; font-size: 13px; font-weight: bold;")

    def on_wiz_exp_confirmed_toggled(self, checked):
        self.refresh_exposure_done()
        if checked:
            self.parent_app.apply_camera_exposure()
            self.set_status(self.lbl_wiz_exp_status, "wizard.slides.slide_exposure.status_confirmed",
                            "color: #00e676; font-size: 14px; font-weight: bold;")
        else:
            self.status_texts.pop(self.lbl_wiz_exp_status, None)
            self.lbl_wiz_exp_status.setText("")
            self.lbl_wiz_exp_status.setStyleSheet("color: #ff9800; font-size: 13px; font-weight: bold;")

    # -----------------------------------------
    # Exposure slide: robot poses that show a marker to the camera
    # -----------------------------------------
    EXPOSURE_MOTION_STYLES = {"idle": "#9e9e9e", "moving": "#2196f3", "done": "#4caf50", "failed": "#f44336"}

    def set_exposure_motion_status(self, state, target="", detail=""):
        """state: idle / moving / done / failed; target: ready_right / ready_left / zero."""
        self.exposure_motion_status = (state, target, detail)
        name = tr(f"wizard.slides.slide_exposure.target_{target}") if target else ""
        text = tr(f"wizard.slides.slide_exposure.motion_status_{state}", target=name)
        self.lbl_exp_motion_status.setText(f"{text} ({detail})" if detail else text)
        self.lbl_exp_motion_status.setStyleSheet(
            f"color: {self.EXPOSURE_MOTION_STYLES[state]}; font-size: 13px; font-weight: bold;")

    def set_exposure_motion_buttons(self, enabled):
        for btn in (self.btn_exp_auto, self.btn_exp_zero):
            btn.setEnabled(enabled)

    def set_exposure_choice(self, key, style="color: #dddddd; font-size: 14px; font-weight: bold;", can_accept=False, **kwargs):
        """Box 2 question after 1); Yes is offered only when 1) left a value to keep."""
        self.set_status(self.lbl_exp_choice, f"wizard.slides.slide_exposure.{key}", style, **kwargs)
        self.btn_exp_accept.setEnabled(can_accept)

    def show_manual_exposure(self, checked):
        # Yes is not part of the hand path; hiding it keeps the slide inside a maximized laptop window.
        self.btn_exp_accept.setVisible(not checked)
        self.lbl_wiz_exp_status.setVisible(checked)
        self.exp_manual_panel.setVisible(checked)

    def accept_auto_exposure(self):
        self.chk_wiz_exp_confirmed.setChecked(True)   # applies the exposure and ticks 3)
        self.exposure_move_zero()

    def exposure_run_auto(self):
        # A new run replaces the value the operator was asked about.
        self.chk_wiz_exp_confirmed.setChecked(False)
        self.set_exposure_choice("choice_idle")
        self.start_exposure_motion("auto", lambda done: self.parent_app.run_auto_exposure(
            lambda success, error="", result=None: (self.show_auto_exposure_result(success, result or {}),
                                                    done(success, error))))

    def show_auto_exposure_result(self, success, result):
        """Mirror the applied exposure in both exposure controls and ask in box 2 whether to keep it."""
        if not success:
            return
        if result.get("applied") == "manual":
            value = int(result["exposure"])
            self.chk_wiz_auto_exp.setChecked(False)
            self.spin_wiz_exp.setValue(value)            # also updates the Camera tab controls
            if hasattr(self.parent_app, "chk_auto_exposure"):
                self.parent_app.chk_auto_exposure.setChecked(False)
            self.parent_app.reset_marker_monitor_confirmation()
            self.set_exposure_choice("auto_applied", "color: #00e676; font-size: 14px; font-weight: bold;", True,
                                     value=value, ms=f"{self.exposure_ms(value):.1f}")
        elif result.get("applied") == "auto":
            self.chk_wiz_auto_exp.setChecked(True)
            self.set_exposure_choice("auto_camera_auto", "color: #00e5ff; font-size: 14px; font-weight: bold;", True,
                                     model=result.get("camera_model") or "?")
        else:
            self.chk_wiz_auto_exp.setChecked(True)
            self.set_exposure_choice("auto_none", "color: #ff9800; font-size: 14px; font-weight: bold;")
            self.chk_exp_manual.setChecked(True)

    def exposure_move_zero(self):
        self.start_exposure_motion("zero", self.parent_app.move_to_zero_for_exposure)

    def refresh_exposure_done(self):
        """Next on the exposure slide needs 3) the check and 4) the zero pose (the home offset reset starts from it)."""
        self.step_completed[self.SLIDE_EXPOSURE] = self.exposure_zero_done and self.chk_wiz_exp_confirmed.isChecked()
        self.update_navigation(self.stacked_widget.currentIndex())

    def start_exposure_motion(self, target, start):
        def on_finished(success, error=""):
            self.set_wizard_busy(False)
            self.set_exposure_motion_buttons(True)
            self.set_exposure_motion_status("done" if success else "failed", target, "" if success else error)
            if target == "zero":
                self.exposure_zero_done = success
                self.refresh_exposure_done()
            # The robot move ran as a core sequence, which stops the marker monitor.
            self.sync_marker_monitor()

        self.exposure_zero_done = False
        self.refresh_exposure_done()
        if start(on_finished):
            self.set_exposure_motion_status("moving", target)
            self.set_exposure_motion_buttons(False)
            self.set_wizard_busy(True)
        elif not self.parent_app.robot:
            self.set_exposure_motion_status("failed", target, tr("wizard.st.robot_not_connected"))

    def exposure_stop_motion(self):
        self.parent_app.stop_motion()

    GRIPPER_POSE_STYLES = {"idle": "#9e9e9e", "moving": "#2196f3", "done": "#4caf50", "failed": "#f44336"}

    def gripper_power_kept(self):
        """A v1.3 marker bracket goes on the gripper, so the core leaves the tool flange power on."""
        return self.robot_version() == "1.3"

    def refresh_gripper_pose_desc(self):
        suffix = "_v13" if self.gripper_power_kept() else ""
        self.lbl_gp_desc.setText(tr(f"wizard.slides.slide_gripper_pose.desc{suffix}"))

    def set_gripper_pose_status(self, state, detail=""):
        """state: idle / moving / done (pose reached, tool power off unless v1.3) / failed."""
        self.gripper_pose_status = (state, detail)
        suffix = "_v13" if state == "done" and self.gripper_power_kept() else ""
        text = tr(f"wizard.slides.slide_gripper_pose.status_{state}{suffix}")
        self.lbl_gripper_pose_status.setText(f"{text} ({detail})" if detail else text)
        self.lbl_gripper_pose_status.setStyleSheet(
            f"color: {self.GRIPPER_POSE_STYLES[state]}; font-size: 15px; font-weight: bold;")

    def move_to_gripper_pose(self):
        def on_finished(success, error="", voltages=None):
            self.set_wizard_busy(False)
            self.btn_gripper_pose.setEnabled(True)
            if success:
                self.set_gripper_pose_status("done", ", ".join(f"{side} {value:.0f} V" for side, value in (voltages or {}).items()))
            else:
                self.set_gripper_pose_status("failed", error)
            self.mark_step_completed(self.SLIDE_GRIPPER_POSE, success)

        if self.parent_app.move_to_gripper_pose(on_finished):
            self.set_gripper_pose_status("moving")
            self.btn_gripper_pose.setEnabled(False)
            self.set_wizard_busy(True)
        elif not self.parent_app.robot:
            self.set_gripper_pose_status("failed", tr("wizard.st.robot_not_connected"))

    def sync_marker_monitor(self):
        """The app runs the core marker monitor while this slide or Camera > Brightness is shown."""
        if hasattr(self.parent_app, "sync_marker_monitor"):
            self.parent_app.sync_marker_monitor()

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_marker_monitor()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.sync_marker_monitor()

    def mark_step_completed(self, step_idx, success=True, msg=""):
        if step_idx < len(self.step_completed):
            self.step_completed[step_idx] = success
        self.update_navigation(self.stacked_widget.currentIndex())

        # Map step index to status label
        lbl_name = {
            self.SLIDE_EXPOSURE: "lbl_wiz_exp_status",
            self.SLIDE_INTRINSICS_CALIB: "lbl_step1_status",
            self.SLIDE_ROBOT_CONNECT: "lbl_step2_status",
            self.SLIDE_HOME_OFFSET: "lbl_step7_status",
            self.SLIDE_CALIBRATION: "lbl_step4_status",
            self.SLIDE_APPLY: "lbl_step6_status",
        }.get(step_idx)

        if lbl_name:
            lbl = getattr(self, lbl_name, None)
            if lbl:
                # msg: plain text (e.g. an error from the robot) or a (i18n key, kwargs) pair.
                if success:
                    self.set_status(lbl, "wizard.st.success_msg" if msg else "wizard.st.success",
                                    "color: #4caf50; font-weight: bold; font-size: 16px;", msg=msg)
                else:
                    self.set_status(lbl, "wizard.st.error_msg", "color: #f44336; font-weight: bold; font-size: 16px;",
                                    msg=msg)

    def ask_external_robot_ip(self):
        dlg = RobotIpDialog(self.wizard_ip_input.text(), self)
        if dlg.exec() == QDialog.Accepted and dlg.address():
            self.wizard_ip_input.setText(dlg.address())

    def intrinsics_prep_steps(self):
        return tr("wizard.slides.slide_intrinsics_prep.steps", file=self.CHARUCO_BOARD_PDF)

    def open_charuco_board(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(get_asset_path(self.CHARUCO_BOARD_PDF)))

    def intrinsics_registered(self):
        """The connected camera has intrinsics of its own (by serial number), so slides 0 and 0-1 are passed over."""
        info = self.camera_info()
        return bool(info and info.get("intrinsics_serial_matched"))

    def skip_passed_over_start(self):
        """Opening the wizard steps past slides passed over at the start (a camera registered by serial)."""
        idx = self.stacked_widget.currentIndex()
        while idx < self.SLIDE_COUNT - 1 and self.slide_passed_over(idx):
            idx += 1
        self.stacked_widget.setCurrentIndex(idx)
        self.update_navigation(idx)

    def refresh_stage_progress(self, idx):
        stage = self.SLIDE_STAGE[idx]
        states = ["done" if i < stage else "now" if i == stage else "todo" for i in range(len(self.STAGE_KEYS))]
        self.stage_progress.update_steps([tr(f"wizard.progress.{key}") for key in self.STAGE_KEYS], states,
                                         [tr(f"wizard.progress.{key}_tip") for key in self.STAGE_KEYS])

    def refresh_run_progress(self):
        states, names = [], []
        for i, key in enumerate(self.RUN_KEYS):
            if i in self.run_skipped:
                state = "skipped"
            elif self.run_outcome == "completed" or i < self.run_phase:
                state = "done"
            elif i == self.run_phase:
                state = "failed" if self.run_outcome == "failed" else "now"
            else:
                state = "todo"
            name = tr(f"wizard.progress.{key}")
            if key == "run_collect" and self.run_detail:
                name = tr("wizard.progress.run_collect_n", n=self.run_detail[0], total=self.run_detail[1])
            if state == "skipped":
                name = tr("wizard.progress.skipped", name=name)
            states.append(state)
            names.append(name)
        self.run_progress.update_steps(names, states)

    def on_core_progress(self, value):
        """Move the calibration slide's line along with the Full Auto run (Qt-queued, GUI thread)."""
        if not self.unified_timer.isActive():
            return
        phase = self.run_phase
        if isinstance(value, dict) and value.get("stage") == "collect":
            phase = max(phase, 2)
            self.run_detail = (value.get("pose_index", 0), value.get("pose_total", 0))
        else:
            name = getattr(value, "name", None)
            if name == "step1":
                phase = max(phase, 0)
            elif name == "step1_5":
                phase = max(phase, 1)
            elif name == "step2":
                solving = "dataset_path" in value.partial or "optimization" in value.partial
                phase = max(phase, 3 if solving else 2)
            elif name == "full":
                head = getattr(value.completed.get("step1_5"), "completed", {}).get("head_camera", {})
                if isinstance(head, dict) and head.get("skipped"):
                    self.run_skipped.add(1)
        self.run_phase = phase
        self.refresh_run_progress()

    def robot_version(self):
        getter = getattr(self.parent_app, "get_robot_version", None)
        return str(getter()).removeprefix("v") if getter else None

    @staticmethod
    def home_reset_pending():
        """(pending, created_at): a Home Offset Reset is on record whose calibration was not applied."""
        return home_reset_pending(StoragePaths.root / "config" / "home_reset_baseline.json")

    def slide_passed_over(self, idx):
        """Slides Next / Previous step over."""
        if idx in (self.SLIDE_INTRINSICS_PREP, self.SLIDE_INTRINSICS_CALIB):
            return self.intrinsics_registered()
        if idx == self.SLIDE_GRIPPER_REMOVAL:
            return self.robot_version() == "1.3"
        if idx == self.SLIDE_HOME_OFFSET:
            return self.home_reset_pending()[0]
        return False

    def confirm_home_reset_passed_over(self):
        """True to pass over the reset slide, False to open it anyway (e.g. another robot's record)."""
        created_at = self.home_reset_pending()[1] or "?"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(tr("wizard.slides.slide_6.pending_title"))
        box.setText(tr("wizard.slides.slide_6.pending_msg", created_at=created_at))
        box.addButton(QMessageBox.Ok)
        go_reset = box.addButton(tr("wizard.slides.slide_6.btn_go_reset"), QMessageBox.ActionRole)
        box.exec()
        return box.clickedButton() is not go_reset

    def go_prev(self):
        idx = self.stacked_widget.currentIndex()
        target = idx - 1
        while target >= 0 and self.slide_passed_over(target):
            target -= 1
        if target >= 0:
            self.stacked_widget.setCurrentIndex(target)
        else:
            if hasattr(self.parent_app, 'overview_container') and self.parent_app.overview_container:
                self.parent_app.overview_container.setVisible(True)
            else:
                if hasattr(self.parent_app, 'overview_title') and self.parent_app.overview_title:
                    self.parent_app.overview_title.setVisible(True)
                if hasattr(self.parent_app, 'overview_link') and self.parent_app.overview_link:
                    self.parent_app.overview_link.setVisible(True)
                if hasattr(self.parent_app, 'overview_duration') and self.parent_app.overview_duration:
                    self.parent_app.overview_duration.setVisible(True)
                self.parent_app.btn_start_wizard.setVisible(True)
                if hasattr(self.parent_app, 'overview_img') and self.parent_app.overview_img:
                    self.parent_app.overview_img.setVisible(True)
            self.setVisible(False)

    def go_next(self):
        idx = self.stacked_widget.currentIndex()
        if idx < self.stacked_widget.count() - 1:
            if self.sender() == self.btn_skip:
                self.step_completed[idx] = True
            target = idx + 1
            while target < self.SLIDE_COUNT - 1 and self.slide_passed_over(target):
                target += 1
            if idx < self.SLIDE_HOME_OFFSET < target and not self.confirm_home_reset_passed_over():
                target = self.SLIDE_HOME_OFFSET
            self.stacked_widget.setCurrentIndex(target)
            self.update_navigation(target)
        else:
            self.parent_app.log_msg("Calibration Wizard Finished.")
            if hasattr(self.parent_app, 'overview_container') and self.parent_app.overview_container:
                self.parent_app.overview_container.setVisible(True)
            else:
                if hasattr(self.parent_app, 'overview_title') and self.parent_app.overview_title:
                    self.parent_app.overview_title.setVisible(True)
                if hasattr(self.parent_app, 'overview_link') and self.parent_app.overview_link:
                    self.parent_app.overview_link.setVisible(True)
                if hasattr(self.parent_app, 'overview_duration') and self.parent_app.overview_duration:
                    self.parent_app.overview_duration.setVisible(True)
                self.parent_app.btn_start_wizard.setVisible(True)
                if hasattr(self.parent_app, 'overview_img') and self.parent_app.overview_img:
                    self.parent_app.overview_img.setVisible(True)
            self.setVisible(False)
            self.stacked_widget.setCurrentIndex(0)

    def update_navigation(self, idx):
        if idx != self.SLIDE_APPLY:
            self.check_pose_init_done = False

        if hasattr(self, "parent_app") and hasattr(self.parent_app, "on_left_tab_changed"):
            self.parent_app.on_left_tab_changed(self.parent_app.left_tabs.currentIndex())
        self.sync_marker_monitor()
        if idx == self.SLIDE_APPLY and hasattr(self, "btn_apply_new_offset"):
            self.refresh_apply_gate()
        if idx == self.SLIDE_MARKER_BRACKET:
            self.sync_bracket_version_from_robot()
        if idx == self.SLIDE_GRIPPER_POSE:
            self.refresh_gripper_pose_desc()
        self.refresh_stage_progress(idx)
        if idx == self.SLIDE_INTRINSICS_CALIB:
            self.refresh_intrinsics_serial_status()

        # Update shared top title dynamically to prevent title layout shifts
        if hasattr(self, 'lbl_wizard_title') and idx in self.TITLE_KEYS:
            self.lbl_wizard_title.setText(tr(self.TITLE_KEYS[idx]))

        self.btn_prev.setVisible(True)
        self.btn_prev.setText(tr("wizard.btn_prev"))

        show_skip = idx in (self.SLIDE_GRIPPER_POSE, self.SLIDE_INTRINSICS_CALIB, self.SLIDE_HOME_OFFSET)
        self.btn_skip.setVisible(show_skip)
        self.btn_skip.setText(tr("wizard.btn_skip"))

        if hasattr(self, 'lbl_skip_hint1'):
            self.lbl_skip_hint1.setVisible(idx == self.SLIDE_INTRINSICS_CALIB)

        enabled = self.step_completed[idx]
        self.btn_next.setEnabled(enabled)

        if enabled:
            self.btn_next.setStyleSheet("background-color: #1e88e5; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setStyleSheet("background-color: #222222; color: #616161; font-weight: bold; font-size: 15px; border-radius: 6px; border: 1px solid #333333;")

        if idx == self.stacked_widget.count() - 1:
            self.btn_next.setText(tr("wizard.btn_finish"))
            self.btn_next.setEnabled(True)
            self.btn_next.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setText(tr("wizard.btn_next"))

    # Step 1: Intrinsics
    def step1_capture(self):
        self.parent_app.capture_intrinsics_frame()
        frames = len(self.parent_app.captured_images)
        self.set_status(self.lbl_captured, "wizard.st.int_captured", frames=frames)
        # The 16th capture starts the solve by itself; its result sets the status when it arrives.
        if self.parent_app.intrinsics_busy:
            return
        if frames >= 16:
            err = self.parent_app.intrinsics_calibrator.rms_error
            if err is not None and err > 0.0:
                self.set_status(self.lbl_step1_status, "wizard.st.int_ok", "color: #d35400; font-weight: bold; font-size: 16px;",
                                rms=f"{err:.4f}")
            else:
                self.set_status(self.lbl_step1_status, "wizard.st.int_ready", "color: #27ae60; font-weight: bold; font-size: 16px;",
                                frames=frames)
        else:
            self.set_status(self.lbl_step1_status, "wizard.st.int_need", "color: #b0bec5; font-weight: bold; font-size: 16px;",
                            frames=frames)

    def step1_run(self):
        if len(self.parent_app.captured_images) < 16:
            self.set_status(self.lbl_step1_status, "wizard.st.int_need_all", "color: #c0392b; font-weight: bold; font-size: 16px;")
            QMessageBox.warning(self, tr("wizard.st.int_insufficient_title"),
                                tr("wizard.st.int_insufficient_run", frames=len(self.parent_app.captured_images)))
            return

        # The solve runs in a worker; UnifiedCalibrationApp._on_intrinsics_finished sets the status.
        # Reading rms_error here showed the previous run's value (or "failed") until it finished.
        self.parent_app.run_intrinsics_calibration()

    def step1_save(self):
        if len(self.parent_app.captured_images) < 16:
            QMessageBox.warning(self, tr("wizard.st.int_insufficient_title"),
                                tr("wizard.st.int_insufficient_save", frames=len(self.parent_app.captured_images)))
            self.mark_step_completed(self.SLIDE_INTRINSICS_CALIB, False, ("wizard.st.int_need_save", {}))
            return

        if self.parent_app.intrinsics_calibrator.cameraMatrix is not None and float(self.parent_app.intrinsics_calibrator.rms_error) > 0.0:
            self.parent_app.save_intrinsics_calibration()
            self.refresh_intrinsics_serial_status()
            self.mark_step_completed(self.SLIDE_INTRINSICS_CALIB, True, ("wizard.st.int_saved", {}))
        else:
            QMessageBox.warning(self, tr("wizard.st.int_invalid_title"), tr("wizard.st.int_invalid_msg"))
            self.mark_step_completed(self.SLIDE_INTRINSICS_CALIB, False, ("wizard.st.int_not_run", {}))

    # Step 2: Robot Connection
    def step2_connect(self):
        self.btn_wizard_connect.setText("CONNECTING...")
        self.btn_wizard_connect.setStyleSheet("background-color: #d35400; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px; border-radius: 6px; border: 1px solid #111111;")
        self.btn_wizard_connect.setEnabled(False)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self.parent_app.ip_input.setText(self.wizard_ip_input.text())
        self.parent_app.chk_servo_head.setChecked(self.wizard_chk_head.isChecked())

        self.parent_app.connect_robot()

        self.btn_wizard_connect.setEnabled(True)
        if self.parent_app.robot is not None:
            self.btn_wizard_connect.setText(tr("wizard.btn_connected"))
            self.btn_wizard_connect.setStyleSheet("background-color: #34495e; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px; border-radius: 6px; border: 1px solid #111111;")
            self.mark_step_completed(self.SLIDE_ROBOT_CONNECT, True, ("wizard.st.connected", {}))
        else:
            self.btn_wizard_connect.setText(tr("wizard.btn_connect"))
            self.btn_wizard_connect.setStyleSheet("background-color: #2b5278; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px; border-radius: 6px; border: 1px solid #111111;")
            self.mark_step_completed(self.SLIDE_ROBOT_CONNECT, False, ("wizard.st.connect_failed", {}))

    def sync_bracket_radio(self):
        is_head = self.wizard_chk_head.isChecked()
        self.rdo_bracket_yes.blockSignals(True)
        self.rdo_bracket_no.blockSignals(True)
        self.rdo_bracket_yes.setChecked(not is_head)
        self.rdo_bracket_no.setChecked(is_head)
        self.rdo_bracket_yes.blockSignals(False)
        self.rdo_bracket_no.blockSignals(False)

    def on_bracket_radio_changed(self):
        if not self.sender().isChecked():
            return
        is_yes = self.rdo_bracket_yes.isChecked()
        self.wizard_chk_head.blockSignals(True)
        self.wizard_chk_head.setChecked(not is_yes)
        self.wizard_chk_head.blockSignals(False)
        if hasattr(self.parent_app, 'on_head_checkbox_changed'):
            self.parent_app.on_head_checkbox_changed(not is_yes)

    # Step 3: Home Offset Reset
    def step3_reset(self):
        reply = QMessageBox.question(
            self,
            tr("wizard.st.reset_confirm_title"),
            tr("wizard.st.reset_confirm_msg"),
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        cancelled_style = "color: #aaaaaa; font-weight: bold; font-size: 16px;"
        if reply != QMessageBox.Ok:
            self.set_status(self.lbl_step7_status, "wizard.st.reset_cancelled", cancelled_style)
            return

        if self.parent_app.home_offset_reset(confirm_dialog=False):
            self.set_status(self.lbl_step7_status, "wizard.st.reset_progress", "color: #1e88e5; font-weight: bold; font-size: 16px;")
            self.set_wizard_busy(True)
        else:
            if not self.parent_app.robot:
                self.mark_step_completed(self.SLIDE_HOME_OFFSET, False, ("wizard.st.robot_not_connected", {}))
            else:
                self.set_status(self.lbl_step7_status, "wizard.st.reset_cancelled", cancelled_style)

    def set_wizard_busy(self, busy):
        self.btn_prev.setEnabled(not busy)
        self.btn_skip.setEnabled(not busy)
        if busy:
            self.btn_next.setEnabled(False)
            self.btn_next.setStyleSheet("background-color: #2a2a2a; color: #757575; font-weight: bold; font-size: 15px; border-radius: 6px; border: 1px solid #3d3d3d;")
        else:
            self.update_navigation(self.stacked_widget.currentIndex())
        if hasattr(self, 'btn_step3_reset'):
            self.btn_step3_reset.setEnabled(not busy)

    # -----------------------------------------
    # Unified Step 1, Step 1.5 & Step 2 Calibration Execution
    # -----------------------------------------
    def start_unified_calibration(self):
        self.unified_elapsed = 0
        self.run_phase, self.run_detail, self.run_skipped, self.run_outcome = 0, None, set(), None
        self.refresh_run_progress()
        self.set_status(self.lbl_step4_status, "wizard.st.calib_progress", "color: #1e88e5; font-weight: bold; font-size: 16px;",
                        time="00:00")
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(False)
            self.btn_start_unified.setStyleSheet("background-color: #2a2a2a; color: #757575; font-weight: bold; font-size: 18px; border-radius: 6px; border: 1px solid #3d3d3d;")

        self.unified_timer.start(1000)
        self.parent_app.start_full_auto()
        if hasattr(self.parent_app, 'active_worker') and self.parent_app.active_worker:
            self.parent_app.active_worker.finished_result.connect(self.on_unified_finished)
        else:
            self.stop_unified_calibration_error(("wizard.st.calib_not_started", {}))

    def on_unified_finished(self, result):
        self.stop_step5(result.success, result.error)

    def update_unified_time(self):
        self.unified_elapsed += 1
        m = self.unified_elapsed // 60
        s = self.unified_elapsed % 60
        self.set_status(self.lbl_step4_status, "wizard.st.calib_progress", time=f"{m:02d}:{s:02d}")


    def stop_step5(self, success=True, err_msg=""):
        self.unified_timer.stop()
        self.run_outcome = "completed" if success else "failed"
        self.refresh_run_progress()
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(True)
            self.btn_start_unified.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 18px; border-radius: 6px; padding: 0 15px;")

        m = self.unified_elapsed // 60
        s = self.unified_elapsed % 60
        time_str = f"{m:02d}:{s:02d}"
        if success:
            self.mark_step_completed(self.SLIDE_CALIBRATION, True, ("wizard.st.calib_done", {"time": time_str}))
        else:
            self.mark_step_completed(self.SLIDE_CALIBRATION,False, err_msg)

    def stop_unified_calibration(self):
        self.parent_app.stop_full_auto()
        if hasattr(self.parent_app, 'head_cam_stop_event') and self.parent_app.head_cam_stop_event is not None:
            self.parent_app.head_cam_stop_event.set()
        self.parent_app.request_stop_all_auto_motion()
        self.stop_unified_calibration_error(("wizard.st.calib_cancelled", {}))

    def stop_unified_calibration_error(self, err_msg=""):
        self.unified_timer.stop()
        self.run_outcome = "failed"
        self.refresh_run_progress()
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(True)
            self.btn_start_unified.setStyleSheet("background-color: #43a047; color: white; font-weight: bold; font-size: 18px; border-radius: 6px; padding: 0 15px;")
        self.mark_step_completed(self.SLIDE_CALIBRATION,False, err_msg)

    def get_apply_paths(self):
        result_path = self.parent_app.get_latest_result_path()
        baseline_path = self.parent_app.get_home_reset_path_for_result(result_path)
        return result_path, baseline_path

    @idle_core_action
    def wizard_move_check(self, state):
        self.wizard_preview_move(state, "check")

    @idle_core_action
    def wizard_move_zero(self, state):
        self.wizard_preview_move(state, "zero")

    def wizard_preview_move(self, state, target):
        """Move to the zero pose or the check pose under the baseline or optimized offsets (preview only)."""
        result_path, baseline_path = self.get_apply_paths()
        path = baseline_path if state == "baseline" else result_path

        if not path or not os.path.exists(path):
            QMessageBox.warning(self, tr("common.status_error"),
                                tr("wizard.step6.no_json", state=state))
            return

        self.set_wizard_buttons_enabled(False)
        self.lbl_step6_status.setText(tr("wizard.step6.moving_check" if target == "check" else "wizard.step6.moving_zero",
                                         state=state))
        self.lbl_step6_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")

        from ui.core_bridge import Step2ApplyHomeOffsetWorker
        if target == "check":
            self.wizard_worker = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "move_check",
                json_path=path,
                label=f"{state.capitalize()} Check Position",
                arm="both",
                include_head=self.parent_app.include_head_motion,
                skip_init_pose=self.check_pose_init_done
            )
        else:
            self.wizard_worker = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "move_zero",
                json_path=path,
                label=f"{state.capitalize()} Zero",
                arm="both",
                include_head=self.parent_app.include_head_motion
            )
        self.wizard_worker.log_signal.connect(self.parent_app.log_msg)

        def on_finished(success, error_msg, res):
            self.set_wizard_buttons_enabled(True)
            if success:
                # From the zero pose the next check move must pass the joint ready pose again, as the
                # first one did; only check -> check may skip it.
                self.check_pose_init_done = target == "check"
                self.lbl_step6_status.setText(tr("wizard.step6.arrived_check" if target == "check" else "wizard.step6.arrived_zero",
                                                 state=state))
                self.lbl_step6_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                QMessageBox.information(self, tr("wizard.step6.preview_complete_title"),
                                        tr("wizard.step6.preview_complete_msg" if target == "check" else "wizard.step6.preview_zero_msg",
                                           state=state))
            else:
                self.check_pose_init_done = False   # stopped somewhere on the way
                self.lbl_step6_status.setText(tr("wizard.step6.preview_error_status"))
                self.lbl_step6_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
                QMessageBox.critical(self, tr("wizard.step6.preview_error_title"), error_msg)

        self.wizard_worker.finished_signal.connect(on_finished)
        self.wizard_worker.start()

    @idle_core_action
    def wizard_apply_offset(self, state):
        result_path, baseline_path = self.get_apply_paths()
        path = baseline_path if state == "baseline" else result_path

        if not path or not os.path.exists(path):
            QMessageBox.warning(self, tr("common.status_error"),
                                tr("wizard.step6.no_json", state=state))
            return
        if self.optimized_blocked(state):
            return

        confirm_msg = tr("wizard.step6.confirm_msg", state=state.upper())
        confirm = QMessageBox.question(
            self,
            tr("wizard.step6.confirm_title"),
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        self.set_wizard_buttons_enabled(False)
        self.parent_app.log_msg(f"[INFO] Moving robot to '{state.upper()}' Zero Pose before applying home offset...")
        self.lbl_step6_status.setText(tr("wizard.step6.moving_zero", state=state))
        self.lbl_step6_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")

        from ui.core_bridge import Step2ApplyHomeOffsetWorker
        # Inferred arm
        current_apply_arm = self.parent_app.infer_home_offset_apply_arm("both", result_path)

        self.wizard_worker_move = Step2ApplyHomeOffsetWorker(
            self.parent_app,
            "move_zero",
            json_path=path,
            label=f"{state.capitalize()} Zero",
            arm="both",
            include_head=self.parent_app.include_head_motion
        )
        self.wizard_worker_move.log_signal.connect(self.parent_app.log_msg)

        def on_move_finished(success, error_msg, res):
            # The arm has left the check pose (for the zero pose, or stopped part way).
            self.check_pose_init_done = False
            if not success:
                self.set_wizard_buttons_enabled(True)
                self.lbl_step6_status.setText(tr("wizard.step6.move_zero_error_status"))
                self.lbl_step6_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
                QMessageBox.critical(self, tr("wizard.step6.move_zero_error_title"),
                                     tr("wizard.step6.move_zero_error_msg", error_msg=error_msg))
                return

            arm_to_apply = res.get("arm", current_apply_arm)
            self.parent_app.log_msg(f"[INFO] Arrived at '{state.upper()}' Zero Pose. Now resetting and applying home offset...")
            self.lbl_step6_status.setText(tr("wizard.step6.applying_offset", state=state))

            self.wizard_worker_apply = Step2ApplyHomeOffsetWorker(
                self.parent_app,
                "apply",
                arm=arm_to_apply,
                include_head=self.parent_app.include_head_motion,
                json_path=result_path if state == "optimized" else None
            )
            self.wizard_worker_apply.log_signal.connect(self.parent_app.log_msg)

            def on_apply_finished(app_success, app_error_msg, app_res):
                self.set_wizard_buttons_enabled(True)
                if app_success:
                    if app_res.get("needs_reconnect", False):
                        self.parent_app.log_msg("Re-connecting and initializing robot...")
                        if self.parent_app.robot:
                            self.parent_app.connect_robot()
                            from PySide6.QtWidgets import QApplication
                            QApplication.processEvents()
                        self.parent_app.connect_robot()
                        self.parent_app.log_msg("Current pose home offset apply complete.")

                    if app_res.get("success", False) or app_res.get("needs_reconnect", False):
                        # Reset software joint offsets to 0.0 for the applied arm(s) since they are now physically absorbed
                        for arm in ["left", "right"]:
                            if arm_to_apply == "both" or arm_to_apply == arm:
                                self.parent_app.joint_offsets_store[arm]["joint3"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint5"] = 0.0
                                self.parent_app.joint_offsets_store[arm]["joint6"] = 0.0

                                self.parent_app.joint_offsets[arm]["wrist_pitch"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_roll"] = 0.0
                                self.parent_app.joint_offsets[arm]["wrist_yaw2"] = 0.0
                                self.parent_app.joint_offsets[arm]["elbow"] = 0.0

                        # Save zeroed offsets to setting.yaml and update GUI
                        self.parent_app.save_offsets_to_yaml()
                        self.parent_app.update_applied_offset_label()

                        # Zero out baseline json if it exists to prevent accidental unsafe rollback later
                        if baseline_path and os.path.exists(baseline_path):
                            try:
                                import json
                                data = ResultStorage.load(baseline_path)

                                if "right_arm_joint_offset_deg" in data and (arm_to_apply == "both" or arm_to_apply == "right"):
                                    data["right_arm_joint_offset_deg"] = [0.0] * len(data["right_arm_joint_offset_deg"])
                                if "left_arm_joint_offset_deg" in data and (arm_to_apply == "both" or arm_to_apply == "left"):
                                    data["left_arm_joint_offset_deg"] = [0.0] * len(data["left_arm_joint_offset_deg"])
                                if "head_joint_offset_deg" in data and data["head_joint_offset_deg"] is not None and self.parent_app.include_head_motion:
                                    data["head_joint_offset_deg"] = [0.0] * len(data["head_joint_offset_deg"])

                                if "right_arm_joint_offset_deg" in data and "left_arm_joint_offset_deg" in data:
                                    data["joint_offset_deg"] = data["right_arm_joint_offset_deg"] + data["left_arm_joint_offset_deg"]
                                elif "joint_offset_deg" in data:
                                    data["joint_offset_deg"] = [0.0] * len(data["joint_offset_deg"])

                                ResultStorage.save(baseline_path, data)
                                self.parent_app.log_msg(f"[INFO] Zeroed out applied arm offsets in baseline json: {baseline_path}")
                            except Exception as e:
                                self.parent_app.log_msg(f"[WARN] Failed to zero out baseline json: {e}")

                        self.lbl_step6_status.setText(tr("wizard.step6.success_status", state=state.upper()))
                        self.lbl_step6_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                        self.mark_step_completed(self.SLIDE_APPLY,True, f"'{state.upper()}' home offset applied.")

                        QMessageBox.information(self, tr("wizard.step6.success_title"),
                                                tr("wizard.step6.success_msg", state=state.upper()))
                    else:
                        self.lbl_step6_status.setText(tr("wizard.step6.partial_fail_status"))
                        self.lbl_step6_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 16px;")
                        QMessageBox.warning(self, tr("wizard.step6.partial_fail_title"),
                                            tr("wizard.step6.partial_fail_msg"))
                else:
                    self.lbl_step6_status.setText(tr("wizard.step6.apply_error_status"))
                    self.lbl_step6_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
                    QMessageBox.critical(self, tr("wizard.step6.apply_error_title"), app_error_msg)

            self.wizard_worker_apply.finished_signal.connect(on_apply_finished)
            self.wizard_worker_apply.start()

        self.wizard_worker_move.finished_signal.connect(on_move_finished)
        self.wizard_worker_move.start()

    def optimized_apply_allowed(self):
        """The optimized result may be applied only while the core allows it (solved from samples at
        the current robot zero in this run, not applied yet). Previewing it is always possible."""
        core = getattr(self.parent_app, "core", None)
        if core is None or not hasattr(core, "result_apply_allowed") or core.applicable_result_path is None:
            return False   # fails closed; the lock label says why
        result_path, _ = self.get_apply_paths()
        return bool(result_path and os.path.exists(result_path) and core.result_apply_allowed(result_path))

    def refresh_apply_gate(self, enabled=True):
        allowed = self.optimized_apply_allowed()
        result_path, _ = self.get_apply_paths()
        exists = bool(result_path and os.path.exists(result_path))
        for btn in (self.btn_new_offset_zero, self.btn_new_offset_preview):
            btn.setEnabled(enabled and exists)
        self.btn_apply_new_offset.setEnabled(enabled and allowed)
        if hasattr(self, "lbl_wiz_opt_locked"):
            self.lbl_wiz_opt_locked.setVisible(not allowed)

    def optimized_blocked(self, state):
        if state != "optimized" or self.optimized_apply_allowed():
            return False
        QMessageBox.warning(self, tr("dialogs.apply_home_offset.opt_locked_title"),
                            tr("dialogs.apply_home_offset.opt_locked"))
        self.refresh_apply_gate()
        return True

    def set_wizard_buttons_enabled(self, enabled):
        self.btn_rollback_zero.setEnabled(enabled)
        self.btn_rollback_preview.setEnabled(enabled)
        self.btn_rollback_joint.setEnabled(enabled)
        self.refresh_apply_gate(enabled)
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
