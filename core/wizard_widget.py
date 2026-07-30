from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QGroupBox, QCheckBox, QLineEdit, QMessageBox, QDialog,
    QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap
from core.i18n import I18nManager, tr

class HowToMoveArmsDialog(QDialog):
    def __init__(self, parent=None, is_ko=False):
        super().__init__(parent)
        is_ko = (I18nManager.instance().current_lang == "ko")
        self.setWindowTitle("팔 이동 방법 (Direct Teaching)" if is_ko else "How to Move Arms (Direct Teaching)")
        self.resize(750, 520)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #ffffff; font-size: 14px; }
            QGroupBox { border: 2px solid #2d2d2d; border-radius: 8px; margin-top: 15px; font-weight: bold; font-size: 15px; color: #00e5ff; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 15px; padding: 0 5px; }
            QPushButton { background-color: #1565c0; color: white; font-weight: bold; font-size: 14px; padding: 8px 16px; border-radius: 6px; }
            QPushButton:hover { background-color: #1e88e5; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        lbl_title = QLabel("직접 교시 버튼 사용 안내" if is_ko else "Direct Teaching Button Usage")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffeb3b;")
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)
        
        img_lbl = QLabel()
        pix = QPixmap("img/teaching_button.png")
        if not pix.isNull():
            img_lbl.setPixmap(pix.scaled(550, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_lbl.setText("[img/teaching_button.png]")
        img_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_lbl)
        
        box = QGroupBox("직접 교시 사용 안내" if is_ko else "Direct Teaching Steps")
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)
        
        insts = [
            "1. 각 팔의 직접 교시 버튼을 눌러 수동으로 주요 관절 위치를 설정합니다." if is_ko else "1. Press the direct teaching button on each arm to manually adjust key joint postures.",
            "2. 중요: 보정값이 반대 방향으로 계산되어 오작동을 일으키지 않도록, 지정된 주요 관절을 수작업으로 옮겨주어야 합니다." if is_ko else "2. Important: Manually move target joints so calibration offsets are calculated in the correct direction."
        ]
        for txt in insts:
            lbl = QLabel(txt)
            lbl.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
            lbl.setWordWrap(True)
            box_layout.addWidget(lbl)
            
        warn_lbl = QLabel("⚠️ 경고: 양팔의 직접 교시 버튼을 절대로 동시에 누르지 마십시오!" if is_ko else "⚠️ Warning: NEVER press teaching buttons on both arms simultaneously!")
        warn_lbl.setStyleSheet("font-size: 16px; color: #ff5252; font-weight: bold;")
        warn_lbl.setWordWrap(True)
        warn_lbl.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(warn_lbl)
        
        layout.addWidget(box)
        
        btn_close = QPushButton("확인 (Close)" if is_ko else "Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignCenter)

class CalibrationWizardWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(12)
        
        self.lbl_wizard_title = QLabel()
        self.lbl_wizard_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
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
            btn.setFixedSize(140, 45)
            
        self.btn_prev.setStyleSheet("background-color: #555555; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_skip.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_next.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        
        self.btn_prev.clicked.connect(self.go_prev)
        self.btn_skip.clicked.connect(self.go_next)
        self.btn_next.clicked.connect(self.go_next)
        
        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_skip)
        self.nav_layout.addWidget(self.btn_next)
        
        self.layout.addLayout(self.nav_layout)
        
        # State tracking for each step to enable Next (9 slides total)
        self.step_completed = [False] * 9
        self.step_completed[0] = True   # 1-1 Camera Mounting
        self.step_completed[1] = True   # 1-2 Marker Attachment
        self.step_completed[2] = True   # 1-3 Intrinsics Check
        self.step_completed[3] = False  # Intrinsics Calibration (Optional)
        self.step_completed[4] = False  # Robot Connection
        self.step_completed[5] = False  # 3-1 Initial Zero (Must move zero position to complete)
        self.step_completed[6] = False  # 3-2 Home Offset Position Setup
        self.step_completed[7] = False  # 4. Calibration Start (Step 1 + Step 2 Unified)
        self.step_completed[8] = True   # 5. Apply Home Offset
        
        # Unified Timer for Step 1 + Step 2 Calibration
        self.unified_timer = QTimer(self)
        self.unified_timer.timeout.connect(self.update_unified_time)
        self.unified_elapsed = 0
        
        # Connect language changed signal
        I18nManager.instance().language_changed.connect(self.on_language_changed)
        
        self.setup_slides()
        self.stacked_widget.currentChanged.connect(self.update_navigation)
        self.update_navigation(0)

    def on_language_changed(self, lang):
        self.update_navigation(self.stacked_widget.currentIndex())
        
        # Slide 0
        if hasattr(self, 't0'): self.t0.setText(tr("wizard.slides.slide_0.title"))
        if hasattr(self, 'd0_box'): self.d0_box.setTitle(tr("wizard.slides.slide_0.box_title"))
        if hasattr(self, 'lbl_inst0_1'): self.lbl_inst0_1.setText(tr("wizard.slides.slide_0.inst1"))
        if hasattr(self, 'lbl_inst0_2'): self.lbl_inst0_2.setText(tr("wizard.slides.slide_0.inst2"))
        
        # Slide 1
        if hasattr(self, 't1_2'): self.t1_2.setText(tr("wizard.slides.slide_1.title"))
        if hasattr(self, 'd1_2_box'): self.d1_2_box.setTitle(tr("wizard.slides.slide_1.box_title"))
        if hasattr(self, 'lbl_m1'): self.lbl_m1.setText(tr("wizard.slides.slide_1.inst1"))
        if hasattr(self, 'lbl_m2'): self.lbl_m2.setText(tr("wizard.slides.slide_1.inst2"))
        if hasattr(self, 'lbl_m3'): self.lbl_m3.setText(tr("wizard.slides.slide_1.inst3"))
        
        # Slide 2
        if hasattr(self, 't1_3'): self.t1_3.setText(tr("wizard.slides.slide_2.title"))
        if hasattr(self, 'lbl_intrinsics_hint'): self.lbl_intrinsics_hint.setText(tr("wizard.slides.slide_2.skip_note"))
        if hasattr(self, 'd1_3'): self.d1_3.setText(tr("wizard.slides.slide_2.inst1"))
        if hasattr(self, 'btn_go_intrinsics'): self.btn_go_intrinsics.setText(tr("wizard.slides.slide_2.btn_go"))
        
        # Slide 3
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
        
        # Slide 4
        if hasattr(self, 't2'): self.t2.setText(tr("wizard.slides.slide_4.title"))
        if hasattr(self, 'd2'): self.d2.setText(tr("wizard.slides.slide_4.inst1"))
        if hasattr(self, 'head_desc'): self.head_desc.setText(tr("wizard.slides.slide_4.head_note"))
        if hasattr(self, 'lbl_bracket_query'): self.lbl_bracket_query.setText(tr("wizard.slides.slide_4.additional_bracket_query"))
        if hasattr(self, 'rdo_bracket_yes'): self.rdo_bracket_yes.setText(tr("wizard.slides.slide_4.yes"))
        if hasattr(self, 'rdo_bracket_no'): self.rdo_bracket_no.setText(tr("wizard.slides.slide_4.no"))
        if hasattr(self, 'conn_box'): self.conn_box.setTitle(tr("wizard.slides.slide_4.box_title"))
        
        # Slide 5
        if hasattr(self, 't3_1'): self.t3_1.setText(tr("wizard.slides.slide_5.title"))
        if hasattr(self, 'd3_1'): self.d3_1.setText(tr("wizard.slides.slide_5.inst1"))
        if hasattr(self, 'btn_move_zero_init'): self.btn_move_zero_init.setText(tr("wizard.slides.slide_5.btn_move_zero"))
        
        # Slide 6
        if hasattr(self, 't3_2'): self.t3_2.setText(tr("wizard.slides.slide_6.title"))
        if hasattr(self, 'lbl_skip_hint7'): self.lbl_skip_hint7.setText(tr("wizard.slides.slide_6.skip_hint"))
        if hasattr(self, 'btn_how_to_move'): self.btn_how_to_move.setText(tr("wizard.slides.slide_6.btn_how_to_move"))
        if hasattr(self, 'inst3_2_box'): self.inst3_2_box.setTitle(tr("wizard.slides.slide_6.box_title"))
        if hasattr(self, 'lbl_p1'): self.lbl_p1.setText(tr("wizard.slides.slide_6.inst1"))
        if hasattr(self, 'lbl_p2'): self.lbl_p2.setText(tr("wizard.slides.slide_6.inst2"))
        if hasattr(self, 'lbl_p3'): self.lbl_p3.setText(tr("wizard.slides.slide_6.inst3"))
        if hasattr(self, 'btn_step3_reset'): self.btn_step3_reset.setText(tr("wizard.slides.slide_6.btn_reset"))
        
        # Slide 7
        if hasattr(self, 't4'): self.t4.setText(tr("wizard.slides.slide_7.title"))
        if hasattr(self, 'd4_step1'): self.d4_step1.setText(tr("wizard.slides.slide_7.desc"))
        if hasattr(self, 'btn_start_unified'): self.btn_start_unified.setText(tr("wizard.btn_start_calibration"))
        if hasattr(self, 'aux_box4'): self.aux_box4.setTitle(tr("wizard.safety_title"))
        if hasattr(self, 'feed_desc'): self.feed_desc.setText(tr("wizard.slides.slide_7.feed_desc"))
        if hasattr(self, 'btn_feed4'): self.btn_feed4.setText(tr("wizard.btn_open_feed"))
        if hasattr(self, 'stop_desc'): self.stop_desc.setText(tr("wizard.slides.slide_7.stop_desc"))
        if hasattr(self, 'btn_stop4'): self.btn_stop4.setText(tr("wizard.btn_stop_motion"))
        
        # Slide 8
        if hasattr(self, 't6'): self.t6.setText(tr("wizard.slides.slide_8.title"))
        if hasattr(self, 'd6'): self.d6.setText(tr("wizard.slides.slide_8.desc"))
        if hasattr(self, 'apply_instructions_box'): self.apply_instructions_box.setTitle(tr("wizard.slides.slide_8.box_title"))
        if hasattr(self, 'lbl_apply1'): self.lbl_apply1.setText(tr("wizard.slides.slide_8.inst1"))
        if hasattr(self, 'lbl_apply2'): self.lbl_apply2.setText(tr("wizard.slides.slide_8.inst2"))
        if hasattr(self, 'lbl_apply3'): self.lbl_apply3.setText(tr("wizard.slides.slide_8.inst3"))
        if hasattr(self, 'lbl_apply4'): self.lbl_apply4.setText(tr("wizard.slides.slide_8.inst4"))
        if hasattr(self, 'btn_step6_apply'): self.btn_step6_apply.setText(tr("wizard.slides.slide_8.btn_apply"))
        
    def setup_slides(self):
        # -----------------------------------------
        # Slide 0: 1-1. Camera Mounting Check
        # -----------------------------------------
        slide0 = QWidget()
        l0 = QVBoxLayout(slide0)
        l0.setSpacing(14)
        l0.setAlignment(Qt.AlignCenter)
        
        self.t0 = QLabel(tr("wizard.slides.slide_0.title"))
        self.t0.setVisible(False)
        
        img0 = QLabel()
        pix0 = QPixmap("img/head_onoff.png")
        if not pix0.isNull():
            img0.setPixmap(pix0.scaled(700, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img0.setText("[img/head_onoff.png not found]")
        img0.setAlignment(Qt.AlignCenter)
        l0.addWidget(img0)
        
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
        self.stacked_widget.addWidget(slide0)

        # -----------------------------------------
        # Slide 1: 1-2. Marker Attachment Check
        # -----------------------------------------
        slide1_2 = QWidget()
        l1_2 = QVBoxLayout(slide1_2)
        l1_2.setSpacing(14)
        l1_2.setAlignment(Qt.AlignCenter)
        
        self.t1_2 = QLabel(tr("wizard.slides.slide_1.title"))
        self.t1_2.setVisible(False)
        
        img1_2 = QLabel()
        pix1_2 = QPixmap("img/marker_connect.png")
        if not pix1_2.isNull():
            img1_2.setPixmap(pix1_2.scaled(700, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img1_2.setText("[img/marker_connect.png not found]")
        img1_2.setAlignment(Qt.AlignCenter)
        l1_2.addWidget(img1_2)
        
        self.d1_2_box = QGroupBox(tr("wizard.slides.slide_1.box_title"))
        self.d1_2_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        self.d1_2_box.setFixedWidth(750)
        d1_2_layout = QVBoxLayout(self.d1_2_box)
        d1_2_layout.setSpacing(8)
        
        self.lbl_m1 = QLabel(tr("wizard.slides.slide_1.inst1"))
        self.lbl_m1.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_m1.setWordWrap(True)
        self.lbl_m1.setOpenExternalLinks(True)
        d1_2_layout.addWidget(self.lbl_m1)
        
        self.lbl_m2 = QLabel(tr("wizard.slides.slide_1.inst2"))
        self.lbl_m2.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_m2.setWordWrap(True)
        d1_2_layout.addWidget(self.lbl_m2)
        
        self.lbl_m3 = QLabel(tr("wizard.slides.slide_1.inst3"))
        self.lbl_m3.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        self.lbl_m3.setWordWrap(True)
        d1_2_layout.addWidget(self.lbl_m3)
        
        l1_2.addWidget(self.d1_2_box, alignment=Qt.AlignCenter)
        self.stacked_widget.addWidget(slide1_2)

        # -----------------------------------------
        # Slide 2: 1-3. Camera Intrinsics Check
        # -----------------------------------------
        slide1_3 = QWidget()
        l1_3 = QVBoxLayout(slide1_3)
        l1_3.setSpacing(14)
        l1_3.setAlignment(Qt.AlignCenter)
        
        self.t1_3 = QLabel(tr("wizard.slides.slide_2.title"))
        self.t1_3.setVisible(False)
        
        self.lbl_intrinsics_hint = QLabel(tr("wizard.slides.slide_2.skip_note"))
        self.lbl_intrinsics_hint.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 16px;")
        self.lbl_intrinsics_hint.setWordWrap(True)
        self.lbl_intrinsics_hint.setAlignment(Qt.AlignCenter)
        l1_3.addWidget(self.lbl_intrinsics_hint)
        
        img_row1_3 = QHBoxLayout()
        
        img1_3_left = QLabel()
        pix1_3_left = QPixmap("img/CHARUCOBOARD.png")
        if not pix1_3_left.isNull():
            img1_3_left.setPixmap(pix1_3_left.scaled(380, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img1_3_left.setText("[img/CHARUCOBOARD.png not found]")
        img1_3_left.setAlignment(Qt.AlignCenter)
        img_row1_3.addWidget(img1_3_left)

        img1_3_right = QLabel()
        pix1_3_right = QPixmap("img/camera_intrinsics.png")
        if not pix1_3_right.isNull():
            img1_3_right.setPixmap(pix1_3_right.scaled(380, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img1_3_right.setText("[img/camera_intrinsics.png not found]")
        img1_3_right.setAlignment(Qt.AlignCenter)
        img_row1_3.addWidget(img1_3_right)

        l1_3.addLayout(img_row1_3)
        
        self.d1_3 = QLabel(tr("wizard.slides.slide_2.inst1"))
        self.d1_3.setStyleSheet("font-size: 16px; color: #dddddd; font-weight: bold;")
        self.d1_3.setAlignment(Qt.AlignCenter)
        self.d1_3.setWordWrap(True)
        l1_3.addWidget(self.d1_3)
        
        self.btn_go_intrinsics = QPushButton(tr("wizard.slides.slide_3.title"))
        self.btn_go_intrinsics.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; font-size: 15px; padding: 10px 20px; border-radius: 6px;")
        self.btn_go_intrinsics.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        l1_3.addWidget(self.btn_go_intrinsics, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide1_3)

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
        slide1_layout.addLayout(header1)
        
        content1_layout = QHBoxLayout()
        
        int_left = QVBoxLayout()
        self.wizard_video_label = QLabel("Camera Feed Loading...")
        self.wizard_video_label.setAlignment(Qt.AlignCenter)
        self.wizard_video_label.setMinimumSize(480, 300)
        self.wizard_video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        int_left.addWidget(self.wizard_video_label, 3)
        
        self.instr_box = QGroupBox(tr("wizard.slides.slide_3.box_guidelines"))
        self.instr_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
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
        self.controls_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        controls_layout = QVBoxLayout()
        
        self.chk_int_guide = QCheckBox(tr("wizard.slides.slide_3.guide_overlay"))
        self.chk_int_guide.setChecked(True)
        self.chk_int_guide.setStyleSheet("color: #00e5ff; font-size: 15px; font-weight: bold;")
        self.chk_int_guide.stateChanged.connect(self.parent_app.on_guide_changed)
        controls_layout.addWidget(self.chk_int_guide)
        
        self.btn_int_capture = QPushButton(tr("wizard.slides.slide_3.btn_capture"))
        self.btn_int_capture.setMinimumHeight(45)
        self.btn_int_capture.setStyleSheet("background-color: #1565c0; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_capture.clicked.connect(self.step1_capture)
        controls_layout.addWidget(self.btn_int_capture)
        
        self.btn_int_calibrate = QPushButton(tr("wizard.slides.slide_3.btn_calibrate"))
        self.btn_int_calibrate.setMinimumHeight(45)
        self.btn_int_calibrate.setStyleSheet("background-color: #2e7d32; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_calibrate.clicked.connect(self.step1_run)
        controls_layout.addWidget(self.btn_int_calibrate)
        
        self.btn_int_save = QPushButton(tr("wizard.slides.slide_3.btn_save"))
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #e65100; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_save.clicked.connect(self.step1_save)
        controls_layout.addWidget(self.btn_int_save)
        
        self.btn_int_reset = QPushButton(tr("wizard.slides.slide_3.btn_reset"))
        self.btn_int_reset.setMinimumHeight(35)
        self.btn_int_reset.setStyleSheet("background-color: #37474f; color: white; font-weight: bold; font-size: 13px;")
        self.btn_int_reset.clicked.connect(self.parent_app.reset_intrinsics_captures)
        controls_layout.addWidget(self.btn_int_reset)
        
        self.controls_box.setLayout(controls_layout)
        
        int_right = QVBoxLayout()
        
        self.stats_box2 = QGroupBox(tr("wizard.slides.slide_3.box_stats"))
        self.stats_box2.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        stats_layout2 = QHBoxLayout()
        self.lbl_captured = QLabel(tr("wizard.slides.slide_3.lbl_captured") + "0 / 16")
        self.lbl_captured.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #2979ff;")
        
        self.lbl_temp = QLabel(tr("wizard.slides.slide_3.lbl_temp") + "-- °C")
        self.lbl_temp.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #ff5500;")
        
        stats_layout2.addWidget(self.lbl_captured)
        stats_layout2.addStretch()
        stats_layout2.addWidget(self.lbl_temp)
        self.stats_box2.setLayout(stats_layout2)
        
        self.lbl_step1_status = QLabel("Status: Waiting for Capture (Need 16 Frames)")
        self.lbl_step1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step1_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")
        
        int_right.addWidget(self.stats_box2)
        int_right.addWidget(self.controls_box)
        int_right.addStretch()
        int_right.addWidget(self.lbl_step1_status)
        int_right.addStretch()
        int_right.addWidget(self.lbl_step1_status)
        
        content1_layout.addLayout(int_left, 2)
        content1_layout.addLayout(int_right, 1)
        slide1_layout.addLayout(content1_layout, 1)
        self.stacked_widget.addWidget(slide1)
        
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
        
        self.lbl_step2_status = QLabel("Status: Waiting")
        self.lbl_step2_status.setAlignment(Qt.AlignCenter)
        self.lbl_step2_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l2.addWidget(self.lbl_step2_status)
        
        # Question Label (Centered, Yellow, Bold)
        self.lbl_bracket_query = QLabel(tr("wizard.slides.slide_4.additional_bracket_query"))
        self.lbl_bracket_query.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffeb3b; margin-top: 10px; margin-bottom: 5px;")
        self.lbl_bracket_query.setAlignment(Qt.AlignCenter)
        l2.addWidget(self.lbl_bracket_query)
        
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
                background-color: #ff9800;
                color: #000000;
                border: 2px solid #ff9800;
            }
            QRadioButton:hover {
                border: 2px solid #ff9800;
            }
        """
        
        # Left Column: Yes (Additional Bracket)
        col_yes = QVBoxLayout()
        col_yes.setAlignment(Qt.AlignCenter)
        col_yes.setSpacing(10)
        
        img_yes = QLabel()
        pix_yes = QPixmap("img/additional_head_bracket.png")
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
        pix_no = QPixmap("img/standard_head_bracket.png")
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
        l2.addLayout(bracket_layout)
        
        # Group them
        self.bracket_btn_group = QButtonGroup(self)
        self.bracket_btn_group.addButton(self.rdo_bracket_yes)
        self.bracket_btn_group.addButton(self.rdo_bracket_no)
        
        self.rdo_bracket_no.setChecked(True)
        self.rdo_bracket_yes.toggled.connect(self.on_bracket_radio_changed)
        self.rdo_bracket_no.toggled.connect(self.on_bracket_radio_changed)
        
        self.conn_box = QGroupBox(tr("wizard.slides.slide_4.box_title"))
        self.conn_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        self.conn_box.setFixedWidth(600)
        conn_layout = QVBoxLayout()
        conn_layout.setSpacing(10)
        
        ip_row = QHBoxLayout()
        lbl_ip = QLabel("IP/Port:")
        lbl_ip.setStyleSheet("font-size: 15px; font-weight: bold;")
        ip_row.addWidget(lbl_ip)
        
        self.wizard_ip_input = QLineEdit("192.168.30.1:50051")
        if self.parent_app.ui_only:
            self.wizard_ip_input.setText("127.0.0.1:50051")
        self.wizard_ip_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px; font-size: 15px;")
        ip_row.addWidget(self.wizard_ip_input)
        conn_layout.addLayout(ip_row)
        
        connect_row = QHBoxLayout()
        self.btn_wizard_connect = QPushButton("CONNECT")
        self.btn_wizard_connect.setMinimumWidth(160)
        self.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
        self.btn_wizard_connect.clicked.connect(self.step2_connect)
        connect_row.addWidget(self.btn_wizard_connect)
        
        self.wizard_chk_head = QCheckBox("Head")
        self.wizard_chk_head.setChecked(True)
        self.wizard_chk_head.setVisible(False)
        self.wizard_chk_head.toggled.connect(self.sync_bracket_radio)
        connect_row.addWidget(self.wizard_chk_head)
        conn_layout.addLayout(connect_row)
        
        self.conn_box.setLayout(conn_layout)
        l2.addWidget(self.conn_box, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide2)

        # -----------------------------------------
        # Slide 5: 3-1. Initial Zero Position
        # -----------------------------------------
        slide3_1 = QWidget()
        l3_1 = QVBoxLayout(slide3_1)
        l3_1.setSpacing(14)
        l3_1.setAlignment(Qt.AlignCenter)
        
        self.t3_1 = QLabel(tr("wizard.slides.slide_5.title"))
        self.t3_1.setVisible(False)
        
        self.d3_1 = QLabel(tr("wizard.slides.slide_5.inst1"))
        self.d3_1.setStyleSheet("font-size: 16px; color: #dddddd; font-weight: bold;")
        self.d3_1.setWordWrap(True)
        self.d3_1.setAlignment(Qt.AlignCenter)
        l3_1.addWidget(self.d3_1)
        
        self.lbl_step3_1_status = QLabel("Status: Waiting for Zero Position Move")
        self.lbl_step3_1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step3_1_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l3_1.addWidget(self.lbl_step3_1_status)
        
        self.btn_move_zero_init = QPushButton("Move to Zero Position")
        self.btn_move_zero_init.setMinimumWidth(260)
        self.btn_move_zero_init.setMinimumHeight(45)
        self.btn_move_zero_init.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 15px;")
        self.btn_move_zero_init.clicked.connect(self.step3_1_move_zero)
        l3_1.addWidget(self.btn_move_zero_init, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide3_1)

        # -----------------------------------------
        # Slide 6: 3-2. Home Offset Position Setup
        # -----------------------------------------
        slide3_2 = QWidget()
        l3_2 = QVBoxLayout(slide3_2)
        l3_2.setSpacing(10)
        
        self.t3_2 = QLabel(tr("wizard.slides.slide_6.title"))
        self.t3_2.setVisible(False)
        
        self.lbl_skip_hint7 = QLabel(tr("wizard.slides.slide_6.skip_hint"))
        self.lbl_skip_hint7.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 20px;")
        self.lbl_skip_hint7.setWordWrap(True)
        self.lbl_skip_hint7.setAlignment(Qt.AlignCenter)
        l3_2.addWidget(self.lbl_skip_hint7)
        
        self.lbl_step7_status = QLabel("Status: Waiting")
        self.lbl_step7_status.setAlignment(Qt.AlignCenter)
        self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l3_2.addWidget(self.lbl_step7_status)
        
        row3_2 = QHBoxLayout()
        row3_2.setSpacing(15)
        
        img3_2 = QLabel()
        pix3_2 = QPixmap("img/home_offset_position.png")
        if not pix3_2.isNull():
            img3_2.setPixmap(pix3_2.scaled(550, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img3_2.setText("[img/home_offset_position.png not found]")
        img3_2.setAlignment(Qt.AlignCenter)
        row3_2.addWidget(img3_2)
        
        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        
        self.btn_how_to_move = QPushButton(tr("wizard.slides.slide_6.btn_how_to_move"))
        self.btn_how_to_move.setMinimumHeight(45)
        self.btn_how_to_move.setStyleSheet("background-color: #0288d1; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 15px;")
        self.btn_how_to_move.clicked.connect(self.show_how_to_move_arms_dialog)
        right_col.addWidget(self.btn_how_to_move, alignment=Qt.AlignRight)
        
        self.inst3_2_box = QGroupBox(tr("wizard.slides.slide_6.box_title"))
        self.inst3_2_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        inst3_2_layout = QVBoxLayout(self.inst3_2_box)
        inst3_2_layout.setSpacing(10)
        
        self.lbl_p1 = QLabel(tr("wizard.slides.slide_6.inst1"))
        self.lbl_p1.setStyleSheet("font-size: 15px; color: #ffffff; font-weight: bold;")
        self.lbl_p1.setWordWrap(True)
        inst3_2_layout.addWidget(self.lbl_p1)

        self.lbl_p2 = QLabel(tr("wizard.slides.slide_6.inst2"))
        self.lbl_p2.setStyleSheet("font-size: 15px; color: #ffffff; font-weight: bold;")
        self.lbl_p2.setWordWrap(True)
        inst3_2_layout.addWidget(self.lbl_p2)

        self.lbl_p3 = QLabel(tr("wizard.slides.slide_6.inst3"))
        self.lbl_p3.setStyleSheet("font-size: 15px; color: #ffeb3b; font-weight: bold;")
        self.lbl_p3.setWordWrap(True)
        inst3_2_layout.addWidget(self.lbl_p3)
            
        right_col.addWidget(self.inst3_2_box)
        row3_2.addLayout(right_col)
        l3_2.addLayout(row3_2)
        
        self.btn_step3_reset = QPushButton(tr("wizard.slides.slide_6.btn_reset"))
        self.btn_step3_reset.setMinimumWidth(260)
        self.btn_step3_reset.setMinimumHeight(45)
        self.btn_step3_reset.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 15px;")
        self.btn_step3_reset.clicked.connect(self.step3_reset)
        l3_2.addWidget(self.btn_step3_reset, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide3_2)
        
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
        
        self.lbl_step4_status = QLabel("Status: Waiting")
        self.lbl_step4_status.setAlignment(Qt.AlignCenter)
        self.lbl_step4_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l4.addWidget(self.lbl_step4_status)
        
        action_row4 = QHBoxLayout()
        self.btn_start_unified = QPushButton(tr("wizard.btn_start_calibration"))
        self.btn_start_unified.setMinimumHeight(50)
        self.btn_start_unified.setMinimumWidth(260)
        self.btn_start_unified.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 18px; border-radius: 6px; padding: 0 15px;")
        self.btn_start_unified.clicked.connect(self.start_unified_calibration)
        
        action_row4.addStretch()
        action_row4.addWidget(self.btn_start_unified)
        action_row4.addStretch()
        l4.addLayout(action_row4)
        
        self.aux_box4 = QGroupBox(tr("wizard.safety_title"))
        self.aux_box4.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 15px;}")
        self.aux_box4.setFixedWidth(640)
        aux_layout4 = QVBoxLayout()
        aux_layout4.setSpacing(10)
        
        feed_row = QHBoxLayout()
        feed_row.addStretch()
        self.feed_desc = QLabel(tr("wizard.slides.slide_7.feed_desc"))
        self.feed_desc.setStyleSheet("font-size: 14px; color: #ffeb3b; font-weight: bold;")
        feed_row.addWidget(self.feed_desc)
        self.btn_feed4 = QPushButton(tr("wizard.btn_open_feed"))
        self.btn_feed4.setMinimumWidth(160)
        self.btn_feed4.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
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
        self.btn_stop4.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        self.btn_stop4.clicked.connect(self.stop_unified_calibration)
        stop_row.addWidget(self.btn_stop4)
        aux_layout4.addLayout(stop_row)
        
        self.aux_box4.setLayout(aux_layout4)
        l4.addWidget(self.aux_box4, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide4)
        
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
        
        self.lbl_step6_status = QLabel("Status: Waiting")
        self.lbl_step6_status.setAlignment(Qt.AlignCenter)
        self.lbl_step6_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l6.addWidget(self.lbl_step6_status)
        
        apply_row = QHBoxLayout()
        apply_row.setSpacing(15)
        
        img_apply = QLabel()
        pix_apply = QPixmap("img/apply_offset.png")
        if not pix_apply.isNull():
            img_apply.setPixmap(pix_apply.scaled(520, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_apply.setText("[img/apply_offset.png not found]")
        img_apply.setAlignment(Qt.AlignCenter)
        apply_row.addWidget(img_apply)
        
        self.apply_instructions_box = QGroupBox(tr("wizard.slides.slide_8.box_title"))
        self.apply_instructions_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
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
        
        self.btn_step6_apply = QPushButton(tr("wizard.slides.slide_8.btn_apply"))
        self.btn_step6_apply.setMinimumWidth(260)
        self.btn_step6_apply.setMinimumHeight(45)
        self.btn_step6_apply.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; font-size: 16px; border-radius: 6px; padding: 0 15px;")
        self.btn_step6_apply.clicked.connect(self.parent_app.apply_home_offset)
        l6.addWidget(self.btn_step6_apply, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide6)

    def show_how_to_move_arms_dialog(self):
        dlg = HowToMoveArmsDialog(self, is_ko=False)
        dlg.exec()

    def mark_step_completed(self, step_idx, success=True, msg=""):
        if step_idx < len(self.step_completed):
            self.step_completed[step_idx] = success
        if (step_idx == 7 or step_idx == 6) and len(self.step_completed) > 6:
            self.step_completed[6] = success
        self.update_navigation(self.stacked_widget.currentIndex())
        
        # Map step index to status label
        lbl_name = None
        if step_idx == 3:
            lbl_name = "lbl_step1_status"
        elif step_idx == 4:
            lbl_name = "lbl_step2_status"
        elif step_idx == 5:
            lbl_name = "lbl_step3_1_status"
        elif step_idx == 7:
            lbl_name = "lbl_step4_status"
        elif step_idx == 8:
            lbl_name = "lbl_step6_status"

        if lbl_name:
            lbl = getattr(self, lbl_name, None)
            if lbl:
                if success:
                    lbl.setText(f"Status: SUCCESS - {msg}" if msg else "Status: SUCCESS")
                    lbl.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                else:
                    lbl.setText(f"Status: ERROR - {msg}")
                    lbl.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def step3_1_move_zero(self):
        if self.parent_app.move_to_zero_pose():
            self.lbl_step3_1_status.setText("Status: Moving to Zero Position...")
            self.lbl_step3_1_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
            self.set_wizard_busy(True)
        else:
            if not self.parent_app.robot:
                self.mark_step_completed(5, False, "Robot Not Connected")

    def go_prev(self):
        idx = self.stacked_widget.currentIndex()
        if idx == 4:
            self.stacked_widget.setCurrentIndex(2)
        elif idx == 3:
            self.stacked_widget.setCurrentIndex(2)
        elif idx > 0:
            self.stacked_widget.setCurrentIndex(idx - 1)
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
        if idx == 2:
            self.stacked_widget.setCurrentIndex(4)
        elif idx < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(idx + 1)
            if self.sender() == self.btn_skip:
                self.step_completed[idx] = True
                self.update_navigation(idx + 1)
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
        if hasattr(self, "parent_app") and hasattr(self.parent_app, "on_left_tab_changed"):
            self.parent_app.on_left_tab_changed(self.parent_app.left_tabs.currentIndex())
        
        # Update shared top title dynamically to prevent title layout shifts
        title_keys = [
            "wizard.slides.slide_0.title",
            "wizard.slides.slide_1.title",
            "wizard.slides.slide_2.title",
            "wizard.slides.slide_3.title",
            "wizard.slides.slide_4.title",
            "wizard.slides.slide_5.title",
            "wizard.slides.slide_6.title",
            "wizard.slides.slide_7.title",
            "wizard.slides.slide_8.title",
        ]
        if hasattr(self, 'lbl_wizard_title') and idx < len(title_keys):
            self.lbl_wizard_title.setText(tr(title_keys[idx]))
            
        self.btn_prev.setVisible(True)
        self.btn_prev.setText(tr("wizard.btn_prev"))
            
        show_skip = (idx == 3 or idx == 6)
        self.btn_skip.setVisible(show_skip)
        self.btn_skip.setText(tr("wizard.btn_skip"))
        
        if hasattr(self, 'lbl_skip_hint1'):
            self.lbl_skip_hint1.setVisible(idx == 3)
        if hasattr(self, 'lbl_skip_hint7'):
            self.lbl_skip_hint7.setVisible(idx == 6)
        
        enabled = self.step_completed[idx]
        self.btn_next.setEnabled(enabled)
        
        if enabled:
            self.btn_next.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setStyleSheet("background-color: #444444; color: #888888; font-weight: bold; font-size: 15px; border-radius: 6px;")
        
        if idx == self.stacked_widget.count() - 1:
            self.btn_next.setText(tr("wizard.btn_finish"))
            self.btn_next.setEnabled(True)
            self.btn_next.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setText(tr("wizard.btn_next"))

    # Step 1: Intrinsics
    def step1_capture(self):
        self.parent_app.capture_intrinsics_frame()
        frames = len(self.parent_app.captured_images)
        self.lbl_captured.setText(f"Captured Frames: {frames} / 16")
        if frames >= 16:
            self.lbl_step1_status.setText(f"Status: Captured {frames} / 16 frames. Ready to calibrate.")
            self.lbl_step1_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_step1_status.setText(f"Status: Captured {frames} / 16 frames (Need 16)")
            self.lbl_step1_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")

    def step1_run(self):
        if len(self.parent_app.captured_images) < 16:
            self.lbl_step1_status.setText("Status: Need all 16 frames to run calibration!")
            self.lbl_step1_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
            QMessageBox.warning(self, "Insufficient Data", f"Cannot run calibration: Only {len(self.parent_app.captured_images)} / 16 frames collected.\nPlease capture all 16 frames first.")
            return

        self.parent_app.run_intrinsics_calibration()
        err = self.parent_app.intrinsics_calibrator.rms_error
        if err is not None and err > 0.0:
            self.lbl_step1_status.setText(f"Status: Calibration OK (RMS: {err:.4f})")
            self.lbl_step1_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_step1_status.setText("Status: Calibration Failed (Check board settings)")
            self.lbl_step1_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def step1_save(self):
        if len(self.parent_app.captured_images) < 16:
            QMessageBox.warning(self, "Insufficient Data", f"Cannot save parameters: Only {len(self.parent_app.captured_images)} / 16 frames collected.")
            self.mark_step_completed(3, False, "Need 16 frames to save")
            return

        if self.parent_app.intrinsics_calibrator.cameraMatrix is not None and float(self.parent_app.intrinsics_calibrator.rms_error) > 0.0:
            self.parent_app.save_intrinsics_calibration()
            self.mark_step_completed(3, True, "Parameters Saved")
        else:
            QMessageBox.warning(self, "Invalid Calibration", "Calibration must be successfully executed before saving parameters.")
            self.mark_step_completed(3, False, "Calibration not run yet")

    # Step 2: Robot Connection
    def step2_connect(self):
        self.btn_wizard_connect.setText("CONNECTING...")
        self.btn_wizard_connect.setStyleSheet("background-color: #ffb74d; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
        self.btn_wizard_connect.setEnabled(False)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self.parent_app.ip_input.setText(self.wizard_ip_input.text())
        self.parent_app.chk_servo_head.setChecked(self.wizard_chk_head.isChecked())
        
        self.parent_app.connect_robot()
        
        self.btn_wizard_connect.setEnabled(True)
        if self.parent_app.robot is not None:
            self.btn_wizard_connect.setText("CONNECTED")
            self.btn_wizard_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px;")
            self.mark_step_completed(4, True, "Connected to Robot")
        else:
            self.btn_wizard_connect.setText("CONNECT")
            self.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
            self.mark_step_completed(4, False, "Connection Failed")

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
            "Confirm Home Offset Reset",
            "Are you sure you want to proceed?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            self.lbl_step7_status.setText("Status: Reset cancelled")
            self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")
            return

        if self.parent_app.home_offset_reset(confirm_dialog=False):
            self.lbl_step7_status.setText("Status: Reset in progress...")
            self.lbl_step7_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
            self.set_wizard_busy(True)
        else:
            if not self.parent_app.robot:
                self.mark_step_completed(6, False, "Robot Not Connected")
            else:
                self.lbl_step7_status.setText("Status: Reset cancelled")
                self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")

    def set_wizard_busy(self, busy):
        self.btn_prev.setEnabled(not busy)
        self.btn_skip.setEnabled(not busy)
        if busy:
            self.btn_next.setEnabled(False)
            self.btn_next.setStyleSheet("background-color: #444444; color: #888888; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.update_navigation(self.stacked_widget.currentIndex())
        if hasattr(self, 'btn_step3_reset'):
            self.btn_step3_reset.setEnabled(not busy)

    # -----------------------------------------
    # Unified Step 1 & Step 2 Calibration Execution
    # -----------------------------------------
    def start_unified_calibration(self):
        self.unified_elapsed = 0
        self.lbl_step4_status.setText("Status: [Step 1/2] Full Auto In Progress (00:00)")
        self.lbl_step4_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(False)
            self.btn_start_unified.setStyleSheet("background-color: #555555; color: #888888; font-weight: bold; font-size: 18px; border-radius: 6px;")
        
        self.unified_timer.start(1000)
        self.parent_app.start_full_auto()
        if hasattr(self.parent_app, 'active_worker') and self.parent_app.active_worker:
            self.parent_app.active_worker.finished_signal.connect(self.on_unified_step1_finished)
        else:
            self.stop_unified_calibration_error("Worker not started")

    def update_unified_time(self):
        self.unified_elapsed += 1
        m = self.unified_elapsed // 60
        s = self.unified_elapsed % 60
        curr_text = self.lbl_step4_status.text()
        if "Step 1/2" in curr_text:
            self.lbl_step4_status.setText(f"Status: [Step 1/2] Full Auto In Progress ({m:02d}:{s:02d})")
        elif "Moving to Init Pose" in curr_text:
            self.lbl_step4_status.setText(f"Status: [Step 2/2] Moving to Init Pose ({m:02d}:{s:02d})")
        elif "Auto Motion In Progress" in curr_text:
            self.lbl_step4_status.setText(f"Status: [Step 2/2] Auto Motion In Progress ({m:02d}:{s:02d})")
        elif "Optimization" in curr_text:
            self.lbl_step4_status.setText(f"Status: [Step 2/2] Hand-Eye Optimization Calculation ({m:02d}:{s:02d})")

    def on_unified_step1_finished(self):
        was_stopped = False
        if hasattr(self.parent_app, "full_auto_stop_event") and self.parent_app.full_auto_stop_event is not None:
            was_stopped = self.parent_app.full_auto_stop_event.is_set()
        error_msg = getattr(self.parent_app.active_worker, "error_msg", None) if hasattr(self.parent_app, "active_worker") and self.parent_app.active_worker else None

        if not was_stopped and not error_msg:
            m = self.unified_elapsed // 60
            s = self.unified_elapsed % 60
            self.lbl_step4_status.setText(f"Status: [Step 2/2] Moving to Init Pose ({m:02d}:{s:02d})")
            
            # Automatically apply Step 1 joint offsets & marker brackets (silent)
            self.parent_app.apply_full_auto_results(silent=True)
            
            # Start Step 2 Init Pose
            self.parent_app.step2_init_pose(silent=True)
            if hasattr(self.parent_app, 'auto_motion_thread') and self.parent_app.auto_motion_thread:
                self.parent_app.auto_motion_thread.finished_signal.connect(self.on_unified_init_finished)
            else:
                self.stop_unified_calibration_error("Step 2 Init worker not started")
        else:
            self.stop_unified_calibration_error(error_msg or "Cancelled by User")

    def on_unified_init_finished(self, success, err):
        if success:
            m = self.unified_elapsed // 60
            s = self.unified_elapsed % 60
            self.lbl_step4_status.setText(f"Status: [Step 2/2] Auto Motion In Progress ({m:02d}:{s:02d})")
            
            # Ensure previous worker status is reset before launching step 2 auto motion
            self.parent_app.auto_motion_running = False
            self.parent_app.auto_motion_thread = None
            
            # Start Step 2 Auto Motion
            self.parent_app.step2_auto_motion()
            if hasattr(self.parent_app, 'auto_motion_thread') and self.parent_app.auto_motion_thread:
                self.parent_app.auto_motion_thread.finished_signal.connect(self.on_unified_auto_motion_finished)
            else:
                self.stop_unified_calibration_error("Step 2 Auto Motion worker not started")
        else:
            self.stop_unified_calibration_error(err)

    def on_unified_auto_motion_finished(self, success, err_msg=""):
        if success:
            m = self.unified_elapsed // 60
            s = self.unified_elapsed % 60
            self.lbl_step4_status.setText(f"Status: [Step 2/2] Hand-Eye Optimization Calculation ({m:02d}:{s:02d})")
        else:
            self.stop_unified_calibration_error(err_msg)

    def stop_step5(self, success=True, err_msg=""):
        self.unified_timer.stop()
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(True)
            self.btn_start_unified.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 18px; border-radius: 6px;")
            
        m = self.unified_elapsed // 60
        s = self.unified_elapsed % 60
        time_str = f"{m:02d}:{s:02d}"
        if success:
            self.mark_step_completed(7, True, f"Calibration Pipeline Complete! Total Time: {time_str}")
        else:
            self.mark_step_completed(7, False, err_msg)

    def stop_unified_calibration(self):
        self.parent_app.stop_full_auto()
        self.parent_app.request_stop_all_auto_motion()
        self.stop_unified_calibration_error("Cancelled by User")

    def stop_unified_calibration_error(self, err_msg=""):
        self.unified_timer.stop()
        if hasattr(self, 'btn_start_unified'):
            self.btn_start_unified.setEnabled(True)
            self.btn_start_unified.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 18px; border-radius: 6px;")
        self.mark_step_completed(7, False, err_msg)
