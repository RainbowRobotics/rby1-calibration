from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QGroupBox, QCheckBox, QLineEdit, QComboBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap

class CalibrationWizardWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget, stretch=1)
        
        # Navigation Layout
        self.nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_skip = QPushButton("Skip")
        self.btn_next = QPushButton("Next")
        
        # Apply standard styling
        self.nav_style = "font-weight: bold; font-size: 14px; padding: 6px 12px; border-radius: 4px;"
        self.btn_prev.setStyleSheet("background-color: #555; color: white; " + self.nav_style)
        
        # Make Skip button much larger and more prominent
        self.btn_skip.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 18px; padding: 10px 24px; border-radius: 6px;")
        
        self.btn_prev.clicked.connect(self.go_prev)
        self.btn_skip.clicked.connect(self.go_next)
        self.btn_next.clicked.connect(self.go_next)
        
        self.lbl_skip_hint = QLabel("If you have already completed this step, please click the skip button.")
        self.lbl_skip_hint.setStyleSheet("color: red; font-weight: bold; font-size: 18px;")
        
        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.lbl_skip_hint)
        self.nav_layout.addWidget(self.btn_skip)
        self.nav_layout.addWidget(self.btn_next)
        
        self.layout.addLayout(self.nav_layout)
        
        # State tracking for each step to enable Next
        self.step_completed = [False] * 6
        self.step_completed[0] = False # Must complete camera intrinsics (or skip)
        self.step_completed[2] = False # Must complete home offset reset (or skip)
        self.step_completed[5] = True  # Last step can just finish
        
        # Timers
        self.step4_timer = QTimer(self)
        self.step4_timer.timeout.connect(self.update_step4_time)
        self.step4_elapsed = 0
        
        self.step5_timer = QTimer(self)
        self.step5_timer.timeout.connect(self.update_step5_time)
        self.step5_elapsed = 0
        
        self.setup_slides()
        self.stacked_widget.currentChanged.connect(self.update_navigation)
        self.update_navigation(0)
        
    def setup_slides(self):
        # -----------------------------------------
        # Slide 1: Camera Intrinsics (Replicated layout)
        # -----------------------------------------
        slide1 = QWidget()
        slide1_layout = QHBoxLayout(slide1)
        
        int_left = QVBoxLayout()
        self.wizard_video_label = QLabel("Camera Feed Loading...")
        self.wizard_video_label.setAlignment(Qt.AlignCenter)
        self.wizard_video_label.setMinimumSize(640, 480)
        self.wizard_video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        int_left.addWidget(self.wizard_video_label, 3)
        
        instr_box = QGroupBox("Calibration Guidelines")
        instr_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 14px;}")
        instr_layout = QVBoxLayout()
        instructions = [
            "1. Ensure the calibration board is recognized correctly (green overlay).",
            "2. Tilt the board at various angles while capturing.",
            "3. Acquire data covering the entire camera field of view.",
            "4. Keep the board as steady as possible during each capture."
        ]
        for text in instructions:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #ddd; font-weight: bold;")
            instr_layout.addWidget(lbl)
        instr_box.setLayout(instr_layout)
        int_left.addWidget(instr_box, 1)
        
        controls_box = QGroupBox("Calibration Controls")
        controls_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 14px;}")
        controls_layout = QVBoxLayout()
        
        self.chk_int_guide = QCheckBox("Show Guide Overlay")
        self.chk_int_guide.setChecked(True)
        self.chk_int_guide.setStyleSheet("color: #00e5ff; font-weight: bold;")
        self.chk_int_guide.stateChanged.connect(self.parent_app.on_guide_changed)
        controls_layout.addWidget(self.chk_int_guide)
        
        btn_int_capture = QPushButton("CAPTURE FRAME (C)")
        btn_int_capture.setMinimumHeight(45)
        btn_int_capture.setStyleSheet("background-color: #1565c0; color: white; font-size: 13px; font-weight: bold;")
        btn_int_capture.clicked.connect(self.step1_capture)
        controls_layout.addWidget(btn_int_capture)
        
        btn_int_calibrate = QPushButton("RUN CALIBRATION")
        btn_int_calibrate.setMinimumHeight(45)
        btn_int_calibrate.setStyleSheet("background-color: #2e7d32; color: white; font-size: 13px; font-weight: bold;")
        btn_int_calibrate.clicked.connect(self.step1_run)
        controls_layout.addWidget(btn_int_calibrate)
        
        self.btn_int_save = QPushButton("SAVE PARAMETERS")
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #e65100; color: white; font-size: 13px; font-weight: bold;")
        self.btn_int_save.clicked.connect(self.step1_save)
        controls_layout.addWidget(self.btn_int_save)
        
        btn_int_reset = QPushButton("RESET CAPTURES")
        btn_int_reset.setMinimumHeight(30)
        btn_int_reset.setStyleSheet("background-color: #37474f; color: white; font-weight: bold;")
        btn_int_reset.clicked.connect(self.parent_app.reset_intrinsics_captures)
        controls_layout.addWidget(btn_int_reset)
        
        controls_box.setLayout(controls_layout)
        
        int_right = QVBoxLayout()
        
        stats_box2 = QGroupBox("Capture Stats")
        stats_box2.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 14px;}")
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
        
        self.lbl_step1_status = QLabel("Status: Waiting for Capture")
        self.lbl_step1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step1_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 14px;")
        
        int_right.addWidget(stats_box2)
        int_right.addWidget(controls_box)
        int_right.addStretch()
        int_right.addWidget(self.lbl_step1_status)
        
        slide1_layout.addLayout(int_left, 2)
        slide1_layout.addLayout(int_right, 1)
        self.stacked_widget.addWidget(slide1)
        
        # Slide 2: Robot Connection
        slide2 = QWidget()
        l2 = QVBoxLayout(slide2)
        l2.setAlignment(Qt.AlignCenter)
        
        t2 = QLabel("2. Robot Connection")
        t2.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffeb3b;")
        t2.setAlignment(Qt.AlignCenter)
        l2.addWidget(t2)
        
        d2 = QLabel("Connect to the robot using the configured IP address.")
        d2.setStyleSheet("font-size: 14px;")
        d2.setWordWrap(True)
        d2.setAlignment(Qt.AlignCenter)
        l2.addWidget(d2)
        
        self.lbl_step2_status = QLabel("Status: Waiting")
        self.lbl_step2_status.setAlignment(Qt.AlignCenter)
        self.lbl_step2_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l2.addWidget(self.lbl_step2_status)
        
        # Connection Box
        conn_box = QGroupBox("Robot Connection")
        conn_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 14px;}")
        conn_box.setFixedWidth(400)
        conn_layout = QVBoxLayout()
        
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP/Port:"))
        self.wizard_ip_input = QLineEdit("192.168.30.1:50051")
        if self.parent_app.ui_only:
            self.wizard_ip_input.setText("127.0.0.1:50051")
        self.wizard_ip_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 4px;")
        ip_row.addWidget(self.wizard_ip_input)
        conn_layout.addLayout(ip_row)
        
        connect_row = QHBoxLayout()
        self.btn_wizard_connect = QPushButton("CONNECT")
        self.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 6px; font-size: 14px;")
        self.btn_wizard_connect.clicked.connect(self.step2_connect)
        connect_row.addWidget(self.btn_wizard_connect)
        
        self.wizard_chk_head = QCheckBox("Head")
        self.wizard_chk_head.setChecked(True)
        self.wizard_chk_head.setStyleSheet("color: #cccccc;")
        connect_row.addWidget(self.wizard_chk_head)
        conn_layout.addLayout(connect_row)
        
        conn_box.setLayout(conn_layout)
        l2.addWidget(conn_box, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide2)
        
        # Slide 3: Home Offset Reset
        slide3, self.lbl_step3_status = self.create_slide_with_status(
            "3. Home Offset Reset", 
            "Reset the current home offset to baseline before starting calibration.",
            "Reset Home Offset", self.step3_reset, "#c62828"
        )
        self.stacked_widget.addWidget(slide3)
        
        # Slide 4: Start Full Auto
        slide4 = QWidget()
        l4 = QVBoxLayout(slide4)
        l4.setAlignment(Qt.AlignCenter)
        
        t4 = QLabel("4. Full Auto Calibration (Step 1)")
        t4.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffeb3b;")
        t4.setAlignment(Qt.AlignCenter)
        l4.addWidget(t4)
        
        d4 = QLabel("Run the automated sequence for Step 1 (Marker Offset & Joint 3,5,6 Calibration).")
        d4.setStyleSheet("font-size: 14px;")
        d4.setWordWrap(True)
        d4.setAlignment(Qt.AlignCenter)
        l4.addWidget(d4)
        
        self.lbl_step4_status = QLabel("Status: Waiting")
        self.lbl_step4_status.setAlignment(Qt.AlignCenter)
        self.lbl_step4_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l4.addWidget(self.lbl_step4_status)
        
        btn_layout4 = QHBoxLayout()
        btn_feed4 = QPushButton("Open Camera Feed")
        btn_feed4.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; font-size: 14px; padding: 6px;")
        btn_feed4.clicked.connect(self.parent_app.toggle_camera_feed_dialog)
        
        btn_start_full = QPushButton("Start Full Auto")
        btn_start_full.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 14px; padding: 6px;")
        btn_start_full.clicked.connect(self.start_step4)
        
        btn_stop4 = QPushButton("Stop Motion")
        btn_stop4.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 14px; padding: 6px;")
        btn_stop4.clicked.connect(self.parent_app.stop_full_auto)
        
        btn_layout4.addStretch()
        btn_layout4.addWidget(btn_feed4)
        btn_layout4.addWidget(btn_start_full)
        btn_layout4.addWidget(btn_stop4)
        btn_layout4.addStretch()
        l4.addLayout(btn_layout4)
        self.stacked_widget.addWidget(slide4)
        
        # Slide 5: Init Pose & Auto Motion
        slide5 = QWidget()
        l5 = QVBoxLayout(slide5)
        l5.setAlignment(Qt.AlignCenter)
        
        t5 = QLabel("5. Init Pose & Auto Motion (Step 2)")
        t5.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffeb3b;")
        t5.setAlignment(Qt.AlignCenter)
        l5.addWidget(t5)
        
        d5 = QLabel("Run Step 2 (Joint Offset Calibration).\nFirst move to Init Pose, then execute Auto Motion to gather dataset.")
        d5.setStyleSheet("font-size: 14px;")
        d5.setWordWrap(True)
        d5.setAlignment(Qt.AlignCenter)
        l5.addWidget(d5)
        
        self.lbl_step5_status = QLabel("Status: Waiting")
        self.lbl_step5_status.setAlignment(Qt.AlignCenter)
        self.lbl_step5_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l5.addWidget(self.lbl_step5_status)
        
        btn_layout5 = QHBoxLayout()
        btn_feed5 = QPushButton("Open Camera Feed")
        btn_feed5.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; font-size: 14px; padding: 6px;")
        btn_feed5.clicked.connect(self.parent_app.toggle_camera_feed_dialog)
        
        btn_init = QPushButton("Move Init Pose")
        btn_init.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; font-size: 14px; padding: 6px;")
        btn_init.clicked.connect(self.start_step5_init)
        
        btn_auto = QPushButton("Start Auto Motion")
        btn_auto.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 14px; padding: 6px;")
        btn_auto.clicked.connect(self.start_step5_auto)
        
        btn_stop5 = QPushButton("Stop Motion")
        btn_stop5.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 14px; padding: 6px;")
        btn_stop5.clicked.connect(self.parent_app.request_stop_all_auto_motion)
        
        btn_layout5.addStretch()
        btn_layout5.addWidget(btn_feed5)
        btn_layout5.addWidget(btn_init)
        btn_layout5.addWidget(btn_auto)
        btn_layout5.addWidget(btn_stop5)
        btn_layout5.addStretch()
        l5.addLayout(btn_layout5)
        
        self.stacked_widget.addWidget(slide5)
        
        # Slide 6: Apply Home Offset
        slide6, self.lbl_step6_status = self.create_slide_with_status(
            "6. Apply Home Offset", 
            "Review and apply the calculated optimized home offset.",
            "Apply Home Offset", self.parent_app.apply_home_offset, "#1976d2"
        )
        self.stacked_widget.addWidget(slide6)
        
    def create_slide_with_status(self, title, desc, btn_text, btn_callback, btn_color):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setAlignment(Qt.AlignCenter)
        
        t = QLabel(title)
        t.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffeb3b;")
        t.setAlignment(Qt.AlignCenter)
        l.addWidget(t)
        
        d = QLabel(desc)
        d.setStyleSheet("font-size: 14px;")
        d.setWordWrap(True)
        d.setAlignment(Qt.AlignCenter)
        l.addWidget(d)
        
        lbl_status = QLabel("Status: Waiting")
        lbl_status.setAlignment(Qt.AlignCenter)
        lbl_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l.addWidget(lbl_status)
        
        b = QPushButton(btn_text)
        b.setFixedWidth(200)
        b.setStyleSheet(f"background-color: {btn_color}; color: white; font-weight: bold; font-size: 14px; padding: 6px;")
        b.clicked.connect(btn_callback)
        l.addWidget(b, alignment=Qt.AlignCenter)
        
        return w, lbl_status

    def mark_step_completed(self, step_idx, success=True, msg=""):
        self.step_completed[step_idx] = success
        self.update_navigation(self.stacked_widget.currentIndex())
        lbl = getattr(self, f"lbl_step{step_idx+1}_status", None)
        if lbl:
            if success:
                lbl.setText(f"Status: SUCCESS - {msg}" if msg else "Status: SUCCESS")
                lbl.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
            else:
                lbl.setText(f"Status: ERROR - {msg}")
                lbl.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def go_prev(self):
        idx = self.stacked_widget.currentIndex()
        if idx > 0:
            self.stacked_widget.setCurrentIndex(idx - 1)
        else:
            # Go back to overview
            self.parent_app.btn_start_wizard.setVisible(True)
            self.parent_app.overview_img.setVisible(True)
            self.setVisible(False)
            
    def go_next(self):
        idx = self.stacked_widget.currentIndex()
        if idx < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(idx + 1)
            # If skipping, mark as completed just in case
            if self.sender() == self.btn_skip:
                self.step_completed[idx] = True
                self.update_navigation(idx + 1)
        else:
            self.parent_app.log_msg("Calibration Wizard Finished.")
            self.parent_app.btn_start_wizard.setVisible(True)
            self.parent_app.overview_img.setVisible(True)
            self.setVisible(False)
            self.stacked_widget.setCurrentIndex(0) # reset
            
    def update_navigation(self, idx):
        if hasattr(self, "parent_app") and hasattr(self.parent_app, "on_left_tab_changed"):
            self.parent_app.on_left_tab_changed(self.parent_app.left_tabs.currentIndex())
        self.btn_prev.setVisible(True)
        if idx == 0:
            self.btn_prev.setText("Back to Overview")
        else:
            self.btn_prev.setText("Previous")
        # Skip allowed on Slide 1 and Slide 3
        show_skip = (idx == 0 or idx == 2)
        self.btn_skip.setVisible(show_skip)
        self.lbl_skip_hint.setVisible(show_skip)
        
        enabled = self.step_completed[idx]
        self.btn_next.setEnabled(enabled)
        
        if enabled:
            self.btn_next.setStyleSheet("background-color: #1976d2; color: white; " + self.nav_style)
        else:
            self.btn_next.setStyleSheet("background-color: #444444; color: #888888; " + self.nav_style)
        
        if idx == self.stacked_widget.count() - 1:
            self.btn_next.setText("Finish")
            self.btn_next.setEnabled(True)
            self.btn_next.setStyleSheet("background-color: #1976d2; color: white; " + self.nav_style)
        else:
            self.btn_next.setText("Next")

    # Step 1: Intrinsics
    def step1_capture(self):
        self.parent_app.capture_intrinsics_frame()
        frames = len(self.parent_app.captured_images)
        self.lbl_captured.setText(f"Captured Frames: {frames}")
        self.lbl_step1_status.setText(f"Status: Captured {frames} frames")
        self.lbl_step1_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")

    def step1_run(self):
        self.parent_app.run_intrinsics_calibration()
        err = self.parent_app.intrinsics_calibrator.rms_error
        if err is not None:
            self.lbl_step1_status.setText(f"Status: Calibration OK (RMS: {err:.4f})")
            self.lbl_step1_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_step1_status.setText("Status: Calibration Failed (Need 6+ frames)")
            self.lbl_step1_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def step1_save(self):
        self.parent_app.save_intrinsics_calibration()
        if self.parent_app.intrinsics_calibrator.cameraMatrix is not None:
            self.mark_step_completed(0, True, "Parameters Saved")
        else:
            self.mark_step_completed(0, False, "Calibration not run yet")

    # Step 2: Robot Connection
    def step2_connect(self):
        # Update connection button to loading state
        self.btn_wizard_connect.setText("CONNECTING...")
        self.btn_wizard_connect.setStyleSheet("background-color: #ffb74d; color: #000000; font-weight: bold; padding: 6px; font-size: 14px;")
        self.btn_wizard_connect.setEnabled(False)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        # Sync to main UI
        self.parent_app.ip_input.setText(self.wizard_ip_input.text())
        self.parent_app.chk_servo_head.setChecked(self.wizard_chk_head.isChecked())
        
        self.parent_app.connect_robot()
        
        self.btn_wizard_connect.setEnabled(True)
        if self.parent_app.robot is not None:
            self.btn_wizard_connect.setText("CONNECTED")
            self.btn_wizard_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 6px; font-size: 14px;")
            self.mark_step_completed(1, True, "Connected to Robot")
        else:
            self.btn_wizard_connect.setText("CONNECT")
            self.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 6px; font-size: 14px;")
            self.mark_step_completed(1, False, "Connection Failed")

    # Step 3: Home Offset Reset
    def step3_reset(self):
        self.parent_app.home_offset_reset()
        if self.parent_app.robot is not None:
            self.mark_step_completed(2, True, "Home Offset Reset")
        else:
            self.mark_step_completed(2, False, "Robot Not Connected")

    # Step 4: Full Auto
    def start_step4(self):
        self.step4_elapsed = 0
        self.lbl_step4_status.setText("Status: Full Auto In Progress (00:00)")
        self.lbl_step4_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
        self.step4_timer.start(1000)
        self.parent_app.start_full_auto()
        if hasattr(self.parent_app, 'active_worker') and self.parent_app.active_worker:
            self.parent_app.active_worker.finished_signal.connect(self.stop_step4)
        else:
            self.stop_step4(False, "Worker not started")

    def update_step4_time(self):
        self.step4_elapsed += 1
        m = self.step4_elapsed // 60
        s = self.step4_elapsed % 60
        self.lbl_step4_status.setText(f"Status: Full Auto In Progress ({m:02d}:{s:02d})")

    def stop_step4(self):
        self.step4_timer.stop()
        was_stopped = False
        if hasattr(self.parent_app, "full_auto_stop_event") and self.parent_app.full_auto_stop_event is not None:
            was_stopped = self.parent_app.full_auto_stop_event.is_set()
            
        error_msg = getattr(self.parent_app.active_worker, "error_msg", None) if hasattr(self.parent_app, "active_worker") and self.parent_app.active_worker else None
        
        if not was_stopped and not error_msg:
            self.mark_step_completed(3, True, f"Done ({self.step4_elapsed//60:02d}:{self.step4_elapsed%60:02d})")
        else:
            if was_stopped:
                self.mark_step_completed(3, False, "Cancelled by User")
            else:
                self.mark_step_completed(3, False, error_msg or "Unknown Error")

    # Step 5: Auto Motion
    def start_step5_init(self):
        self.parent_app.step2_init_pose()
        if hasattr(self.parent_app, 'auto_motion_thread') and self.parent_app.auto_motion_thread:
            self.parent_app.auto_motion_thread.finished_signal.connect(
                lambda success, err: self.lbl_step5_status.setText("Status: Init Pose Reached" if success else f"Status: Error - {err}")
            )
        
    def start_step5_auto(self):
        self.step5_elapsed = 0
        self.lbl_step5_status.setText("Status: Auto Motion In Progress (00:00)")
        self.lbl_step5_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
        self.step5_timer.start(1000)
        self.parent_app.step2_auto_motion()
        if hasattr(self.parent_app, 'auto_motion_thread') and self.parent_app.auto_motion_thread:
            self.parent_app.auto_motion_thread.finished_signal.connect(self.stop_step5)
        else:
            self.stop_step5(False, "Worker not started")
            
    def update_step5_time(self):
        self.step5_elapsed += 1
        m = self.step5_elapsed // 60
        s = self.step5_elapsed % 60
        self.lbl_step5_status.setText(f"Status: Auto Motion In Progress ({m:02d}:{s:02d})")
        
    def stop_step5(self, success=True, err_msg=""):
        self.step5_timer.stop()
        if success:
            self.mark_step_completed(4, True, f"Done ({self.step5_elapsed//60:02d}:{self.step5_elapsed%60:02d})")
        else:
            self.mark_step_completed(4, False, err_msg)

