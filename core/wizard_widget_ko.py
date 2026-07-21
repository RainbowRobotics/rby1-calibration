# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QGroupBox, QCheckBox, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap

class CalibrationWizardWidgetKO(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(12)
        
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget, stretch=1)
        
        # 내비게이션 레이아웃
        self.nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("이전")
        self.btn_skip = QPushButton("스킵")
        self.btn_next = QPushButton("다음")
        
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
        
        # 스텝 진행 상태 (총 11개 슬라이드)
        self.step_completed = [False] * 11
        self.step_completed[0] = True   # 1-1 카메라 마운팅
        self.step_completed[1] = True   # 1-2 마커 부착
        self.step_completed[2] = True   # 1-3 내부 파라미터 확인
        self.step_completed[3] = False  # 내부 파라미터 보정 (선택)
        self.step_completed[4] = False  # 로봇 연결
        self.step_completed[5] = False  # 3-1 초기 제로 포즈 (이동 완료 필요)
        self.step_completed[6] = True   # 3-2 직접 교시 버튼 안내
        self.step_completed[7] = False  # 3-3 홈 오프셋 설정 포즈
        self.step_completed[8] = False  # 풀 오토 캘리브레이션
        self.step_completed[9] = False  # 자동 모션
        self.step_completed[10] = True  # 홈 오프셋 적용
        
        # 타이머
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
        # Slide 0: 1-1. 카메라 마운팅 확인
        # -----------------------------------------
        slide0 = QWidget()
        l0 = QVBoxLayout(slide0)
        l0.setSpacing(14)
        l0.setAlignment(Qt.AlignCenter)
        
        t0 = QLabel("1-1. 카메라 마운팅 확인")
        t0.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t0.setAlignment(Qt.AlignCenter)
        l0.addWidget(t0)
        
        img0 = QLabel()
        pix0 = QPixmap("img/head_onoff.png")
        if not pix0.isNull():
            img0.setPixmap(pix0.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img0.setText("[img/head_onoff.png 찾을 수 없음]")
        img0.setAlignment(Qt.AlignCenter)
        l0.addWidget(img0)
        
        d0_box = QGroupBox("카메라 장착 지침 사항")
        d0_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        d0_box.setFixedWidth(750)
        d0_layout = QVBoxLayout(d0_box)
        d0_layout.setSpacing(8)
        
        inst0 = [
            "1. 카메라가 헤드 또는 정밀 구동부에 정상적으로 장착되었는지 확인하세요.",
            "2. 브라켓이 필요한 경우: 1.0 버전, 혹은 헤드가 없는 모델",
            "3. 브라켓이 없어도 되는 경우: 1.1 ~ 1.3 버전, 헤드가 정밀 구동부인 모델"
        ]
        for txt in inst0:
            lbl = QLabel(txt)
            lbl.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
            lbl.setWordWrap(True)
            d0_layout.addWidget(lbl)
            
        l0.addWidget(d0_box, alignment=Qt.AlignCenter)
        self.stacked_widget.addWidget(slide0)

        # -----------------------------------------
        # Slide 1: 1-2. 마커 부착 확인
        # -----------------------------------------
        slide1_2 = QWidget()
        l1_2 = QVBoxLayout(slide1_2)
        l1_2.setSpacing(14)
        l1_2.setAlignment(Qt.AlignCenter)
        
        t1_2 = QLabel("1-2. 마커 부착 확인")
        t1_2.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t1_2.setAlignment(Qt.AlignCenter)
        l1_2.addWidget(t1_2)
        
        img1_2 = QLabel()
        pix1_2 = QPixmap("img/marker_connect.png")
        if not pix1_2.isNull():
            img1_2.setPixmap(pix1_2.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img1_2.setText("[img/marker_connect.png 찾을 수 없음]")
        img1_2.setAlignment(Qt.AlignCenter)
        l1_2.addWidget(img1_2)
        
        d1_2_box = QGroupBox("마커 부착 지침 사항")
        d1_2_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        d1_2_box.setFixedWidth(750)
        d1_2_layout = QVBoxLayout(d1_2_box)
        d1_2_layout.setSpacing(8)
        
        lbl_m1 = QLabel("1. 그리퍼를 분해하고 캘리브레이션 마커가 툴 플랜지에 견고하게 부착되었는지 확인하세요.")
        lbl_m1.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        d1_2_layout.addWidget(lbl_m1)
        
        lbl_m2 = QLabel('2. 그리퍼 분해 관련 상세 내용은 매뉴얼을 참고하세요: <a href="https://rainbowco-my.sharepoint.com/:p:/g/personal/support_rainbow-robotics_com/IQD86a950aEVQqclH8O9vQxrAcXAZA4gEQ3921tDqwQIGH8?e=qRn473" style="color: #00e5ff; font-weight: bold;">그리퍼 분해 및 조립 가이드</a>')
        lbl_m2.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
        lbl_m2.setOpenExternalLinks(True)
        lbl_m2.setWordWrap(True)
        d1_2_layout.addWidget(lbl_m2)
        
        l1_2.addWidget(d1_2_box, alignment=Qt.AlignCenter)
        self.stacked_widget.addWidget(slide1_2)

        # -----------------------------------------
        # Slide 2: 1-3. 카메라 내부 파라미터 확인
        # -----------------------------------------
        slide1_3 = QWidget()
        l1_3 = QVBoxLayout(slide1_3)
        l1_3.setSpacing(14)
        l1_3.setAlignment(Qt.AlignCenter)
        
        t1_3 = QLabel("1-3. 카메라 내부 파라미터 확인")
        t1_3.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t1_3.setAlignment(Qt.AlignCenter)
        l1_3.addWidget(t1_3)
        
        img_row1_3 = QHBoxLayout()
        
        img1_3_left = QLabel()
        pix1_3_left = QPixmap("img/CHARUCOBOARD.png")
        if not pix1_3_left.isNull():
            img1_3_left.setPixmap(pix1_3_left.scaled(380, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img1_3_left.setText("[img/CHARUCOBOARD.png 찾을 수 없음]")
        img1_3_left.setAlignment(Qt.AlignCenter)
        img_row1_3.addWidget(img1_3_left)

        img1_3_right = QLabel()
        pix1_3_right = QPixmap("img/camera_intrinsics.png")
        if not pix1_3_right.isNull():
            img1_3_right.setPixmap(pix1_3_right.scaled(380, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img1_3_right.setText("[img/camera_intrinsics.png 찾을 수 없음]")
        img1_3_right.setAlignment(Qt.AlignCenter)
        img_row1_3.addWidget(img1_3_right)

        l1_3.addLayout(img_row1_3)
        
        d1_3 = QLabel("카메라 렌즈 왜곡은 추적 정밀도에 영향을 줄 수 있습니다. 필요에 따라 내부 파라미터를 보정하세요.\n\n이미 사용중인 카메라 내부 파라미터 가 있는 경우, 다음으로 넘어가시길 바랍니다.")
        d1_3.setStyleSheet("font-size: 16px; color: #dddddd; font-weight: bold;")
        d1_3.setAlignment(Qt.AlignCenter)
        l1_3.addWidget(d1_3)
        
        btn_go_intrinsics = QPushButton("카메라 내부 파라미터 보정하기")
        btn_go_intrinsics.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; font-size: 15px; padding: 10px 20px; border-radius: 6px;")
        btn_go_intrinsics.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        l1_3.addWidget(btn_go_intrinsics, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide1_3)

        # -----------------------------------------
        # Slide 3: 카메라 내부 파라미터 보정 (선택 사항)
        # -----------------------------------------
        slide1 = QWidget()
        slide1_layout = QVBoxLayout(slide1)
        
        header1 = QVBoxLayout()
        t1 = QLabel("카메라 내부 파라미터 보정 (선택 사항)")
        t1.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t1.setAlignment(Qt.AlignCenter)
        header1.addWidget(t1)
        
        self.lbl_skip_hint1 = QLabel("이미 진행했다면 다음(Next) 혹은 스킵(Skip) 버튼을 눌러 다음 단계를 진행하십시오.")
        self.lbl_skip_hint1.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 16px;")
        self.lbl_skip_hint1.setAlignment(Qt.AlignCenter)
        header1.addWidget(self.lbl_skip_hint1)
        slide1_layout.addLayout(header1)
        
        content1_layout = QHBoxLayout()
        
        int_left = QVBoxLayout()
        self.wizard_video_label = QLabel("카메라 피드 로딩 중...")
        self.wizard_video_label.setAlignment(Qt.AlignCenter)
        self.wizard_video_label.setMinimumSize(640, 480)
        self.wizard_video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #2d2d2d; border-radius: 8px;")
        int_left.addWidget(self.wizard_video_label, 3)
        
        instr_box = QGroupBox("보정 지침 사항")
        instr_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        instr_layout = QVBoxLayout()
        instructions = [
            "1. 체커보드가 정상적으로 인식되는지 확인하세요 (녹색 오버레이 표시).",
            "2. 화면 내 16개 가이드 영역을 커버하도록 보정판 위치를 변경해 데이터를 수집하세요.",
            "3. 16개 유효 프레임이 모두 수집되어야 캘리브레이션 및 파라미터 저장이 가능합니다.",
            "4. 프레임 캡처 시 보정판을 움직이지 말고 고정하세요."
        ]
        for text in instructions:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #dddddd; font-size: 15px; font-weight: bold;")
            lbl.setWordWrap(True)
            instr_layout.addWidget(lbl)
        instr_box.setLayout(instr_layout)
        int_left.addWidget(instr_box, 1)
        
        controls_box = QGroupBox("보정 제어")
        controls_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        controls_layout = QVBoxLayout()
        
        self.chk_int_guide = QCheckBox("가이드 오버레이 표시")
        self.chk_int_guide.setChecked(True)
        self.chk_int_guide.setStyleSheet("color: #00e5ff; font-size: 15px; font-weight: bold;")
        self.chk_int_guide.stateChanged.connect(self.parent_app.on_guide_changed)
        controls_layout.addWidget(self.chk_int_guide)
        
        btn_int_capture = QPushButton("프레임 캡처 (C)")
        btn_int_capture.setMinimumHeight(45)
        btn_int_capture.setStyleSheet("background-color: #1565c0; color: white; font-size: 14px; font-weight: bold;")
        btn_int_capture.clicked.connect(self.step1_capture)
        controls_layout.addWidget(btn_int_capture)
        
        btn_int_calibrate = QPushButton("캘리브레이션 실행")
        btn_int_calibrate.setMinimumHeight(45)
        btn_int_calibrate.setStyleSheet("background-color: #2e7d32; color: white; font-size: 14px; font-weight: bold;")
        btn_int_calibrate.clicked.connect(self.step1_run)
        controls_layout.addWidget(btn_int_calibrate)
        
        self.btn_int_save = QPushButton("파라미터 저장")
        self.btn_int_save.setMinimumHeight(45)
        self.btn_int_save.setStyleSheet("background-color: #e65100; color: white; font-size: 14px; font-weight: bold;")
        self.btn_int_save.clicked.connect(self.step1_save)
        controls_layout.addWidget(self.btn_int_save)
        
        btn_int_reset = QPushButton("캡처 초기화")
        btn_int_reset.setMinimumHeight(35)
        btn_int_reset.setStyleSheet("background-color: #37474f; color: white; font-weight: bold; font-size: 13px;")
        btn_int_reset.clicked.connect(self.parent_app.reset_intrinsics_captures)
        controls_layout.addWidget(btn_int_reset)
        
        controls_box.setLayout(controls_layout)
        
        int_right = QVBoxLayout()
        
        stats_box2 = QGroupBox("캡처 통계")
        stats_box2.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        stats_layout2 = QHBoxLayout()
        self.lbl_captured = QLabel("수집된 프레임: 0 / 16")
        self.lbl_captured.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_captured.setStyleSheet("color: #2979ff;")
        
        self.lbl_temp = QLabel("카메라 온도: -- °C")
        self.lbl_temp.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_temp.setStyleSheet("color: #ff5500;")
        
        stats_layout2.addWidget(self.lbl_captured)
        stats_layout2.addStretch()
        stats_layout2.addWidget(self.lbl_temp)
        stats_box2.setLayout(stats_layout2)
        
        self.lbl_step1_status = QLabel("상태: 프레임 수집 대기 중 (16개 필요)")
        self.lbl_step1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step1_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")
        
        int_right.addWidget(stats_box2)
        int_right.addWidget(controls_box)
        int_right.addStretch()
        int_right.addWidget(self.lbl_step1_status)
        
        content1_layout.addLayout(int_left, 2)
        content1_layout.addLayout(int_right, 1)
        slide1_layout.addLayout(content1_layout, 1)
        self.stacked_widget.addWidget(slide1)
        
        # -----------------------------------------
        # Slide 4: 로봇 연결
        # -----------------------------------------
        slide2 = QWidget()
        l2 = QVBoxLayout(slide2)
        l2.setSpacing(12)
        l2.setAlignment(Qt.AlignCenter)
        
        t2 = QLabel("2. 로봇 연결")
        t2.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t2.setAlignment(Qt.AlignCenter)
        l2.addWidget(t2)
        
        d2 = QLabel("설정된 IP 주소를 사용하여 로봇에 연결하세요.")
        d2.setStyleSheet("font-size: 16px; color: #dddddd;")
        d2.setWordWrap(True)
        d2.setAlignment(Qt.AlignCenter)
        l2.addWidget(d2)
        
        self.lbl_step2_status = QLabel("상태: 대기 중")
        self.lbl_step2_status.setAlignment(Qt.AlignCenter)
        self.lbl_step2_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l2.addWidget(self.lbl_step2_status)
        
        # 헤드 옵션 다이어그램
        head_box = QWidget()
        head_layout = QVBoxLayout(head_box)
        head_layout.setContentsMargins(0, 0, 0, 0)
        
        head_img_label = QLabel()
        pix_head = QPixmap("img/head_onoff.png")
        if not pix_head.isNull():
            head_img_label.setPixmap(pix_head.scaled(750, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            head_img_label.setText("[img/head_onoff.png 찾을 수 없음]")
        head_img_label.setAlignment(Qt.AlignCenter)
        head_layout.addWidget(head_img_label)
        
        head_desc = QLabel("참고: 카메라 전용 브라켓을 사용하는 경우 'Head' 체크박스를 해제하세요.\n카메라가 정밀 구동부 상단에 직접 마운트된 경우 'Head'를 체크하세요.")
        head_desc.setStyleSheet("font-size: 16px; color: #ffecb3; font-weight: bold;")
        head_desc.setWordWrap(True)
        head_desc.setAlignment(Qt.AlignCenter)
        head_layout.addWidget(head_desc)
        
        l2.addWidget(head_box, alignment=Qt.AlignCenter)
        
        # 로봇 연결 설정 박스
        conn_box = QGroupBox("로봇 연결 설정")
        conn_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        conn_box.setFixedWidth(600)
        conn_layout = QVBoxLayout()
        conn_layout.setSpacing(10)
        
        ip_row = QHBoxLayout()
        lbl_ip = QLabel("IP/포트:")
        lbl_ip.setStyleSheet("font-size: 15px; font-weight: bold;")
        ip_row.addWidget(lbl_ip)
        
        self.wizard_ip_input = QLineEdit("192.168.30.1:50051")
        if self.parent_app.ui_only:
            self.wizard_ip_input.setText("127.0.0.1:50051")
        self.wizard_ip_input.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px; font-size: 15px;")
        ip_row.addWidget(self.wizard_ip_input)
        conn_layout.addLayout(ip_row)
        
        connect_row = QHBoxLayout()
        self.btn_wizard_connect = QPushButton("로봇 연결 (CONNECT)")
        self.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
        self.btn_wizard_connect.clicked.connect(self.step2_connect)
        connect_row.addWidget(self.btn_wizard_connect)
        
        self.wizard_chk_head = QCheckBox("Head (헤드 구동부 포함)")
        self.wizard_chk_head.setChecked(True)
        self.wizard_chk_head.setStyleSheet("color: #00e5ff; font-size: 15px; font-weight: bold;")
        connect_row.addWidget(self.wizard_chk_head)
        conn_layout.addLayout(connect_row)
        
        conn_box.setLayout(conn_layout)
        l2.addWidget(conn_box, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide2)

        # -----------------------------------------
        # Slide 5: 3-1. 초기 제로 포즈 이동
        # -----------------------------------------
        slide3_1 = QWidget()
        l3_1 = QVBoxLayout(slide3_1)
        l3_1.setSpacing(14)
        l3_1.setAlignment(Qt.AlignCenter)
        
        t3_1 = QLabel("3-1. 초기 제로 포즈 이동")
        t3_1.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t3_1.setAlignment(Qt.AlignCenter)
        l3_1.addWidget(t3_1)
        
        d3_1 = QLabel("기준 초기화 설정을 위해 먼저 로봇 팔을 초기 제로 포즈로 이동시킵니다.")
        d3_1.setStyleSheet("font-size: 16px; color: #dddddd; font-weight: bold;")
        d3_1.setAlignment(Qt.AlignCenter)
        l3_1.addWidget(d3_1)
        
        self.lbl_step3_1_status = QLabel("상태: 제로 포즈 이동 대기 중")
        self.lbl_step3_1_status.setAlignment(Qt.AlignCenter)
        self.lbl_step3_1_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l3_1.addWidget(self.lbl_step3_1_status)
        
        self.btn_move_zero_init = QPushButton("제로 포즈로 이동")
        self.btn_move_zero_init.setFixedWidth(250)
        self.btn_move_zero_init.setMinimumHeight(45)
        self.btn_move_zero_init.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_move_zero_init.clicked.connect(self.step3_1_move_zero)
        l3_1.addWidget(self.btn_move_zero_init, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide3_1)

        # -----------------------------------------
        # Slide 6: 3-2. 직접 교시 버튼 사용 안내
        # -----------------------------------------
        slide3_2 = QWidget()
        l3_2 = QVBoxLayout(slide3_2)
        l3_2.setSpacing(14)
        l3_2.setAlignment(Qt.AlignCenter)
        
        t3_2 = QLabel("3-2. 직접 교시 버튼 사용 안내")
        t3_2.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t3_2.setAlignment(Qt.AlignCenter)
        l3_2.addWidget(t3_2)
        
        img3_2 = QLabel()
        pix3_2 = QPixmap("img/teaching_button.png")
        if not pix3_2.isNull():
            img3_2.setPixmap(pix3_2.scaled(550, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img3_2.setText("[img/teaching_button.png 찾을 수 없음]")
        img3_2.setAlignment(Qt.AlignCenter)
        l3_2.addWidget(img3_2)
        
        d3_2_box = QGroupBox("직접 교시 사용 안내")
        d3_2_box.setStyleSheet("QGroupBox::title { color: #00e5ff; font-weight: bold; font-size: 16px;}")
        d3_2_box.setFixedWidth(750)
        d3_2_layout = QVBoxLayout(d3_2_box)
        d3_2_layout.setSpacing(8)
        
        inst3_2 = [
            "1. 각 팔의 직접 교시 버튼을 눌러 수동으로 주요 관절 위치를 설정합니다.",
            "2. 중요: 보정값이 반대 방향으로 계산되어 오작동을 일으키지 않도록, 지정된 주요 관절을 수작업으로 옮겨주어야 합니다."
        ]
        for txt in inst3_2:
            lbl = QLabel(txt)
            lbl.setStyleSheet("font-size: 15px; color: #dddddd; font-weight: bold;")
            lbl.setWordWrap(True)
            d3_2_layout.addWidget(lbl)
            
        warn3_2 = QLabel("⚠️ 경고: 양팔의 직접 교시 버튼을 절대로 동시에 누르지 마십시오!")
        warn3_2.setStyleSheet("font-size: 16px; color: #ff5252; font-weight: bold;")
        warn3_2.setAlignment(Qt.AlignCenter)
        d3_2_layout.addWidget(warn3_2)
        
        l3_2.addWidget(d3_2_box, alignment=Qt.AlignCenter)
        self.stacked_widget.addWidget(slide3_2)

        # -----------------------------------------
        # Slide 7: 3-3. 홈 오프셋 설정 포즈 맞추기
        # -----------------------------------------
        slide3_3 = QWidget()
        l3_3 = QVBoxLayout(slide3_3)
        l3_3.setSpacing(10)
        
        t3_3 = QLabel("3-3. 홈 오프셋 설정 포즈 맞추기")
        t3_3.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t3_3.setAlignment(Qt.AlignCenter)
        l3_3.addWidget(t3_3)
        
        self.lbl_skip_hint7 = QLabel("이미 진행했다면 다음(Next) 혹은 스킵(Skip) 버튼을 눌러 다음 단계를 진행하십시오.")
        self.lbl_skip_hint7.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 16px;")
        self.lbl_skip_hint7.setAlignment(Qt.AlignCenter)
        l3_3.addWidget(self.lbl_skip_hint7)
        
        self.lbl_step7_status = QLabel("상태: 대기 중")
        self.lbl_step7_status.setAlignment(Qt.AlignCenter)
        self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l3_3.addWidget(self.lbl_step7_status)
        
        # 2열 배치
        row3_3 = QHBoxLayout()
        row3_3.setSpacing(15)
        
        img3_3 = QLabel()
        pix3_3 = QPixmap("img/home_offset_position.png")
        if not pix3_3.isNull():
            img3_3.setPixmap(pix3_3.scaled(550, 340, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img3_3.setText("[img/home_offset_position.png 찾을 수 없음]")
        img3_3.setAlignment(Qt.AlignCenter)
        row3_3.addWidget(img3_3)
        
        inst3_3_box = QGroupBox("수동 위치 교시 가이드라인")
        inst3_3_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        inst3_3_layout = QVBoxLayout(inst3_3_box)
        inst3_3_layout.setSpacing(10)
        
        posture_texts = [
            "1. Shoulder Roll(어깨 회전)을 몸 안쪽으로 넣어주세요.",
            "2. Elbow(팔꿈치)를 물리적으로 닿을 때까지 뒤로 펼쳐주세요.",
            "3. Wrist Pitch(손목)는 선택 사항이며, 회전시킬 경우 10도 정도로만 살짝 틀어주세요.",
            "4. 1~3번이 완료되면 아래 '홈 오프셋 리셋' 버튼을 눌러 영점을 초기화하세요."
        ]
        for p_txt in posture_texts:
            lbl_p = QLabel(p_txt)
            if p_txt.startswith("4."):
                lbl_p.setStyleSheet("font-size: 16px; color: #ffeb3b; font-weight: bold;")
            else:
                lbl_p.setStyleSheet("font-size: 16px; color: #ffffff; font-weight: bold;")
            lbl_p.setWordWrap(True)
            inst3_3_layout.addWidget(lbl_p)
            
        row3_3.addWidget(inst3_3_box)
        l3_3.addLayout(row3_3)
        
        self.btn_step3_reset = QPushButton("홈 오프셋 리셋")
        self.btn_step3_reset.setFixedWidth(240)
        self.btn_step3_reset.setMinimumHeight(45)
        self.btn_step3_reset.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_step3_reset.clicked.connect(self.step3_reset)
        l3_3.addWidget(self.btn_step3_reset, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide3_3)
        
        # -----------------------------------------
        # Slide 8: 풀 오토 캘리브레이션 (Step 1)
        # -----------------------------------------
        slide4 = QWidget()
        l4 = QVBoxLayout(slide4)
        l4.setAlignment(Qt.AlignCenter)
        l4.setSpacing(14)
        
        t4 = QLabel("4. 풀 오토 캘리브레이션 (Step 1)")
        t4.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t4.setAlignment(Qt.AlignCenter)
        l4.addWidget(t4)
        
        d4_layout = QVBoxLayout()
        d4_step1 = QLabel("1. '풀 오토 시작'을 눌러 자동 데이터 수집 및 보정을 진행하세요.")
        d4_step1.setStyleSheet("font-size: 16px; color: #ffffff; font-weight: bold;")
        d4_step1.setAlignment(Qt.AlignCenter)
        d4_layout.addWidget(d4_step1)
        
        d4_step2 = QLabel("2. 완료가 되면 '적용'을 눌러 오프셋 및 브라켓을 저장하세요.")
        d4_step2.setStyleSheet("font-size: 16px; color: #00e5ff; font-weight: bold;")
        d4_step2.setAlignment(Qt.AlignCenter)
        d4_layout.addWidget(d4_step2)
        l4.addLayout(d4_layout)
        
        self.lbl_step4_status = QLabel("상태: 대기 중")
        self.lbl_step4_status.setAlignment(Qt.AlignCenter)
        self.lbl_step4_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l4.addWidget(self.lbl_step4_status)
        
        action_row4 = QHBoxLayout()
        btn_start_full = QPushButton("풀 오토 시작")
        btn_start_full.setMinimumHeight(45)
        btn_start_full.setFixedWidth(200)
        btn_start_full.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        btn_start_full.clicked.connect(self.start_step4)
        
        btn_apply4 = QPushButton("적용 (Apply)")
        btn_apply4.setMinimumHeight(45)
        btn_apply4.setFixedWidth(200)
        btn_apply4.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        btn_apply4.clicked.connect(self.parent_app.apply_full_auto_results)
        
        action_row4.addStretch()
        action_row4.addWidget(btn_start_full)
        action_row4.addWidget(btn_apply4)
        action_row4.addStretch()
        l4.addLayout(action_row4)
        
        aux_box4 = QGroupBox("안전 및 모니터링 제어")
        aux_box4.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 15px;}")
        aux_box4.setFixedWidth(750)
        aux_layout4 = QVBoxLayout()
        aux_layout4.setSpacing(10)
        
        feed_row = QHBoxLayout()
        feed_desc = QLabel("마커가 잘 인식되는지 확인하려면 이 버튼을 누르세요 ->")
        feed_desc.setStyleSheet("font-size: 15px; color: #ffeb3b; font-weight: bold;")
        feed_row.addWidget(feed_desc)
        feed_row.addStretch()
        btn_feed4 = QPushButton("카메라 피드 열기")
        btn_feed4.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        btn_feed4.clicked.connect(self.parent_app.toggle_camera_feed_dialog)
        feed_row.addWidget(btn_feed4)
        aux_layout4.addLayout(feed_row)
        
        stop_row = QHBoxLayout()
        stop_desc = QLabel("로봇이 비정상적으로 동작하거나 충돌 위험이 발생할 경우 이 버튼을 누르세요 ->")
        stop_desc.setStyleSheet("font-size: 15px; color: #ff5252; font-weight: bold;")
        stop_row.addWidget(stop_desc)
        stop_row.addStretch()
        btn_stop4 = QPushButton("모션 정지")
        btn_stop4.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        btn_stop4.clicked.connect(self.parent_app.stop_full_auto)
        stop_row.addWidget(btn_stop4)
        aux_layout4.addLayout(stop_row)
        
        aux_box4.setLayout(aux_layout4)
        l4.addWidget(aux_box4, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide4)
        
        # -----------------------------------------
        # Slide 9: 초기 포즈 & 자동 모션 (Step 2)
        # -----------------------------------------
        slide5 = QWidget()
        l5 = QVBoxLayout(slide5)
        l5.setAlignment(Qt.AlignCenter)
        l5.setSpacing(14)
        
        t5 = QLabel("5. 초기 포즈 & 자동 모션 (Step 2)")
        t5.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t5.setAlignment(Qt.AlignCenter)
        l5.addWidget(t5)
        
        d5_layout = QVBoxLayout()
        d5_step1 = QLabel("1. '초기 포즈 이동'을 눌러 로봇 팔을 시작 위치로 이동시킵니다.")
        d5_step1.setStyleSheet("font-size: 16px; color: #ffffff; font-weight: bold;")
        d5_step1.setAlignment(Qt.AlignCenter)
        d5_layout.addWidget(d5_step1)
        
        d5_step2 = QLabel("2. '자동 모션 시작'을 눌러 자동 궤적 데이터 수집을 실행하세요.")
        d5_step2.setStyleSheet("font-size: 16px; color: #00e5ff; font-weight: bold;")
        d5_step2.setAlignment(Qt.AlignCenter)
        d5_layout.addWidget(d5_step2)
        l5.addLayout(d5_layout)
        
        self.lbl_step5_status = QLabel("상태: 대기 중")
        self.lbl_step5_status.setAlignment(Qt.AlignCenter)
        self.lbl_step5_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l5.addWidget(self.lbl_step5_status)
        
        action_row5 = QHBoxLayout()
        btn_init = QPushButton("초기 포즈 이동")
        btn_init.setMinimumHeight(45)
        btn_init.setFixedWidth(200)
        btn_init.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        btn_init.clicked.connect(self.start_step5_init)
        
        btn_auto = QPushButton("자동 모션 시작")
        btn_auto.setMinimumHeight(45)
        btn_auto.setFixedWidth(200)
        btn_auto.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        btn_auto.clicked.connect(self.start_step5_auto)
        
        action_row5.addStretch()
        action_row5.addWidget(btn_init)
        action_row5.addWidget(btn_auto)
        action_row5.addStretch()
        l5.addLayout(action_row5)
        
        aux_box5 = QGroupBox("안전 및 모니터링 제어")
        aux_box5.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 15px;}")
        aux_box5.setFixedWidth(750)
        aux_layout5 = QVBoxLayout()
        aux_layout5.setSpacing(10)
        
        feed_row5 = QHBoxLayout()
        feed_desc5 = QLabel("마커가 잘 인식되는지 확인하려면 이 버튼을 누르세요 ->")
        feed_desc5.setStyleSheet("font-size: 15px; color: #ffeb3b; font-weight: bold;")
        feed_row5.addWidget(feed_desc5)
        feed_row5.addStretch()
        btn_feed5 = QPushButton("카메라 피드 열기")
        btn_feed5.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        btn_feed5.clicked.connect(self.parent_app.toggle_camera_feed_dialog)
        feed_row5.addWidget(btn_feed5)
        aux_layout5.addLayout(feed_row5)
        
        stop_row5 = QHBoxLayout()
        stop_desc5 = QLabel("로봇이 비정상적으로 동작하거나 충돌 위험이 발생할 경우 이 버튼을 누르세요 ->")
        stop_desc5.setStyleSheet("font-size: 15px; color: #ff5252; font-weight: bold;")
        stop_row5.addWidget(stop_desc5)
        stop_row5.addStretch()
        btn_stop5 = QPushButton("모션 정지")
        btn_stop5.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 14px; padding: 6px 14px; border-radius: 4px;")
        btn_stop5.clicked.connect(self.parent_app.request_stop_all_auto_motion)
        stop_row5.addWidget(btn_stop5)
        aux_layout5.addLayout(stop_row5)
        
        aux_box5.setLayout(aux_layout5)
        l5.addWidget(aux_box5, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide5)
        
        # -----------------------------------------
        # Slide 10: 홈 오프셋 적용
        # -----------------------------------------
        slide6 = QWidget()
        l6 = QVBoxLayout(slide6)
        l6.setAlignment(Qt.AlignCenter)
        l6.setSpacing(12)
        
        t6 = QLabel("6. 홈 오프셋 적용")
        t6.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b;")
        t6.setAlignment(Qt.AlignCenter)
        l6.addWidget(t6)
        
        d6 = QLabel("계산된 최적화 홈 오프셋을 검토하고 로봇에 적용합니다.")
        d6.setStyleSheet("font-size: 16px; color: #dddddd;")
        d6.setWordWrap(True)
        d6.setAlignment(Qt.AlignCenter)
        l6.addWidget(d6)
        
        self.lbl_step6_status = QLabel("상태: 대기 중")
        self.lbl_step6_status.setAlignment(Qt.AlignCenter)
        self.lbl_step6_status.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        l6.addWidget(self.lbl_step6_status)
        
        apply_row = QHBoxLayout()
        apply_row.setSpacing(15)
        
        img_apply = QLabel()
        pix_apply = QPixmap("img/apply_offset.png")
        if not pix_apply.isNull():
            img_apply.setPixmap(pix_apply.scaled(520, 290, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_apply.setText("[img/apply_offset.png 찾을 수 없음]")
        img_apply.setAlignment(Qt.AlignCenter)
        apply_row.addWidget(img_apply)
        
        apply_instructions_box = QGroupBox("홈 오프셋 적용 지침 사항")
        apply_instructions_box.setStyleSheet("QGroupBox::title { color: #ffeb3b; font-weight: bold; font-size: 16px;}")
        apply_instr_layout = QVBoxLayout()
        apply_instr_layout.setSpacing(8)
        
        instructions_list = [
            "1. 비교: 'Baseline' 및 'Optimized'를 선택하여 보정 전후 오프셋을 비교해 볼 수 있습니다.",
            "2. 제로 포즈 검사: 'Move to Zero'를 눌러 관절을 이동시켜 조인트가 제대로 보정되었는지 확인하세요.",
            "3. 대칭성 검사: 'Move to Check'를 눌러 두 팔이 대칭을 이루는지 확인하세요.",
            "4. 적용/롤백: 적용을 원하는 값(Baseline 또는 Optimized)을 선택한 뒤 '선택한 오프셋 적용' 버튼을 클릭하세요."
        ]
        for inst_text in instructions_list:
            lbl_inst = QLabel(inst_text)
            lbl_inst.setStyleSheet("font-size: 14px; color: #dddddd; font-weight: bold;")
            lbl_inst.setWordWrap(True)
            apply_instr_layout.addWidget(lbl_inst)
            
        apply_instructions_box.setLayout(apply_instr_layout)
        apply_row.addWidget(apply_instructions_box)
        
        l6.addLayout(apply_row)
        
        self.btn_step6_apply = QPushButton("홈 오프셋 적용")
        self.btn_step6_apply.setFixedWidth(240)
        self.btn_step6_apply.setMinimumHeight(45)
        self.btn_step6_apply.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        self.btn_step6_apply.clicked.connect(self.parent_app.apply_home_offset)
        l6.addWidget(self.btn_step6_apply, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(slide6)

    def mark_step_completed(self, step_idx, success=True, msg=""):
        self.step_completed[step_idx] = success
        self.update_navigation(self.stacked_widget.currentIndex())
        
        lbl_name = None
        if step_idx == 3:
            lbl_name = "lbl_step1_status"
        elif step_idx == 4:
            lbl_name = "lbl_step2_status"
        elif step_idx == 5:
            lbl_name = "lbl_step3_1_status"
        elif step_idx == 7:
            lbl_name = "lbl_step7_status"
        elif step_idx == 8:
            lbl_name = "lbl_step4_status"
        elif step_idx == 9:
            lbl_name = "lbl_step5_status"
        elif step_idx == 10:
            lbl_name = "lbl_step6_status"

        if lbl_name:
            lbl = getattr(self, lbl_name, None)
            if lbl:
                if success:
                    lbl.setText(f"상태: 성공 - {msg}" if msg else "상태: 성공")
                    lbl.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
                else:
                    lbl.setText(f"상태: 오류 - {msg}")
                    lbl.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def step3_1_move_zero(self):
        if self.parent_app.move_to_zero_pose():
            self.lbl_step3_1_status.setText("상태: 제로 포즈로 이동 중...")
            self.lbl_step3_1_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
            self.set_wizard_busy(True)
        else:
            if not self.parent_app.robot:
                self.mark_step_completed(5, False, "로봇이 연결되지 않음")

    def go_prev(self):
        idx = self.stacked_widget.currentIndex()
        if idx == 4:
            self.stacked_widget.setCurrentIndex(2)
        elif idx == 3:
            self.stacked_widget.setCurrentIndex(2)
        elif idx > 0:
            self.stacked_widget.setCurrentIndex(idx - 1)
        else:
            if hasattr(self.parent_app, 'overview_title') and self.parent_app.overview_title:
                self.parent_app.overview_title.setVisible(True)
            if hasattr(self.parent_app, 'overview_link') and self.parent_app.overview_link:
                self.parent_app.overview_link.setVisible(True)
            if hasattr(self.parent_app, 'overview_duration') and self.parent_app.overview_duration:
                self.parent_app.overview_duration.setVisible(True)
            self.parent_app.btn_start_wizard.setVisible(True)
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
            self.parent_app.log_msg("캘리브레이션 위자드가 완료되었습니다.")
            if hasattr(self.parent_app, 'overview_title') and self.parent_app.overview_title:
                self.parent_app.overview_title.setVisible(True)
            if hasattr(self.parent_app, 'overview_link') and self.parent_app.overview_link:
                self.parent_app.overview_link.setVisible(True)
            if hasattr(self.parent_app, 'overview_duration') and self.parent_app.overview_duration:
                self.parent_app.overview_duration.setVisible(True)
            self.parent_app.btn_start_wizard.setVisible(True)
            self.parent_app.overview_img.setVisible(True)
            self.setVisible(False)
            self.stacked_widget.setCurrentIndex(0)
            
    def update_navigation(self, idx):
        if hasattr(self, "parent_app") and hasattr(self.parent_app, "on_left_tab_changed"):
            self.parent_app.on_left_tab_changed(self.parent_app.left_tabs.currentIndex())
        self.btn_prev.setVisible(True)
        if idx == 0:
            self.btn_prev.setText("오버뷰로 돌아가기")
        else:
            self.btn_prev.setText("이전")
            
        show_skip = (idx == 3 or idx == 7)
        self.btn_skip.setVisible(show_skip)
        
        if hasattr(self, 'lbl_skip_hint1'):
            self.lbl_skip_hint1.setVisible(idx == 3)
        if hasattr(self, 'lbl_skip_hint7'):
            self.lbl_skip_hint7.setVisible(idx == 7)
        
        enabled = self.step_completed[idx]
        self.btn_next.setEnabled(enabled)
        
        if enabled:
            self.btn_next.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setStyleSheet("background-color: #444444; color: #888888; font-weight: bold; font-size: 15px; border-radius: 6px;")
        
        if idx == self.stacked_widget.count() - 1:
            self.btn_next.setText("완료 (Finish)")
            self.btn_next.setEnabled(True)
            self.btn_next.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        else:
            self.btn_next.setText("다음")

    def step1_capture(self):
        self.parent_app.capture_intrinsics_frame()
        frames = len(self.parent_app.captured_images)
        self.lbl_captured.setText(f"수집된 프레임: {frames} / 16")
        if frames >= 16:
            self.lbl_step1_status.setText(f"상태: {frames} / 16 프레임 수집 완료. 캘리브레이션 준비 됨.")
            self.lbl_step1_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_step1_status.setText(f"상태: {frames} / 16 프레임 수집됨 (16개 필요)")
            self.lbl_step1_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")

    def step1_run(self):
        if len(self.parent_app.captured_images) < 16:
            self.lbl_step1_status.setText("상태: 캘리브레이션을 실행하려면 16개 프레임이 모두 필요합니다!")
            self.lbl_step1_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")
            QMessageBox.warning(self, "데이터 부족", f"캘리브레이션을 실행할 수 없습니다: {len(self.parent_app.captured_images)} / 16 프레임만 수집됨.\n16개 프레임을 먼저 모두 캡처하세요.")
            return

        self.parent_app.run_intrinsics_calibration()
        err = self.parent_app.intrinsics_calibrator.rms_error
        if err is not None and err > 0.0:
            self.lbl_step1_status.setText(f"상태: 캘리브레이션 완료 (RMS: {err:.4f})")
            self.lbl_step1_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_step1_status.setText("상태: 캘리브레이션 실패 (보정판 설정 확인 필요)")
            self.lbl_step1_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 16px;")

    def step1_save(self):
        if len(self.parent_app.captured_images) < 16:
            QMessageBox.warning(self, "데이터 부족", f"파라미터를 저장할 수 없습니다: {len(self.parent_app.captured_images)} / 16 프레임만 수집됨.")
            self.mark_step_completed(3, False, "저장하려면 16개 프레임 필요")
            return

        if self.parent_app.intrinsics_calibrator.cameraMatrix is not None and float(self.parent_app.intrinsics_calibrator.rms_error) > 0.0:
            self.parent_app.save_intrinsics_calibration()
            self.mark_step_completed(3, True, "파라미터 저장 완료")
        else:
            QMessageBox.warning(self, "유효하지 않은 캘리브레이션", "파라미터를 저장하기 전에 캘리브레이션을 성공적으로 실행해야 합니다.")
            self.mark_step_completed(3, False, "캘리브레이션 미실행")

    def step2_connect(self):
        self.btn_wizard_connect.setText("연결 중...")
        self.btn_wizard_connect.setStyleSheet("background-color: #ffb74d; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
        self.btn_wizard_connect.setEnabled(False)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self.parent_app.ip_input.setText(self.wizard_ip_input.text())
        self.parent_app.chk_servo_head.setChecked(self.wizard_chk_head.isChecked())
        
        self.parent_app.connect_robot()
        
        self.btn_wizard_connect.setEnabled(True)
        if self.parent_app.robot is not None:
            self.btn_wizard_connect.setText("연결됨 (CONNECTED)")
            self.btn_wizard_connect.setStyleSheet("background-color: #757575; color: #ffffff; font-weight: bold; padding: 8px 16px; font-size: 15px;")
            self.mark_step_completed(4, True, "로봇 연결 완료")
        else:
            self.btn_wizard_connect.setText("로봇 연결 (CONNECT)")
            self.btn_wizard_connect.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 8px 16px; font-size: 15px;")
            self.mark_step_completed(4, False, "로봇 연결 실패")

    def step3_reset(self):
        reply = QMessageBox.question(
            self,
            "홈 오프셋 리셋 확인",
            "진행하시겠습니까?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            self.lbl_step7_status.setText("상태: 영점 초기화 취소됨")
            self.lbl_step7_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 16px;")
            return

        if self.parent_app.home_offset_reset(confirm_dialog=False):
            self.lbl_step7_status.setText("상태: 영점 초기화 진행 중...")
            self.lbl_step7_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
            self.set_wizard_busy(True)
        else:
            if not self.parent_app.robot:
                self.mark_step_completed(7, False, "로봇 연결되지 않음")
            else:
                self.lbl_step7_status.setText("상태: 영점 초기화 취소됨")
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

    def start_step4(self):
        self.step4_elapsed = 0
        self.lbl_step4_status.setText("상태: 풀 오토 진행 중 (00:00)")
        self.lbl_step4_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
        self.step4_timer.start(1000)
        self.parent_app.start_full_auto()
        if hasattr(self.parent_app, 'active_worker') and self.parent_app.active_worker:
            self.parent_app.active_worker.finished_signal.connect(self.stop_step4)
        else:
            self.stop_step4(False, "워커가 시작되지 않음")

    def update_step4_time(self):
        self.step4_elapsed += 1
        m = self.step4_elapsed // 60
        s = self.step4_elapsed % 60
        self.lbl_step4_status.setText(f"상태: 풀 오토 진행 중 ({m:02d}:{s:02d})")

    def get_calibrated_joint_summary_ko(self):
        store = getattr(self.parent_app, 'joint_offsets_store', {})
        r_store = store.get("right", {})
        l_store = store.get("left", {})
        
        r_j6 = r_store.get("joint6", 0.0)
        r_j5 = r_store.get("joint5", 0.0)
        r_j3 = r_store.get("joint3", 0.0)
        
        l_j6 = l_store.get("joint6", 0.0)
        l_j5 = l_store.get("joint5", 0.0)
        l_j3 = l_store.get("joint3", 0.0)
        
        return f"[오른팔] J6={r_j6:+.2f}°, J5={r_j5:+.2f}°, J3={r_j3:+.2f}° | [왼팔] J6={l_j6:+.2f}°, J5={l_j5:+.2f}°, J3={l_j3:+.2f}°"

    def stop_step4(self):
        self.step4_timer.stop()
        was_stopped = False
        if hasattr(self.parent_app, "full_auto_stop_event") and self.parent_app.full_auto_stop_event is not None:
            was_stopped = self.parent_app.full_auto_stop_event.is_set()
            
        error_msg = getattr(self.parent_app.active_worker, "error_msg", None) if hasattr(self.parent_app, "active_worker") and self.parent_app.active_worker else None
        
        if not was_stopped and not error_msg:
            summary = self.get_calibrated_joint_summary_ko()
            self.lbl_step4_status.setText(f"상태: 계산 완료! 보정 결과: {summary}\n⚠️ 다음 단계 이동을 위해 반드시 '적용 (Apply)' 버튼을 눌러주세요.")
            self.lbl_step4_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 15px;")
            self.step_completed[8] = False
            self.update_navigation(self.stacked_widget.currentIndex())
        else:
            if was_stopped:
                self.mark_step_completed(8, False, "사용자에 의해 취소됨")
            else:
                self.mark_step_completed(8, False, error_msg or "알 수 없는 오류")

    def on_step4_applied(self):
        summary = self.get_calibrated_joint_summary_ko()
        self.mark_step_completed(8, True, f"적용 완료! {summary}")

    def start_step5_init(self):
        self.parent_app.step2_init_pose()
        if hasattr(self.parent_app, 'auto_motion_thread') and self.parent_app.auto_motion_thread:
            self.parent_app.auto_motion_thread.finished_signal.connect(
                lambda success, err: self.lbl_step5_status.setText("상태: 초기 포즈 도달 완료" if success else f"상태: 오류 - {err}")
            )
        
    def start_step5_auto(self):
        self.step5_elapsed = 0
        self.lbl_step5_status.setText("상태: 자동 모션 진행 중 (00:00)")
        self.lbl_step5_status.setStyleSheet("color: #2196f3; font-weight: bold; font-size: 16px;")
        self.step5_timer.start(1000)
        self.parent_app.step2_auto_motion()
        if hasattr(self.parent_app, 'auto_motion_thread') and self.parent_app.auto_motion_thread:
            self.parent_app.auto_motion_thread.finished_signal.connect(self.on_step5_motion_finished)
        else:
            self.stop_step5(False, "워커가 시작되지 않음")

    def on_step5_motion_finished(self, success=True, err_msg=""):
        if success:
            self.lbl_step5_status.setText("상태: 모션 완료. 최적화 연산 수행 중...")
            self.lbl_step5_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 16px;")
        else:
            self.stop_step5(False, err_msg)
            
    def update_step5_time(self):
        self.step5_elapsed += 1
        m = self.step5_elapsed // 60
        s = self.step5_elapsed % 60
        if "최적화 연산" in self.lbl_step5_status.text():
            self.lbl_step5_status.setText(f"상태: 최적화 연산 수행 중 ({m:02d}:{s:02d})")
        else:
            self.lbl_step5_status.setText(f"상태: 자동 모션 진행 중 ({m:02d}:{s:02d})")
        
    def stop_step5(self, success=True, err_msg=""):
        self.step5_timer.stop()
        if success:
            self.mark_step_completed(9, True, f"완료 ({self.step5_elapsed//60:02d}:{self.step5_elapsed%60:02d})")
        else:
            self.mark_step_completed(9, False, err_msg)
