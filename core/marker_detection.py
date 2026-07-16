import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading
import os, yaml

#debugging flag : 실사용시 모두 false여야함
imshow_when_detect = False
tcpip_send = False
use_calib_int = False # 세밀하게 보정된 파일(camera_intrinsics.yaml)을 사용할지 여부
# see_depth_sensors_depth = False
# see_stereo_depth = False

# 유틸리티 클래스
class TCPClient:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        try:
            self.sock.connect((ip, port))
            print("Connected to Python Server!")
            self.connected = True
        except ConnectionRefusedError:
            print("Connection Failed. (Is Python Server running?)")
        except Exception as e:
            print(f"Connection Error: {e}")

    def __del__(self):
        if self.connected:
            self.sock.close()

    def send_pose(self, T):
        if not self.connected:
            return
        
        # T is expected to be a flat list or numpy array of 16 floats
        if isinstance(T, np.ndarray):
            T = T.flatten().tolist()
            
        # Pack 16 floats (4 bytes each) -> 64 bytes
        try:
            packed_data = struct.pack('16f', *T)
            self.sock.send(packed_data)
        except Exception as e:
            print(f"Send Error: {e}")
            self.connected = False
        
# 카메라 클래스
"""
해당 클래스는 realsense 카메라만 호환
다른 카메라를 사용할 경우 아래의 get이란 이름이 들어간 함수들의 양식을 동일하게 맞춰주고 구현하는것을 권장
"""
class RealSenseCamera:
    # serial_number : 해당 넘버의 카메라를 사용, 입력하지 않으면 첫번째 카메라 사용
    """카메라의 시리얼 넘버는 realsense_check.py를 통해 검색할 수 있음"""
    def __init__(self, serial_number=None):
        # 연결되어있는 카메라 검색
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices found!")
            raise RuntimeError("No RealSense connected")
        # 카메라 선택 : 시리얼 넘버가 정해져있으면 해당 카메라 사용, 없으면 첫번째 카메라 사용
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
            if serial_number == dev.get_info(rs.camera_info.serial_number) or serial_number is None:
                self.device_number = i
                break

        # 안전한 사용을 위해 선택한 카메라 재연결
        print("Resetting Realsense device...")
        devices[self.device_number].hardware_reset()
        # 카메라가 다시 연결될 때까지 대기
        time.sleep(3)

        # 카메라 정보 재확인(하드웨어 리셋했으므로 재확인 필요)
        ctx = rs.context()
        devices = ctx.query_devices()

        self.device_name = devices[self.device_number].get_info(rs.camera_info.name)
        self.serial_number = devices[self.device_number].get_info(rs.camera_info.serial_number)
        print("Using camera is : ", self.device_name)

        # depth 해상도 확인 : D435는 1mm, D405 는 0.1mm. 즉 모델에 따라 다름
        depth_sensor = devices[self.device_number].first_depth_sensor()
        if depth_sensor.supports(rs.option.thermal_compensation):
            depth_sensor.set_option(rs.option.thermal_compensation, 1.0) # 온도보정기능 On
            depth_sensor.set_option(rs.option.visual_preset, 3) # High Accuracy
        depth_scale = depth_sensor.get_depth_scale()
        print("depth scale : ", depth_scale)
        

        # 카메라 구동을 위한 파라미터 설정
        self.depth_resolution = depth_scale*1000                # 각 픽셀의 depth 값 해상도
        self.pipeline = rs.pipeline()                           # 카메라 스트리밍을 위한 파이프라인
        self.config = rs.config()                               # 카메라 설정구조체
        self.spatial = rs.spatial_filter()                      # 평활화 필터 (노이즈 제거)
        self.spatial.set_option(rs.option.filter_magnitude, 2)  # 평활화 필터의 강도 설정
        self.temporal = rs.temporal_filter()                    # 깜빡임 방지필터
        self.hole_filling = rs.hole_filling_filter()            # 노이즈에 의한 빈 공간 채우는 필터

        # 상태 flag
        self.camera_running = False                             # 카메라 구동상태 flag
        self.camera_monitoring = False                          # 카메라 모니터링 상태 flag
        self.Infrared = True                                    # 적외선 카메라 사용 flag

        #이미지 저장 변수
        self.color_image = None
        self.depth_image = None
        # 기본해상도
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30

        #카메라 내부 파라미터 : 해당 파라미터로 depth map 및 마커좌표 계산됨
        self.fx = 0.0                                           # 초점거리 x
        self.fy = 0.0                                           # 초점거리 y
        self.principal_point = [0.0, 0.0]                       # 주점(이미지 중심)
        self.intrinsics = None                                  # 내부 파라미터 행렬
        self.profile = None                                     # 카메라 프로파일
        self.baseline = 0.065                                   # 스테레오 카메라 베이스라인(m)
        self.dist_coeffs = None                                 # 카메라 왜곡 계수

        # 스레드 동기화를 위한 Lock
        self.lock = threading.Lock()
        self.thread = None

    def initialize_camera(self, set_width, set_height, set_fps):
        self.width = set_width
        self.height = set_height
        self.fps = set_fps
        
        try:
            self.config.enable_device(self.serial_number)
            # 사용할 카메라(컬러, depth, ir) 스트리밍 활성화. depth는 모든상황에 사용됨
            # cpu 부하를 줄이기 위해 적외선 사용 시 ir1, ir2 를, 사용하지 않을 경우 color만 추가로 스트리밍 진행
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            if self.Infrared:
                self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
                self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
            
            # 파이프라인 시작
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"Failed to start pipeline with {self.width}x{self.height}@{self.fps}. Error: {e}")
            print("Attempting fallback resolution (848x480 @ 30fps)...")
            try:
                self.config = rs.config() # Reset config
                self.config.enable_device(self.serial_number)
                self.width, self.height, self.fps = 848, 480, 30
                self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self.profile = self.pipeline.start(self.config)
            except Exception as e2:
                print(f"Fallback 1 failed: {e2}. Attempting 640x480 @ 30fps...")
                try:
                    self.config = rs.config()
                    self.config.enable_device(self.serial_number)
                    self.width, self.height, self.fps = 640, 480, 30
                    self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                    self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                    self.profile = self.pipeline.start(self.config)
                except Exception as e3:
                    print(f"All profile attempts failed: {e3}")
                    raise e3

        try:
            # 구동 후 카메라 안정화를 위해 10프레임 정도는 버리고 사용
            for i in range(10):
                self.pipeline.wait_for_frames()
            
            # [NEW] 센서 수동 노출(Manual Exposure) 설정 - 모션 블러 방지 및 Gain 활용
            device = self.profile.get_device()
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                if sensor.supports(rs.option.exposure):
                    try:
                        sensor.set_option(rs.option.exposure, 8000) # 8ms (모션 블러 방지와 밝기 밸런스)
                    except Exception as e:
                        print(f"Warning: Failed to set exposure on sensor {sensor.get_info(rs.camera_info.name)}: {e}")
                if sensor.supports(rs.option.gain):
                    try:
                        sensor.set_option(rs.option.gain, 80) # Gain 상향으로 밝기 확보
                    except Exception as e:
                        print(f"Warning: Failed to set gain on sensor {sensor.get_info(rs.camera_info.name)}: {e}")

            # depth 카메라 내부 파라미터 얻기 : baseline, fx, fy, principal_point 를 위해 사용
            color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

            left_ir_stream = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            right_ir_stream = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            
            extrinsics = left_ir_stream.get_extrinsics_to(right_ir_stream)
            self.baseline = abs(extrinsics.translation[0]) # m
            self.intrinsics = color_stream.get_intrinsics()
            self.fx = self.intrinsics.fx
            self.fy = self.intrinsics.fy
            self.principal_point = [self.intrinsics.ppx, self.intrinsics.ppy] #pixel
            self.dist_coeffs = np.array(self.intrinsics.coeffs)

            print(f"Successfully initialized: {self.width}x{self.height} @ {self.fps}fps")
            print(f"Focal Length: fx={self.fx}, fy={self.fy}")
            print(f"Principal Point: {self.principal_point[0]}, {self.principal_point[1]}")
            print("Baseline: ", self.baseline)
            self.camera_running = True
        except Exception as e:
            print(f"Camera didn't initialize post-start: {e}")
            raise e

    # Monitoring 함수로 스트리밍을 켜고 끔
    def monitoring(self, Flag=True):
        self.camera_monitoring = Flag
        if self.camera_monitoring:
            
            self.thread = threading.Thread(target=self.stream_on)
            self.thread.start()
        else:
            self.stream_off()
            if self.thread is not None:
                self.thread.join()

    def stream_on(self , fps = 30):
        align_to = rs.stream.color
        align = rs.align(align_to)
        frame_sleep = 1/fps
        try:
            while self.camera_running:
                self.capture_image()
                 # Visualization Logic
                result_list = []
                if self.color_image is not None:
                    result_list.append(self.color_image)
                if self.depth_image is not None:
                    if np.max(self.depth_image) > 0:
                        # min_dist = float(np.min(self.depth_image[self.depth_image > 0]))
                        # max_dist = float(np.max(self.depth_image))
                        min_dist = 700.0
                        max_dist = 5000.0
                        alpha = (0.0 - 255.0) / (max_dist - min_dist)
                        beta = 255.0 - (min_dist * alpha)
                    else:
                        min_dist = 700.0
                        max_dist = 5000.0
                        alpha = (0.0 - 255.0) / (max_dist - min_dist)
                        beta = 255.0 - (min_dist * alpha)
                    depth_re_img = self.depth_image.astype(np.float32)
                    depth_re_img = depth_re_img * alpha + beta
                    depth_re_img = np.clip(depth_re_img, 0, 255).astype(np.uint8)
                    # depth_re_img = depth_re_img.astype(np.uint8)
                    depth_re_img[self.depth_image == 0] = 0
                    depth_re_img_bgr = cv2.cvtColor(depth_re_img, cv2.COLOR_GRAY2BGR)
                    #depth_re_img_bgr = cv2.applyColorMap(depth_re_img, cv2.COLORMAP_BONE)
                    result_list.append(depth_re_img_bgr)
                if self.Infrared == True and self.left_ir_image is not None and self.right_ir_image is not None:
                    result_list.append(cv2.cvtColor(self.left_ir_image, cv2.COLOR_GRAY2BGR))
                    result_list.append(cv2.cvtColor(self.right_ir_image, cv2.COLOR_GRAY2BGR))
                
                if len(result_list) > 0:
                    if len(result_list) == 1:
                        # 이미지가 한개일 때 화면 크기 기본 해상도의 1/2
                        resize_height = self.height // 2
                        resize_width = self.width // 2
                    else:
                        # 이미지가 두개 이상일 때 화면 크기 기본 해상도의 1/n
                        resize_height = self.height // len(result_list)
                        resize_width = (self.width // len(result_list)) * len(result_list)
                    concat_image = cv2.hconcat(result_list)
                    concat_image = cv2.resize(concat_image, (resize_width, resize_height))
                    cv2.imshow("Preview", concat_image)
                    key = cv2.waitKey(1)
                    if key == 27 or key == ord('q'): # ESC or q
                        raise KeyboardInterrupt
                    
                    if cv2.getWindowProperty('Preview', cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt
                time.sleep(frame_sleep)
        except RuntimeError as e:
            print(f"Error: {e}")

    def stream_off(self):
        self.camera_running = False
        if self.thread is not None:
            self.thread.join()
        try:
            self.pipeline.stop()
        except:
            pass

    def capture_image(self):
        # 복잡한 스레드 검사 로직(if self.thread is not None...)을 전부 삭제합니다.
        if not self.camera_running:
            return

        # 이 함수는 stream_on(백그라운드 스레드)에서만 단독으로 실행됩니다.
        try:
            frames = self.pipeline.wait_for_frames()
            
            # 카메라의 depth map을 사용할 경우 컬러 이미지와 해상도, 사이즈를 맞추기 위해 align을 사용.
            align_to = rs.stream.color
            align = rs.align(align_to)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            # 추가 필터적용
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
            if not color_frame or not depth_frame:
                print("no frame")
                return
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())
        
            with self.lock:
                self.color_image = color_data
                self.depth_image = depth_data
                if self.Infrared:
                    ir_frame_left = frames.get_infrared_frame(1)
                    ir_frame_right = frames.get_infrared_frame(2)
                    if ir_frame_left and ir_frame_right:
                        self.left_ir_image = np.asanyarray(ir_frame_left.get_data())
                        self.right_ir_image = np.asanyarray(ir_frame_right.get_data())
        except Exception as e:
            pass # 간헐적인 프레임 드랍 무시


    # 연산을 위해 호출돼야하는 함수들
    def get_color_image(self):
        with self.lock:
            if self.color_image is None:
                return None
            return self.color_image.copy()

    def get_depth_image(self):
        with self.lock:
            if self.depth_image is None:
                return None
            return self.depth_image.copy()

    def get_infrared_images(self):
        with self.lock:
            if not hasattr(self, 'left_ir_image') or not hasattr(self, 'right_ir_image'):
                return None, None
            return self.left_ir_image.copy(), self.right_ir_image.copy()

    def get_principal_point_and_focal_length(self):
        return [self.principal_point[0], self.principal_point[1], self.fx, self.fy]

    def get_depth_resolution(self):
        return self.depth_resolution

    def get_baseline(self):
        return self.baseline

    def get_camera_temperature(self):
        try:
            if not self.camera_running or self.profile is None:
                return None
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.projector_temperature):
                proj_temp = depth_sensor.get_option(rs.option.projector_temperature)
                return proj_temp
            # 만약 asic 온도를 대신 지원하는 모델일 경우 (D435 등은 둘 다 지원하거나 하나만 지원)
            elif depth_sensor.supports(rs.option.asic_temperature):
                asic_temp = depth_sensor.get_option(rs.option.asic_temperature)
                return asic_temp
        except Exception as e:
            print(f"Failed to get temperature: {e}")
            return None
        return None

    def get_dist_coeffs(self):
        return self.dist_coeffs


class Marker_Detection:
    def __init__(self):
        # 어떤 마커를 인식시킬건지 정의
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()
        # 마커 인식 정밀도 향상을 위한 파라미터 튜닝
        # 1. 서브픽셀 정밀도 극대화
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.01

        # 2. 이진화 세밀화 (조명 및 그림자 대응)
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 3  # 세밀한 스캔
        self.parameters.adaptiveThreshConstant = 7     # 노이즈 대비 약간 낮춤

        # 3. 형상 근사 및 필터링
        self.parameters.polygonalApproxAccuracyRate = 0.01 # 더 엄격한 사각형 검사
        self.parameters.minDistanceToBorder = 3
        self.parameters.minMarkerPerimeterRate = 0.01

        # 4. 내부 비트 샘플링 향상
        self.parameters.perspectiveRemovePixelPerCell = 12 # 정밀한 비트 추출

        # 계산에 쓰일 카메라 내부 파라미터
        self.principal_point = [0, 0]
        self.fx = 0
        self.fy = 0
        self.dist_coeffs = None
        self.depth_resolution = 1
        self.rpy = [0, 0, 0]
        
        self.lpf_alpha = 0.5
        self.prev_pts_dict = {}

        self.focal_scale = 1.0# 0.99 # Focal length scaling factor for fine-tuning

        # 인식할 마커 타입과 ID
        self.marker_type = None
        self.marker_id = None
        
        self.marker_size_mm = 36.0 # 기본값, config에서 덮어씌워짐
        self.markers_config = {}
        if tcpip_send:
            self.tcp_client = TCPClient("127.0.0.1", 5000)

        self.marker_depth = 0
        self.stereo_depth = 0

        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.detector = None

    # 연산에 필요한 카메라 파라미터 설정
    def set_intrinsics_param(self, param):
        self.principal_point = [param[0], param[1]]
        self.fx = param[2]
        self.fy = param[3]

    def set_depth_resolution(self, depth_resolution):
        self.depth_resolution = depth_resolution

    def set_baseline(self, baseline):
        self.baseline = baseline

    def set_dist_coeffs(self, dist_coeffs):
        self.dist_coeffs = dist_coeffs

    def set_marker_type(self, marker_type="plate"):
        self.marker_type = marker_type
        if marker_type == "plate":
            self.plate_left_ids = self.markers_config.get("plate", {}).get("left_ids", [])
            self.plate_right_ids = self.markers_config.get("plate", {}).get("right_ids", [])
            self.marker_id = self.plate_left_ids + self.plate_right_ids
            self.marker_size_mm = self.markers_config.get("plate", {}).get("plate_size_mm", 100.0) * 0.8
        else:
            self.marker_id = []

    def get_depth_from_depth_img(self, depth_image, center_pixel):
        x, y = int(center_pixel[0]), int(center_pixel[1])
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            # Use a small window (e.g., 3x3) to get a more stable depth value
            roi = depth_image[max(0, y-1):min(depth_image.shape[0], y+2),
                              max(0, x-1):min(depth_image.shape[1], x+2)]
            valid_depths = roi[roi > 0]
            if len(valid_depths) > 0:
                return float(np.median(valid_depths))*self.depth_resolution
            return float(depth_image[y, x])*self.depth_resolution
        return 0.0

    # 마커들의 중심좌표(4*4행렬)
    def detect(self, color_image, lpf = False, logging = False, depth_image = None, use_filter = True):
        pnp_dist_coeffs = self.dist_coeffs
        if self.dist_coeffs is not None and np.any(self.dist_coeffs != 0):
            base_cam_mat = np.array([
                [self.fx, 0, self.principal_point[0]],
                [0, self.fy, self.principal_point[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            # 이미지 전체 렌즈 왜곡 펴기 (마커 모서리 직선 복원)
            color_image = cv2.undistort(color_image, base_cam_mat, self.dist_coeffs, None, base_cam_mat)
            # 이미 왜곡을 폈으므로, 이후 solvePnP에서는 왜곡 파라미터 무시 (이중 보정 방지)
            pnp_dist_coeffs = None
            
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        
        # Step 2: 서브픽셀 정밀도 강제화
        if corners is not None and len(corners) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            for i in range(len(corners)):
                cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), criteria)
                
        marker_centers_result = []
        

        # 등록된 마커만 필터링 (tuple과 ndarray는 pop()을 쓸 수 없으므로 새로 리스트를 만듭니다)
        if ids is not None and len(ids) > 0 and self.marker_id is not None:
            valid_indices = []
            for i, mid in enumerate(ids):
                val = mid[0] if isinstance(mid, (np.ndarray, list)) else mid
                if val in self.marker_id:
                    valid_indices.append(i)
            if len(valid_indices) > 0:
                corners = tuple(corners[i] for i in valid_indices)
                ids = np.array([ids[i] for i in valid_indices])
            else:
                corners, ids = (), None

        # debugging (필터링이 끝난 마커들만 화면에 표시되게 위치를 조정합니다)
        if imshow_when_detect:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            cv2.imshow("Detected Markers", color_image)
            cv2.waitKey(1)
            
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            unique_ids, counts = np.unique(ids_flat, return_counts=True)
            duplicate_ids = unique_ids[counts > 1]
            if len(duplicate_ids) > 0:
                print(f"Warning: Duplicate marker IDs detected: {duplicate_ids}. Ignoring these duplicates in this frame.")
                
            comp_fx = self.fx * self.focal_scale
            comp_fy = self.fy * self.focal_scale
            cam_mat = np.array([
                [comp_fx, 0, self.principal_point[0]],
                [0, comp_fy, self.principal_point[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            half_m = self.marker_size_mm / 2.0

            obj_pts = np.array([
                [-half_m, -half_m, 0], [ half_m, -half_m, 0],
                [ half_m,  half_m, 0], [-half_m,  half_m, 0]
            ], dtype=np.float32)
            
            for i in range(len(ids)):
                marker_id = ids_flat[i]
                if marker_id in duplicate_ids:
                    continue
                c = corners[i][0]
                
                # # Check depth if see_depth_sensors_depth is True and it's a plate
                # if see_depth_sensors_depth and self.marker_type == "plate" and depth_image is not None:
                #     center_px = np.mean(c, axis=0)
                #     self.marker_depth = self.get_depth_from_depth_img(depth_image, center_px)
                #     print(f"[Depth] Marker ID {marker_id} Center Depth: {self.marker_depth:.1f} mm")

                success, rvec, tvec = cv2.solvePnP(obj_pts, c, cam_mat, pnp_dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                if not success:
                    continue
                rot_matrix, _ = cv2.Rodrigues(rvec)
                center_pos = tvec.flatten().tolist()
                
                # --- EMA & Slerp Smoothing ---
                if use_filter or lpf:
                    alpha = self.lpf_alpha
                    if marker_id in self.prev_pts_dict:
                        prev_pos, prev_rot = self.prev_pts_dict[marker_id]
                        
                        # Adaptive EMA: Calculate position delta
                        dist = np.linalg.norm(np.array(center_pos) - np.array(prev_pos))
                        # If movement is larger than 2mm per frame (fast motor movement), disable smoothing to avoid lag
                        if dist > 2.0:
                            alpha = 1.0
                            
                        # 1. Position EMA
                        center_pos = [
                            alpha * center_pos[0] + (1 - alpha) * prev_pos[0],
                            alpha * center_pos[1] + (1 - alpha) * prev_pos[1],
                            alpha * center_pos[2] + (1 - alpha) * prev_pos[2]
                        ]
                        # 2. Rotation Slerp
                        try:
                            from scipy.spatial.transform import Rotation as R
                            r_curr = R.from_matrix(rot_matrix)
                            r_prev = R.from_matrix(prev_rot)
                            
                            delta_rot = r_curr * r_prev.inv()
                            r_smoothed = R.from_rotvec(delta_rot.as_rotvec() * alpha) * r_prev
                            rot_matrix_smoothed = r_smoothed.as_matrix()
                        except ImportError:
                            rot_matrix_smoothed = rot_matrix
                    else:
                        rot_matrix_smoothed = rot_matrix
                        
                    # Update previous state
                    self.prev_pts_dict[marker_id] = (center_pos, rot_matrix_smoothed)
                else:
                    rot_matrix_smoothed = rot_matrix
                
                transform = [
                    rot_matrix_smoothed[0][0], rot_matrix_smoothed[0][1], rot_matrix_smoothed[0][2], center_pos[0],
                    rot_matrix_smoothed[1][0], rot_matrix_smoothed[1][1], rot_matrix_smoothed[1][2], center_pos[1],
                    rot_matrix_smoothed[2][0], rot_matrix_smoothed[2][1], rot_matrix_smoothed[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]
                
                # Plate group identification
                if marker_id in getattr(self, 'plate_left_ids', []):
                    marker_centers_result.append(("plate_left", transform))
                elif marker_id in getattr(self, 'plate_right_ids', []):
                    marker_centers_result.append(("plate_right", transform))
                else:
                    marker_centers_result.append((marker_id, transform))
                # if tcpip_send:
                #     self.tcp_client.send_pose(transform)
                
        return marker_centers_result



class Marker_Transform:
    def __init__(self, serial_number = None):
        # Initialize
        self.camera = RealSenseCamera(serial_number=serial_number)
        self.marker_detection = Marker_Detection()
        
        # Load configs globally in the wrapper class
        self._load_all_configs()
        
        # Setup Transforms
        tf_vec_l = self.camera_config.get("Tf_to_marker_left", self.camera_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        tf_vec_r = self.camera_config.get("Tf_to_marker_right", self.camera_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        head_base_vec = self.camera_config.get("head_base_to_cam", [0.009, -0.09, -0.085, 159.0, 0.0, 180.0])
        print(tf_vec_l)
        
        self.Tf_to_marker_tf_left = self.make_transform(tf_vec_l)
        self.Tf_to_marker_tf_right = self.make_transform(tf_vec_r)
        self.head_base_to_cam_tf = self.make_transform(head_base_vec)
        
        self.width = self.camera_config.get("width", 1280)
        self.height = self.camera_config.get("height", 720)
        self.fps = self.camera_config.get("fps", 30)
        
        print("Initializing Camera...")
        self.camera.initialize_camera(self.width, self.height, self.fps)
        
        intrinsics = self.camera.get_principal_point_and_focal_length()
        self.marker_detection.set_intrinsics_param(intrinsics)

        depth_resolution = self.camera.get_depth_resolution()
        self.marker_detection.set_depth_resolution(depth_resolution)

        dist_coeffs = self.camera.get_dist_coeffs()
        self.marker_detection.set_dist_coeffs(dist_coeffs)

        self.marker_detection.set_baseline(self.camera.baseline)
        
        # [NEW] 내부 파라미터 보정 파일(camera_intrinsics.yaml) 사용 설정
        if use_calib_int:
            # Search for config in current dir or parent dir (to support both source and installed structures)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            calib_file = os.path.join(base_dir, "config", "camera_intrinsics.yaml")
            if not os.path.exists(calib_file):
                calib_file = os.path.join(os.path.dirname(base_dir), "config", "camera_intrinsics.yaml")
            if os.path.exists(calib_file):
                try:
                    with open(calib_file, "r") as f:
                        calib_data = yaml.safe_load(f)
                    
                    mtx = np.array(calib_data["camera_matrix"])
                    dist = np.array(calib_data["dist_coeffs"])
                    
                    calib_w = calib_data.get("width")
                    calib_h = calib_data.get("height")
                    
                    # 해상도가 다를 경우 비례적으로 스케일 조정
                    if calib_w and calib_h and (calib_w != self.width or calib_h != self.height):
                        scale_x = self.width / calib_w
                        scale_y = self.height / calib_h
                        
                        if abs(scale_x - scale_y) > 0.03:
                            print(f"\n[WARNING] Aspect ratio mismatch! Calibration: {calib_w}x{calib_h}, Current: {self.width}x{self.height}")
                        
                        mtx[0,0] *= scale_x # fx
                        mtx[1,1] *= scale_y # fy
                        mtx[0,2] *= scale_x # ppx
                        mtx[1,2] *= scale_y # ppy
                        print(f"\n[INFO] Scaled intrinsics from {calib_w}x{calib_h} to {self.width}x{self.height} (Scale X:{scale_x:.2f}, Y:{scale_y:.2f})")

                    # Marker_Detection에 보정된 파라미터 주입
                    # 신규 인터페이스 [ppx, ppy, fx, fy]
                    new_intrinsics = [mtx[0,2], mtx[1,2], mtx[0,0], mtx[1,1]]
                    self.marker_detection.set_intrinsics_param(new_intrinsics)
                    self.marker_detection.set_dist_coeffs(dist)
                    
                    print(f"[INFO] --- Loaded Calibrated Intrinsics from {calib_file} ---")
                    print(f"       fx: {mtx[0,0]:.2f}, fy: {mtx[1,1]:.2f}, ppx: {mtx[0,2]:.2f}, ppy: {mtx[1,2]:.2f}")
                    print(f"       dist: {dist}")
                except Exception as e:
                    print(f"\n[ERROR] Failed to load {calib_file}: {e}")
            else:
                print(f"\n[WARNING] Calibrated Intrinsics file {calib_file} NOT FOUND. Using factory defaults.")
        
        self.temp_history = []

    def _load_all_configs(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        setting_config_path = os.path.join(base_dir, "config", "setting.yaml")
        if not os.path.exists(setting_config_path):
            setting_config_path = os.path.join(os.path.dirname(base_dir), "config", "setting.yaml")
            
        try:
            with open(setting_config_path, "r") as f:
                config_data = yaml.safe_load(f)
                self.camera_config = config_data.get("camera", {})
                self.markers_config = config_data.get("marker", config_data) # fallback
                self.marker_detection.markers_config = self.markers_config
                
                print(f"- Loaded Setting Config from {os.path.basename(setting_config_path)}")
        except Exception as e:
            print(f"- Warning: Could not load {setting_config_path}: {e}")
            self.camera_config = {}
            self.markers_config = {}
    def set_marker_type(self, marker_type="plate"):
        self.marker_detection.set_marker_type(marker_type)
    def make_transform(self, data):
        # data: [x, y, z, roll, pitch, yaw] (x,y,z in meters, r,p,y in degrees)
        x, y, z = data[0]*1000, data[1]*1000, data[2]*1000 
        roll = data[3] * math.pi / 180
        pitch = data[4] * math.pi / 180
        yaw = data[5] * math.pi / 180
        
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)
        
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = cy * cp
        m[0, 1] = sr * sp * cy - cr * sy
        m[0, 2] = cr * sp * cy + sr * sy
        m[0, 3] = x
        
        m[1, 0] = sy * cp
        m[1, 1] = sr * sp * sy + cr * cy
        m[1, 2] = cr * sp * sy - sr * cy
        m[1, 3] = y
        
        m[2, 0] = -sp
        m[2, 1] = cp * sr
        m[2, 2] = cp * cr
        m[2, 3] = z
        
        return m

    def calc_cam_to_tool(self, camera_to_marker_tf, side="left"):
        try:
            target_tf = self.Tf_to_marker_tf_left if side == "left" else self.Tf_to_marker_tf_right
            # target_tf is in meters now. camera_to_marker_tf is also in meters.
            tf_to_marker_inv = np.linalg.inv(target_tf)
        
            if tcpip_send:
                cam_to_tool_tf = camera_to_marker_tf @ tf_to_marker_inv
            else:
                cam_to_tool_tf = camera_to_marker_tf
            cam_to_tool_vec = cam_to_tool_tf.flatten()
            
            cam_to_tool_vec[3] /= 1000
            cam_to_tool_vec[7] /= 1000
            cam_to_tool_vec[11] /= 1000
            if tcpip_send and len(cam_to_tool_vec) > 0:
                self.marker_detection.tcp_client.send_pose(cam_to_tool_vec) 
            return cam_to_tool_vec
        except np.linalg.LinAlgError:
            print("Singular matrix, cannot invert")
            return None

    def get_marker_transform(self, sampling_time=0, side="left", use_filter=None):
        if use_filter is None:
            use_filter = (sampling_time == 0)
        lpf = False
        # Collection array for sampling -> dict of lists
        collected_transforms = {} # { marker_id: [tf_vectors...] }
        sampled_temps = []
        start_time = time.time()

        if sampling_time > 0:
            self.marker_detection.prev_pts_dict = {}
            lpf = True

        while True:
            try:
                if not self.camera.camera_monitoring:
                    self.camera.capture_image()
                color_img = self.camera.get_color_image()
                depth_img = self.camera.get_depth_image()
                if color_img is None:
                    time.sleep(0.01)
                    if sampling_time == 0 : return None
                    continue
                
                marker_transforms = self.marker_detection.detect(color_img, lpf=lpf, depth_image=depth_img, use_filter=use_filter)
                for marker_id_or_group, tf_list in marker_transforms:
                    if marker_id_or_group not in collected_transforms:
                        collected_transforms[marker_id_or_group] = []
                    collected_transforms[marker_id_or_group].append(tf_list)
                # -------------------------------------------------------------
                
                # Check timeout if sampling
                if sampling_time == 0 or (sampling_time > 0 and (time.time() - start_time > sampling_time)):
                    break
                        
            except KeyboardInterrupt:
                raise
            
            # CPU 점유율을 낮추기 위한 미세한 대기
            time.sleep(0.01)
            
        final_results = {}
        # Post-processing for sampling
        if sampling_time > 0:
            if not collected_transforms:
                return None
            
            for marker_id, tfs in collected_transforms.items():
                data = np.array(tfs) # Shape (N, 16)
                
                # Separate translation and rotation for CAMERA_TO_MARKER (NOT inverted yet)
                translations = data[:, [3, 7, 11]]
                
                # Median for translation is robust
                final_translation = np.median(translations, axis=0)
                
                # Average rotations using SVD (chordal L2 mean) to maintain orthogonality
                rotations = []
                for vec in data:
                    R = np.array([
                        [vec[0], vec[1], vec[2]],
                        [vec[4], vec[5], vec[6]],
                        [vec[8], vec[9], vec[10]]
                    ])
                    rotations.append(R)
                
                sum_R = np.sum(rotations, axis=0)
                U, S, Vt = np.linalg.svd(sum_R)
                final_R = U @ Vt
                
                # Ensure det(R) = 1 (proper rotation)
                if np.linalg.det(final_R) < 0:
                    U[:, 2] *= -1
                    final_R = U @ Vt
                
                avg_cam_to_marker_tf = np.eye(4, dtype=np.float32)
                avg_cam_to_marker_tf[0:3, 0:3] = final_R
                avg_cam_to_marker_tf[0:3, 3] = final_translation
                
                calc_side = "left" if "left" in str(marker_id) else "right"
                
                cam_to_tool_vec = self.calc_cam_to_tool(avg_cam_to_marker_tf, side=calc_side)
                if cam_to_tool_vec is not None:
                    final_results[marker_id] = cam_to_tool_vec
        elif sampling_time == 0:
            for marker_id, tfs in collected_transforms.items():
                # For sampling_time == 0, there is only one frame of transforms
                # and thus tfs should have only length 1
                camera_to_marker_tf = np.array(tfs[-1], dtype=np.float32).reshape(4, 4)
                
                calc_side = "left" if "left" in str(marker_id) else "right"
                cam_to_tool_vec = self.calc_cam_to_tool(camera_to_marker_tf, side=calc_side)
                if cam_to_tool_vec is not None:
                    final_results[marker_id] = cam_to_tool_vec
        
        if len(final_results) > 0:
            if self.marker_detection.marker_type == "plate":
                if side == "left":
                    res = final_results.get("plate_left")
                    return [res] if res is not None else None
                elif side == "right":
                    res = final_results.get("plate_right")
                    return [res] if res is not None else None
                elif side == "all":
                    out = []
                    res_r = final_results.get("plate_right")
                    res_l = final_results.get("plate_left")
                    if res_r is not None: out.append(res_r)
                    if res_l is not None: out.append(res_l)
                    return out if out else None
            return final_results
        else:
            return None
