from core.storage import camera_intrinsics_path, camera_intrinsics_serial_path
from core.storage import ConfigStorage
from core.camera_processing import RealSenseCamera, CameraUnavailableError
import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading
import sys
import os
import re

#debugging flag : must be all false in production
imshow_when_detect = False
tcpip_send = False
# Which values from the calibrated intrinsics file (camera_intrinsics.yaml) override the factory ones:
#   "off"             : factory fx, fy, cx, cy and distortion
#   "principal_point" : calibrated cx, cy only (factory focal length and distortion kept)
#   "full"            : calibrated fx, fy, cx, cy and distortion
# The principal point sets where the optical axis is in the image, so it directly biases the head
# tilt/pan pointing estimate (2026-09-15: factory cy 347.9 vs calibrated 350.6 px = 0.24 deg of tilt).
calib_intrinsics_mode = "full"
# see_depth_sensors_depth = False
# see_stereo_depth = False

# Utility classes

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
        
# Camera class
"""
This class is only compatible with RealSense cameras
If using another camera, it is recommended to implement functions with the same signature.
"""


class Marker_Detection:
    def __init__(self):
        # Define which markers to detect
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()
        # Parameter tuning for improving marker detection precision
        # 1. Maximize sub-pixel precision
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.01

        # 2. Refine binarization (for lighting and shadow robust)
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 3  # Fine scan
        self.parameters.adaptiveThreshConstant = 7     # Adjusted lower for noise

        # 3. Shape approximation and filtering
        self.parameters.polygonalApproxAccuracyRate = 0.01 # Stricter square check
        self.parameters.minDistanceToBorder = 3
        self.parameters.minMarkerPerimeterRate = 0.01

        # 4. Enhance internal bit sampling
        self.parameters.perspectiveRemovePixelPerCell = 12 # Fine bit extraction

        # Intrinsic parameters used for calculation
        self.principal_point = [0, 0]
        self.fx = 0
        self.fy = 0
        self.dist_coeffs = None
        self.depth_resolution = 1
        self.rpy = [0, 0, 0]
        
        self.lpf_alpha = 0.5
        self.prev_pts_dict = {}
        self.prev_pnp_rot = {}

        self.focal_scale = 1.0# 0.99 # Focal length scaling factor for fine-tuning

        # Marker type and ID to detect
        self.marker_type = None
        self.marker_id = None
        
        self.marker_size_mm = 36.0 # Default value, overwritten by config
        self.markers_config = {}
        if tcpip_send:
            self.tcp_client = TCPClient("127.0.0.1", 5000)

        self.marker_depth = 0
        self.stereo_depth = 0

        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.detector = None

    def reset_tracking(self):
        self.prev_pts_dict = {}
        self.prev_pnp_rot = {}

    # Set camera parameters required for calculation
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

    # Center coordinates of markers (4x4 matrix)
    def detect(self, color_image, lpf = False, logging = False, depth_image = None, use_filter = True):
        # [LEGACY] Image-wide lens undistortion (causes CPU processing bottleneck -> Commented out)
        # pnp_dist_coeffs = self.dist_coeffs
        # if self.dist_coeffs is not None and np.any(self.dist_coeffs != 0):
        #     base_cam_mat = np.array([
        #         [self.fx, 0, self.principal_point[0]],
        #         [0, self.fy, self.principal_point[1]],
        #         [0, 0, 1]
        #     ], dtype=np.float32)
        #     # Image-wide lens undistortion (to restore marker edges to straight lines)
        #     color_image = cv2.undistort(color_image, base_cam_mat, self.dist_coeffs, None, base_cam_mat)
        #     # Since distortion is already corrected, ignore distortion params in solvePnP to avoid double correction
        #     pnp_dist_coeffs = None

        # [OPTIMIZED] Pass distortion parameters to solvePnP instead of doing full undistort (takes <0.01ms)
        pnp_dist_coeffs = self.dist_coeffs
            
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        
        # Step 2: Enforce sub-pixel precision
        if corners is not None and len(corners) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            for i in range(len(corners)):
                cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), criteria)
                
        marker_centers_result = []
        

        # Filter only registered markers (create a new list since tuple and ndarray don't support pop)
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

        # debugging (Adjust positions so only filtered markers are displayed on screen)
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

                # --- IPPE Planar Ambiguity Disambiguation ---
                pnp_res = cv2.solvePnPGeneric(obj_pts, c, cam_mat, pnp_dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                num_sol = pnp_res[0] if len(pnp_res) > 0 else 0
                if num_sol == 0 or len(pnp_res[1]) == 0:
                    continue
                
                rvecs = pnp_res[1]
                tvecs = pnp_res[2]
                reproj_errs = pnp_res[3] if len(pnp_res) > 3 else None

                # Pick the lower-reprojection-error solution independently per frame (same as
                # cv2.solvePnP(SOLVEPNP_IPPE), used up to 2026-09-07). Choosing the solution
                # closest to the previous frame (introduced 2026-09-13) locks onto a wrong
                # mirrored solution for whole runs of frames near a fronto-parallel view, which
                # the per-frame spike filter in CalibratorBase.filter_sweep_dataset cannot remove.
                best_idx = 0
                if num_sol > 1 and reproj_errs is not None and len(reproj_errs) > 1:
                    best_idx = int(np.argmin(np.asarray(reproj_errs).ravel()))
                rvec = rvecs[best_idx]
                tvec = tvecs[best_idx]
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
    def __init__(self, serial_number=None, *, sim=False, robot=None, robot_version='1.2', camera_factory=RealSenseCamera,
                 simulation_factory=None):
        # simulation_factory(version) -> the simulated pose sensor (core.calibration.simulation
        # SimulationModel.create). Passed in so this comm module does not import the calibration maths.
        self.simulation_factory = simulation_factory
        self.sim = bool(sim)
        self.camera = None
        self.robot = robot
        self.robot_version = str(robot_version).removeprefix('v')
        self.rng = np.random.default_rng(42)
        self.marker_detection = Marker_Detection()
        
        # The camera has to be open before the configs load: which model is connected decides which
        # camera_info.yaml extrinsics and which camera_intrinsics_<model>.yaml apply.
        # _load_all_configs() used to run first, so self.camera was still None and the connected
        # model was never detected -- neither the bracket extrinsics nor the intrinsics were ever
        # switched when the camera was swapped, and the mismatch warning could never fire.
        if not self.sim:
            try:
                self.camera = camera_factory(serial_number=serial_number)
            except (CameraUnavailableError, RuntimeError) as e:
                print(f"[WARN] Camera unavailable: {e}. Falling back to simulation model.")
                self.sim = True

        # Load configs globally in the wrapper class
        self._load_all_configs()
        
        # Setup Transforms
        tf_vec_l = self.markers_config.get("Tf_to_marker_left", self.markers_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        tf_vec_r = self.markers_config.get("Tf_to_marker_right", self.markers_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        head_base_vec = self.camera_config.get("head_base_to_cam", [0.009, -0.09, -0.085, 159.0, 0.0, 180.0])
        print(tf_vec_l)
        
        self.Tf_to_marker_tf_left = self.make_transform(tf_vec_l)
        self.Tf_to_marker_tf_right = self.make_transform(tf_vec_r)
        self.head_base_to_cam_tf = self.make_transform(head_base_vec)
        
        self.width = self.camera_config.get("width", 1280)
        self.height = self.camera_config.get("height", 720)
        self.fps = self.camera_config.get("fps", 30)

        if self.sim:
            self.simulation_model = self._create_simulation_model(self.robot_version)

        if self.camera is not None:
            print("Initializing Camera...")
            self.camera.initialize_camera(self.width, self.height, self.fps)
            # initialize_camera now refuses to run at anything but the requested profile, so this
            # should never trigger; kept as a guard. If the frames ever arrive at a different
            # resolution than setting.yaml says, the calibrated intrinsics would be scaled to the
            # wrong size and put a constant factor on every solvePnP range.
            actual = (getattr(self.camera, "width", None), getattr(self.camera, "height", None))
            if all(actual) and actual != (self.width, self.height):
                print(f"[WARNING] Camera negotiated {actual[0]}x{actual[1]}, not the "
                      f"{self.width}x{self.height} requested in setting.yaml. Using what it gave.")
                self.width, self.height = actual
                self.fps = getattr(self.camera, "fps", self.fps)

            # Factory values first; apply_calibrated_intrinsics() layers the calibration on top and
            # can be called again later (after a save) without reconnecting the camera.
            self._factory_intrinsics = list(self.camera.get_principal_point_and_focal_length())
            self._factory_dist_coeffs = self.camera.get_dist_coeffs()
            self.marker_detection.set_intrinsics_param(self._factory_intrinsics)
            self.marker_detection.set_dist_coeffs(self._factory_dist_coeffs)

            depth_resolution = self.camera.get_depth_resolution()
            self.marker_detection.set_depth_resolution(depth_resolution)

            self.marker_detection.set_baseline(self.camera.baseline)

            # Apply the calibrated intrinsics that _load_all_configs resolved for this camera model.
            # active_intrinsics_path is None when no calibration file matches the connected model;
            # the factory intrinsics set above then stay in use. Applying another model's file is
            # worse than having none: D405 fx 660 on a D435 (factory fx 916) makes every solvePnP
            # range read 28% short, which is what wrecked the marker bracket poses.
            self.apply_calibrated_intrinsics(self.active_intrinsics_path)

            # Always default to Auto Exposure on initialization
            self.camera.set_exposure(6000.0, auto_exposure=True)

        self.temp_supported = bool(self.camera is not None and getattr(self, 'temp_supported', False))
        self.temp_history = []
        self.set_marker_type("plate")

    def apply_calibrated_intrinsics(self, calib_file):
        """Make the detector use `calib_file`'s intrinsics now, on top of the factory values.

        Runs at camera start-up, and again right after the intrinsics calibration is saved: the
        detector used to pick up a saved calibration only on the next start, so a session that
        calibrated, saved and carried straight on ran its sweeps on the old numbers while the file
        already held the new ones. With no usable file the factory values are restored.
        Returns True when a calibration file was applied.
        """
        factory = list(getattr(self, "_factory_intrinsics", None) or [])
        if not factory:
            return False
        self.marker_detection.set_intrinsics_param(factory)
        self.marker_detection.set_dist_coeffs(getattr(self, "_factory_dist_coeffs", None))
        if calib_intrinsics_mode not in ("principal_point", "full"):
            return False
        if not calib_file:
            print(f"\n[WARNING] No calibrated intrinsics for '{self.camera_model}'. Using the camera's factory intrinsics.")
            return False
        try:
            calib_data = ConfigStorage.load(calib_file)

            mtx = np.array(calib_data["camera_matrix"], dtype=float)
            dist = np.array(calib_data["dist_coeffs"], dtype=float)

            calib_w = calib_data.get("width")
            calib_h = calib_data.get("height")

            # Proportionally adjust scale if resolution differs
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

            # Inject calibrated parameters to Marker_Detection
            # Interface [ppx, ppy, fx, fy]
            if calib_intrinsics_mode == "full":
                new_intrinsics = [mtx[0,2], mtx[1,2], mtx[0,0], mtx[1,1]]
                self.marker_detection.set_dist_coeffs(dist)
            else:
                new_intrinsics = [mtx[0,2], mtx[1,2], factory[2], factory[3]]
            self.marker_detection.set_intrinsics_param(new_intrinsics)
            self.active_intrinsics_path = calib_file

            print(f"[INFO] --- Calibrated intrinsics ({calib_intrinsics_mode}) from {calib_file} ---")
            print(f"       factory   : fx {factory[2]:.2f}, fy {factory[3]:.2f}, ppx {factory[0]:.2f}, ppy {factory[1]:.2f}")
            print(f"       in use    : fx {new_intrinsics[2]:.2f}, fy {new_intrinsics[3]:.2f}, ppx {new_intrinsics[0]:.2f}, ppy {new_intrinsics[1]:.2f}")
            if calib_intrinsics_mode == "full":
                print(f"       dist: {dist}")
            else:
                print(f"       dist (factory, in use): {getattr(self, '_factory_dist_coeffs', None)}  "
                      f"| calibrated file had: {dist}")
            return True
        except Exception as e:
            print(f"\n[ERROR] Failed to load {calib_file}: {e}")
            return False

    def bind_robot(self, robot, robot_version):
        version = str(robot_version).removeprefix('v')
        if self.sim and (getattr(self, 'simulation_model', None) is None or self.simulation_model.version != version):
            self.simulation_model = self._create_simulation_model(version)
            self.rng = np.random.default_rng(self.simulation_model.config.get('seed', 42))
        self.robot, self.robot_version = robot, version
        if robot is not None:
            self._apply_marker_bracket_version(version)

    def _create_simulation_model(self, version):
        if self.simulation_factory is None:
            raise RuntimeError("Simulation mode (or no camera) needs simulation_factory; "
                               "CalibrationCore.connect_camera passes SimulationModel.create.")
        return self.simulation_factory(version)

    def _apply_marker_bracket_version(self, version):
        """Make the live `Tf_to_marker_<side>` belong to this robot version's bracket.

        The v1.2 and v1.3 marker brackets are mounted at a different place and angle, and the
        live value is what the detector turns a marker pose into a tool pose with, so a leftover
        from the other version puts every observation ~67 mm and 90 deg out. The version-suffixed
        nominals already in setting.yaml say which is which.

        Only ever called for a connected robot: that is the one authoritative source of the
        version. Disconnecting falls back to the '1.2' default, which says nothing about the
        brackets, and acting on it would throw away a v1.3 setup's calibration.
        """
        changes = resolve_marker_brackets(self.markers_config, version)
        if not changes:
            return
        for side, value in changes.items():
            self.markers_config[f"Tf_to_marker_{side}"] = list(value)
            setattr(self, f"Tf_to_marker_tf_{side}", self.make_transform(value))
        self.marker_detection.markers_config = self.markers_config
        from core.storage import CONFIG_PATHS
        setting_path = CONFIG_PATHS.get("setting_yaml")
        if not setting_path or not os.path.exists(setting_path):
            return
        try:
            ConfigStorage.update_values(setting_path, {("marker", f"Tf_to_marker_{side}"): list(value)
                                                       for side, value in changes.items()})
            print(f"[INFO] Updated setting.yaml marker brackets for robot v{version}: "
                  + ", ".join(f"Tf_to_marker_{side}={value}" for side, value in changes.items()))
        except Exception as e:
            print(f"[WARNING] Could not write the v{version} marker brackets to setting.yaml: {e}")

    def set_marker_type(self, marker_type):
        if self.marker_detection is not None:
            self.marker_detection.set_marker_type(marker_type)

    def set_camera_exposure(self, exposure_val, auto_exposure=False):
        if self.camera is None: return False
        return self.camera.set_exposure(exposure_val, auto_exposure)

    def get_camera_exposure(self):
        if self.camera is None: return True, 6000.0
        return self.camera.get_exposure()

    def get_camera_info(self):
        """Connected camera, the model family its settings come from, stream and intrinsics file."""
        if self.camera is None or not hasattr(self.camera, "get_stream_info"):
            return None
        info = dict(self.camera.get_stream_info())
        info["camera_model"] = getattr(self, "camera_model", None)
        path = getattr(self, "active_intrinsics_path", None)
        info["intrinsics_file"] = os.path.basename(path) if path else None
        info["intrinsics_serial_matched"] = bool(getattr(self, "intrinsics_serial_matched", False))
        info["intrinsics_source"] = getattr(self, "intrinsics_source", None)
        return info

    def get_camera_exposure_range(self):
        """(min, max, step, default) this camera's colour sensor accepts, or None."""
        if self.camera is None: return None
        return self.camera.get_exposure_range()

    def get_camera_exposure_unit_us(self):
        """Microseconds per step of this camera's exposure setpoint (100 on a D435 RGB module)."""
        if self.camera is None or not hasattr(self.camera, "get_exposure_unit_us"): return 1.0
        return self.camera.get_exposure_unit_us()

    def get_actual_exposure(self):
        if self.camera is None: return 6000.0
        return self.camera.get_actual_exposure()

    def _load_all_configs(self):
        """Resolve every camera-dependent config from the connected camera's model.

        Order matters: the RealSense is already open here, so `camera.device_name` names the
        model. Everything keys off the model *family* (`D435i`/`D435f` -> `D435`), because the
        variants share one housing, one bracket and one set of optics; only the three-digit body
        number matters. Both stores follow the same rule:

          * bracket extrinsics -> the family's entry in camera_info.yaml
          * intrinsics         -> config/camera_intrinsics_<family>.yaml

        When a store has nothing for the connected family, this records it on
        `extrinsics_missing` / `intrinsics_missing` for the UI to report, and leaves the values
        alone rather than guessing. Startup continues either way.
        """
        from core.storage import CONFIG_PATHS
        setting_config_path = CONFIG_PATHS.get("setting_yaml")
        if not setting_config_path or not os.path.exists(setting_config_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            setting_config_path = os.path.join(base_dir, "config", "setting.yaml")
            if not os.path.exists(setting_config_path):
                setting_config_path = os.path.join(os.path.dirname(base_dir), "config", "setting.yaml")

        self.extrinsics_missing = False
        self.intrinsics_missing = False
        self.intrinsics_mismatch = False
        self.calib_device_name = ""
        self.active_intrinsics_path = None
        self.camera_model = None
        # Serial of the connected unit (None in simulation), whether its own intrinsics file was
        # found, and where the intrinsics in use came from: "serial", "model", "file" (simulation /
        # no camera) or "factory".
        self.camera_serial = getattr(self.camera, "serial_number", None) or None
        self.intrinsics_serial_matched = False
        self.intrinsics_source = None

        try:
            config_data = ConfigStorage.load(setting_config_path)

            camera_config = config_data.get("camera", {})
            yaml_device_name = camera_config.get("device_name")
            connected_device_name = getattr(self.camera, "device_name", None)

            # The family, not the full name: an 'Intel RealSense D435i' is stored and looked up as
            # 'D435', so a variant never creates a second, divergent set of entries.
            self.camera_model = camera_model_family(connected_device_name)
            if connected_device_name:
                print(f"[INFO] Connected camera '{connected_device_name}' -> model family '{self.camera_model}'")

            self.temp_supported = False
            info_file = CONFIG_PATHS.get("camera_info")
            if not info_file or not os.path.exists(info_file):
                info_file = os.path.join(os.path.dirname(setting_config_path), "camera_info.yaml")
                if not os.path.exists(info_file) and getattr(sys, "frozen", False):
                    info_file = os.path.join(sys._MEIPASS, "config", "camera_info.yaml")

            info_data = {}
            if os.path.exists(info_file):
                try:
                    info_data = ConfigStorage.load(info_file)
                except Exception as e:
                    print(f"[WARNING] Failed to read camera_info.yaml: {e}")

            target_model = self.camera_model or camera_model_family(yaml_device_name)
            matched_info_key = lookup_camera_info_key(info_data, target_model)

            if matched_info_key:
                cam_ext = info_data[matched_info_key]
                self.temp_supported = cam_ext.get("temp_supported", False)
                info_head_base = cam_ext.get("head_base_to_cam")
                info_mount = cam_ext.get("mount_to_cam")
                info_mount_link = cam_ext.get("camera_mount_link", "link_head_2")

                current_dev = camera_config.get("device_name")
                current_head_base = camera_config.get("head_base_to_cam")
                current_mount = camera_config.get("mount_to_cam")
                current_mount_link = camera_config.get("camera_mount_link")

                def is_diff(v1, v2):
                    if v1 is None or v2 is None:
                        return v1 != v2
                    try:
                        return not np.allclose(v1, v2, atol=1e-5)
                    except Exception:
                        return v1 != v2

                # camera_info.yaml holds each model's CAD nominal. The *_nominal keys in
                # setting.yaml record which model the live extrinsics were derived from, so they
                # are what decides whether the live values still belong to this camera:
                #
                #   nominals match  -> the live values are this model's nominal plus a Step 1.5 /
                #                      Step 2 calibration (or a hand measurement). Keep them;
                #                      overwriting on every launch silently threw them away.
                #   nominals differ -> they came from a different camera. Reload the nominal.
                #
                # Comparing device_name alone was not enough: editing device_name by hand in
                # setting.yaml made the two agree while the extrinsics stayed on the old model.
                nominal_diff = (is_diff(camera_config.get("mount_to_cam_nominal"), info_mount)
                                or is_diff(camera_config.get("head_base_to_cam_nominal"), info_head_base))
                dev_diff = (self.camera_model is not None
                            and camera_model_family(current_dev) != self.camera_model)
                missing = current_head_base is None or current_mount is None or current_mount_link is None
                model_changed = dev_diff or nominal_diff or missing

                if not model_changed and (is_diff(current_head_base, info_head_base)
                                          or is_diff(current_mount, info_mount)):
                    print(f"[INFO] setting.yaml camera extrinsics differ from the '{matched_info_key}' nominal in "
                          f"camera_info.yaml; keeping setting.yaml (calibrated or edited values).")

                changes = {}
                if model_changed:
                    if dev_diff:
                        print(f"[INFO] Connected camera '{connected_device_name}' (family '{self.camera_model}') "
                              f"differs from setting.yaml '{yaml_device_name}'. Loading its nominal extrinsics...")
                    elif nominal_diff:
                        print(f"[INFO] setting.yaml extrinsics were derived from another camera model "
                              f"(mount_to_cam_nominal {camera_config.get('mount_to_cam_nominal')} != "
                              f"'{matched_info_key}' {info_mount}). Reloading the nominal...")
                    if missing:
                        print(f"[INFO] setting.yaml has no camera extrinsics yet. Loading the "
                              f"'{matched_info_key}' nominal from camera_info.yaml...")

                    camera_config["device_name"] = self.camera_model or matched_info_key
                    changes[("camera", "device_name")] = camera_config["device_name"]
                    if info_head_base is not None:
                        camera_config["head_base_to_cam"] = list(info_head_base)
                        changes[("camera", "head_base_to_cam")] = list(info_head_base)
                    if info_mount is not None:
                        camera_config["mount_to_cam"] = list(info_mount)
                        changes[("camera", "mount_to_cam")] = list(info_mount)
                    if info_mount_link is not None:
                        camera_config["camera_mount_link"] = info_mount_link
                        changes[("camera", "camera_mount_link")] = info_mount_link

                # The nominals are the CAD numbers for whichever camera is mounted now, so they
                # track camera_info.yaml unconditionally. Nothing refreshed them before, and
                # Step 1.5 / Step 2 anchor their baseline to them -- a D405 nominal left behind
                # meant a D435 was being calibrated against the wrong CAD pose.
                if info_mount is not None and is_diff(camera_config.get("mount_to_cam_nominal"), info_mount):
                    camera_config["mount_to_cam_nominal"] = list(info_mount)
                    changes[("camera", "mount_to_cam_nominal")] = list(info_mount)
                if info_head_base is not None and is_diff(camera_config.get("head_base_to_cam_nominal"), info_head_base):
                    camera_config["head_base_to_cam_nominal"] = list(info_head_base)
                    changes[("camera", "head_base_to_cam_nominal")] = list(info_head_base)

                if changes:
                    config_data["camera"] = camera_config
                    ConfigStorage.update_values(setting_config_path, changes)
                    print(f"[INFO] Updated setting.yaml for {matched_info_key} from camera_info.yaml "
                          f"(head_base_to_cam: {camera_config.get('head_base_to_cam')}, "
                          f"mount_to_cam: {camera_config.get('mount_to_cam')})")
            else:
                self.extrinsics_missing = bool(target_model)
                if not os.path.exists(info_file):
                    print(f"[WARNING] camera_info.yaml not found at {info_file}")
                elif target_model:
                    print(f"[ERROR] camera_info.yaml has no entry for '{target_model}'. The bracket "
                          f"extrinsics (head_base_to_cam / mount_to_cam) for this model have to be "
                          f"measured and added to camera_info.yaml; setting.yaml is left untouched.")

            self.camera_config = camera_config
            self.markers_config = config_data.get("marker", {}) or {}
            # Legacy fallback
            for lk in ["Tf_to_marker_left", "Tf_to_marker_right", "Tf_to_marker_left_v12", "Tf_to_marker_right_v12", "Tf_to_marker_left_v13", "Tf_to_marker_right_v13"]:
                if lk not in self.markers_config and lk in camera_config:
                    self.markers_config[lk] = camera_config[lk]
            self.marker_detection.markers_config = self.markers_config
            print(f"- Loaded Setting Config from {os.path.basename(setting_config_path)}")
            print(f"  * head_base_to_cam: {camera_config.get('head_base_to_cam')}")
            print(f"  * mount_to_cam: {camera_config.get('mount_to_cam')}")

            self._resolve_camera_intrinsics()
        except Exception as e:
            print(f"- Warning: Could not load {setting_config_path}: {e}")
            self.camera_config = {}
            self.markers_config = {}

    def _resolve_camera_intrinsics(self):
        """Point `active_intrinsics_path` at the intrinsics belonging to the connected model.

        camera_intrinsics.yaml is the working file the detector reads; camera_intrinsics_<family>.yaml
        is each model's saved calibration. When the working file belongs to another model, the
        outgoing one is archived to its own store and this model's store is copied in. With no
        store for this model, `active_intrinsics_path` stays None so the camera's factory
        intrinsics are used -- another model's numbers are far worse than none.
        """
        working = camera_intrinsics_path()
        working_data = {}
        if os.path.exists(working):
            try:
                working_data = ConfigStorage.load(working) or {}
            except Exception as e:
                print(f"[WARNING] Failed to parse camera_intrinsics.yaml: {e}")
        self.calib_device_name = working_data.get("device_name", "")
        working_family = camera_model_family(self.calib_device_name)

        if self.camera_model is None:
            # No camera (simulation, or it failed to open): keep whatever is on file.
            self.active_intrinsics_path = working if working_data else None
            self.intrinsics_source = "file" if working_data else None
            return

        if self._use_serial_intrinsics(working, working_data, working_family):
            return

        if working_family == self.camera_model:
            self.active_intrinsics_path = working if working_data else None
            self.intrinsics_missing = not working_data
            self.intrinsics_source = "model" if working_data else "factory"
            return

        self.intrinsics_mismatch = bool(working_family)
        if working_family:
            print(f"[WARNING] camera_intrinsics.yaml holds '{self.calib_device_name}' intrinsics but a "
                  f"'{self.camera_model}' is connected.")

        store = camera_intrinsics_path(self.camera_model)
        source = store
        if not os.path.exists(source) and getattr(sys, "frozen", False):
            # A frozen build seeds config/ next to the executable from the bundle on first run.
            # If that copy never happened -- a read-only install folder, a partial first run --
            # the model's store is still readable inside the bundle, same fallback camera_info
            # already uses. Without it a packaged build silently drops to factory intrinsics.
            bundled = os.path.join(sys._MEIPASS, "config", os.path.basename(store))
            if os.path.exists(bundled):
                source = bundled
                print(f"[INFO] config/ was not seeded; reading {os.path.basename(store)} from the bundle.")
        store_data = None
        if os.path.exists(source):
            try:
                store_data = ConfigStorage.load(source) or {}
                if "camera_matrix" not in store_data:
                    raise KeyError("camera_matrix")
            except Exception as e:
                store_data = None
                print(f"[ERROR] Failed to read {source}: {e}")

        if store_data is None:
            self.intrinsics_missing = True
            self.active_intrinsics_path = None
            self.intrinsics_source = "factory"
            print(f"[ERROR] No intrinsics calibration for '{self.camera_model}' "
                  f"({os.path.basename(store)} not found or unusable). Falling back to the camera's "
                  f"factory intrinsics -- run the Step 1 intrinsics calibration and save it.")
            return

        # Archive whatever camera_intrinsics.yaml held before replacing it, so swapping back does
        # not lose that calibration. A file with no device_name cannot be attributed to a model --
        # versions before the model stores existed wrote none -- and gating the archive on a known
        # model meant exactly those files were overwritten without a copy. Park them under a
        # timestamp instead; the replacement is stamped with a device_name, so this runs once.
        if working_data:
            outgoing = (camera_intrinsics_path(working_family) if working_family
                        else camera_intrinsics_path(f"unknown_{time.strftime('%Y%m%d_%H%M%S')}"))
            try:
                ConfigStorage.save(outgoing, working_data)
                print(f"[INFO] Archived the previous "
                      f"{working_family or 'unidentified'} intrinsics to {os.path.basename(outgoing)}")
            except Exception as e:
                print(f"[WARNING] Could not archive the previous intrinsics to {outgoing}: {e}")

        store_data["device_name"] = self.camera_model
        self.intrinsics_source = "model"
        try:
            ConfigStorage.save(working, store_data)
            ConfigStorage.save(store, store_data)
            self.active_intrinsics_path = working
            self.intrinsics_mismatch = False
            self.calib_device_name = self.camera_model
            print(f"[INFO] Loaded the '{self.camera_model}' intrinsics from {os.path.basename(store)} "
                  f"into camera_intrinsics.yaml")
        except Exception as e:
            # Still usable read-only even if the working copy could not be written.
            self.active_intrinsics_path = store
            print(f"[WARNING] Could not update camera_intrinsics.yaml ({e}); "
                  f"reading {os.path.basename(store)} directly.")

    def _use_serial_intrinsics(self, working, working_data, working_family):
        """Use this unit's own calibration, config/camera_intrinsics/<serial>.yaml, when it has one
        (or when the working file is already stamped with its serial). Returns False to fall back
        to the per-model store. The outgoing working file is archived first, as for a model swap.
        """
        serial = self.camera_serial
        if not serial:
            return False
        path = camera_intrinsics_serial_path(serial)
        data = None
        if path and os.path.exists(path):
            try:
                data = ConfigStorage.load(path) or {}
                if "camera_matrix" not in data:
                    raise KeyError("camera_matrix")
            except Exception as e:
                data = None
                print(f"[ERROR] Failed to read {path}: {e}")
        if data is None:
            if working_data and str(working_data.get("serial_number") or "") == str(serial) \
                    and "camera_matrix" in working_data:
                data, path = working_data, working
            else:
                print(f"[INFO] No intrinsics calibrated for camera S/N {serial}; using the "
                      f"'{self.camera_model}' model calibration if there is one.")
                return False

        if working_data and path != working:
            other = working_data.get("serial_number")
            if other and str(other) != str(serial):
                outgoing = camera_intrinsics_serial_path(other)
                if outgoing and os.path.exists(outgoing):
                    outgoing = None      # that unit's own file already holds it
            elif not other and working_family != self.camera_model:
                outgoing = (camera_intrinsics_path(working_family) if working_family
                            else camera_intrinsics_path(f"unknown_{time.strftime('%Y%m%d_%H%M%S')}"))
            else:
                outgoing = None          # same model, no serial: the model store keeps it
            if outgoing:
                try:
                    os.makedirs(os.path.dirname(outgoing), exist_ok=True)
                    ConfigStorage.save(outgoing, working_data)
                    print(f"[INFO] Archived the previous intrinsics to {outgoing}")
                except Exception as e:
                    print(f"[WARNING] Could not archive the previous intrinsics to {outgoing}: {e}")

        data = dict(data)
        data["device_name"] = self.camera_model
        data["serial_number"] = str(serial)
        self.active_intrinsics_path = working
        if path != working:
            try:
                ConfigStorage.save(working, data)
            except Exception as e:
                self.active_intrinsics_path = path
                print(f"[WARNING] Could not update camera_intrinsics.yaml ({e}); reading {path} directly.")
        self.calib_device_name = self.camera_model
        self.intrinsics_missing = False
        self.intrinsics_mismatch = False
        self.intrinsics_serial_matched = True
        self.intrinsics_source = "serial"
        print(f"[INFO] Loaded the intrinsics calibrated for camera S/N {serial} ({self.camera_model}).")
        return True

    def reload_saved_intrinsics(self):
        """Re-read the intrinsics files and apply them to the detector, as a reconnect would.

        For a save made while the camera is open: the unit's new S/N file was otherwise not
        recognized (source, registered or not) until the camera was reconnected.
        Returns True when a calibration file was applied."""
        self.intrinsics_missing = False
        self.intrinsics_mismatch = False
        self.intrinsics_serial_matched = False
        self.intrinsics_source = None
        self._resolve_camera_intrinsics()
        return self.apply_calibrated_intrinsics(self.active_intrinsics_path)

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
            # target_tf is in meters. camera_to_marker_tf is also in meters.
            tf_to_marker_inv = np.linalg.inv(target_tf)
        
            if tcpip_send:
                cam_to_tool_tf = camera_to_marker_tf @ tf_to_marker_inv
            else:
                cam_to_tool_tf = camera_to_marker_tf
            cam_to_tool_vec = cam_to_tool_tf.flatten()
            
            if tcpip_send and len(cam_to_tool_vec) > 0:
                self.marker_detection.tcp_client.send_pose(cam_to_tool_vec) 
            return cam_to_tool_vec
        except np.linalg.LinAlgError:
            print("Singular matrix, cannot invert")
            return None

    def get_marker_transform(self, sampling_time=0, side="left", use_filter=None, *, q_encoder=None):
        if not np.isfinite(sampling_time) or sampling_time < 0:
            raise ValueError('sampling_time must be finite and nonnegative')
        if self.sim and self.robot is None and q_encoder is None:
            raise RuntimeError('A connected SDK robot or q_encoder is required for simulated observations')
        if use_filter is None:
            use_filter = (sampling_time == 0)
        lpf = False
        # Collection array for sampling -> dict of lists
        collected_transforms = {} # { marker_id: [tf_vectors...] }
        sampled_temps = []
        start_time = time.monotonic()

        if sampling_time > 0:
            self.marker_detection.prev_pts_dict = {}
            self.marker_detection.prev_pnp_rot = {}
            lpf = True

        while True:
            try:
                if self.sim:
                    if q_encoder is not None:
                        q = np.asarray(q_encoder)
                    elif self.robot is not None:
                        q = np.asarray(self.robot.get_state().position)
                    else:
                        q = np.zeros(26)
                    sides = ['right', 'left'] if side == 'all' else [side]
                    marker_transforms = [
                        ('plate_' + arm, self.simulation_model.marker_pose(self.robot, q, arm, self.rng).ravel())
                        for arm in sides]
                else:
                    if self.camera is None:
                        time.sleep(0.01)
                        if sampling_time == 0: return None
                        continue
                    if not self.camera.camera_monitoring:
                        self.camera.capture_image()
                    color_img = self.camera.get_color_image()
                    depth_img = self.camera.get_depth_image()
                    if color_img is None:
                        time.sleep(0.01)
                        if sampling_time == 0: return None
                        continue
                    
                    raw_transforms = self.marker_detection.detect(color_img, lpf=lpf, depth_image=depth_img, use_filter=use_filter)
                    # Real detector coordinates are millimetres. Normalize to meters at observation boundary.
                    normalized = []
                    for marker_id, values in raw_transforms:
                        transform = np.asarray(values, dtype=float).reshape(4, 4).copy()
                        transform[:3, 3] /= 1000.0
                        normalized.append((marker_id, transform.ravel()))
                    marker_transforms = normalized

                for marker_id_or_group, tf_list in marker_transforms:
                    if marker_id_or_group not in collected_transforms:
                        collected_transforms[marker_id_or_group] = []
                    collected_transforms[marker_id_or_group].append(tf_list)
                
                # Check timeout if sampling
                if sampling_time == 0 or (sampling_time > 0 and (time.monotonic() - start_time >= sampling_time)):
                    break
                        
            except KeyboardInterrupt:
                raise
            
            # Small sleep to reduce CPU utilization
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
                
                avg_cam_to_marker_tf = np.eye(4, dtype=float)
                avg_cam_to_marker_tf[0:3, 0:3] = final_R
                avg_cam_to_marker_tf[0:3, 3] = final_translation
                
                calc_side = "left" if "left" in str(marker_id) else "right"
                
                cam_to_tool_vec = self.calc_cam_to_tool(avg_cam_to_marker_tf, side=calc_side)
                if cam_to_tool_vec is not None:
                    final_results[marker_id] = cam_to_tool_vec
        elif sampling_time == 0:
            for marker_id, tfs in collected_transforms.items():
                camera_to_marker_tf = np.array(tfs[-1], dtype=float).reshape(4, 4)
                
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


def camera_model_family(name):
    """'Intel RealSense D435i' -> 'D435'. Variants of one model (D435i has an IMU, D435f an IR
    filter) share the housing and optics, so they share the mount-to-camera extrinsics; only the
    number identifies the body. D405 and D455 are different housings and stay separate."""
    if not name:
        return None
    match = re.search(r"[Dd](\d{3})", str(name))
    return f"D{match.group(1)}" if match else None


def lookup_camera_info_key(info_data, target_model):
    """Exact entry for this model if camera_info.yaml has one, otherwise the model family.

    Letting the family answer means a D435f (or any future variant) works without its own entry,
    and a variant only needs an entry when its numbers genuinely differ.
    """
    if not target_model or not info_data:
        return None
    for key in info_data:
        if key.lower() == str(target_model).lower():
            return key
    family = camera_model_family(target_model)
    if family:
        for key in info_data:
            if key.lower() == family.lower():
                return key
    return None


MARKER_SIDES = ("left", "right")
# How much closer another version's nominal has to be before the live bracket is declared to
# belong to it. A Step 1 fit moves a bracket by well under a millimetre and a fraction of a
# degree, while the v1.2 and v1.3 brackets sit ~67 mm and 90 deg apart, so this sits in the
# empty space between the two and never mistakes a calibration for a version change.
BRACKET_VERSION_MARGIN_M = 0.005
BRACKET_VERSION_MARGIN_DEG = 2.0


def normalize_robot_version(version):
    """'v1.3' / '1.3' / 1.3 -> '1.3'."""
    return str(version).replace("v", "").strip()


def marker_bracket_nominal_key(side, version):
    """setting.yaml key holding the CAD nominal marker bracket for one robot version."""
    return f"Tf_to_marker_{side}_v{normalize_robot_version(version).replace('.', '')}"


def _is_pose(value):
    return isinstance(value, (list, tuple)) and len(value) >= 6


def _pose_rotation(pose):
    from scipy.spatial.transform import Rotation
    # Matches make_transform: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), i.e. extrinsic xyz.
    return Rotation.from_euler("xyz", [pose[3], pose[4], pose[5]], degrees=True)


def bracket_pose_delta(pose_a, pose_b):
    """How far apart two [x, y, z, roll, pitch, yaw] brackets are, as (metres, degrees)."""
    distance = float(np.linalg.norm(np.asarray(pose_a[:3], dtype=float)
                                    - np.asarray(pose_b[:3], dtype=float)))
    relative = _pose_rotation(pose_a).inv() * _pose_rotation(pose_b)
    return distance, float(np.degrees(np.linalg.norm(relative.as_rotvec())))


def marker_bracket_nominals(markers_config, side):
    """{robot version: nominal bracket} for every `Tf_to_marker_<side>_v<tag>` in the config."""
    nominals = {}
    for key, value in (markers_config or {}).items():
        match = re.fullmatch(rf"Tf_to_marker_{side}_v(\d)(\d+)", str(key))
        if match and _is_pose(value):
            nominals[f"{match.group(1)}.{match.group(2)}"] = list(value)
    return nominals


def bracket_version_of(pose, nominals):
    """Which robot version's bracket `pose` was derived from, or None when it is ambiguous.

    The v1.2 and v1.3 brackets are mounted at a different place and a different angle, so a live
    Tf_to_marker value sits right next to the nominal it came from and nowhere near the other. The
    nearest nominal therefore names the version, as long as it wins by more than a calibration's
    worth of movement -- otherwise this says nothing rather than guessing.
    """
    if not _is_pose(pose) or not nominals:
        return None
    ranked = sorted(((bracket_pose_delta(pose, nominal), version)
                     for version, nominal in nominals.items()),
                    key=lambda item: (item[0][0], item[0][1]))
    (best_pos, best_rot), best_version = ranked[0]
    if len(ranked) == 1:
        return best_version
    (next_pos, next_rot), _ = ranked[1]
    if (next_pos - best_pos) < BRACKET_VERSION_MARGIN_M and (next_rot - best_rot) < BRACKET_VERSION_MARGIN_DEG:
        return None
    return best_version


def resolve_marker_brackets(markers_config, version, log=print):
    """Which live marker brackets belong to another robot version, and what to replace them with.

    `Tf_to_marker_<side>_v12` / `_v13` are the CAD nominals; `Tf_to_marker_<side>` is the live
    value the detector and the calibrators actually use, carrying whatever Step 1 fitted on top of
    one of them. Nothing tied the live value to a robot version, so moving a setting.yaml between
    a v1.2 and a v1.3 robot silently kept a bracket ~67 mm and 90 deg wrong.

    Returns {side: nominal} for the sides that have to be reset; an empty dict means every live
    bracket already belongs to `version`.
    """
    version = normalize_robot_version(version)
    changes = {}
    for side in MARKER_SIDES:
        nominals = marker_bracket_nominals(markers_config, side)
        target = nominals.get(version)
        if target is None:
            continue
        live = (markers_config or {}).get(f"Tf_to_marker_{side}")
        if not _is_pose(live):
            changes[side] = list(target)
            continue
        belongs_to = bracket_version_of(live, nominals)
        if belongs_to == version:
            continue
        if belongs_to is None:
            distance, angle = bracket_pose_delta(live, target)
            log(f"[WARNING] Tf_to_marker_{side} matches no robot version's bracket closely enough "
                f"to tell them apart ({distance * 1000:.1f} mm / {angle:.2f} deg from the v{version} "
                f"nominal); leaving it as it is.")
            continue
        distance, angle = bracket_pose_delta(live, target)
        log(f"[INFO] Tf_to_marker_{side} is the v{belongs_to} bracket but this robot is v{version} "
            f"({distance * 1000:.1f} mm / {angle:.2f} deg apart). Loading the v{version} nominal...")
        changes[side] = list(target)
    return changes
