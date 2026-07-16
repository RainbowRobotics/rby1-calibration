import cv2
import numpy as np
import os
import glob
import yaml

class IntrinsicsCalibrator:
    class BoardPattern:
        NONE = 0
        CHESSBOARD = 1
        CHARUCOBOARD = 2

    CALIB_GUIDELINES = [
        # Center Poses
        {"name": "Center (Flat)", "pts": np.array([[0.30, 0.25], [0.70, 0.25], [0.70, 0.70], [0.30, 0.70]])},
        {"name": "Center (Tilt Up)", "pts": np.array([[0.30, 0.25], [0.70, 0.25], [0.66, 0.70], [0.34, 0.70]])},
        {"name": "Center (Tilt Down)", "pts": np.array([[0.34, 0.25], [0.66, 0.25], [0.70, 0.70], [0.30, 0.70]])},
        {"name": "Center (Tilt Left)", "pts": np.array([[0.30, 0.25], [0.70, 0.30], [0.70, 0.65], [0.30, 0.70]])},
        {"name": "Center (Tilt Right)", "pts": np.array([[0.30, 0.30], [0.70, 0.25], [0.70, 0.70], [0.30, 0.65]])},
        # Four Corners Flat
        {"name": "Top-Left Corner (Flat)", "pts": np.array([[0.05, 0.05], [0.43, 0.05], [0.43, 0.45], [0.05, 0.45]])},
        {"name": "Top-Right Corner (Flat)", "pts": np.array([[0.57, 0.05], [0.95, 0.05], [0.95, 0.45], [0.57, 0.45]])},
        {"name": "Bottom-Left Corner (Flat)", "pts": np.array([[0.05, 0.50], [0.43, 0.50], [0.43, 0.90], [0.05, 0.90]])},
        {"name": "Bottom-Right Corner (Flat)", "pts": np.array([[0.57, 0.50], [0.95, 0.50], [0.95, 0.90], [0.57, 0.90]])},
        # Four Corners Tilted
        {"name": "Top-Left Corner (Tilt Up-Left)", "pts": np.array([[0.05, 0.05], [0.43, 0.08], [0.43, 0.42], [0.05, 0.45]])},
        {"name": "Top-Right Corner (Tilt Up-Right)", "pts": np.array([[0.57, 0.08], [0.95, 0.05], [0.95, 0.45], [0.57, 0.42]])},
        {"name": "Bottom-Left Corner (Tilt Down-Left)", "pts": np.array([[0.05, 0.50], [0.43, 0.53], [0.43, 0.87], [0.05, 0.90]])},
        {"name": "Bottom-Right Corner (Tilt Down-Right)", "pts": np.array([[0.57, 0.53], [0.95, 0.50], [0.95, 0.90], [0.57, 0.87]])},
        # Edge Midpoints Flat
        {"name": "Left Edge (Flat)", "pts": np.array([[0.05, 0.25], [0.43, 0.25], [0.43, 0.70], [0.05, 0.70]])},
        {"name": "Right Edge (Flat)", "pts": np.array([[0.57, 0.25], [0.95, 0.25], [0.95, 0.70], [0.57, 0.70]])},
        {"name": "Top Edge (Flat)", "pts": np.array([[0.30, 0.05], [0.70, 0.05], [0.70, 0.50], [0.30, 0.50]])},
    ]

    def __init__(self):
        self.cameraMatrix = np.eye(3, dtype=np.float64)
        self.distCoeffs = np.zeros((5, 1), dtype=np.float64)
        self.board_size = (0, 0)
        self.pattern = self.BoardPattern.NONE
        self.square_size = 0.0
        self.marker_size = 0.0
        self.aruco_dict = None
        self.charuco_board = None
        self.b_set_board = False
        
        self.use_rational_model = False
        self.camera_matrix_guess = None

        # Results and verification data
        self.rvecs = None
        self.tvecs = None
        self.all_obj_points = []
        self.all_img_points = []
        self.all_ids = []
        self.rms_error = 0.0
        self.std_fx = 0.0
        self.std_fy = 0.0
        self.std_cx = 0.0
        self.std_cy = 0.0
        self.test_rmse = None

    def set_board(self, width, height, pattern, square_size, marker_size, aruco_dict_name):
        self.board_size = (width, height)
        self.pattern = pattern
        self.square_size = square_size
        self.marker_size = marker_size
        
        if pattern == self.BoardPattern.CHARUCOBOARD:
            # Map string dictionary name to cv2.aruco constants
            try:
                dict_attr = getattr(cv2.aruco, aruco_dict_name)
            except AttributeError:
                dict_attr = cv2.aruco.DICT_5X5_100
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
            self.charuco_board = cv2.aruco.CharucoBoard((width, height), square_size, marker_size, self.aruco_dict)
        
        self.b_set_board = True
        return True

    def set_calibration_flags(self, use_rational_model=False):
        self.use_rational_model = use_rational_model

    def set_intrinsic_guess(self, fx, fy, cx, cy):
        self.camera_matrix_guess = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

    def _get_dynamic_win_size(self, corners, img_size):
        if corners is None or len(corners) < 2:
            ws = max(5, int(min(img_size) * 0.01))
            return (ws, ws)
        pts = corners.reshape(-1, 2)
        diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        diffs = diffs[diffs > 1.0] # Ignore duplicate-like points
        if len(diffs) == 0:
            ws = max(5, int(min(img_size) * 0.01))
            return (ws, ws)
        avg_dist = np.mean(diffs)
        # Typically window size is about 1/4 to 1/2 of the distance between corners
        win_size = int(max(5, min(31, avg_dist / 4)))
        return (win_size, win_size)

    def run_calibration(self, image_dir, output_yaml):
        if not self.b_set_board:
            print("Board not set!")
            return False

        image_paths = sorted(glob.glob(os.path.join(image_dir, "calib_*.png")))
        if not image_paths:
            print(f"No images found in {image_dir}")
            return False

        all_obj_points = []
        all_img_points = []
        all_ids = []
        img_size = None

        for path in image_paths:
            img = cv2.imread(path)
            if img is None: continue
            
            if img_size is None:
                img_size = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.pattern == self.BoardPattern.CHESSBOARD:
                ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
                if ret:
                    win_size = self._get_dynamic_win_size(corners, img_size)
                    cv2.cornerSubPix(gray, corners, win_size, (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
                    objp *= self.square_size
                    all_obj_points.append(objp)
                    all_img_points.append(corners)
                    all_ids.append(np.arange(len(corners)))

            elif self.pattern == self.BoardPattern.CHARUCOBOARD:
                dp = cv2.aruco.DetectorParameters()
                dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                detector = cv2.aruco.CharucoDetector(self.charuco_board)
                detector.setDetectorParameters(dp)
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

                if charuco_ids is not None and len(charuco_ids) > 4:
                    win_size = self._get_dynamic_win_size(charuco_corners, img_size)
                    cv2.cornerSubPix(gray, charuco_corners, win_size, (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    all_img_points.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    all_obj_points.append(self.charuco_board.getChessboardCorners()[charuco_ids.flatten()])

        success = self._calibrate_and_validate(all_obj_points, all_img_points, img_size, all_ids)
        if success:
            print(f"Calibration successful! RMS error: {self.rms_error:.4f}")
            if output_yaml is not None:
                self._save_results(output_yaml, img_size[0], img_size[1])
        return success

    def run_calibration_with_images(self, images, output_yaml):
        """
        Runs calibration using a list of images (numpy arrays) instead of reading from disk.
        """
        if not self.b_set_board:
            print("Board not set!")
            return False

        if not images:
            print("No images provided for calibration.")
            return False

        all_obj_points = []
        all_img_points = []
        all_ids = []
        img_size = None

        for i, img in enumerate(images):
            if img is None: continue
            
            if img_size is None:
                img_size = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.pattern == self.BoardPattern.CHESSBOARD:
                ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
                if ret:
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
                    objp *= self.square_size
                    all_obj_points.append(objp)
                    all_img_points.append(corners)
                    all_ids.append(np.arange(len(corners)))

            elif self.pattern == self.BoardPattern.CHARUCOBOARD:
                dp = cv2.aruco.DetectorParameters()
                dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                detector = cv2.aruco.CharucoDetector(self.charuco_board)
                detector.setDetectorParameters(dp)
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

                if charuco_ids is not None and len(charuco_ids) > 4:
                    cv2.cornerSubPix(gray, charuco_corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    all_img_points.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    all_obj_points.append(self.charuco_board.getChessboardCorners()[charuco_ids.flatten()])

        success = self._calibrate_and_validate(all_obj_points, all_img_points, img_size, all_ids)
        if success:
            print(f"Calibration successful! RMS error: {self.rms_error:.4f}")
            if output_yaml is not None:
                self._save_results(output_yaml, img_size[0], img_size[1])
        return success

    def _calibrate_and_validate(self, all_obj_points, all_img_points, img_size, all_ids):
        if len(all_obj_points) < 5:
            print(f"Not enough valid frames for calibration (detected {len(all_obj_points)} valid frames, minimum 5 required).")
            return False

        # Set up calibration flags
        flags = 0
        if self.use_rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL

        # Initialize intrinsic guess
        initial_camera_matrix = np.eye(3, dtype=np.float64)
        if self.camera_matrix_guess is not None:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            initial_camera_matrix = self.camera_matrix_guess.copy()
        else:
            # Fallback guess based on image size
            initial_camera_matrix[0, 2] = img_size[0] / 2.0
            initial_camera_matrix[1, 2] = img_size[1] / 2.0
            focal_guess = max(img_size[0], img_size[1]) * 0.8
            initial_camera_matrix[0, 0] = focal_guess
            initial_camera_matrix[1, 1] = focal_guess
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        dist_len = 8 if self.use_rational_model else 5
        initial_dist_coeffs = np.zeros((dist_len, 1), dtype=np.float64)

        # Iteratively filter outliers (worst-view-first)
        current_obj_points = list(all_obj_points)
        current_img_points = list(all_img_points)
        current_ids = list(all_ids)

        max_iters = 5
        for iter_idx in range(max_iters):
            n_views = len(current_obj_points)
            if n_views < 5:
                break

            try:
                ret, mtx, dist, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors = cv2.calibrateCameraExtended(
                    current_obj_points, current_img_points, img_size, initial_camera_matrix.copy(), initial_dist_coeffs.copy(), flags=flags
                )
                per_view_err = perViewErrors.flatten()
            except Exception as e:
                # Fallback to standard calibrateCamera
                try:
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        current_obj_points, current_img_points, img_size, initial_camera_matrix.copy(), initial_dist_coeffs.copy(), flags=flags
                    )
                    # Manually compute per-view errors since standard calibrateCamera doesn't return them
                    per_view_err = []
                    for obj_pts, img_pts, rvec, tvec in zip(current_obj_points, current_img_points, rvecs, tvecs):
                        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
                        err = np.linalg.norm(img_pts - proj_pts, axis=2)
                        rms = np.sqrt(np.mean(err ** 2))
                        per_view_err.append(rms)
                    per_view_err = np.array(per_view_err)
                    stdIntrinsics = None
                except Exception as inner_e:
                    print(f"Calibration calculation failed at iter {iter_idx}: {inner_e}")
                    return False

            # Update calibration results with current iteration
            self.cameraMatrix = mtx
            self.distCoeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.all_obj_points = current_obj_points
            self.all_img_points = current_img_points
            self.all_ids = current_ids
            self.rms_error = ret

            if stdIntrinsics is not None:
                std_int = stdIntrinsics.flatten()
                self.std_fx = float(std_int[0])
                self.std_fy = float(std_int[1])
                self.std_cx = float(std_int[2])
                self.std_cy = float(std_int[3])
            else:
                self.std_fx = 0.0
                self.std_fy = 0.0
                self.std_cx = 0.0
                self.std_cy = 0.0

            # Find the view with the maximum error
            max_idx = np.argmax(per_view_err)
            max_err = per_view_err[max_idx]
            median_err = np.median(per_view_err)
            outlier_threshold = max(0.35, median_err * 1.5)

            # If the worst view exceeds the outlier threshold, filter it out
            if max_err > outlier_threshold:
                # Ensure we keep a minimum number of views
                min_keep = max(5, min(10, int(len(all_obj_points) * 0.7)))
                if len(current_obj_points) - 1 >= min_keep:
                    print(f"[Iter {iter_idx}] Filtered out worst outlier view (index {max_idx}, error: {max_err:.4f} px). Remaining: {len(current_obj_points) - 1}")
                    current_obj_points.pop(max_idx)
                    current_img_points.pop(max_idx)
                    current_ids.pop(max_idx)
                    continue

            # If no outlier was removed (or min_keep prevents it), stop early
            break

        # Run cross-validation on final inlier points
        self.test_rmse = self.compute_cross_validation_rmse(self.all_obj_points, self.all_img_points, img_size, flags, initial_camera_matrix, initial_dist_coeffs)
        return True

    def compute_cross_validation_rmse(self, all_obj_points, all_img_points, img_size, flags=0, mtx_init=None, dist_init=None):
        if len(all_obj_points) < 6:
            return None
            
        # Determinisitic split: test set is every 4th frame (indices 3, 7, etc.)
        test_indices = list(range(3, len(all_obj_points), 4))
        train_indices = [i for i in range(len(all_obj_points)) if i not in test_indices]
        
        train_obj = [all_obj_points[i] for i in train_indices]
        train_img = [all_img_points[i] for i in train_indices]
        
        test_obj = [all_obj_points[i] for i in test_indices]
        test_img = [all_img_points[i] for i in test_indices]
        
        # Calibrate on train set
        if mtx_init is None:
            mtx_init = np.eye(3, dtype=np.float64)
        else:
            mtx_init = mtx_init.copy()
            
        if dist_init is None:
            dist_init = np.zeros((5, 1), dtype=np.float64)
        else:
            dist_init = dist_init.copy()
            
        try:
            ret_t, mtx_t, dist_t, rvecs_t, tvecs_t = cv2.calibrateCamera(
                train_obj, train_img, img_size, mtx_init, dist_init, flags=flags
            )
            
            # Evaluate on test set
            test_errors = []
            for obj_pts, img_pts in zip(test_obj, test_img):
                # Solve PnP to find test view extrinsics
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx_t, dist_t)
                if ok:
                    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx_t, dist_t)
                    err = np.linalg.norm(img_pts - proj_pts, axis=2)
                    test_errors.extend(err.flatten())
            
            if test_errors:
                return float(np.sqrt(np.mean(np.array(test_errors)**2)))
        except Exception:
            pass
        return None

    def _save_results(self, output_yaml, width, height):
        data = {
            "width": int(width),
            "height": int(height),
            "camera_matrix": self.cameraMatrix.tolist(),
            "dist_coeffs": self.distCoeffs.flatten().tolist()
        }
        with open(output_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Results saved to {output_yaml} ({width}x{height})")

    def generate_verification_image(self, test_img, save_path):
        """
        Generates an undistorted side-by-side comparison with grids and saves it.
        """
        if test_img is None:
            return False
            
        h, w = test_img.shape[:2]
        
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(test_img, self.cameraMatrix, self.distCoeffs, None, new_mtx)
        
        combined_res = np.vstack((test_img, undistorted))
        h_res, w_res = combined_res.shape[:2]

        grid_size = 60
        for y in range(0, h_res, grid_size):
            cv2.line(combined_res, (0, y), (w_res, y), (0, 255, 0), 1)
        for x in range(0, w_res, grid_size):
            cv2.line(combined_res, (x, 0), (x, h_res), (0, 255, 0), 1)

        cv2.putText(combined_res, "Original", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(combined_res, "Undistorted", (30, h + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(combined_res, f"RMS Error: {self.rms_error:.4f}", (30, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, combined_res)
        return True
