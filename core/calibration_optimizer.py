import math

import numpy as np

try:
    import qpsolvers
except ImportError:
    qpsolvers = None

from core.sag_compensation import SagEstimator


DEFAULT_LAMBDA_CAM_POS = 1.0
DEFAULT_LAMBDA_CAM_ROT = 1.0
DEFAULT_ESTIMATE_MEASUREMENT_NOISE = False
DEFAULT_NOISE_UPDATE_RATE = 0.5
DEFAULT_INITIAL_NOISE_STD_ROT_RAD = np.deg2rad(0.5)
DEFAULT_INITIAL_NOISE_STD_POS_M = 5e-4
DEFAULT_MIN_NOISE_STD_ROT_RAD = np.deg2rad(0.01)
DEFAULT_MIN_NOISE_STD_POS_M = 1e-5
DEFAULT_MAX_NOISE_STD_ROT_RAD = np.deg2rad(2.0)
DEFAULT_MAX_NOISE_STD_POS_M = 1e-3

ARM_SIDES = ("right", "left")
ARM_OPTIMIZATION_NDOF = {14, 16, 20, 22}
HEAD_OPTIMIZATION_NDOF = {2, 16, 22}
CAMERA_OPTIMIZATION_NDOF = {6, 20, 22}

D2R = np.pi / 180.0


class ResidualNoiseEstimator:
    def __init__(
        self,
        enabled=DEFAULT_ESTIMATE_MEASUREMENT_NOISE,
        update_rate=DEFAULT_NOISE_UPDATE_RATE,
        initial_rot_std_rad=DEFAULT_INITIAL_NOISE_STD_ROT_RAD,
        initial_pos_std_m=DEFAULT_INITIAL_NOISE_STD_POS_M,
        min_rot_std_rad=DEFAULT_MIN_NOISE_STD_ROT_RAD,
        min_pos_std_m=DEFAULT_MIN_NOISE_STD_POS_M,
        max_rot_std_rad=DEFAULT_MAX_NOISE_STD_ROT_RAD,
        max_pos_std_m=DEFAULT_MAX_NOISE_STD_POS_M,
    ):
        self.enabled = bool(enabled)
        self.update_rate = float(np.clip(update_rate, 0.0, 1.0))
        self.min_rot_std_rad = float(min_rot_std_rad)
        self.min_pos_std_m = float(min_pos_std_m)
        self.max_rot_std_rad = float(max_rot_std_rad)
        self.max_pos_std_m = float(max_pos_std_m)
        self.rot_std_rad = self._clamp_rot(initial_rot_std_rad)
        self.pos_std_m = self._clamp_pos(initial_pos_std_m)

    def _clamp_rot(self, value):
        return float(np.clip(value, self.min_rot_std_rad, self.max_rot_std_rad))

    def _clamp_pos(self, value):
        return float(np.clip(value, self.min_pos_std_m, self.max_pos_std_m))

    def weights(self):
        if not self.enabled:
            return np.ones(6, dtype=np.float64)

        rot_std = self._clamp_rot(self.rot_std_rad)
        pos_std = self._clamp_pos(self.pos_std_m)
        return np.array([
            1.0 / rot_std,
            1.0 / rot_std,
            1.0 / rot_std,
            1.0 / pos_std,
            1.0 / pos_std,
            1.0 / pos_std,
        ], dtype=np.float64)

    def update(self, residuals):
        if not self.enabled or not residuals:
            return

        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1, 6)
        rot_std = np.sqrt(np.mean(residuals[:, :3] ** 2))
        pos_std = np.sqrt(np.mean(residuals[:, 3:] ** 2))

        rot_std = self._clamp_rot(rot_std)
        pos_std = self._clamp_pos(pos_std)
        alpha = self.update_rate
        self.rot_std_rad = self._clamp_rot(
            (1.0 - alpha) * self.rot_std_rad + alpha * rot_std
        )
        self.pos_std_m = self._clamp_pos(
            (1.0 - alpha) * self.pos_std_m + alpha * pos_std
        )

    def format(self):
        if not self.enabled:
            return "noise=off"

        return (
            f"sigma_rot={np.rad2deg(self.rot_std_rad):.4g}deg, "
            f"sigma_pos={self.pos_std_m * 1e3:.4g}mm"
        )

    def as_dict(self):
        return {
            "measurement_noise_enabled": self.enabled,
            "measurement_noise_rot_std_rad": float(self.rot_std_rad),
            "measurement_noise_rot_std_deg": float(np.rad2deg(self.rot_std_rad)),
            "measurement_noise_pos_std_m": float(self.pos_std_m),
            "measurement_noise_pos_std_mm": float(self.pos_std_m * 1e3),
            "measurement_noise_max_rot_std_rad": float(self.max_rot_std_rad),
            "measurement_noise_max_rot_std_deg": float(np.rad2deg(self.max_rot_std_rad)),
            "measurement_noise_max_pos_std_m": float(self.max_pos_std_m),
            "measurement_noise_max_pos_std_mm": float(self.max_pos_std_m * 1e3),
        }


def add_weighted_normal_equation(H, g, J, xi, weights):
    J_weighted = J * weights[:, None]
    xi_weighted = xi * weights
    H += J_weighted.T @ J_weighted
    g += J_weighted.T @ xi_weighted


def adjoint(T):
    R = T[:3, :3]
    p = T[:3, 3]
    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0],
    ])
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = p_hat @ R
    return Ad


def make_transform(data):
    # data: [x, y, z, roll, pitch, yaw] (xyz: m, rpy: deg)
    x, y, z = data[:3]
    roll, pitch, yaw = np.deg2rad(data[3:])

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    T = np.eye(4, dtype=np.float64)
    T[0, 0] = cy * cp
    T[0, 1] = sr * sp * cy - cr * sy
    T[0, 2] = cr * sp * cy + sr * sy
    T[0, 3] = x

    T[1, 0] = sy * cp
    T[1, 1] = sr * sp * sy + cr * cy
    T[1, 2] = cr * sp * sy - sr * cy
    T[1, 3] = y

    T[2, 0] = -sp
    T[2, 1] = cp * sr
    T[2, 2] = cp * cr
    T[2, 3] = z
    return T


def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)

    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def se3_exp(xi):
    w = xi[:3]
    v = xi[3:]
    R = so3_exp(w)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        V = np.eye(3)
    else:
        K = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0],
        ]) / theta

        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K)
        )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def so3_log(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    w_hat = (R - R.T) / (2 * np.sin(theta))
    return theta * np.array([
        w_hat[2, 1],
        w_hat[0, 2],
        w_hat[1, 0],
    ])


def se3_log(T):
    R = T[:3, :3]
    t = T[:3, 3]

    w = so3_log(R)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        v = t
    else:
        w_hat = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0],
        ]) / theta

        A = (
            np.eye(3)
            - 0.5 * w_hat
            + (1 / theta**2)
            * (1 - theta / (2 * np.tan(theta / 2)))
            * (w_hat @ w_hat)
        )
        v = A @ t

    return np.hstack([w, v])


def rot_to_euler_zyx(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw])


def compute_fk(robot, dyn_model, q_full, ee_link, base_link="link_torso_5"):
    state = dyn_model.make_state(
        [base_link, ee_link],
        robot.model().robot_joint_names,
    )
    state.set_q(q_full)
    dyn_model.compute_forward_kinematics(state)
    return state, dyn_model.compute_transformation(state, 0, 1)


def prepare_q_full(
    q_nominal,
    arm_idx,
    q_cmd,
    q_offset=None,
    head_idx=None,
    q_head=None,
    q_head_offset=None,
):
    q_full = q_nominal.copy()
    q_full[arm_idx] = q_cmd if q_offset is None else (q_cmd + q_offset)
    if head_idx is not None and q_head is not None:
        q_full[head_idx] = q_head if q_head_offset is None else (q_head + q_head_offset)
    return q_full


class QPCalibrationOptimizer:
    def __init__(
        self,
        robot,
        arm_idx,
        ee_links,
        mount_to_cam_nom,
        head_base_to_cam_nom=None,
        ee_to_marker_nom=None,
        head_idx=None,
        camera_link="link_head_2",
        max_iter=500,
        eps=1e-7,
        lambda_cam_pos=DEFAULT_LAMBDA_CAM_POS,
        lambda_cam_rot=DEFAULT_LAMBDA_CAM_ROT,
        qp_solver="osqp",
        qp_regularization=1e-9,
        qp_kwargs=None,
        enforce_joint_offset_limits=True,
        joint_step_bound_rad=None,
        joint_offset_bound_rad=None,
        camera_rot_step_bound_rad=None,
        camera_pos_step_bound_m=None,
        camera_rot_bound_rad=None,
        camera_pos_bound_m=None,
        use_sag=False,
        optimize_arm=True, 
        optimize_head=False,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=DEFAULT_ESTIMATE_MEASUREMENT_NOISE,
        measurement_noise_update_rate=DEFAULT_NOISE_UPDATE_RATE,
        apply_joint_offset_limits=False,
        joint_offsets_to_apply=None,
    ):
        self.robot = robot
        self.dyn_model = robot.get_dynamics()
        self.model = robot.model()
        self.use_sag = use_sag
        self.active_arms = active_arms
        self.apply_joint_offset_limits = apply_joint_offset_limits
        self.joint_offsets_to_apply = joint_offsets_to_apply

        self.arm_idx = np.array(arm_idx, dtype=int)
        self.head_idx = np.array(head_idx, dtype=int) if head_idx is not None else None
        self.ee_links = dict(ee_links)
        self.mount_to_cam_nom = mount_to_cam_nom
        self.head_base_to_cam_nom = head_base_to_cam_nom
        self.ee_to_marker_nom = dict(ee_to_marker_nom)
        self.camera_link = camera_link

        self.use_head_kinematics = (self.head_idx is not None) and optimize_head
        self.optimize_arm = optimize_arm
        self.optimize_head = optimize_head
        self.optimize_camera = optimize_camera

        self.max_iter = max_iter
        self.eps = eps
        self.lambda_cam_pos = lambda_cam_pos
        self.lambda_cam_rot = lambda_cam_rot
        self.noise_estimator = ResidualNoiseEstimator(
            enabled=estimate_measurement_noise,
            update_rate=measurement_noise_update_rate,
        )
        self.q_nominal = robot.get_state().position.copy()
        
        if self.use_head_kinematics:
            self.base_link = self.camera_link
            self.T_mount_to_cam_nom = make_transform(self.mount_to_cam_nom) if self.mount_to_cam_nom else np.eye(4)
        else:
            self.base_link = "link_head_0"
            self.T_mount_to_cam_nom = make_transform(self.head_base_to_cam_nom) if self.head_base_to_cam_nom else np.eye(4)

        # Sag Estimators
        self.sag_estimators = {
            "right": SagEstimator(robot, self.arm_idx, self.ee_links["right"]),
            "left": SagEstimator(robot, self.arm_idx, self.ee_links["left"]),
        }
        self.numeric_jac_eps = 1e-7

        self.q_lower, self.q_upper = self.get_joint_limit()
        self.joint_offset_lower, self.joint_offset_upper = self.get_joint_offset_limits()

        self.qp_solver = qp_solver
        self.qp_regularization = float(qp_regularization)
        self.qp_kwargs = {} if qp_kwargs is None else dict(qp_kwargs)
        self.enforce_joint_offset_limits = enforce_joint_offset_limits
        self.joint_step_bound_rad = joint_step_bound_rad
        self.joint_offset_bound_rad = joint_offset_bound_rad
        self.camera_rot_step_bound_rad = camera_rot_step_bound_rad
        self.camera_pos_step_bound_m = camera_pos_step_bound_m
        self.camera_rot_bound_rad = camera_rot_bound_rad
        self.camera_pos_bound_m = camera_pos_bound_m

    def get_joint_limit(self):
        links = [self.base_link] + list(self.ee_links.values())
        state = self.dyn_model.make_state(links, self.model.robot_joint_names)
        q_lower = self.dyn_model.get_limit_q_lower(state)
        q_upper = self.dyn_model.get_limit_q_upper(state)
        q_lower = np.asarray(q_lower, dtype=np.float64).reshape(-1)
        q_upper = np.asarray(q_upper, dtype=np.float64).reshape(-1)
        return q_lower, q_upper

    def get_joint_offset_limits(self):
        lower_parts = []
        upper_parts = []
        if self.optimize_arm:
            urdf_q_lower_arm = self.q_lower[self.arm_idx]
            q_lower = urdf_q_lower_arm.copy()
            q_lower[ 1] =  0.0 * D2R # rsr
            # q_lower[ 5] =  0.0 * D2R # rsr
            q_lower[ 8] =-10.0 * D2R # lsr
            q_lower[ 3] =  0.0 * D2R # rep
            q_lower[10] =  0.0 * D2R # le[]
            # q_lower[12] =  0.0 * D2R # le[]

            urdf_q_upper_arm = self.q_upper[self.arm_idx]
            q_upper = urdf_q_upper_arm.copy()
            q_upper[ 1] = 10.0 * D2R# rsr
            # q_upper[ 5] = 0.1 * D2R# rsr
            q_upper[ 8] =  0.0 * D2R# lsr
            q_upper[ 3] =  2.0 * D2R# rep
            q_upper[10] =  2.0 * D2R# lep
            # q_upper[12] =  0.1 * D2R# lep

            if getattr(self, 'apply_joint_offset_limits', False) and getattr(self, 'joint_offsets_to_apply', None) is not None:
                jo = self.joint_offsets_to_apply  
                r_j3 = jo.get("right", {}).get("joint3", 0.0)
                l_j3 = jo.get("left", {}).get("joint3", 0.0)

                v1_r3 = (-r_j3 - 0.001) * D2R
                v2_r3 = (-r_j3 + 0.001) * D2R
                q_lower[3] = min(v1_r3, v2_r3)
                q_upper[3] = max(v1_r3, v2_r3)

                v1_l3 = (-l_j3 - 0.001) * D2R
                v2_l3 = (-l_j3 + 0.001) * D2R
                q_lower[10] = min(v1_l3, v2_l3)
                q_upper[10] = max(v1_l3, v2_l3)
                # In 1.2, Joint 5 is calibrated
                r_j5 = jo.get("right", {}).get("joint5", 0.0)
                l_j5 = jo.get("left", {}).get("joint5", 0.0)
                    
                v1_r5 = (-r_j5 - 0.001) * D2R
                v2_r5 = (-r_j5 + 0.001) * D2R
                q_lower[5] = min(v1_r5, v2_r5)
                q_upper[5] = max(v1_r5, v2_r5)

                v1_l5 = (-l_j5 - 0.001) * D2R
                v2_l5 = (-l_j5 + 0.001) * D2R
                q_lower[12] = min(v1_l5, v2_l5)
                q_upper[12] = max(v1_l5, v2_l5)

                # Joint 6 오프셋 제한 범위 추가 (오른쪽: index 6, 왼쪽: index 13)
                r_j6 = jo.get("right", {}).get("joint6", 0.0)
                l_j6 = jo.get("left", {}).get("joint6", 0.0)

                v1_r6 = (-r_j6 - 0.001) * D2R
                v2_r6 = (-r_j6 + 0.001) * D2R

                q_lower[6] = min(v1_r6, v2_r6)
                q_upper[6] = max(v1_r6, v2_r6)

                v1_l6 = (-l_j6 - 0.001) * D2R
                v2_l6 = (-l_j6 + 0.001) * D2R

                q_lower[13] = min(v1_l6, v2_l6)
                q_upper[13] = max(v1_l6, v2_l6)

            lower_parts.append(q_lower)
            upper_parts.append(q_upper)

        if self.optimize_head:
            lower_parts.append(self.q_lower[self.head_idx])
            upper_parts.append(self.q_upper[self.head_idx])

        if not lower_parts:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        return np.concatenate(lower_parts), np.concatenate(upper_parts)

    def joint_param_dim(self):
        dim = 0
        if self.optimize_arm:
            dim += len(self.arm_idx)
        if self.optimize_head:
            dim += len(self.head_idx)
        return dim

    def total_dim(self):
        dim = self.joint_param_dim()
        if self.optimize_camera:
            dim += 6
        return dim

    def get_nominal_mount_to_cam(self):
        return self.T_mount_to_cam_nom.copy()

    def get_nominal_ee_to_marker(self, arm_side):
        return make_transform(self.ee_to_marker_nom[str(arm_side)])

    def unpack_params(self, dx):
        cursor = 0
        q_arm_offset = np.zeros(len(self.arm_idx))
        q_head_offset = np.zeros(len(self.head_idx)) if self.head_idx is not None else None
        xi_mount_cam = np.zeros(6)

        if self.optimize_arm:
            q_arm_offset = dx[cursor:cursor + len(self.arm_idx)]
            cursor += len(self.arm_idx)
        if self.optimize_head:
            q_head_offset = dx[cursor:cursor + len(self.head_idx)]
            cursor += len(self.head_idx)
        if self.optimize_camera:
            xi_mount_cam = dx[cursor:cursor + 6]

        return q_arm_offset, q_head_offset, xi_mount_cam

    def pack_joint_jacobian(self, Jb):
        parts = []
        if self.optimize_arm:
            parts.append(Jb[:, self.arm_idx])
        if self.optimize_head:
            parts.append(Jb[:, self.head_idx])
        if not parts:
            return np.zeros((6, 0))
        return np.concatenate(parts, axis=1)

    def evaluate_sample(
        self,
        q_arm,
        q_head,
        arm_side,
        q_arm_offset,
        q_head_offset,
        xi_mount_cam,
    ):
        q_full = prepare_q_full(
            q_nominal=self.q_nominal,
            arm_idx=self.arm_idx,
            q_cmd=q_arm,
            q_offset=q_arm_offset if self.optimize_arm else None,
            head_idx=self.head_idx,
            q_head=q_head,
            q_head_offset=q_head_offset if self.optimize_head else None,
        )

        state = self.dyn_model.make_state(
            [self.base_link, self.ee_links[str(arm_side)]],
            self.model.robot_joint_names,
        )

        state.set_q(q_full)

        if self.use_sag:
            q_full_sagged, _ = self.sag_estimators[str(arm_side)].get_sagged_joints(q_full)
            state.set_q(q_full_sagged)

        self.dyn_model.compute_forward_kinematics(state)
        self.dyn_model.compute_diff_forward_kinematics(state)

        T_fk = self.dyn_model.compute_transformation(state, 0, 1)
        Jb_full = self.dyn_model.compute_body_jacobian(state, 0, 1)
        Jb_joint = self.pack_joint_jacobian(Jb_full)

        T_mount_to_cam_nom = self.get_nominal_mount_to_cam()
        T_mount_to_cam = (
            T_mount_to_cam_nom @ se3_exp(xi_mount_cam)
            if self.optimize_camera else T_mount_to_cam_nom
        )
        T_ee_to_marker = self.get_nominal_ee_to_marker(arm_side)

        T_model = np.linalg.inv(T_mount_to_cam) @ T_fk @ T_ee_to_marker
        return Jb_joint, T_mount_to_cam, T_ee_to_marker, T_model

    def build_camera_jacobian_numeric(
        self,
        q_arm,
        q_head,
        arm_side,
        q_arm_offset,
        q_head_offset,
        xi_mount_cam,
        T_model_ref,
    ):
        J_cam = np.zeros((6, 6))

        for i in range(6):
            delta = np.zeros(6)
            delta[i] = self.numeric_jac_eps

            _, _, _, T_model_plus = self.evaluate_sample(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam + delta,
            )
            _, _, _, T_model_minus = self.evaluate_sample(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam - delta,
            )

            xi_plus = se3_log(np.linalg.inv(T_model_ref) @ T_model_plus)
            xi_minus = se3_log(np.linalg.inv(T_model_ref) @ T_model_minus)
            J_cam[:, i] = (xi_plus - xi_minus) / (2 * self.numeric_jac_eps)

        return J_cam

    def build_jacobian(
        self,
        q_arm,
        q_head,
        arm_side,
        q_arm_offset,
        q_head_offset,
        xi_mount_cam,
        Jb_joint,
        T_ee_to_marker,
        T_model,
    ):
        joint_dim = self.joint_param_dim()
        if joint_dim > 0:
            J_joint = adjoint(np.linalg.inv(T_ee_to_marker)) @ Jb_joint
        else:
            J_joint = np.zeros((6, 0))

        if joint_dim > 0 and self.optimize_camera:
            J = np.zeros((6, joint_dim + 6))
            J[:, :joint_dim] = J_joint
            J[:, joint_dim:] = self.build_camera_jacobian_numeric(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                T_model,
            )
            return J

        if self.optimize_camera:
            return self.build_camera_jacobian_numeric(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                T_model,
            )

        return J_joint

    @staticmethod
    def _as_bound_array(bound, size):
        if bound is None:
            return None
        arr = np.asarray(bound, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(size, float(arr), dtype=np.float64)
        arr = arr.reshape(-1)
        if arr.size != size:
            raise ValueError(f"Expected bound size {size}, got {arr.size}")
        return arr

    def _apply_step_bound(self, lb, ub, slc, bound):
        if bound is None:
            return
        size = slc.stop - slc.start
        bound_arr = self._as_bound_array(bound, size)
        lb[slc] = np.maximum(lb[slc], -bound_arr)
        ub[slc] = np.minimum(ub[slc], bound_arr)

    def _apply_absolute_bound(self, lb, ub, slc, current_value, bound):
        if bound is None:
            return
        size = slc.stop - slc.start
        bound_arr = self._as_bound_array(bound, size)
        current_value = np.asarray(current_value, dtype=np.float64).reshape(-1)
        if current_value.size != size:
            raise ValueError(
                f"Expected current value size {size}, got {current_value.size}"
            )
        lb[slc] = np.maximum(lb[slc], -bound_arr - current_value)
        ub[slc] = np.minimum(ub[slc], bound_arr - current_value)

    def _apply_absolute_limits(self, lb, ub, slc, current_value, lower, upper):
        size = slc.stop - slc.start
        current_value = np.asarray(current_value, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)

        if current_value.size != size:
            raise ValueError(
                f"Expected current value size {size}, got {current_value.size}"
            )
        if lower.size != size or upper.size != size:
            raise ValueError(
                f"Expected limit size {size}, got lower={lower.size}, upper={upper.size}"
            )

        lb[slc] = np.maximum(lb[slc], lower - current_value)
        ub[slc] = np.minimum(ub[slc], upper - current_value)

    def _build_qp_bounds(self, dim, q_arm_offset, q_head_offset, xi_mount_cam):
        lb = np.full(dim, -np.inf, dtype=np.float64)
        ub = np.full(dim, np.inf, dtype=np.float64)
        any_bound = False

        joint_dim = self.joint_param_dim()
        if joint_dim > 0:
            joint_slice = slice(0, joint_dim)
            joint_current = []
            if self.optimize_arm:
                joint_current.append(q_arm_offset)
            if self.optimize_head and q_head_offset is not None:
                joint_current.append(q_head_offset)
            joint_current = np.concatenate(joint_current)
            
            # total_current = q_nominal + q_offset
            total_current = joint_current #q_nom_subset + joint_current

            # if self.joint_step_bound_rad is not None:
                # self._apply_step_bound(lb, ub, joint_slice, self.joint_step_bound_rad)
                # any_bound = True
            # if self.joint_offset_bound_rad is not None:
                # self._apply_absolute_bound(
                    # lb, ub, joint_slice, joint_current, self.joint_offset_bound_rad
                # )
                # any_bound = True
            if self.enforce_joint_offset_limits:
                self._apply_absolute_limits(
                    lb,
                    ub,
                    joint_slice,
                    joint_current,
                    self.joint_offset_lower,
                    self.joint_offset_upper,
                )
                any_bound = True

        if self.optimize_camera:
            rot_slice = slice(dim - 6, dim - 3)
            pos_slice = slice(dim - 3, dim)
            if self.camera_rot_step_bound_rad is not None:
                self._apply_step_bound(lb, ub, rot_slice, self.camera_rot_step_bound_rad)
                any_bound = True
            if self.camera_pos_step_bound_m is not None:
                self._apply_step_bound(lb, ub, pos_slice, self.camera_pos_step_bound_m)
                any_bound = True
            if self.camera_rot_bound_rad is not None:
                self._apply_absolute_bound(
                    lb, ub, rot_slice, xi_mount_cam[:3], self.camera_rot_bound_rad
                )
                any_bound = True
            if self.camera_pos_bound_m is not None:
                self._apply_absolute_bound(
                    lb, ub, pos_slice, xi_mount_cam[3:], self.camera_pos_bound_m
                )
                any_bound = True

        if not any_bound:
            return None, None
        if np.any(lb > ub):
            raise RuntimeError("QP bounds are infeasible for this iteration.")
        return lb, ub

    def compute_step(
        self,
        q_arm_list,
        q_head_list,
        T_meas_list,
        q_arm_offset,
        q_head_offset,
        xi_mount_cam,
    ):
        if qpsolvers is None:
            raise RuntimeError(
                "qpsolvers is required for QPCalibrationOptimizer. "
                "Install it in the runtime environment first."
            )

        dim = self.total_dim()
        H = np.zeros((dim, dim), dtype=np.float64)
        g = np.zeros(dim, dtype=np.float64)
        total_err = 0.0
        weights = self.noise_estimator.weights()
        residual_samples = []

        if q_head_list is None:
            q_head_iter = [None] * len(q_arm_list)
        else:
            q_head_iter = q_head_list

        for q_arm, q_head, T_meas_pair in zip(q_arm_list, q_head_iter, T_meas_list):
            for side_idx, arm_side in enumerate(ARM_SIDES):
                if arm_side not in self.active_arms:
                    continue

                # Handle both single-arm (N, 4, 4) and dual-arm (N, 2, 4, 4) data shapes
                if T_meas_pair.ndim == 3:
                    if T_meas_pair.shape == (2, 4, 4):
                        T_meas = T_meas_pair[side_idx]
                    else:
                        T_meas = T_meas_pair
                elif T_meas_pair.ndim == 2:
                    T_meas = T_meas_pair
                else:
                    T_meas = T_meas_pair[side_idx]

                Jb_joint, _, T_ee_to_marker, T_model = self.evaluate_sample(
                    q_arm,
                    q_head,
                    arm_side,
                    q_arm_offset,
                    q_head_offset,
                    xi_mount_cam,
                )

                T_err = np.linalg.inv(T_model) @ T_meas
                xi = se3_log(T_err)
                J = self.build_jacobian(
                    q_arm,
                    q_head,
                    arm_side,
                    q_arm_offset,
                    q_head_offset,
                    xi_mount_cam,
                    Jb_joint,
                    T_ee_to_marker,
                    T_model,
                )

                add_weighted_normal_equation(H, g, J, xi, weights)
                total_err += np.linalg.norm(xi)
                residual_samples.append(xi)

        self.noise_estimator.update(residual_samples)

        if self.optimize_camera:
            rot_slice = slice(dim - 6, dim - 3)
            pos_slice = slice(dim - 3, dim)

            if self.lambda_cam_rot > 0.0:
                H[rot_slice, rot_slice] += self.lambda_cam_rot * np.eye(3)
                g[rot_slice] += -self.lambda_cam_rot * xi_mount_cam[:3]

            if self.lambda_cam_pos > 0.0:
                H[pos_slice, pos_slice] += self.lambda_cam_pos * np.eye(3)
                g[pos_slice] += -self.lambda_cam_pos * xi_mount_cam[3:]



        P = 0.5 * (H + H.T)
        if self.qp_regularization > 0.0:
            P += self.qp_regularization * np.eye(dim)
        q = -g
        lb, ub = self._build_qp_bounds(dim, q_arm_offset, q_head_offset, xi_mount_cam)

        import scipy.sparse as spa
        dx = qpsolvers.solve_qp(
            spa.csc_matrix(P),
            q,
            lb=lb,
            ub=ub,
            solver=self.qp_solver,
            #eps_abs=1e-8,
            #eps_rel=1e-8,
            **self.qp_kwargs,
        )
        if dx is None:
            raise RuntimeError(f"QP solver '{self.qp_solver}' failed to find a solution.")

        return np.asarray(dx, dtype=np.float64).reshape(-1), total_err

    def apply_update(self, q_arm_offset, q_head_offset, xi_mount_cam, dx):
        dq_arm, dq_head, dxi = self.unpack_params(dx)
        if self.optimize_arm:
            q_arm_offset += dq_arm
        if self.optimize_head and q_head_offset is not None:
            q_head_offset += dq_head
        if self.optimize_camera:
            xi_mount_cam += dxi
        return q_arm_offset, q_head_offset, xi_mount_cam

    def get_calibrated_head_base_to_cam(self, xi_mount_cam):
        T_mount_to_cam = self.get_nominal_mount_to_cam() @ se3_exp(xi_mount_cam)
        if self.use_head_kinematics:
            _, T_head_base_to_mount = compute_fk(
                robot=self.robot,
                dyn_model=self.dyn_model,
                q_full=self.q_nominal,
                ee_link=self.camera_link,
                base_link="link_head_0",
            )
            T_calib = T_head_base_to_mount @ T_mount_to_cam
        else:
            T_calib = T_mount_to_cam

        p = T_calib[:3, 3]
        rpy = rot_to_euler_zyx(T_calib[:3, :3])

        return [
            p[0],
            p[1],
            p[2],
            np.rad2deg(rpy[0]),
            np.rad2deg(rpy[1]),
            np.rad2deg(rpy[2]),
        ]

    def get_calibrated_mount_to_cam(self, xi_mount_cam):
        T_mount_to_cam = self.get_nominal_mount_to_cam() @ se3_exp(xi_mount_cam)

        p = T_mount_to_cam[:3, 3]
        rpy = rot_to_euler_zyx(T_mount_to_cam[:3, :3])

        return [
            p[0],
            p[1],
            p[2],
            np.rad2deg(rpy[0]),
            np.rad2deg(rpy[1]),
            np.rad2deg(rpy[2]),
        ]

    def optimize(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset_init=None, q_head_offset_init=None, xi_mount_cam_init=None):
        if self.use_head_kinematics and q_head_list is None:
            raise RuntimeError(
                "Head kinematics are enabled for this ndof, but q_head_list is missing."
            )

        q_arm_offset = q_arm_offset_init.copy() if q_arm_offset_init is not None else np.zeros(len(self.arm_idx))
        if self.optimize_head:
            q_head_offset = q_head_offset_init.copy() if q_head_offset_init is not None else np.zeros(len(self.head_idx))
        else:
            q_head_offset = q_head_offset_init.copy() if q_head_offset_init is not None else None
            
        xi_mount_cam = xi_mount_cam_init.copy() if xi_mount_cam_init is not None else np.zeros(6)

        for it in range(self.max_iter):
            dx, total_err = self.compute_step(
                q_arm_list,
                q_head_list,
                T_meas_list,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
            )
            q_arm_offset, q_head_offset, xi_mount_cam = self.apply_update(
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                dx,
            )

            # Split total_err into translation (m) and rotation (rad) components for better debugging
            print(f"[{it:02d}] |dx|={np.linalg.norm(dx):.3e}, |err|={total_err:.3e}")

            if np.linalg.norm(dx) < self.eps:
                print("Converged.")
                break

        mount_to_cam_new = self.get_calibrated_mount_to_cam(xi_mount_cam)
        head_base_to_cam_new = self.get_calibrated_head_base_to_cam(xi_mount_cam)
        return q_arm_offset, q_head_offset, xi_mount_cam, mount_to_cam_new, head_base_to_cam_new

class CalibrationOptimizer:
    def __init__(
        self,
        robot,
        arm_idx,
        ee_links,
        mount_to_cam_nom,
        head_base_to_cam_nom,
        ee_to_marker_nom,
        active_arms=["right", "left"],
        optimize_arm=True,
        optimize_head=False,
        optimize_camera=False,
        head_idx=None,
        camera_link="link_head_2",
        use_head_kinematics=True,
        max_iter=500,
        eps=1e-6,
        lambda_cam_pos=DEFAULT_LAMBDA_CAM_POS,
        lambda_cam_rot=DEFAULT_LAMBDA_CAM_ROT,
        use_sag=False,
        estimate_measurement_noise=DEFAULT_ESTIMATE_MEASUREMENT_NOISE,
        measurement_noise_update_rate=DEFAULT_NOISE_UPDATE_RATE,
    ):
        self.robot = robot
        self.dyn_model = robot.get_dynamics()
        self.model = robot.model()
        self.use_sag = use_sag

        self.arm_idx = np.array(arm_idx, dtype=int)
        self.head_idx = np.array(head_idx, dtype=int) if head_idx is not None else None
        self.ee_links = dict(ee_links)
        self.mount_to_cam_nom = mount_to_cam_nom
        self.head_base_to_cam_nom = head_base_to_cam_nom
        self.ee_to_marker_nom = dict(ee_to_marker_nom)
        self.camera_link = camera_link
        self.active_arms = active_arms

        self.use_head_kinematics = use_head_kinematics and (self.head_idx is not None)
        self.optimize_arm = optimize_arm
        self.optimize_head = optimize_head and self.use_head_kinematics
        self.optimize_camera = optimize_camera

        self.max_iter = max_iter
        self.eps = eps
        self.lambda_cam_pos = lambda_cam_pos
        self.lambda_cam_rot = lambda_cam_rot
        self.noise_estimator = ResidualNoiseEstimator(
            enabled=estimate_measurement_noise,
            update_rate=measurement_noise_update_rate,
        )
        self.q_nominal = robot.get_state().position.copy()
        
        # Sag Estimators
        self.sag_estimators = {}
        if "right" in self.ee_links:
            r_idx = self.arm_idx[:7] if len(self.arm_idx) > 7 else self.arm_idx
            self.sag_estimators["right"] = SagEstimator(robot, r_idx, self.ee_links["right"])
        if "left" in self.ee_links:
            l_idx = self.arm_idx[7:14] if len(self.arm_idx) > 7 else self.arm_idx
            self.sag_estimators["left"] = SagEstimator(robot, l_idx, self.ee_links["left"])
        
        self.numeric_jac_eps = 1e-7

        if self.use_head_kinematics:
            self.base_link = self.camera_link
            self.T_cam_nom = make_transform(self.mount_to_cam_nom) if self.mount_to_cam_nom else np.eye(4)
        else:
            self.base_link = "link_head_0"
            self.T_cam_nom = make_transform(self.head_base_to_cam_nom) if self.head_base_to_cam_nom else np.eye(4)

    def joint_param_dim(self):
        dim = 0
        if self.optimize_arm:
            dim += len(self.arm_idx)
        if self.optimize_head:
            dim += len(self.head_idx)
        return dim

    def total_dim(self):
        dim = self.joint_param_dim()
        if self.optimize_camera:
            dim += 6
        return dim

    def get_nominal_cam_transform(self):
        return self.T_cam_nom.copy()

    def get_nominal_ee_to_marker(self, arm_side):
        return make_transform(self.ee_to_marker_nom[str(arm_side)])

    def unpack_params(self, dx):
        cursor = 0
        q_arm_offset = np.zeros(len(self.arm_idx))
        q_head_offset = np.zeros(len(self.head_idx)) if self.head_idx is not None else None
        xi_cam = np.zeros(6)

        if self.optimize_arm:
            q_arm_offset = dx[cursor:cursor + len(self.arm_idx)]
            cursor += len(self.arm_idx)
        if self.optimize_head:
            q_head_offset = dx[cursor:cursor + len(self.head_idx)]
            cursor += len(self.head_idx)
        if self.optimize_camera:
            xi_cam = dx[cursor:cursor + 6]

        return q_arm_offset, q_head_offset, xi_cam

    def pack_joint_jacobian(self, Jb):
        parts = []
        if self.optimize_arm:
            parts.append(Jb[:, self.arm_idx])
        if self.optimize_head:
            parts.append(Jb[:, self.head_idx])
        if not parts:
            return np.zeros((6, 0))
        return np.concatenate(parts, axis=1)

    def evaluate_sample(self, q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_cam):
        q_full = prepare_q_full(
            q_nominal=self.q_nominal,
            arm_idx=self.arm_idx,
            q_cmd=q_arm,
            q_offset=q_arm_offset if self.optimize_arm else None,
            head_idx=self.head_idx,
            q_head=q_head,
            q_head_offset=q_head_offset if self.optimize_head else None,
        )

        state = self.dyn_model.make_state(
            [self.base_link, self.ee_links[str(arm_side)]],
            self.model.robot_joint_names
        )
        state.set_q(q_full)
        
        if self.use_sag:
            q_full_sagged, _ = self.sag_estimators[str(arm_side)].get_sagged_joints(q_full)
            state.set_q(q_full_sagged)
            
        self.dyn_model.compute_forward_kinematics(state)
        self.dyn_model.compute_diff_forward_kinematics(state)

        T_fk = self.dyn_model.compute_transformation(state, 0, 1)
        Jb_full = self.dyn_model.compute_body_jacobian(state, 0, 1)
        Jb_joint = self.pack_joint_jacobian(Jb_full)

        T_cam_nom = self.get_nominal_cam_transform()
        T_cam = (
            T_cam_nom @ se3_exp(xi_cam)
            if self.optimize_camera else T_cam_nom
        )
        T_ee_to_marker = self.get_nominal_ee_to_marker(arm_side)

        T_model = np.linalg.inv(T_cam) @ T_fk @ T_ee_to_marker
        return Jb_joint, T_cam, T_ee_to_marker, T_model

    def build_camera_jacobian_numeric(self, q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam, T_model_ref):
        J_cam = np.zeros((6, 6))

        for i in range(6):
            delta = np.zeros(6)
            delta[i] = self.numeric_jac_eps

            _, _, _, T_model_plus = self.evaluate_sample(q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam + delta)
            _, _, _, T_model_minus = self.evaluate_sample(q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam - delta)

            xi_plus = se3_log(np.linalg.inv(T_model_ref) @ T_model_plus)
            xi_minus = se3_log(np.linalg.inv(T_model_ref) @ T_model_minus)
            J_cam[:, i] = (xi_plus - xi_minus) / (2 * self.numeric_jac_eps)

        return J_cam

    def build_jacobian(self, q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam, Jb_joint, T_ee_to_marker, T_model):
        joint_dim = self.joint_param_dim()
        J_joint = adjoint(np.linalg.inv(T_ee_to_marker)) @ Jb_joint if joint_dim > 0 else np.zeros((6, 0))

        if joint_dim > 0 and self.optimize_camera:
            J = np.zeros((6, joint_dim + 6))
            J[:, :joint_dim] = J_joint
            J[:, joint_dim:] = self.build_camera_jacobian_numeric(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                T_model,
            )
            return J

        if self.optimize_camera:
            return self.build_camera_jacobian_numeric(
                q_arm,
                q_head,
                arm_side,
                q_arm_offset,
                q_head_offset,
                xi_mount_cam,
                T_model,
            )

        return J_joint

    def compute_step(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam):
        dim = self.total_dim()
        H = np.zeros((dim, dim))
        g = np.zeros(dim)
        total_err = 0.0
        weights = self.noise_estimator.weights()
        residual_samples = []

        if q_head_list is None:
            q_head_iter = [None] * len(q_arm_list)
        else:
            q_head_iter = q_head_list

        for q_arm, q_head, T_meas_pair in zip(q_arm_list, q_head_iter, T_meas_list):
            for side_idx, arm_side in enumerate(ARM_SIDES):
                if arm_side not in self.active_arms:
                    continue

                # Handle both single-arm (N, 4, 4) and dual-arm (N, 2, 4, 4) data shapes
                if T_meas_pair.ndim == 3: # (2, 4, 4) flattened or single (4, 4)? 
                    # Wait, T_meas_list from data['T_meas'] usually has shape (N, 2, 4, 4) or (N, 4, 4)
                    if T_meas_pair.shape == (2, 4, 4):
                        T_meas = T_meas_pair[side_idx]
                    else:
                        T_meas = T_meas_pair
                elif T_meas_pair.ndim == 2: # (4, 4)
                    T_meas = T_meas_pair
                else:
                    T_meas = T_meas_pair[side_idx]

                Jb_joint, _, T_ee_to_marker, T_model = self.evaluate_sample(
                    q_arm,
                    q_head,
                    arm_side,
                    q_arm_offset,
                    q_head_offset,
                    xi_mount_cam,
                )

                T_err = np.linalg.inv(T_model) @ T_meas
                xi = se3_log(T_err)
                J = self.build_jacobian(
                    q_arm,
                    q_head,
                    arm_side,
                    q_arm_offset,
                    q_head_offset,
                    xi_mount_cam,
                    Jb_joint,
                    T_ee_to_marker,
                    T_model,
                )

                add_weighted_normal_equation(H, g, J, xi, weights)
                total_err += np.linalg.norm(xi)
                residual_samples.append(xi)

        self.noise_estimator.update(residual_samples)

        if self.optimize_camera:
            rot_slice = slice(dim - 6, dim - 3)
            pos_slice = slice(dim - 3, dim)

            if self.lambda_cam_rot > 0.0:
                H[rot_slice, rot_slice] += self.lambda_cam_rot * np.eye(3)
                g[rot_slice] += -self.lambda_cam_rot * xi_mount_cam[:3]

            if self.lambda_cam_pos > 0.0:
                H[pos_slice, pos_slice] += self.lambda_cam_pos * np.eye(3)
                g[pos_slice] += -self.lambda_cam_pos * xi_mount_cam[3:]



        dx = np.linalg.pinv(H) @ g
        return dx, total_err

    def apply_update(self, q_arm_offset, q_head_offset, xi_mount_cam, dx):
        dq_arm, dq_head, dxi = self.unpack_params(dx)
        if self.optimize_arm:
            q_arm_offset += dq_arm
        if self.optimize_head and q_head_offset is not None:
            q_head_offset += dq_head
        if self.optimize_camera:
            xi_mount_cam += dxi
        return q_arm_offset, q_head_offset, xi_mount_cam

    def get_calibrated_head_base_to_cam(self, xi_cam):
        T_cam_calib = self.get_nominal_cam_transform() @ se3_exp(xi_cam)
        if self.use_head_kinematics:
            _, T_head_base_to_mount = compute_fk(
                robot=self.robot,
                dyn_model=self.dyn_model,
                q_full=self.q_nominal,
                ee_link=self.camera_link,
                base_link="link_head_0",
            )
            T_calib = T_head_base_to_mount @ T_cam_calib
        else:
            T_calib = T_cam_calib

        p = T_calib[:3, 3]
        rpy = rot_to_euler_zyx(T_calib[:3, :3])

        return [
            p[0], p[1], p[2],
            np.rad2deg(rpy[0]),
            np.rad2deg(rpy[1]),
            np.rad2deg(rpy[2]),
        ]
        
    def get_calibrated_mount_to_cam(self, xi_cam):
        if not self.use_head_kinematics:
            return None

        T_calib = self.get_nominal_cam_transform() @ se3_exp(xi_cam)
        p = T_calib[:3, 3]
        rpy = rot_to_euler_zyx(T_calib[:3, :3])

        return [
            p[0], p[1], p[2],
            np.rad2deg(rpy[0]),
            np.rad2deg(rpy[1]),
            np.rad2deg(rpy[2]),
        ]

    def optimize(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset_init=None, q_head_offset_init=None, xi_cam_init=None):
        if self.use_head_kinematics and q_head_list is None:
            raise RuntimeError(
                "Head kinematics are enabled for this ndof, but q_head_list is missing."
            )

        q_arm_offset = q_arm_offset_init.copy() if q_arm_offset_init is not None else np.zeros(len(self.arm_idx))
        if self.optimize_head:
            q_head_offset = q_head_offset_init.copy() if q_head_offset_init is not None else np.zeros(len(self.head_idx))
        else:
            q_head_offset = q_head_offset_init.copy() if q_head_offset_init is not None else None
            
        xi_cam = xi_cam_init.copy() if xi_cam_init is not None else np.zeros(6)

        for it in range(self.max_iter):
            dx, total_err = self.compute_step(
                q_arm_list,
                q_head_list,
                T_meas_list,
                q_arm_offset,
                q_head_offset,
                xi_cam,
            )
            q_arm_offset, q_head_offset, xi_cam = self.apply_update(
                q_arm_offset,
                q_head_offset,
                xi_cam,
                dx,
            )

            print(f"[{it}] |dx|={np.linalg.norm(dx):.3e}, |err|={total_err:.3e}")

            if np.linalg.norm(dx) < self.eps:
                print("Converged.")
                break

        mount_to_cam_new = self.get_calibrated_mount_to_cam(xi_cam)
        head_base_to_cam_new = self.get_calibrated_head_base_to_cam(xi_cam)
        return q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new