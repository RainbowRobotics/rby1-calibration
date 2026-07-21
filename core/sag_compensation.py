import numpy as np

class SagEstimator:
    """
    Handles robot joint deflection (sag) estimation based on gravity torque.
    """
    def __init__(self, robot, arm_idx, ee_link):
        self.robot = robot
        self.model = robot.model()
        self.dyn_model = robot.get_dynamics()
        self.arm_idx = np.array(arm_idx, dtype=int)
        self.ee_link = ee_link

        # Default Joint Stiffness [Nm/rad]
        # J1-J3: 14000, J4-J7: 6100 (Based on RBY1 feature/sag branch)
        self.k_joint = np.array([14000, 14000, 14000, 6100, 6100, 6100, 6100], dtype=float)
        
        # Correction signs
        self.sag_sign = np.ones(7, dtype=float)
        
        # Active flags per joint
        self.sag_active = np.array([True, True, True, True, True, True, True], dtype=bool)
        
        # Safety clip to prevent extreme corrections (rad)
        self.sag_clip_rad = np.deg2rad(0.3)

    def compute_gravity_torque(self, q_full: np.ndarray) -> np.ndarray:
        """
        Calculates full gravity torque vector using Inverse Dynamics.
        """
        state = self.dyn_model.make_state(
            ["link_torso_5", self.ee_link],
            self.model.robot_joint_names
        )
        # Standard gravity vector
        state.set_gravity(np.array([0, 0, 0, 0, 0, -9.81], dtype=float))
        state.set_q(q_full)
        state.set_qdot(np.zeros(self.model.robot_dof, dtype=float))
        state.set_qddot(np.zeros(self.model.robot_dof, dtype=float))

        self.dyn_model.compute_forward_kinematics(state)
        self.dyn_model.compute_diff_forward_kinematics(state)
        self.dyn_model.compute_2nd_diff_forward_kinematics(state)
        self.dyn_model.compute_inverse_dynamics(state)

        return state.get_tau().copy()

    def estimate_sag(self, tau_arm: np.ndarray) -> np.ndarray:
        """
        Calculates deflection (delta_q) based on joint torque and stiffness.
        """
        delta_q = np.zeros(7, dtype=float)
        for i in range(7):
            if self.sag_active[i]:
                delta_q[i] = self.sag_sign[i] * tau_arm[i] / self.k_joint[i]
        
        # Clip to safety range
        delta_q = np.clip(delta_q, -self.sag_clip_rad, self.sag_clip_rad)
        return delta_q

    def get_sagged_joints(self, q_full: np.ndarray) -> np.ndarray:
        """
        Returns a copy of q_full with sag-compensated joint positions for the target arm.
        Note: Actual link position = Measured joint + Sag deflection.
        """
        tau_full = self.compute_gravity_torque(q_full)
        tau_arm = tau_full[self.arm_idx[:7]]
        delta_q_sag = self.estimate_sag(tau_arm)
        
        q_full_sagged = q_full.copy()
        q_full_sagged[self.arm_idx[:7]] += delta_q_sag
        
        return q_full_sagged, delta_q_sag
