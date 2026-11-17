import logging
import time
import numpy as np
import rby1_sdk as rby
from contextlib import contextmanager
from threading import local

_execution = local()

# Tool flange output (gripper power): the state read-back follows the command with a short delay.
TOOL_FLANGE_SIDES = ("right", "left")
TOOL_FLANGE_OFF_TIMEOUT_S = 2.0
TOOL_FLANGE_POLL_S = 0.1


class RobotOperationCancelled(RuntimeError):
    def __init__(self, partial=None):
        super().__init__("Stopped by user")
        self.partial = partial or {}


@contextmanager
def motion_cancellation(stop_event):
    previous = getattr(_execution, "stop_event", None)
    previous_partial = getattr(_execution, "partial", None)
    _execution.stop_event = stop_event
    _execution.partial = None
    try:
        yield
    finally:
        _execution.stop_event = previous
        _execution.partial = previous_partial


def check_motion_cancelled(partial=None):
    if partial is not None:
        _execution.partial = partial
    event = getattr(_execution, "stop_event", None)
    if event is not None and event.is_set():
        raise RobotOperationCancelled(getattr(_execution, "partial", None))


def wait_motion(seconds):
    event = getattr(_execution, "stop_event", None)
    if event is None:
        time.sleep(seconds)
    elif event.wait(seconds):
        check_motion_cancelled()


def cancel_control(robot, address=None, model_name=None):
    """Use a separate SDK connection when available, matching the UI stop path."""
    if robot is None:
        return
    if address and model_name:
        temporary = rby.create_robot(address, model_name)
        try:
            if temporary.connect():
                return temporary.cancel_control()
        finally:
            temporary.disconnect()
    return robot.cancel_control()


class RobotOperations:
    """SDK operations shared by calibration controllers; no UI dependencies."""
    @staticmethod
    def initialize_robot(address, model, power=".*", servo=None, include_head=True):
        robot = rby.create_robot(address, model)
        if not robot.connect():
            logging.error(f"Failed to connect robot {address}")
            return None
        
        # Safety check: Verify actual connected robot model matches expected model
        try:
            robot_info = robot.get_robot_info()
            actual_model = robot_info.robot_model_name.lower()
            expected_model = model.lower()
            if actual_model != expected_model:
                logging.warning(f"Model mismatch! UI selected model: {model}, but actual robot model is: {robot_info.robot_model_name}. Auto-reconnecting with actual model...")
                robot.disconnect()
                robot = rby.create_robot(address, robot_info.robot_model_name)
                if not robot.connect():
                    logging.error(f"Failed to connect robot {address} with actual model {robot_info.robot_model_name}")
                    return None
        except Exception as e:
            logging.error(f"Failed to verify robot model: {e}")
            robot.disconnect()
            return None

        # Check if connecting to localhost/simulator
        is_local = any(loc in str(address) for loc in ["127.0.0.1", "localhost", "0.0.0.0"])

        # Check if power is ON; if not, turn on power
        try:
            power_pattern = ".*" if is_local else "48v"
            if not robot.is_power_on(power_pattern):
                logging.info(f"Power ({power_pattern}) is not ON. Turning power on...")
                if not robot.power_on(power_pattern):
                    logging.error(f"Failed to turn power ({power_pattern}) on.")
                    robot.disconnect()
                    return None
                time.sleep(1.0)
            else:
                logging.info(f"Power ({power_pattern}) is already ON.")
        except Exception as e:
            logging.error(f"Failed to check or set power status: {e}")
            robot.disconnect()
            return None

        # Wait 1 second
        time.sleep(1.0)

        # Check and reset control manager fault if necessary
        try:
            cm_state = robot.get_control_manager_state().state
            if cm_state in [
                rby.ControlManagerState.State.MajorFault,
                rby.ControlManagerState.State.MinorFault,
            ]:
                logging.warning("Control manager is in fault state. Resetting...")
                robot.reset_fault_control_manager()
                time.sleep(0.5)
            cm_state = robot.get_control_manager_state().state
            is_cm_enabled = (cm_state == rby.ControlManagerState.State.Enabled)
        except Exception as e:
            logging.warning(f"Failed to check control manager state: {e}")
            is_cm_enabled = False

        # Configure servo pattern based on include_head flag (independent of physical hardware)
        if servo is not None and servo != ".*":
            target_servo_pattern = servo
        elif is_local:
            target_servo_pattern = ".*"
        else:
            target_servo_pattern = "^(?!.*wheel).*$" if include_head else "^(?!.*(head|wheel)).*$"

        # Check if servos are ON
        try:
            is_servo_ok = robot.is_servo_on(target_servo_pattern)
        except Exception as e:
            logging.warning(f"Failed to check servo status: {e}")
            is_servo_ok = False

        def enable_cm_helper(r):
            try:
                cm_state_post = r.get_control_manager_state()
                if cm_state_post.state in [
                    rby.ControlManagerState.State.MajorFault,
                    rby.ControlManagerState.State.MinorFault,
                ]:
                    logging.warning(f"Control manager is in fault state: {cm_state_post.state}. Resetting...")
                    if not r.reset_fault_control_manager():
                        logging.error("Failed to reset control manager")
                
                cm_state_post = r.get_control_manager_state()
                if cm_state_post.state == rby.ControlManagerState.State.Enabled:
                    logging.info("Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...")
                    try:
                        r.disable_control_manager()
                        time.sleep(0.5)
                    except Exception as ex:
                        logging.warning(f"Failed to disable control manager: {ex}")
                
                logging.info("Enabling control manager with unlimited_mode_enabled=True...")
                if not r.enable_control_manager(unlimited_mode_enabled=True):
                    logging.error("Failed to enable control manager with unlimited_mode_enabled=True")
                else:
                    time.sleep(1.0)
            except Exception as ex:
                logging.error(f"Failed to configure control manager: {ex}")

        if is_servo_ok:
            logging.info("Servos are ON. Ensuring Control Manager is enabled with unlimited mode...")
            enable_cm_helper(robot)
        else:
            # Otherwise, disable control manager first, then turn on servos and enable
            logging.info("Servos are not ON. Disabling Control Manager first to turn on servos...")
            if is_cm_enabled:
                try:
                    robot.disable_control_manager()
                    time.sleep(0.5)
                except Exception as e:
                    logging.warning(f"Failed to disable control manager: {e}")
            
            logging.info(f"Turning servos on with pattern '{target_servo_pattern}'...")
            if not robot.servo_on(target_servo_pattern):
                logging.error(f"Failed to turn servos on with pattern '{target_servo_pattern}'.")
            else:
                time.sleep(0.5)
            
            enable_cm_helper(robot)

        return robot


    @staticmethod
    def terminate_robot(robot):
        if robot:
            try:
                robot.disconnect()
                return True
            except Exception as e:
                logging.error(f"Failed to disconnect robot: {e}")
        return False


    @staticmethod
    def tool_flange_output_voltages(robot, sides=TOOL_FLANGE_SIDES):
        state = robot.get_state()
        return {side: float(getattr(state, f"tool_flange_{side}").output_voltage) for side in sides}

    @staticmethod
    def power_off_tool_flanges(robot, sides=TOOL_FLANGE_SIDES):
        """Set the tool flange output (gripper power) to 0 V and return the voltage read back per
        side. Raises when the robot rejects the command or the output is still on afterwards."""
        for side in sides:
            if not robot.set_tool_flange_output_voltage(side, 0):
                raise RuntimeError(f"Tool flange {side}: the robot rejected the 0 V output command")
        deadline = time.monotonic() + TOOL_FLANGE_OFF_TIMEOUT_S
        while True:
            voltages = RobotOperations.tool_flange_output_voltages(robot, sides)
            if all(value == 0 for value in voltages.values()):
                return voltages
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Tool flange output is still on after the 0 V command: {voltages}")
            time.sleep(TOOL_FLANGE_POLL_S)

    @staticmethod
    def compute_fk(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
        model = robot.model()
        state = dyn_model.make_state([base_link, ee_link], model.robot_joint_names)
        num_joints = len(model.robot_joint_names)
        q_arr = np.zeros(num_joints)
        if len(q) >= num_joints:
            q_arr = np.array(q[:num_joints])
        else:
            q_arr[:len(q)] = q
        state.set_q(q_arr)
        dyn_model.compute_forward_kinematics(state)
        T = dyn_model.compute_transformation(state, 0, 1)
        return T


    def movej(self, robot, torso=None, right_arm=None, left_arm=None, head=None, minimum_time=0, apply_offsets=True, priority=10):
        if getattr(self, 'stop_requested', False):
            return False
        if not robot:
            return False
            
        if head is not None:
            model = robot.model()
            has_head = hasattr(model, 'head_idx') and len(model.head_idx) > 0
            if not has_head:
                head = None

        if apply_offsets and hasattr(self, 'joint_offsets') and self.joint_offsets is not None:
            # Offset mapping: Joint 3 (index 3) is elbow
            # For v1.3:
            # - Joint 5 (index 5) is wrist pitch
            # - Joint 6 (index 6) is wrist roll
            # For v1.2:
            # - Joint 5 (index 5) is wrist pitch
            is_v13 = self.is_v13()
            
            # Support both flat and nested left/right dictionary structures
            if "left" in self.joint_offsets and "right" in self.joint_offsets:
                left_offsets = self.joint_offsets["left"]
                right_offsets = self.joint_offsets["right"]
            else:
                left_offsets = self.joint_offsets
                right_offsets = self.joint_offsets
                
            if right_arm is not None:
                right_arm = list(right_arm)
                r_j6_offset = right_offsets.get("wrist_roll", 0.0) if is_v13 else right_offsets.get("wrist_yaw2", 0.0)
                right_arm[6] += np.radians(r_j6_offset)
                right_arm[5] += np.radians(right_offsets.get("wrist_pitch", 0.0))
                right_arm[3] += np.radians(right_offsets.get("elbow", 0.0))
            if left_arm is not None:
                left_arm = list(left_arm)
                l_j6_offset = left_offsets.get("wrist_roll", 0.0) if is_v13 else left_offsets.get("wrist_yaw2", 0.0)
                left_arm[6] += np.radians(l_j6_offset)
                left_arm[5] += np.radians(left_offsets.get("wrist_pitch", 0.0))
                left_arm[3] += np.radians(left_offsets.get("elbow", 0.0))

        comp_cmd = rby.ComponentBasedCommandBuilder()
        
        has_body = False
        body_cmd = rby.BodyComponentBasedCommandBuilder()
        if torso is not None:
            body_cmd.set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(torso)
            )
            has_body = True
        if right_arm is not None:
            body_cmd.set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(right_arm)
            )
            has_body = True
        if left_arm is not None:
            body_cmd.set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(left_arm)
            )
            has_body = True
        
        if has_body:
            comp_cmd.set_body_command(body_cmd)

        if head is not None:
            comp_cmd.set_head_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(head)
            )
        
        cmd = rby.RobotCommandBuilder().set_command(comp_cmd)
        
        try:
            if getattr(self, "stop_requested", False):
                return False
            check_motion_cancelled()
            rv = robot.send_command(cmd, priority).get()
            if rv.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                print(f"[DEBUG MOVEJ ERROR] Failed to conduct movej. Finish code: {rv.finish_code}", flush=True)
                logging.error(f"Failed to conduct movej. Finish code: {rv.finish_code}")
                return False
            return True
        except Exception as e:
            print(f"[DEBUG MOVEJ EXCEPTION] movej exception: {e}", flush=True)
            logging.error(f"movej exception: {e}")
            return False


def connect_robot_session(addr, model, include_head=True, simulated=False, log_callback=print):
    # 1. Create and connect robot (with auto-retry safety guard)
    robot = rby.create_robot(addr, model)
    connected = robot.connect()
    if not connected:
        log_callback("[WARN] Initial connection attempt failed. Waiting 1.0s before retrying...")
        time.sleep(1.0)
        connected = robot.connect()

    if not connected:
        raise ConnectionError(f"Failed to connect robot at {addr}")
    time.sleep(1)

    # 2. Safety check: Verify actual connected robot model matches expected model
    try:
        robot_info = robot.get_robot_info()
        actual_model = robot_info.robot_model_name.lower()
        expected_model = model.lower()
        if actual_model != expected_model:
            log_callback(f"[WARNING] Model mismatch! UI selected model: {model}, but actual robot model is: {robot_info.robot_model_name}. Auto-reconnecting with actual model...")
            robot.disconnect()
            robot = rby.create_robot(addr, robot_info.robot_model_name.lower())
            if not robot.connect():
                raise ConnectionError(f"Failed to connect robot {addr} with actual model {robot_info.robot_model_name}")
            time.sleep(1)
    except Exception as e:
        log_callback(f"[ERROR] Safety check failed: {e}")

    # Check if connecting to localhost/simulator (including WSL2 IP and simulation mode)
    is_local = (
        any(loc in addr for loc in ["127.0.0.1", "localhost", "0.0.0.0"])
        or addr.startswith("172.")
        or simulated
        or simulated
    )

    # 3. Check and turn on power if not already ON
    try:
        power_pattern = ".*" if is_local else "48v"
        if not robot.is_power_on(power_pattern):
            log_callback(f"[INFO] Power ({power_pattern}) is not ON. Turning power on...")
            if not robot.power_on(power_pattern):
                raise RuntimeError(f"Failed to turn power ({power_pattern}) on.")
            time.sleep(1.0)
        else:
            log_callback(f"[INFO] Power ({power_pattern}) is already ON.")
    except Exception as e:
        log_callback(f"[ERROR] Power configuration failed: {e}")

    # 4. Check and reset Control Manager Fault state if needed
    try:
        cm_state = robot.get_control_manager_state()
        if cm_state.state in [
            rby.ControlManagerState.State.MajorFault,
            rby.ControlManagerState.State.MinorFault,
        ]:
            log_callback("[WARNING] Control manager is in fault state. Resetting...")
            robot.reset_fault_control_manager()
            time.sleep(0.5)
    except Exception as e:
        log_callback(f"[WARNING] Failed to check/reset fault state: {e}")

    # 5. Check desired servos and configure Control Manager
    try:
        servo_pattern = "^(?!.*wheel).*$" if include_head else "^(?!.*(head|wheel)).*$"
        is_servo_ok = robot.is_servo_on(servo_pattern)

        cm_state = robot.get_control_manager_state()
        is_cm_enabled = (cm_state.state == rby.ControlManagerState.State.Enabled)

        if is_servo_ok:
            log_callback(f"[INFO] Required servos ({servo_pattern}) are already ON.")
            if is_cm_enabled:
                log_callback("[INFO] Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...")
                robot.disable_control_manager()
                time.sleep(0.5)
            log_callback("[INFO] Enabling control manager with unlimited_mode_enabled=True...")
            if not robot.enable_control_manager(unlimited_mode_enabled=True):
                raise RuntimeError("Failed to enable control manager.")
            time.sleep(1.0)
        else:
            log_callback(f"[INFO] Required servos are not fully ON. Disabling Control Manager first to turn on servos ({servo_pattern})...")
            if is_cm_enabled:
                robot.disable_control_manager()
                time.sleep(0.5)

            log_callback(f"[INFO] Turning servos ({servo_pattern}) on...")
            if not robot.servo_on(servo_pattern):
                raise RuntimeError(f"Failed to turn servos ({servo_pattern}) on.")
            time.sleep(0.5)

            log_callback("[INFO] Enabling control manager with unlimited_mode_enabled=True...")
            if not robot.enable_control_manager(unlimited_mode_enabled=True):
                raise RuntimeError("Failed to enable control manager.")
            time.sleep(1.0)
    except Exception as e:
        log_callback(f"[ERROR] Servo/Control Manager configuration failed: {e}")

    return robot
