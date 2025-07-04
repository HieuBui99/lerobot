import logging
import time
from functools import cached_property
from typing import Any
import threading
import numpy as np
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.ros import RosMotorsBus, RosMotorsBusConfig

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_omy import OMYRobotConfig
from .ros_utils import ensure_rclpy_init

logger = logging.getLogger(__name__)


class OMY(Robot):
    """
    OpenManipulator-Y Robot designed by Robotis.
    """

    config_class = OMYRobotConfig
    name = "omy"

    def __init__(self, config: OMYRobotConfig):
        super().__init__(config)
        self.config = config
        
        self.bus = RosMotorsBus(
            config=RosMotorsBusConfig(
                # observation_topic_name="/arm_controller/joint_trajectory",
                # observation_msg_type="JointTrajectory",
                observation_topic_name="/joint_states",
                observation_msg_type="JointState",
                action_topic_name="/arm_controller/joint_trajectory",               
                motors={
                    "joint1": (1, "joint1"),
                    "joint2": (2, "joint2"),
                    "joint3": (3, "joint3"),
                    "joint4": (4, "joint4"),
                    "joint5": (5, "joint5"),
                    "joint6": (6, "joint6"),
                    "rh_r1_joint": (7, "rh_r1_joint"), #gripper
                },
                fps=15
            )
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self.connected = False

    @property
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {
            "joint1.pos": float,
            "joint2.pos": float,
            "joint3.pos": float,
            "joint4.pos": float,
            "joint5.pos": float,
            "joint6.pos": float,
            "rh_r1_joint.pos": float,  # gripper position
            "usb": (self.config.cameras["usb"].height, self.config.cameras["usb"].width, 3),
            "scene": (self.config.cameras["scene"].height, self.config.cameras["scene"].width, 3)
        }

    @property
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        return self.connected 

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the robot.

        Args:
            calibrate (bool): If True, automatically calibrate the robot after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        ensure_rclpy_init()

        for name in self.cameras:
            print(f"Connecting camera {name}...")
            self.cameras[name].connect()
        
        self.connected = True

        self.bus.connect()
        print(f"{self.robot_type} robot detected")

        self._spin_ros_nodes()
        print("ROS nodes spun")

        self.bus._wait_for_joint_state()
        print("ROS motors bus connected")

        for name in self.cameras:
            self.cameras[name]._wait_for_image()
        print("ROS cameras connected")

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        # no-op
        pass


    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        motors = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"]
        start = time.perf_counter()
        
        observation = self.bus.read(
            "Present_Position",
            motor_names=motors,
        )

        obs_dict = {"observation.state": observation}
        # for i, motor in enumerate(motors):
        #     obs_dict["observation.state"][i] = observation[motor]["position"]

        for cam_name, cam in self.cameras.items():
            obs_dict[f"observation.images.{cam_name}"] = cam.async_read()

        dt_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(f"OMY observation read in {dt_ms:.2f} ms")
        return obs_dict

    def send_action(self, action):

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # # action # (7, )
        motors = list(self.bus.motors.keys())
        
        self.bus.write("Goal_Position", action, motor_names=motors)
        return action
        

    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        self.connected = False

    def _spin_ros_nodes(self):
        # Collect all nodes from arms and cameras
        nodes = []
        nodes.extend(self.bus.get_ros_nodes())
        for cam in self.cameras.values():
            if hasattr(cam, 'get_ros_node'):
                nodes.append(cam.get_ros_node())
        if not nodes:
            return
        self.executor = MultiThreadedExecutor()
        for node in nodes:
            self.executor.add_node(node)
        self._ros_spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._ros_spin_thread.start()