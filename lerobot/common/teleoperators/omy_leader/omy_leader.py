import logging
import time
import threading
from typing import Any
import numpy as np
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors.ros import RosMotorsBus, RosMotorsBusConfig
from lerobot.common.robots.omy.ros_utils import ensure_rclpy_init
from ..teleoperator import Teleoperator
from .config_omy_leader import OMYLeaderConfig
from rclpy.executors import MultiThreadedExecutor
logger = logging.getLogger(__name__)


class OMYLeader(Teleoperator):
    """
    OMY Leader Arm designed by ROBOTIS
    """

    config_class = OMYLeaderConfig
    name = "omy_leader"

    def __init__(self, config: OMYLeaderConfig):
        super().__init__(config)
        self.config = config

        self.bus = RosMotorsBus(
            config=RosMotorsBusConfig(
                observation_topic_name="/leader/joint_trajectory",
                observation_msg_type="JointTrajectory",
                motors={
                    "joint1": (1, "joint1"),
                    "joint2": (2, "joint2"),
                    "joint3": (3, "joint3"),
                    "joint4": (4, "joint4"),
                    "joint5": (5, "joint5"),
                    "joint6": (6, "joint6"),
                    "rh_r1_joint": (7, "rh_r1_joint"),  # gripper
                },
                fps=15, 
            )
        )
        self.connected = False

    @property
    def action_features(self) -> dict:
        return {
            "joint1.pos": float,
            "joint2.pos": float,
            "joint3.pos": float,
            "joint4.pos": float,
            "joint5.pos": float,
            "joint6.pos": float,
            "rh_r1_joint.pos": float,  # gripper position
        }

    @property
    def feedback_features(self) -> dict:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, calibrate: bool = True) -> None:
        ensure_rclpy_init()
        self.bus.connect()
        self.connected = True
        
        self._spin_ros_nodes()
        print("Leader Ros nodes spun...")

        self.bus._wait_for_joint_state()


    @property
    def is_calibrated(self) -> bool:
        """Whether the teleoperator is currently calibrated or not. Should be always `True` if not applicable"""
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the teleoperator.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        start = time.perf_counter()
        motor_names = list(self.bus.motors.keys())
        
        action = self.bus.read("Present_Position", motor_names=motor_names)
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"OMY Leader get_action took {dt_ms:.2f} ms")

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self.connected = False


    def _spin_ros_nodes(self):
        # Collect all nodes from arms and cameras
        nodes = []
        nodes.extend(self.bus.get_ros_nodes())
        if not nodes:
            return
        self.executor = MultiThreadedExecutor()
        for node in nodes:
            self.executor.add_node(node)
        self._ros_spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._ros_spin_thread.start()