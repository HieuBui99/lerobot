from dataclasses import dataclass, field

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.opencv import OpenCVCameraConfig
from lerobot.common.cameras.realsense import RealSenseCameraConfig
from lerobot.common.cameras.ros import RosCameraConfig
from lerobot.common.motors.ros import RosMotorsBus, RosMotorsBusConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("omy")
@dataclass
class OMYRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "usb": RosCameraConfig(
                topic_name="/usb_cam/image_raw/compressed",
                msg_type="CompressedImage",
                fps=15,
                width=640,
                height=480,
                # rotation=90,
            ),
            "scene": RosCameraConfig(
                topic_name="/camera/camera/color/image_raw/compressed",
                msg_type="CompressedImage",
                fps=15,
                width=1280,
                height=720,
                # rotation=90,
            ),
            # "wrist": RosCameraConfig(
            #     topic_name="Intel RealSense D405",
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
        }
    )

    mock: bool = False
