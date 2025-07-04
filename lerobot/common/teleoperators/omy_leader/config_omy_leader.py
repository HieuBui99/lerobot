from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("omy_leader")
@dataclass
class OMYLeaderConfig(TeleoperatorConfig):
    pass