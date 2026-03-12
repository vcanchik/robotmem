"""robotmem Gymnasium 集成 — 通用 Gym 环境记忆包装器

覆盖: Gymnasium-Robotics / ManiSkill / robosuite（所有 Gym 兼容环境）

安装: pip install robotmem
用法: from robotmem.gymnasium import RobotMemWrapper
"""

from robotmem.gymnasium.wrapper import RobotMemWrapper

__all__ = ["RobotMemWrapper"]
