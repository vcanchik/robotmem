"""robotmem stable-baselines3 集成 — SB3 训练回调

安装: pip install robotmem[sb3]
用法: from robotmem.sb3 import RobotMemSB3Callback
"""


def __getattr__(name):
    if name == "RobotMemSB3Callback":
        from robotmem.sb3.callback import RobotMemSB3Callback
        return RobotMemSB3Callback
    raise AttributeError(f"module 'robotmem.sb3' has no attribute {name!r}")


__all__ = ["RobotMemSB3Callback"]
