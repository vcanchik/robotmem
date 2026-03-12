"""robotmem stable-baselines3 回调 — 自动记录 episode 经验

用法:
    from stable_baselines3 import PPO
    from robotmem.sb3 import RobotMemSB3Callback

    model = PPO("MlpPolicy", "CartPole-v1")
    cb = RobotMemSB3Callback(collection="cartpole")
    model.learn(total_timesteps=10000, callback=cb)
    cb.close()

    # 或搭配其他 callback
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    model.learn(callback=CallbackList([cb, eval_cb]))

兼容性:
    - stable-baselines3 >= 2.0
    - 支持所有 SB3 算法 (PPO, SAC, A2C, DQN, TD3, etc.)
    - 支持向量化环境 (SubprocVecEnv, DummyVecEnv)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# lazy import
_BaseCallback = None


def _ensure_sb3():
    global _BaseCallback
    if _BaseCallback is None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
            _BaseCallback = BaseCallback
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for RobotMemSB3Callback. "
                "Install with: pip install robotmem[sb3]"
            )
    return _BaseCallback


class RobotMemSB3Callback(_ensure_sb3()):
    """SB3 训练回调 — 自动记录 episode 经验到 robotmem

    设计原则（宪法）：
    - 第 1 条 机器能用：结构化 context（reward/success/algorithm）
    - 第 3 条 能删就删：不改 SB3 核心逻辑，只用 callback 机制
    - 第 4 条 坏了就喊：robotmem 异常不中断训练，warn 后继续
    """

    def __init__(
        self,
        collection: str = "sb3",
        db_path: str | None = None,
        embed_backend: str = "onnx",
        learn_interval: int = 1,
        recall_at_start: bool = False,
        recall_query: str = "successful training strategy",
        recall_n: int = 5,
        verbose: int = 0,
    ):
        """
        Args:
            collection: robotmem 集合名
            db_path: 自定义 DB 路径
            embed_backend: embedding 后端
            learn_interval: 每 N 个 episode 学习一次（默认每个 episode）
            recall_at_start: 训练开始时 recall 历史经验
            recall_query: recall 查询字符串
            recall_n: recall 返回条数
            verbose: 日志级别
        """
        super().__init__(verbose)
        self._collection = collection
        self._db_path = db_path
        self._embed_backend = embed_backend
        self._learn_interval = learn_interval
        self._recall_at_start = recall_at_start
        self._recall_query = recall_query
        self._recall_n = recall_n

        # 延迟初始化（_on_training_start 中创建）
        self.mem = None
        self._sid: str | None = None
        self._episode_count = 0
        self._total_episodes = 0

        # episode 统计缓冲
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []

    def _on_training_start(self) -> None:
        """训练开始 — 初始化 robotmem（异常不阻塞训练）"""
        try:
            from robotmem.sdk import RobotMemory
            self.mem = RobotMemory(
                db_path=self._db_path,
                collection=self._collection,
                embed_backend=self._embed_backend,
            )
        except Exception as e:
            logger.warning("robotmem 初始化失败，训练将不使用记忆: %s", e)
            self.mem = None
            return

        # 获取算法信息
        algo_name = type(self.model).__name__  # PPO, SAC, etc.

        try:
            self._sid = self.mem.start_session(context={
                "framework": "stable-baselines3",
                "algorithm": algo_name,
                "total_timesteps": self.locals.get("total_timesteps", 0),
            })
            logger.info("robotmem session 开始: %s (algo=%s)", self._sid, algo_name)
        except Exception as e:
            logger.warning("robotmem start_session 失败: %s", e)

        # 可选：recall 历史经验
        if self._recall_at_start:
            tips = self.recall_tips(self._recall_query, n=self._recall_n)
            if tips:
                logger.info("robotmem recall %d 条历史经验", len(tips))

    def _on_step(self) -> bool:
        """每步调用 — 检测 episode 结束并 learn"""
        # SB3 通过 infos 中的 episode 字段通知 episode 结束
        # Monitor wrapper 会设置 info["episode"] = {"r": reward, "l": length, "t": time}
        infos = self.locals.get("infos", [])

        for info in infos:
            if not isinstance(info, dict):
                continue

            ep_info = info.get("episode")
            if ep_info is None:
                continue

            # episode 完成
            self._total_episodes += 1
            ep_reward = ep_info.get("r", 0.0)
            ep_length = ep_info.get("l", 0)
            self._ep_rewards.append(ep_reward)
            self._ep_lengths.append(ep_length)

            # 每 N 个 episode learn 一次
            if self._total_episodes % self._learn_interval == 0:
                self._learn_episodes()

        return True  # 继续训练

    def _on_training_end(self) -> None:
        """训练结束 — 学习剩余 episodes + 结束 session"""
        # 学习剩余未汇总的 episodes
        if self._ep_rewards:
            self._learn_episodes()

        if self.mem and self._sid:
            try:
                self.mem.end_session(
                    session_id=self._sid,
                    outcome_score=None,
                )
                logger.info("robotmem session 结束: %s, %d episodes",
                            self._sid, self._total_episodes)
            except Exception as e:
                logger.warning("robotmem end_session 失败: %s", e)

    def _learn_episodes(self):
        """汇总缓冲中的 episodes 并 learn"""
        if not self._ep_rewards or not self.mem:
            return

        n = len(self._ep_rewards)
        avg_reward = sum(self._ep_rewards) / n
        avg_length = sum(self._ep_lengths) / n
        max_reward = max(self._ep_rewards)

        insight = (
            f"Episodes {self._total_episodes - n + 1}-{self._total_episodes}: "
            f"avg_reward={avg_reward:.2f}, max_reward={max_reward:.2f}, "
            f"avg_length={avg_length:.0f}, n={n}"
        )

        context = {
            "task": {
                "reward": avg_reward,
                "max_reward": max_reward,
                "episodes": n,
                "avg_length": avg_length,
            },
            "params": {
                "timesteps": {
                    "value": self.num_timesteps,
                    "type": "scalar",
                },
            },
        }

        try:
            self.mem.learn(
                insight=insight,
                context=context,
                session_id=self._sid,
            )
        except Exception as e:
            logger.warning("robotmem learn 失败: %s", e)

        # 清空缓冲
        self._ep_rewards.clear()
        self._ep_lengths.clear()

    def recall_tips(
        self,
        query: str,
        n: int = 5,
        context_filter: dict | None = None,
    ) -> list[dict]:
        """手动检索经验"""
        if not self.mem:
            return []
        try:
            return self.mem.recall(
                query=query,
                n=n,
                context_filter=context_filter,
            )
        except Exception as e:
            logger.warning("robotmem recall 失败: %s", e)
            return []

    def close(self):
        """释放 robotmem 资源"""
        if self.mem:
            try:
                self.mem.close()
            except Exception as e:
                logger.warning("robotmem close 失败: %s", e)
            self.mem = None
