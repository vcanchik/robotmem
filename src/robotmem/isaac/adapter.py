"""robotmem Isaac Lab 辅助 — GPU 加速训练循环的记忆集成

Isaac Lab 没有统一的 callback 基类，通过自定义训练循环集成。
本模块提供辅助类，在训练循环的关键位置调用。

用法:
    from robotmem.isaac import RobotMemIsaacHelper

    helper = RobotMemIsaacHelper(collection="isaac_reach")

    # 训练开始
    helper.on_train_begin({"task": "Reach", "robot": "Franka"})

    for iteration in range(num_iterations):
        # ... Isaac Lab 训练步骤 ...
        actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)

        # 每步汇报（自动按 episode 边界学习）
        helper.on_step(rewards, dones, infos, iteration)

    # 训练结束
    helper.on_train_end({"final_reward": avg_reward})
    helper.close()

兼容性:
    - 无 Isaac Lab/Isaac Sim 依赖（纯辅助类）
    - rewards/dones 支持 list、numpy array、torch tensor
    - 适用于 Isaac Lab / Isaac Gym / OmniIsaacGymEnvs
"""

from __future__ import annotations

import logging
from typing import Any

from robotmem.sdk import RobotMemory

logger = logging.getLogger(__name__)


class RobotMemIsaacHelper:
    """Isaac Lab 训练循环辅助 — 自动记录 episode 经验

    设计原则（宪法）：
    - 第 1 条 机器能用：GPU tensor 自动转 Python 数值
    - 第 3 条 能删就删：不依赖 Isaac Lab，纯辅助函数
    - 第 4 条 坏了就喊：异常不中断训练
    - 第 5 条 加不改：适用于任何 Isaac 系训练循环
    """

    def __init__(
        self,
        collection: str = "isaac",
        db_path: str | None = None,
        embed_backend: str = "onnx",
        learn_interval: int = 10,
        recall_at_start: bool = True,
        recall_query: str = "successful policy parameters",
        recall_n: int = 5,
    ):
        """
        Args:
            collection: robotmem 集合名
            db_path: 自定义 DB 路径
            embed_backend: embedding 后端
            learn_interval: 每 N 个 iteration 学习一次
            recall_at_start: 训练开始时 recall 历史经验
            recall_query: recall 查询字符串
            recall_n: recall 返回条数
        """
        # 初始化 robotmem（异常不阻塞训练）
        try:
            self.mem = RobotMemory(
                db_path=db_path,
                collection=collection,
                embed_backend=embed_backend,
            )
            self._mem_available = True
        except Exception as e:
            logger.warning("robotmem 初始化失败，训练将不使用记忆: %s", e)
            self.mem = None
            self._mem_available = False
        self._learn_interval = learn_interval
        self._recall_at_start = recall_at_start
        self._recall_query = recall_query
        self._recall_n = recall_n

        self._sid: str | None = None
        self._total_episodes = 0
        self._iter_episodes = 0
        self._iter_reward_sum = 0.0
        self._recalled_tips: list[dict] = []

    def on_train_begin(self, config: dict[str, Any] | None = None) -> str:
        """训练开始 — 创建 session，可选 recall"""
        if not self._mem_available:
            return ""
        config = config or {}
        try:
            self._sid = self.mem.start_session(context={
                "framework": "isaac-lab",
                "task": config.get("task", "unknown"),
                "robot": config.get("robot", "unknown"),
                **{k: v for k, v in config.items() if k not in ("task", "robot")},
            })
            logger.info("robotmem session 开始: %s", self._sid)
        except Exception as e:
            logger.warning("robotmem start_session 失败: %s", e)
            self._sid = None

        if self._recall_at_start:
            self._recalled_tips = self.recall_tips(
                self._recall_query, n=self._recall_n,
            )
            if self._recalled_tips:
                logger.info("robotmem recall %d 条历史经验", len(self._recalled_tips))

        return self._sid or ""

    def on_step(
        self,
        rewards,
        dones,
        infos: dict | None = None,
        iteration: int = 0,
    ):
        """每步/每 iteration 调用 — 累积统计，按间隔 learn

        Args:
            rewards: 奖励 (tensor/array/list)
            dones: 完成标记 (tensor/array/list)
            infos: 额外信息
            iteration: 当前 iteration 编号
        """
        # 转换 tensor → Python
        n_done = self._to_int_sum(dones)
        reward_sum = self._to_float_sum(rewards)

        self._iter_episodes += n_done
        self._iter_reward_sum += reward_sum

        # 按间隔 learn
        if iteration > 0 and iteration % self._learn_interval == 0:
            self._learn_iteration(iteration)

    def on_train_end(self, metrics: dict[str, Any] | None = None) -> dict | None:
        """训练结束 — 学习剩余数据 + 结束 session"""
        if not self._mem_available:
            return None
        # 学习剩余数据
        if self._iter_episodes > 0:
            self._learn_iteration(-1)

        if not self._sid:
            return None

        try:
            result = self.mem.end_session(
                session_id=self._sid,
                outcome_score=(metrics or {}).get("success_rate"),
            )
            logger.info("robotmem session 结束: %s, %d episodes",
                        self._sid, self._total_episodes)
            return result
        except Exception as e:
            logger.warning("robotmem end_session 失败: %s", e)
            return None

    def recall_tips(
        self,
        query: str,
        n: int = 5,
        context_filter: dict | None = None,
    ) -> list[dict]:
        """检索经验"""
        if not self._mem_available:
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

    @property
    def tips(self) -> list[dict]:
        """训练开始时 recall 的历史经验"""
        return self._recalled_tips

    def close(self):
        """释放资源"""
        if self.mem:
            try:
                self.mem.close()
            except Exception as e:
                logger.warning("robotmem close 失败: %s", e)
            self.mem = None

    # ── 内部方法 ──

    def _learn_iteration(self, iteration: int):
        """汇总并 learn 一批数据"""
        if self._iter_episodes == 0 or not self._mem_available:
            return

        self._total_episodes += self._iter_episodes
        avg_reward = (self._iter_reward_sum / self._iter_episodes
                      if self._iter_episodes > 0 else 0.0)

        insight = (
            f"Iteration {iteration}: "
            f"{self._iter_episodes} episodes, "
            f"avg_reward={avg_reward:.3f}, "
            f"total={self._total_episodes}"
        )

        context = {
            "task": {
                "reward": avg_reward,
                "episodes": self._iter_episodes,
                "total_episodes": self._total_episodes,
            },
            "params": {
                "iteration": {"value": iteration, "type": "scalar"},
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

        # 重置累积
        self._iter_episodes = 0
        self._iter_reward_sum = 0.0

    @staticmethod
    def _to_float_sum(x) -> float:
        """tensor/array/list → float sum"""
        if hasattr(x, 'sum'):
            val = x.sum()
            return float(val.item()) if hasattr(val, 'item') else float(val)
        if isinstance(x, (list, tuple)):
            return float(sum(x))
        return float(x)

    @staticmethod
    def _to_int_sum(x) -> int:
        """tensor/array/list → int sum"""
        if hasattr(x, 'sum'):
            val = x.sum()
            return int(val.item()) if hasattr(val, 'item') else int(val)
        if isinstance(x, (list, tuple)):
            return int(sum(x))
        return int(x)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
