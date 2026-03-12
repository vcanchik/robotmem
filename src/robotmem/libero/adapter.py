"""robotmem LIBERO 辅助 — Lifelong Robot Learning 跨任务记忆管理

LIBERO 的核心场景是 lifelong learning：训练多个任务，需要在任务间迁移知识。
robotmem 为 LIBERO 提供跨任务经验持久化和检索。

用法:
    from robotmem.libero import RobotMemLifelongHelper

    helper = RobotMemLifelongHelper(collection="libero_spatial")

    # 按 LIBERO benchmark 顺序训练多个任务
    for task_id, task_name in enumerate(task_suite):
        helper.on_task_begin(task_name, task_id)

        # 训练前检索跨任务经验
        tips = helper.recall_cross_task(f"skills for {task_name}")

        for episode in range(num_episodes):
            # ... 训练 episode ...
            helper.on_episode_end({
                "episode": episode,
                "reward": reward,
                "success": success,
                "context": {"task": {"name": task_name}},
            })

        helper.on_task_end({"success_rate": task_success_rate})

    helper.close()

兼容性:
    - 无 LIBERO 依赖（纯辅助类）
    - 支持 LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-100
    - 支持任何 lifelong/continual learning 框架
"""

from __future__ import annotations

import json
import logging
from typing import Any

from robotmem.sdk import RobotMemory

logger = logging.getLogger(__name__)


class RobotMemLifelongHelper:
    """LIBERO Lifelong Learning 辅助 — 跨任务记忆管理

    设计原则（宪法）：
    - 第 1 条 机器能用：跨任务 recall 支持 context_filter 按任务名过滤
    - 第 3 条 能删就删：不依赖 LIBERO 代码
    - 第 4 条 坏了就喊：异常不中断训练
    - 第 5 条 加不改：适用于任何 lifelong learning 框架
    """

    def __init__(
        self,
        collection: str = "libero",
        db_path: str | None = None,
        embed_backend: str = "onnx",
        recall_cross_task_n: int = 10,
    ):
        """
        Args:
            collection: robotmem 集合名
            db_path: 自定义 DB 路径
            embed_backend: embedding 后端
            recall_cross_task_n: 跨任务 recall 返回条数
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
        self._recall_n = recall_cross_task_n

        self._sid: str | None = None
        self._current_task: str | None = None
        self._current_task_id: int | None = None
        self._task_episode_count = 0
        self._task_successes = 0
        self._total_tasks = 0

        # 跨任务统计
        self._task_results: list[dict] = []

    def on_task_begin(
        self,
        task_name: str,
        task_id: int = 0,
        config: dict[str, Any] | None = None,
    ) -> str:
        """新任务开始 — 创建 session"""
        self._current_task = task_name
        self._current_task_id = task_id
        self._task_episode_count = 0
        self._task_successes = 0

        if not self._mem_available:
            return ""
        config = config or {}
        try:
            self._sid = self.mem.start_session(context={
                "framework": "libero",
                "task": task_name,
                "task_id": task_id,
                "task_order": self._total_tasks,
                **{k: v for k, v in config.items()
                   if k not in ("task", "task_id")},
            })
            logger.info("robotmem task 开始: %s (id=%d, order=%d)",
                        task_name, task_id, self._total_tasks)
        except Exception as e:
            logger.warning("robotmem on_task_begin 失败: %s", e)
            self._sid = None

        return self._sid or ""

    def on_episode_end(
        self,
        episode_data: dict[str, Any],
        trajectory: list | None = None,
    ) -> dict | None:
        """Episode 结束 — 记录经验"""
        self._task_episode_count += 1
        reward = episode_data.get("reward", 0.0)
        success = episode_data.get("success", reward > 0)

        if success:
            self._task_successes += 1

        if not self._mem_available:
            return None

        ep = episode_data.get("episode", self._task_episode_count)
        insight = (
            f"Task '{self._current_task}' Episode {ep}: "
            f"{'成功' if success else '失败'}, reward={reward:.2f}"
        )

        context = episode_data.get("context", {})
        if "task" not in context:
            context["task"] = {}
        context["task"]["name"] = self._current_task
        context["task"]["task_id"] = self._current_task_id
        context["task"]["success"] = success
        context["task"]["reward"] = reward

        try:
            result = self.mem.learn(
                insight=insight,
                context=context,
                session_id=self._sid,
            )
        except Exception as e:
            logger.warning("robotmem learn 失败: %s", e)
            result = None

        # 可选：保存轨迹
        if trajectory is not None:
            try:
                traj_data = (trajectory[:50] if len(trajectory) > 50
                             else trajectory)
                self.mem.save_perception(
                    description=f"Task '{self._current_task}' ep {ep} "
                                f"轨迹: {len(trajectory)} 步",
                    perception_type="procedural",
                    data=json.dumps(traj_data),
                    session_id=self._sid,
                )
            except Exception as e:
                logger.warning("robotmem save_perception 失败: %s", e)

        return result

    def on_task_end(
        self,
        metrics: dict[str, Any] | None = None,
    ) -> dict | None:
        """任务结束 — 结束 session，记录任务级统计"""
        metrics = metrics or {}
        self._total_tasks += 1

        if not self._mem_available:
            return None

        success_rate = (self._task_successes / self._task_episode_count
                        if self._task_episode_count > 0 else 0.0)

        task_result = {
            "task": self._current_task,
            "task_id": self._current_task_id,
            "episodes": self._task_episode_count,
            "success_rate": success_rate,
            "order": self._total_tasks - 1,
        }
        self._task_results.append(task_result)

        # 学习任务级总结
        try:
            self.mem.learn(
                insight=(
                    f"Task '{self._current_task}' 完成: "
                    f"success_rate={success_rate:.0%}, "
                    f"{self._task_episode_count} episodes, "
                    f"第 {self._total_tasks} 个任务"
                ),
                context={
                    "task": {
                        "name": self._current_task,
                        "success_rate": success_rate,
                        "episodes": self._task_episode_count,
                        "order": self._total_tasks - 1,
                    },
                },
                session_id=self._sid,
            )
        except Exception as e:
            logger.warning("robotmem task summary learn 失败: %s", e)

        # 结束 session
        result = None
        if self._sid:
            try:
                result = self.mem.end_session(
                    session_id=self._sid,
                    outcome_score=metrics.get("success_rate", success_rate),
                )
                logger.info("robotmem task 结束: %s, success_rate=%.0f%%",
                            self._current_task, success_rate * 100)
            except Exception as e:
                logger.warning("robotmem end_session 失败: %s", e)

        return result

    def recall_cross_task(
        self,
        query: str,
        n: int | None = None,
        task_filter: str | None = None,
    ) -> list[dict]:
        """跨任务检索经验 — LIBERO 的核心价值

        Args:
            query: 查询字符串
            n: 返回条数
            task_filter: 按特定任务名过滤（None = 搜索所有任务）
        """
        if not self._mem_available:
            return []
        n = n or self._recall_n
        context_filter = None
        if task_filter:
            context_filter = {"task.name": task_filter}

        try:
            return self.mem.recall(
                query=query,
                n=n,
                context_filter=context_filter,
            )
        except Exception as e:
            logger.warning("robotmem recall 失败: %s", e)
            return []

    def recall_successful(
        self,
        query: str = "successful task strategy",
        n: int = 5,
    ) -> list[dict]:
        """只检索成功的经验"""
        return self.recall_cross_task(
            query=query,
            n=n,
        )

    @property
    def task_results(self) -> list[dict]:
        """所有已完成任务的结果"""
        return self._task_results

    @property
    def forward_transfer_data(self) -> dict:
        """前向迁移数据 — LIBERO 标准评估指标"""
        if len(self._task_results) < 2:
            return {"tasks": self._task_results, "forward_transfer": None}

        rates = [t["success_rate"] for t in self._task_results]
        # 简单的前向迁移指标：后续任务是否比首个任务更快学会
        return {
            "tasks": self._task_results,
            "first_task_rate": rates[0],
            "last_task_rate": rates[-1],
            "avg_rate": sum(rates) / len(rates),
        }

    def close(self):
        """释放资源"""
        if self.mem:
            try:
                self.mem.close()
            except Exception as e:
                logger.warning("robotmem close 失败: %s", e)
            self.mem = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
