"""robotmem Gymnasium Wrapper — 自动记录 episode 经验

任何 Gymnasium 兼容环境都可以用这个 wrapper 自动记录经验到 robotmem。
覆盖 Gymnasium-Robotics (FetchPush/Reach/Slide)、ManiSkill、robosuite 等。

用法:
    import gymnasium as gym
    from robotmem.gymnasium import RobotMemWrapper

    env = gym.make("FetchPush-v4")
    env = RobotMemWrapper(env, collection="fetch_push")

    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            # tips = info.get("robotmem_tips", [])  # recall_on_reset=True 时可用
    env.close()

兼容性:
    - gymnasium >= 0.26
    - 支持 GoalEnv (Gymnasium-Robotics)
    - 支持向量化环境 (需要在 make_vec 之前包装)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# lazy import — 用户没装 gymnasium 时 import robotmem 不报错
_gymnasium = None


def _ensure_gymnasium():
    global _gymnasium
    if _gymnasium is None:
        try:
            import gymnasium
            _gymnasium = gymnasium
        except ImportError:
            raise ImportError(
                "gymnasium is required for RobotMemWrapper. "
                "Install with: pip install gymnasium"
            )
    return _gymnasium


class RobotMemWrapper:
    """通用 Gymnasium 环境记忆包装器 — 自动记录 episode 经验

    设计原则（宪法）：
    - 第 1 条 机器能用：context 结构化，params/spatial/task 三区域
    - 第 3 条 能删就删：不改环境核心逻辑，只在 step/reset 边界加 hook
    - 第 4 条 坏了就喊：robotmem 异常不中断训练，warn 后继续
    - 第 5 条 加不改：任何 Gym 环境直接包装，不改环境代码
    """

    def __init__(
        self,
        env,
        collection: str = "gymnasium",
        db_path: str | None = None,
        embed_backend: str = "onnx",
        learn_on_done: bool = True,
        recall_on_reset: bool = False,
        recall_query: str = "successful episode strategy",
        recall_n: int = 5,
        save_trajectory: bool = False,
        max_trajectory_steps: int = 50,
    ):
        """
        Args:
            env: Gymnasium 环境实例
            collection: robotmem 集合名
            db_path: 自定义 DB 路径（默认 ~/.robotmem/memory.db）
            embed_backend: embedding 后端
            learn_on_done: episode 结束时自动 learn
            recall_on_reset: reset 时自动 recall，放入 info["robotmem_tips"]
            recall_query: recall 查询字符串
            recall_n: recall 返回条数
            save_trajectory: 是否保存轨迹为 perception
            max_trajectory_steps: 轨迹最大步数（截断）
        """
        gymnasium = _ensure_gymnasium()

        # 验证 env 是 Gymnasium 环境
        if not hasattr(env, 'step') or not hasattr(env, 'reset'):
            raise TypeError("env must be a Gymnasium-compatible environment")

        self.env = env
        self._learn_on_done = learn_on_done
        self._recall_on_reset = recall_on_reset
        self._recall_query = recall_query
        self._recall_n = recall_n
        self._save_trajectory = save_trajectory
        self._max_trajectory_steps = max_trajectory_steps

        # robotmem SDK（异常不阻塞环境使用）
        try:
            from robotmem.sdk import RobotMemory
            self.mem = RobotMemory(
                db_path=db_path,
                collection=collection,
                embed_backend=embed_backend,
            )
            self._mem_available = True
        except Exception as e:
            logger.warning("robotmem 初始化失败，环境将不使用记忆: %s", e)
            self.mem = None
            self._mem_available = False
        self._sid: str | None = None
        self._episode_count = 0

        # episode 累积
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_trajectory: list[list[float]] = []
        self._ep_success = False

        # 代理属性
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, 'metadata', {})
        self.render_mode = getattr(env, 'render_mode', None)
        self.spec = getattr(env, 'spec', None)

    # ── Session 管理 ──

    def start_session(self, context: dict[str, Any] | None = None) -> str:
        """手动开始 session（可选，不调用也能用）"""
        if not self._mem_available:
            return ""
        ctx = {"framework": "gymnasium"}
        if context:
            ctx.update(context)
        try:
            self._sid = self.mem.start_session(context=ctx)
            logger.info("robotmem session 开始: %s", self._sid)
        except Exception as e:
            logger.warning("robotmem start_session 失败: %s", e)
            self._sid = None
        return self._sid or ""

    def end_session(self, outcome_score: float | None = None) -> dict | None:
        """手动结束 session（可选）"""
        if not self._mem_available or not self._sid:
            return None
        try:
            result = self.mem.end_session(
                session_id=self._sid,
                outcome_score=outcome_score,
            )
            logger.info("robotmem session 结束: %s, %d episodes",
                        self._sid, self._episode_count)
            return result
        except Exception as e:
            logger.warning("robotmem end_session 失败: %s", e)
            return None

    # ── Gym API 代理 ──

    def step(self, action):
        """执行一步，episode 结束时自动 learn"""
        result = self.env.step(action)

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            # 兼容旧版 Gym API
            obs, reward, done, info = result[:4]
            terminated = done
            truncated = False

        self._ep_reward += reward
        self._ep_steps += 1

        # 检查 GoalEnv 的 is_success
        if isinstance(info, dict) and info.get("is_success"):
            self._ep_success = True

        # 记录轨迹
        if self._save_trajectory and self._ep_steps <= self._max_trajectory_steps:
            act_list = action.tolist() if hasattr(action, 'tolist') else list(action)
            self._ep_trajectory.append(act_list)

        # episode 结束 → learn
        if done and self._learn_on_done and self._mem_available:
            self._on_episode_done(info)

        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def reset(self, **kwargs):
        """重置环境，可选 recall 历史经验"""
        result = self.env.reset(**kwargs)

        # 重置 episode 累积
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_trajectory = []
        self._ep_success = False

        # 可选：recall 经验放入 info
        if self._recall_on_reset and self._mem_available:
            tips = self.recall_tips(self._recall_query, n=self._recall_n)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
                info["robotmem_tips"] = tips
                return obs, info

        return result

    def render(self):
        return self.env.render()

    def close(self):
        """关闭环境和 robotmem"""
        try:
            self.env.close()
        finally:
            if self.mem:
                try:
                    self.mem.close()
                except Exception as e:
                    logger.warning("robotmem close 失败: %s", e)
                self.mem = None

    # ── 记忆操作 ──

    def recall_tips(
        self,
        query: str,
        n: int = 5,
        context_filter: dict | None = None,
    ) -> list[dict]:
        """手动检索经验"""
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

    # ── 内部方法 ──

    def _on_episode_done(self, info: dict):
        """episode 结束时自动 learn"""
        self._episode_count += 1
        success = self._ep_success or info.get("is_success", False)

        insight = (
            f"Episode {self._episode_count}: "
            f"{'成功' if success else '失败'}, "
            f"reward={self._ep_reward:.2f}, "
            f"{self._ep_steps} steps"
        )

        context = {
            "task": {
                "success": success,
                "reward": self._ep_reward,
                "steps": self._ep_steps,
            },
        }

        # 提取 GoalEnv 的目标信息
        if isinstance(info, dict):
            for key in ("achieved_goal", "desired_goal"):
                if key in info:
                    val = info[key]
                    if hasattr(val, 'tolist'):
                        val = val.tolist()
                    context.setdefault("spatial", {})[key] = val

        try:
            self.mem.learn(
                insight=insight,
                context=context,
                session_id=self._sid,
            )
        except Exception as e:
            logger.warning("robotmem learn 失败: %s", e)

        # 可选：保存轨迹
        if self._save_trajectory and self._ep_trajectory:
            try:
                self.mem.save_perception(
                    description=f"Episode {self._episode_count} 轨迹: "
                                f"{len(self._ep_trajectory)} 步",
                    perception_type="procedural",
                    data=json.dumps(self._ep_trajectory),
                    session_id=self._sid,
                )
            except Exception as e:
                logger.warning("robotmem save_perception 失败: %s", e)

    # ── 代理方法 ──

    def __getattr__(self, name):
        """未定义的属性代理到底层 env"""
        return getattr(self.env, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
