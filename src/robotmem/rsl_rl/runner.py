"""robotmem rsl_rl 适配器 — OnPolicyRunner 子类，episode 边界自动记忆

rsl_rl (ETH Zurich) 没有 callback 系统，通过子类化 OnPolicyRunner 在
训练循环中插入 robotmem 的 learn/recall。

用法:
    # 替换 OnPolicyRunner → MemoryOnPolicyRunner（一行改动）
    from robotmem.rsl_rl import MemoryOnPolicyRunner

    runner = MemoryOnPolicyRunner(
        env, train_cfg, log_dir="logs",
        # robotmem 参数
        mem_collection="unitree_g1",
        mem_learn_interval=10,      # 每 10 个 iteration 汇总一次
        mem_recall_at_start=True,   # 训练开始时 recall 历史经验
    )
    runner.learn(num_learning_iterations=1000)

兼容性:
    - rsl_rl >= 2.0（legged_gym / unitree_rl_gym / unitree_rl_lab）
    - torch 为可选依赖，import 时才检查
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from robotmem.sdk import RobotMemory

logger = logging.getLogger(__name__)


class MemoryOnPolicyRunner:
    """OnPolicyRunner + robotmem 记忆增强

    子类化 rsl_rl.runners.OnPolicyRunner，在训练循环中插入：
    - 训练开始：start_session + recall 历史经验
    - 每 N 个 iteration：learn 汇总统计
    - 训练结束：end_session（自动巩固 + proactive recall）

    设计原则（宪法）：
    - 第 1 条 机器能用：记忆 actionable（reward/成功率/参数）
    - 第 3 条 能删就删：不改 OnPolicyRunner 核心逻辑，只加 hook
    - 第 4 条 坏了就喊：robotmem 异常不中断训练，warn 后继续
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        # robotmem 参数
        mem_db_path: str | None = None,
        mem_collection: str = "rsl_rl",
        mem_embed_backend: str = "onnx",
        mem_learn_interval: int = 10,
        mem_recall_at_start: bool = True,
        mem_recall_n: int = 5,
    ) -> None:
        """初始化 MemoryOnPolicyRunner

        Args:
            env, train_cfg, log_dir, device: 原 OnPolicyRunner 参数
            mem_db_path: robotmem DB 路径（None = 默认）
            mem_collection: robotmem collection 名
            mem_embed_backend: embedding 后端
            mem_learn_interval: 每 N 个 iteration 汇总记忆
            mem_recall_at_start: 训练开始时是否 recall
            mem_recall_n: recall 返回条数
        """
        # 延迟 import — rsl_rl/torch 是可选依赖
        try:
            from rsl_rl.runners import OnPolicyRunner as _Base
        except ImportError as e:
            raise ImportError(
                "rsl_rl 未安装。请先安装: pip install rsl_rl\n"
                "或从源码: https://github.com/leggedrobotics/rsl_rl"
            ) from e

        # 保存为实例属性供 learn() 使用
        self._BaseClass = _Base

        # 初始化父类
        self._runner = _Base(env, train_cfg, log_dir=log_dir, device=device)

        # 代理父类属性
        self.env = self._runner.env
        self.cfg = self._runner.cfg
        self.device = self._runner.device
        self.alg = self._runner.alg
        self.logger = self._runner.logger

        # robotmem 配置
        self._mem_collection = mem_collection
        self._mem_learn_interval = max(1, mem_learn_interval)
        self._mem_recall_at_start = mem_recall_at_start
        self._mem_recall_n = mem_recall_n

        # 初始化 robotmem（异常不阻塞训练）
        try:
            self._mem = RobotMemory(
                db_path=mem_db_path,
                collection=mem_collection,
                embed_backend=mem_embed_backend,
            )
            self._mem_available = True
            logger.info("robotmem 初始化成功: collection=%s", mem_collection)
        except Exception as e:
            logger.warning("robotmem 初始化失败，训练将不使用记忆: %s", e)
            self._mem = None
            self._mem_available = False

        # episode 统计追踪
        self._session_id: str | None = None
        self._iter_rewards: list[float] = []
        self._iter_episodes: int = 0
        self._iter_successes: int = 0
        self._total_episodes: int = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """增强的训练循环 — 在 OnPolicyRunner.learn() 基础上插入 robotmem hook

        保持 rsl_rl 原有逻辑不变，在以下位置插入记忆操作：
        1. 训练前：start_session + recall 历史经验
        2. 每步 dones 后：统计 episode 完成
        3. 每 N iteration：learn 汇总
        4. 训练后：end_session
        """
        import torch

        # ── 训练前：robotmem session ──
        self._on_train_begin()

        try:
            self._learn_loop(num_learning_iterations, init_at_random_ep_len)
        finally:
            # 保证异常时也释放资源（宪法第 4 条）
            self._on_train_end()

    def _learn_loop(self, num_learning_iterations: int, init_at_random_ep_len: bool) -> None:
        """内部训练循环 — 分离 try/finally 边界"""
        import torch

        runner = self._runner

        if init_at_random_ep_len:
            runner.env.episode_length_buf = torch.randint_like(
                runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
            )

        obs = runner.env.get_observations().to(runner.device)
        runner.alg.train_mode()

        if runner.is_distributed:
            runner.alg.broadcast_parameters()

        runner.logger.init_logging_writer()

        start_it = runner.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for _ in range(runner.cfg["num_steps_per_env"]):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))

                    if runner.cfg.get("check_for_nan", True):
                        from rsl_rl.utils import check_nan
                        check_nan(obs, rewards, dones)

                    obs = obs.to(runner.device)
                    rewards = rewards.to(runner.device)
                    dones = dones.to(runner.device)

                    runner.alg.process_env_step(obs, rewards, dones, extras)

                    intrinsic_rewards = (
                        runner.alg.intrinsic_rewards
                        if runner.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )
                    runner.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                    # ── robotmem hook: episode 边界统计 ──
                    self._on_env_step(rewards, dones, extras)

                stop = time.time()
                collect_time = stop - start
                start = stop

                runner.alg.compute_returns(obs)

            loss_dict = runner.alg.update()

            stop = time.time()
            learn_time = stop - start
            runner.current_learning_iteration = it

            runner.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=runner.alg.learning_rate,
                action_std=runner.alg.get_policy().output_std,
                rnd_weight=(
                    runner.alg.rnd.weight
                    if runner.cfg["algorithm"]["rnd_cfg"]
                    else None
                ),
            )

            # ── robotmem hook: 每 N iteration 汇总 ──
            if (it + 1) % self._mem_learn_interval == 0:
                self._on_iteration_end(it)

            if runner.logger.writer is not None and it % runner.cfg["save_interval"] == 0:
                runner.save(os.path.join(runner.logger.log_dir, f"model_{it}.pt"))

        # Save final model
        if runner.logger.writer is not None:
            runner.save(
                os.path.join(runner.logger.log_dir, f"model_{runner.current_learning_iteration}.pt")
            )
            runner.logger.stop_logging_writer()

    # ── robotmem hook 方法 ──

    def _on_train_begin(self) -> None:
        """训练开始 — start_session + recall"""
        if not self._mem_available or self._mem is None:
            return

        mem = self._mem
        try:
            env_name = getattr(self.env, '__class__', type(self.env)).__name__
            self._session_id = mem.start_session(context={
                "framework": "rsl_rl",
                "env": env_name,
                "num_envs": getattr(self.env, 'num_envs', 0),
                "policy": self.cfg.get("algorithm", {}).get("class_name", "PPO"),
            })
            logger.info("robotmem session 开始: %s", self._session_id)
        except Exception as e:
            logger.warning("robotmem start_session 失败，跳过 recall: %s", e)
            self._session_id = None
            return

        # recall 历史经验（仅 session 创建成功时）
        if self._mem_recall_at_start:
            self._recall_tips()

    def _on_env_step(self, rewards: Any, dones: Any, extras: dict) -> None:
        """每步环境交互后 — 统计 episode 完成

        Args:
            rewards: shape (num_envs,)
            dones: shape (num_envs,) — True 表示 episode 结束
            extras: 环境附加信息
        """
        import torch

        if not self._mem_available:
            return

        # 统计完成的 episode 数
        if isinstance(dones, torch.Tensor):
            n_done = int(dones.sum().item())
        else:
            n_done = int(sum(dones)) if dones is not None else 0

        if n_done > 0:
            self._iter_episodes += n_done
            self._total_episodes += n_done

            # 提取 episode 奖励（如果 extras 中有）
            extras_has_reward = False
            if isinstance(extras, dict) and "episode" in extras:
                ep_info = extras["episode"]
                if isinstance(ep_info, dict):
                    # 用 sentinel 区分 key 不存在 vs 值为 0.0
                    _sentinel = object()
                    rew = ep_info.get("rew_total", _sentinel)
                    if rew is _sentinel:
                        rew = ep_info.get("reward", _sentinel)
                    if rew is not _sentinel:
                        extras_has_reward = True
                        if isinstance(rew, (int, float)):
                            self._iter_rewards.append(float(rew))
                        elif hasattr(rew, '__iter__'):
                            self._iter_rewards.extend(float(r) for r in rew)

            # 从 rewards tensor 统计（兜底，仅在 extras 无 episode reward 时）
            if isinstance(rewards, torch.Tensor) and not extras_has_reward:
                done_mask = dones.bool() if isinstance(dones, torch.Tensor) else None
                if done_mask is not None and done_mask.any():
                    done_rewards = rewards[done_mask]
                    self._iter_rewards.extend(done_rewards.cpu().tolist())

    def _on_iteration_end(self, iteration: int) -> None:
        """每 N iteration — 汇总记忆"""
        if not self._mem_available or self._mem is None or self._iter_episodes == 0:
            self._reset_iter_stats()
            return

        mem = self._mem
        avg_reward = (
            sum(self._iter_rewards) / len(self._iter_rewards)
            if self._iter_rewards else 0.0
        )

        insight = (
            f"规律：Iteration {iteration}: {self._iter_episodes} episodes 完成, "
            f"avg_reward={avg_reward:.3f}, total_episodes={self._total_episodes}"
        )

        context = {
            "task": {
                "iteration": iteration,
                "episodes": self._iter_episodes,
                "total_episodes": self._total_episodes,
            },
            "params": {
                "avg_reward": {"value": round(avg_reward, 4), "type": "scalar"},
                "learning_rate": {
                    "value": float(self.alg.learning_rate),
                    "type": "scalar",
                },
            },
        }

        try:
            result = mem.learn(
                insight=insight,
                context=context,
                session_id=self._session_id,
            )
            if result.get("status") == "created":
                logger.info(
                    "robotmem learn: iter=%d, episodes=%d, avg_reward=%.3f",
                    iteration, self._iter_episodes, avg_reward,
                )
        except Exception as e:
            logger.warning("robotmem learn 失败: %s", e)

        self._reset_iter_stats()

    def _on_train_end(self) -> None:
        """训练结束 — end_session"""
        if not self._mem_available or self._mem is None or not self._session_id:
            self._close_mem()
            return

        mem = self._mem
        try:
            result = mem.end_session(session_id=self._session_id)
            related = result.get("related_memories", [])
            logger.info(
                "robotmem session 结束: %s, total_episodes=%d, related=%d",
                self._session_id, self._total_episodes, len(related),
            )
            if related:
                logger.info("历史相关经验:")
                for m in related[:3]:
                    logger.info("  - %s", m.get("content", "")[:100])
        except Exception as e:
            logger.warning("robotmem end_session 失败: %s", e)

        self._close_mem()

    def _recall_tips(self) -> None:
        """训练开始时 recall 历史经验"""
        if not self._mem_available or self._mem is None:
            return

        mem = self._mem
        env_name = getattr(self.env, '__class__', type(self.env)).__name__
        query = f"{env_name} training experience reward"

        try:
            tips = mem.recall(query=query, n=self._mem_recall_n)
            if tips:
                logger.info("robotmem recall %d 条历史经验:", len(tips))
                for tip in tips:
                    logger.info("  - [%.2f] %s", tip.get("confidence", 0), tip.get("content", "")[:120])
            else:
                logger.info("robotmem: 无历史经验（首次训练）")
        except Exception as e:
            logger.warning("robotmem recall 失败: %s", e)

    def _reset_iter_stats(self) -> None:
        """重置 iteration 级统计"""
        self._iter_rewards.clear()
        self._iter_episodes = 0
        self._iter_successes = 0

    def _close_mem(self) -> None:
        """释放 robotmem 资源"""
        if self._mem is not None:
            try:
                self._mem.close()
            except Exception as e:
                logger.warning("robotmem close 失败: %s", e)
            self._mem = None
            self._mem_available = False

    # ── 代理方法（透传到底层 runner）──

    def save(self, path: str, infos: dict | None = None) -> None:
        """保存模型"""
        self._runner.save(path, infos)

    def load(self, path: str, **kwargs) -> dict:
        """加载模型"""
        return self._runner.load(path, **kwargs)

    def get_inference_policy(self, device: str | None = None):
        """获取推理策略"""
        return self._runner.get_inference_policy(device)

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """导出 JIT 模型"""
        self._runner.export_policy_to_jit(path, filename)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """导出 ONNX 模型"""
        self._runner.export_policy_to_onnx(path, filename, verbose)

    @property
    def current_learning_iteration(self) -> int:
        return self._runner.current_learning_iteration

    @current_learning_iteration.setter
    def current_learning_iteration(self, value: int) -> None:
        self._runner.current_learning_iteration = value

    def __getattr__(self, name: str):
        """未代理的属性透传到底层 runner"""
        return getattr(self._runner, name)
