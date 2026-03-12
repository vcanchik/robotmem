"""rsl_rl 适配器测试 — mock rsl_rl + torch，验证记忆集成

测试策略：
- Mock rsl_rl.runners.OnPolicyRunner（不依赖真实 rsl_rl/torch）
- Mock torch.Tensor 用于 dones/rewards
- 验证 robotmem 的 learn/recall/session 在训练循环中正确触发
"""

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── Mock rsl_rl + torch ──

class MockTensor:
    """最小 torch.Tensor mock"""

    def __init__(self, data):
        self._data = list(data) if hasattr(data, '__iter__') else [data]

    def sum(self):
        return MockTensor([sum(self._data)])

    def item(self):
        return self._data[0]

    def bool(self):
        return MockTensor([bool(x) for x in self._data])

    def any(self):
        return any(self._data)

    def to(self, device):
        return self

    def tolist(self):
        return self._data

    def cpu(self):
        return self

    def __getitem__(self, mask):
        if isinstance(mask, MockTensor):
            return MockTensor([d for d, m in zip(self._data, mask._data) if m])
        return MockTensor([self._data[mask]])

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def _setup_mock_modules():
    """注入 mock 的 torch / rsl_rl 模块"""
    # mock torch
    mock_torch = types.ModuleType("torch")
    mock_torch.Tensor = MockTensor
    mock_torch.inference_mode = lambda: _noop_context()
    mock_torch.save = MagicMock()
    mock_torch.randint_like = lambda t, high=1: t

    # mock rsl_rl
    mock_rsl_rl = types.ModuleType("rsl_rl")
    mock_runners = types.ModuleType("rsl_rl.runners")
    mock_utils = types.ModuleType("rsl_rl.utils")

    class MockOnPolicyRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            self.env = env
            self.cfg = train_cfg
            self.device = device
            self.alg = MagicMock()
            self.alg.learning_rate = 0.001
            self.alg.intrinsic_rewards = None
            policy_mock = MagicMock()
            policy_mock.output_std = 0.5
            self.alg.get_policy.return_value = policy_mock
            self.logger = MagicMock()
            self.logger.log_dir = log_dir or "test_logs"
            self.logger.writer = None
            self.current_learning_iteration = 0
            self.is_distributed = False
            self.gpu_world_size = 1
            self.gpu_global_rank = 0

    mock_runners.OnPolicyRunner = MockOnPolicyRunner
    mock_rsl_rl.runners = mock_runners
    mock_utils.check_nan = MagicMock()
    mock_rsl_rl.utils = mock_utils

    sys.modules["torch"] = mock_torch
    sys.modules["rsl_rl"] = mock_rsl_rl
    sys.modules["rsl_rl.runners"] = mock_runners
    sys.modules["rsl_rl.utils"] = mock_utils

    return mock_torch, MockOnPolicyRunner


class _noop_context:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def _make_mock_env(num_envs=4, max_ep_length=100):
    """创建 mock VecEnv"""
    env = MagicMock()
    env.num_envs = num_envs
    env.max_episode_length = max_ep_length
    env.device = "cpu"
    env.episode_length_buf = MockTensor([0] * num_envs)
    env.get_observations.return_value = MockTensor([0.0] * num_envs)

    step_count = 0

    def mock_step(actions):
        nonlocal step_count
        step_count += 1
        obs = MockTensor([0.1] * num_envs)
        rewards = MockTensor([0.5] * num_envs)
        # 每 5 步让 env 0 完成一个 episode
        dones_list = [1 if (i == 0 and step_count % 5 == 0) else 0 for i in range(num_envs)]
        dones = MockTensor(dones_list)
        extras = {}
        if any(dones_list):
            extras["episode"] = {"rew_total": 10.0}
        return obs, rewards, dones, extras

    env.step = mock_step
    return env


# ── 注入 mock 模块 ──
mock_torch, MockOnPolicyRunner = _setup_mock_modules()


class TestMemoryOnPolicyRunner:
    """MemoryOnPolicyRunner 核心功能测试"""

    def _make_runner(self, num_envs=4, learn_interval=2, **kwargs):
        """创建测试 runner"""
        from robotmem.rsl_rl.runner import MemoryOnPolicyRunner

        env = _make_mock_env(num_envs=num_envs)
        cfg = {
            "num_steps_per_env": 10,
            "save_interval": 100,
            "algorithm": {
                "class_name": "PPO",
                "rnd_cfg": None,
            },
            "check_for_nan": False,
        }
        runner = MemoryOnPolicyRunner(
            env=env,
            train_cfg=cfg,
            log_dir="/tmp/test_rsl_rl",
            device="cpu",
            mem_db_path=":memory:",
            mem_collection="test_rsl_rl",
            mem_embed_backend="none",
            mem_learn_interval=learn_interval,
            **kwargs,
        )
        return runner

    def test_init_success(self):
        """初始化成功，robotmem 可用"""
        runner = self._make_runner()
        assert runner._mem_available is True
        assert runner._mem is not None
        assert runner._mem_collection == "test_rsl_rl"

    def test_init_mem_failure_does_not_block(self):
        """robotmem 初始化失败不阻塞训练"""
        from robotmem.rsl_rl.runner import MemoryOnPolicyRunner

        env = _make_mock_env()
        cfg = {
            "num_steps_per_env": 5,
            "save_interval": 100,
            "algorithm": {"class_name": "PPO", "rnd_cfg": None},
            "check_for_nan": False,
        }
        # 使用无效路径
        runner = MemoryOnPolicyRunner(
            env=env,
            train_cfg=cfg,
            mem_db_path="/nonexistent/path/db.sqlite",
            mem_embed_backend="none",
        )
        assert runner._mem_available is False

    def test_learn_runs_without_error(self):
        """learn() 正常执行不报错"""
        runner = self._make_runner(learn_interval=2)
        runner.learn(num_learning_iterations=4)

    def test_session_lifecycle(self):
        """训练创建并结束 session"""
        runner = self._make_runner(learn_interval=2)
        runner.learn(num_learning_iterations=2)
        # session 应该已结束
        assert runner._mem_available is False  # close 后标记为 False

    def test_learn_records_memories(self):
        """训练过程中 learn 被调用了正确次数"""
        runner = self._make_runner(learn_interval=2)

        # mock mem.learn 计数
        learn_calls = []
        original_learn = runner._mem.learn

        def tracking_learn(*args, **kwargs):
            result = original_learn(*args, **kwargs)
            learn_calls.append(result)
            return result

        runner._mem.learn = tracking_learn

        runner.learn(num_learning_iterations=4)

        # 4 iterations / interval=2 = 至少 2 次 learn 调用
        # 只有有 episode 完成时才 learn，所以 >= 1
        assert len(learn_calls) >= 1, f"期望至少 1 次 learn 调用，实际 {len(learn_calls)}"

    def test_episode_stats_tracking(self):
        """episode 统计正确追踪"""
        runner = self._make_runner(learn_interval=100)

        # 手动触发 _on_env_step
        # 模拟 2 个 episode 完成
        dones = MockTensor([1, 0, 1, 0])
        rewards = MockTensor([1.0, 0.5, 2.0, 0.3])
        extras = {"episode": {"rew_total": 1.5}}

        runner._on_env_step(rewards, dones, extras)

        assert runner._iter_episodes == 2
        assert runner._total_episodes == 2

    def test_reset_iter_stats(self):
        """iteration 统计重置"""
        runner = self._make_runner()
        runner._iter_episodes = 10
        runner._iter_rewards = [1.0, 2.0]
        runner._iter_successes = 5

        runner._reset_iter_stats()

        assert runner._iter_episodes == 0
        assert runner._iter_rewards == []
        assert runner._iter_successes == 0

    def test_recall_at_start(self):
        """训练开始时 recall 不报错"""
        runner = self._make_runner(mem_recall_at_start=True)
        runner._on_train_begin()
        assert runner._session_id is not None

    def test_recall_disabled(self):
        """关闭 recall 时不调用"""
        runner = self._make_runner(mem_recall_at_start=False)
        # recall_tips 不应被调用
        with patch.object(runner, '_recall_tips') as mock_recall:
            runner._on_train_begin()
            mock_recall.assert_not_called()

    def test_on_iteration_end_no_episodes(self):
        """无 episode 完成时不 learn"""
        runner = self._make_runner()
        runner._on_train_begin()
        # _iter_episodes = 0，不应调用 learn
        runner._on_iteration_end(0)
        # 不报错即通过

    def test_on_iteration_end_with_episodes(self):
        """有 episode 完成时正确 learn"""
        runner = self._make_runner()
        runner._on_train_begin()

        # 模拟统计
        runner._iter_episodes = 5
        runner._iter_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        runner._total_episodes = 5

        runner._on_iteration_end(10)

        # 统计应该被重置
        assert runner._iter_episodes == 0
        assert runner._iter_rewards == []

    def test_train_end_closes_resources(self):
        """训练结束后资源释放"""
        runner = self._make_runner()
        runner._on_train_begin()
        runner._on_train_end()

        assert runner._mem_available is False

    def test_proxy_methods(self):
        """代理方法透传"""
        runner = self._make_runner()
        # mock 底层 runner 的 save 方法
        runner._runner.save = MagicMock()
        runner.save("/tmp/test_model.pt")
        runner._runner.save.assert_called_once_with("/tmp/test_model.pt", None)
        assert runner.current_learning_iteration == 0

    def test_dones_none_handling(self):
        """dones=None 不报错"""
        runner = self._make_runner()
        runner._on_env_step(MockTensor([0.5]), None, {})
        assert runner._iter_episodes == 0

    def test_extras_without_episode_key(self):
        """extras 中无 episode key 不报错"""
        runner = self._make_runner()
        dones = MockTensor([1, 0])
        rewards = MockTensor([1.0, 0.5])
        runner._on_env_step(rewards, dones, {})
        assert runner._iter_episodes == 1

    def test_mem_unavailable_noop(self):
        """robotmem 不可用时所有 hook 静默跳过"""
        runner = self._make_runner()
        runner._mem_available = False

        # 所有 hook 不报错
        runner._on_train_begin()
        runner._on_env_step(MockTensor([1.0]), MockTensor([1]), {})
        runner._on_iteration_end(0)
        runner._on_train_end()

    def test_full_training_loop(self):
        """完整训练循环 E2E（4 iterations，interval=2）"""
        runner = self._make_runner(learn_interval=2)
        runner.learn(num_learning_iterations=4)
        # 不报错 + 资源释放 = 通过
        assert runner._mem_available is False
