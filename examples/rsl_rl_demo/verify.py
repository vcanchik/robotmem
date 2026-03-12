"""rsl_rl 适配器 Docker 验证 — 真实 torch + rsl_rl，三级测试

Level 1: import 测试
Level 2: hook 级测试（真实 torch tensor）
Level 3: 完整 OnPolicyRunner 集成（真实 rsl_rl PPO）
"""

import sys
import traceback

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

results = []


def report(level: str, name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    print(f"  [{level}] {status} {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")
    results.append(ok)


# ── Level 1: Import ──────────────────────────────────────────

print("\n=== Level 1: Import 测试 ===")

try:
    import torch
    report("L1", f"torch {torch.__version__}", True)
except Exception as e:
    report("L1", "torch", False, str(e))

try:
    import rsl_rl
    ver = getattr(rsl_rl, "__version__", "unknown")
    report("L1", f"rsl_rl {ver}", True)
except Exception as e:
    report("L1", "rsl_rl", False, str(e))

try:
    from rsl_rl.runners import OnPolicyRunner
    report("L1", "OnPolicyRunner", True)
except Exception as e:
    report("L1", "OnPolicyRunner", False, str(e))

try:
    from robotmem.rsl_rl import MemoryOnPolicyRunner
    report("L1", "MemoryOnPolicyRunner", True)
except Exception as e:
    report("L1", "MemoryOnPolicyRunner", False, str(e))

try:
    from robotmem.sdk import RobotMemory
    report("L1", "RobotMemory SDK", True)
except Exception as e:
    report("L1", "RobotMemory SDK", False, str(e))


# ── Level 2: Hook 级测试（真实 torch tensor）──────────────────

print("\n=== Level 2: Hook 级测试（真实 torch tensor）===")

try:
    from robotmem.rsl_rl.runner import MemoryOnPolicyRunner

    # 构造最小 mock env（返回真实 torch tensor）
    class TorchMockEnv:
        def __init__(self, num_envs=4, num_obs=8, num_actions=2):
            self.num_envs = num_envs
            self.num_obs = num_obs
            self.num_actions = num_actions
            self.num_privileged_obs = num_obs
            self.max_episode_length = 100
            self.device = "cpu"
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
            self._step = 0

        def get_observations(self):
            return torch.randn(self.num_envs, self.num_obs)

        def step(self, actions):
            self._step += 1
            obs = torch.randn(self.num_envs, self.num_obs)
            rewards = torch.rand(self.num_envs)
            # 每 5 步 env0 完成 episode
            dones_list = [1.0 if (i == 0 and self._step % 5 == 0) else 0.0
                          for i in range(self.num_envs)]
            dones = torch.tensor(dones_list)
            extras = {}
            if any(d > 0 for d in dones_list):
                extras["episode"] = {"rew_total": float(rewards.sum())}
            return obs, rewards, dones, extras

    # 只测试 hook 方法（不走完整 learn 循环）
    # 先用 mock rsl_rl runner 来绕过 OnPolicyRunner 复杂初始化
    from unittest.mock import MagicMock

    env = TorchMockEnv()

    # 直接构造，用 :memory: DB
    runner = MemoryOnPolicyRunner.__new__(MemoryOnPolicyRunner)
    runner.env = env
    runner.cfg = {"algorithm": {"class_name": "PPO"}}
    runner.device = "cpu"
    runner.alg = MagicMock()
    runner.alg.learning_rate = 0.001
    runner.logger = MagicMock()

    # 初始化 robotmem
    runner._mem_collection = "test_docker"
    runner._mem_learn_interval = 2
    runner._mem_recall_at_start = False
    runner._mem_recall_n = 5
    runner._session_id = None
    runner._iter_rewards = []
    runner._iter_episodes = 0
    runner._iter_successes = 0
    runner._total_episodes = 0

    try:
        runner._mem = RobotMemory(
            db_path=":memory:",
            collection="test_docker",
            embed_backend="none",
        )
        runner._mem_available = True
    except Exception as e:
        report("L2", "RobotMemory init", False, str(e))
        raise

    report("L2", "RobotMemory init（:memory:）", True)

    # test _on_train_begin
    runner._on_train_begin()
    report("L2", "_on_train_begin（session 创建）", runner._session_id is not None,
           f"session_id={runner._session_id}")

    # test _on_env_step with real torch tensors
    rewards = torch.tensor([1.0, 0.5, 2.0, 0.3])
    dones = torch.tensor([1.0, 0.0, 1.0, 0.0])
    extras = {"episode": {"rew_total": 1.5}}
    runner._on_env_step(rewards, dones, extras)
    report("L2", "_on_env_step（真实 torch tensor）", runner._iter_episodes == 2,
           f"iter_episodes={runner._iter_episodes}, iter_rewards={runner._iter_rewards}")

    # test _on_iteration_end
    runner._on_iteration_end(10)
    report("L2", "_on_iteration_end（learn 调用）", runner._iter_episodes == 0,
           "统计已重置")

    # test _on_train_end
    runner._on_train_begin()  # 重新 start session
    runner._on_train_end()
    report("L2", "_on_train_end（资源释放）", runner._mem_available is False)

except Exception as e:
    report("L2", "Hook 测试", False, traceback.format_exc())


# ── Level 3: 完整 rsl_rl 集成 ──────────────────────────────

print("\n=== Level 3: 完整 rsl_rl 训练循环 ===")

try:
    from robotmem.rsl_rl import MemoryOnPolicyRunner
    from rsl_rl.runners import OnPolicyRunner
    import copy
    import inspect

    init_sig = inspect.signature(OnPolicyRunner.__init__)
    params = list(init_sig.parameters.keys())
    report("L3", f"OnPolicyRunner.__init__ 参数", True,
           f"params={params}")

    # rsl_rl v5 用 TensorDict + MLPModel（不是 v2 的 ActorCritic）
    from tensordict import TensorDict

    class SimpleVecEnv:
        """最小向量化环境 — 满足 rsl_rl v5 OnPolicyRunner 接口"""

        def __init__(self, num_envs=4, num_obs=8, num_actions=2):
            self.num_envs = num_envs
            self.num_obs = num_obs
            self.num_actions = num_actions
            self.max_episode_length = 50
            self.device = "cpu"
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
            self.cfg = {"env_name": "SimpleVecEnv"}
            self._step = 0

        def get_observations(self):
            return TensorDict({
                "obs": torch.randn(self.num_envs, self.num_obs),
            }, batch_size=[self.num_envs])

        def step(self, actions):
            self._step += 1
            self.episode_length_buf += 1
            obs = TensorDict({
                "obs": torch.randn(self.num_envs, self.num_obs),
            }, batch_size=[self.num_envs])
            rewards = torch.rand(self.num_envs) - 0.5
            dones = torch.zeros(self.num_envs)
            for i in range(self.num_envs):
                if self.episode_length_buf[i] >= self.max_episode_length:
                    dones[i] = 1.0
                    self.episode_length_buf[i] = 0
            if self._step % 10 == 0:
                dones[0] = 1.0
                self.episode_length_buf[0] = 0
            extras = {}
            if dones.any():
                extras["episode"] = {"rew_total": float(rewards[dones.bool()].sum())}
            return obs, rewards, dones, extras

    # rsl_rl v5 配置格式
    num_obs = 8
    num_actions = 2

    train_cfg = {
        "seed": 42,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
        },
        "obs_groups": {
            "actor": ["obs"],
            "critic": ["obs"],
        },
        "multi_gpu": None,
        "num_steps_per_env": 8,
        "save_interval": 100,
        "check_for_nan": False,
    }

    # 先试原生 OnPolicyRunner
    try:
        env1 = SimpleVecEnv(num_envs=4, num_obs=num_obs, num_actions=num_actions)
        base_runner = OnPolicyRunner(env1, copy.deepcopy(train_cfg),
                                     log_dir="/tmp/rsl_test", device="cpu")
        report("L3", "OnPolicyRunner 初始化", True)
    except Exception as e:
        report("L3", "OnPolicyRunner 初始化", False, traceback.format_exc())
        raise

    # 原生 runner 跑 2 个 iteration（确认 env+cfg 正确）
    try:
        base_runner.learn(num_learning_iterations=2)
        report("L3", "OnPolicyRunner.learn(2)", True)
    except Exception as e:
        report("L3", "OnPolicyRunner.learn(2)", False, traceback.format_exc())
        raise

    # 试 MemoryOnPolicyRunner
    try:
        env2 = SimpleVecEnv(num_envs=4, num_obs=num_obs, num_actions=num_actions)
        mem_runner = MemoryOnPolicyRunner(
            env=env2,
            train_cfg=copy.deepcopy(train_cfg),
            log_dir="/tmp/rsl_mem_test",
            device="cpu",
            mem_db_path=":memory:",
            mem_collection="docker_verify",
            mem_embed_backend="none",
            mem_learn_interval=2,
            mem_recall_at_start=True,
        )
        report("L3", "MemoryOnPolicyRunner 初始化", True,
               f"mem_available={mem_runner._mem_available}")
    except Exception as e:
        report("L3", "MemoryOnPolicyRunner 初始化", False, traceback.format_exc())
        raise

    # 跑 4 个 iteration
    try:
        mem_runner.learn(num_learning_iterations=4)
        report("L3", "learn(4 iterations) 完成", True,
               f"mem_available={mem_runner._mem_available}（应为 False，资源已释放）")
    except Exception as e:
        report("L3", "learn(4 iterations)", False, traceback.format_exc())

except Exception as e:
    report("L3", "完整集成", False, traceback.format_exc())


# ── 汇总 ──────────────────────────────────────────────────

print("\n" + "=" * 50)
passed = sum(results)
total = len(results)
all_ok = all(results)
color = "\033[92m" if all_ok else "\033[91m"
print(f"{color}结果: {passed}/{total} 通过\033[0m")

if all_ok:
    print("\n🎉 rsl_rl 适配器验证全部通过！")
else:
    print("\n⚠️  部分测试失败，请检查上方输出")

sys.exit(0 if all_ok else 1)
