"""PushT 消融实验 — 记忆学习值 vs 固定默认值

回答 Paul Graham 的质疑：
  stop-at-peak 用固定值（0.03/200）和记忆学习值效果差多少？
  如果一样 → 记忆在 PushT 上的价值为零。

四组对比：
  A: 基线（无 stop-at-peak）
  B: 写入记忆（无 stop-at-peak）
  C-fixed: stop-at-peak + 固定默认值（不用记忆）
  C-memory: stop-at-peak + 记忆学习值

运行：
  source .venv-pusht/bin/activate
  PYTHONPATH=src python examples/pusht/ablation.py
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import time

import numpy as np

try:
    import gym_pusht  # noqa: F401
    import gymnasium
except ImportError:
    print("需要: pip install gym-pusht 'pymunk>=6.6.0,<7.0'")
    sys.exit(1)

from robotmem.sdk import RobotMemory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import PushTHeuristicPolicy, PushTMemoryPolicy

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-ablation")
DB_PATH = os.path.join(DB_DIR, "memory.db")
COLLECTION = "pusht-ablation"
COVERAGE_THRESHOLD = 0.05
WAYPOINT_INTERVAL = 10


def build_context(obs_initial, obs_final, actions, final_reward, peak_reward, peak_step):
    waypoints = []
    for i in range(0, len(actions), WAYPOINT_INTERVAL):
        waypoints.append(actions[i])
    return {
        "params": {
            "waypoints": {"value": waypoints, "type": "trajectory"},
            "peak_coverage": {"value": float(peak_reward), "type": "scalar"},
            "peak_step": {"value": int(peak_step), "type": "scalar"},
        },
        "spatial": {"x": float(obs_initial[2]), "y": float(obs_initial[3])},
        "task": {
            "name": "pusht",
            "coverage": float(final_reward),
            "peak_coverage": float(peak_reward),
            "success": bool(peak_reward > COVERAGE_THRESHOLD),
        },
    }


def run_episode(env, policy, learn_mem=None, session_id=None):
    """执行单个 episode，返回 final_reward"""
    obs_initial, _ = env.reset()
    obs = obs_initial
    actions = []
    final_reward = 0.0
    peak_reward = 0.0
    peak_step = 0

    if hasattr(policy, "reset_episode"):
        policy.reset_episode()

    for step in range(300):
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(list(action) if isinstance(action, np.ndarray) else action)
        final_reward = reward
        if reward > peak_reward:
            peak_reward = reward
            peak_step = step
        if hasattr(policy, "update_coverage"):
            policy.update_coverage(reward)
        if terminated or truncated:
            break

    if learn_mem is not None:
        ctx = build_context(obs_initial, obs, actions, final_reward, peak_reward, peak_step)
        learn_mem.learn(
            insight=f"PushT: coverage={final_reward:.3f}, peak={peak_reward:.3f}",
            context=ctx,
            session_id=session_id,
        )

    return final_reward


def run_phase(env, policy, episodes, learn_mem=None, session_id=None):
    coverages = []
    for ep in range(episodes):
        cov = run_episode(env, policy, learn_mem, session_id)
        coverages.append(cov)
    return float(np.mean(coverages))


def main():
    seeds = [42, 123, 456, 789, 1024]
    episodes = 50

    print("=" * 60)
    print("PushT 消融实验：记忆学习值 vs 固定默认值")
    print(f"Seeds: {seeds}, Episodes/phase: {episodes}")
    print("=" * 60)

    results = {s: {} for s in seeds}

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        # 清空 DB
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)

        mem = RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx")
        env = gymnasium.make("gym_pusht/PushT-v0", obs_type="state")
        base_policy = PushTHeuristicPolicy(noise_scale=15.0)

        try:
            # Phase A: 基线
            random.seed(seed)
            np.random.seed(seed)
            cov_a = run_phase(env, base_policy, episodes)

            # Phase B: 写入记忆（收集经验）
            random.seed(seed + 1000)
            np.random.seed(seed + 1000)
            with mem.session(context={"task": "pusht"}) as sid:
                cov_b = run_phase(env, base_policy, episodes, learn_mem=mem, session_id=sid)

            # C-fixed: stop-at-peak + 固定默认值（空记忆 → 触发默认 0.03/200）
            random.seed(seed + 2000)
            np.random.seed(seed + 2000)
            fixed_policy = PushTMemoryPolicy(base_policy, [], 0.3)
            print(f"  [seed={seed}] C-fixed 参数: stop={fixed_policy.stop_coverage:.4f}, max_steps={fixed_policy.max_push_steps}")
            cov_c_fixed = run_phase(env, fixed_policy, episodes)

            # C-memory: stop-at-peak + 记忆学习值
            random.seed(seed + 2000)  # 同样的 seed，公平对比
            np.random.seed(seed + 2000)
            recalled = mem.recall(
                "high coverage push trajectory",
                n=5,
                context_filter={"task.success": True},
            )
            memory_policy = PushTMemoryPolicy(base_policy, recalled, 0.3)
            print(f"  [seed={seed}] C-memory 参数: stop={memory_policy.stop_coverage:.4f}, max_steps={memory_policy.max_push_steps}, recalled={len(recalled)}")
            cov_c_memory = run_phase(env, memory_policy, episodes)

            results[seed] = {
                "A": cov_a,
                "B": cov_b,
                "C-fixed": cov_c_fixed,
                "C-memory": cov_c_memory,
            }

            print(f"  Seed {seed}: A={cov_a:.3f}  B={cov_b:.3f}  C-fixed={cov_c_fixed:.3f}  C-memory={cov_c_memory:.3f}")

        finally:
            env.close()
            mem.close()

    # 汇总
    print(f"\n{'=' * 60}")
    print("消融实验结果")
    print(f"{'=' * 60}")
    print(f"{'Seed':>6} | {'A (基线)':>10} | {'B (写入)':>10} | {'C-fixed':>10} | {'C-memory':>10} | {'fixed-A':>10} | {'mem-A':>10} | {'mem-fixed':>10}")
    print("-" * 95)

    all_a, all_b, all_cf, all_cm = [], [], [], []
    for seed in seeds:
        r = results[seed]
        a, b, cf, cm = r["A"], r["B"], r["C-fixed"], r["C-memory"]
        all_a.append(a)
        all_b.append(b)
        all_cf.append(cf)
        all_cm.append(cm)
        print(f"{seed:>6} | {a:>10.3f} | {b:>10.3f} | {cf:>10.3f} | {cm:>10.3f} | {cf-a:>+10.3f} | {cm-a:>+10.3f} | {cm-cf:>+10.3f}")

    ma, mb, mcf, mcm = np.mean(all_a), np.mean(all_b), np.mean(all_cf), np.mean(all_cm)
    print("-" * 95)
    print(f"{'平均':>6} | {ma:>10.3f} | {mb:>10.3f} | {mcf:>10.3f} | {mcm:>10.3f} | {mcf-ma:>+10.3f} | {mcm-ma:>+10.3f} | {mcm-mcf:>+10.3f}")

    print(f"\n关键对比:")
    print(f"  C-fixed vs A:   {mcf-ma:+.3f} ({mcf-ma:+.1%})  ← 固定值 stop-at-peak 的提升")
    print(f"  C-memory vs A:  {mcm-ma:+.3f} ({mcm-ma:+.1%})  ← 记忆学习 stop-at-peak 的提升")
    print(f"  C-memory vs C-fixed: {mcm-mcf:+.3f} ({mcm-mcf:+.1%})  ← 记忆的额外价值")

    print(f"\n判定:")
    diff = mcm - mcf
    if diff > 0.01:
        print(f"  记忆学习值 > 固定值 (+{diff:.1%}) → 记忆有自动调参价值")
    elif diff > -0.01:
        print(f"  记忆学习值 ≈ 固定值 ({diff:+.1%}) → 记忆价值在于发现机制，非调参")
    else:
        print(f"  记忆学习值 < 固定值 ({diff:+.1%}) → 记忆在 PushT 上无额外价值")


if __name__ == "__main__":
    main()
