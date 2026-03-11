"""robotmem PushT Demo — SDK 版

运行方式:
  source .venv-pusht/bin/activate
  PYTHONPATH=src python examples/pusht/demo.py
  PYTHONPATH=src python examples/pusht/demo.py --seed 42

三阶段（默认各 100 episodes）:
  Phase A: 基线（heuristic + 噪声探索，无记忆）
  Phase B: 记忆写入（learn，记录 peak_coverage + peak_step）
  Phase C: 记忆利用（recall → stop-at-peak 停推策略）

指标: 平均 final coverage（0~1），不是二值 success。
PushT reward = T 形块与目标的重叠率，连续值。

记忆改善机制（stop-at-peak）:
  Phase B 记录每个 episode 的 peak_coverage 和 peak_step
  → Phase C recall 成功经验，学习:
    1. peak 通常多高 → 停推阈值（达到后观察下降趋势）
    2. peak 通常在哪步 → 超时保护
  → 当 coverage 从 peak 下降 50% 时停止推送，保留已有 coverage
  → 核心洞察: 启发式经常达到高 peak 后过冲到 0，记忆教 agent 何时该停
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time

try:
    import gym_pusht  # noqa: F401
    import gymnasium
    import numpy as np
except ImportError:
    print("需要安装: pip install gym-pusht 'pymunk>=6.6.0,<7.0'")
    sys.exit(1)

from robotmem.sdk import RobotMemory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import PushTHeuristicPolicy, PushTMemoryPolicy

# 数据隔离
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-demo")
DB_PATH = os.path.join(DB_DIR, "memory.db")

COLLECTION = "pusht-demo"
MEMORY_WEIGHT = 0.3
RECALL_N = 5
COVERAGE_THRESHOLD = 0.05  # peak coverage > 5% 视为可参考经验
WAYPOINT_INTERVAL = 10  # 每 10 步记一个 waypoint


def build_context(obs_initial, obs_final, actions, final_reward, peak_reward, peak_step):
    """构建 context dict — params/spatial/task 三区域"""
    # 轨迹 waypoints: 每 WAYPOINT_INTERVAL 步取一个动作
    waypoints = []
    for i in range(0, len(actions), WAYPOINT_INTERVAL):
        waypoints.append(actions[i])

    return {
        "params": {
            "waypoints": {"value": waypoints, "type": "trajectory"},
            "initial_block_state": {
                "value": [float(obs_initial[2]), float(obs_initial[3]), float(obs_initial[4])],
                "type": "vector",
            },
            "final_block_state": {
                "value": [float(obs_final[2]), float(obs_final[3]), float(obs_final[4])],
                "type": "vector",
            },
            "peak_coverage": {"value": float(peak_reward), "type": "scalar"},
            "peak_step": {"value": int(peak_step), "type": "scalar"},
        },
        "spatial": {
            "x": float(obs_initial[2]),  # 初始 block x (用于 spatial_sort 匹配)
            "y": float(obs_initial[3]),  # 初始 block y
        },
        "task": {
            "name": "pusht",
            "coverage": float(final_reward),
            "peak_coverage": float(peak_reward),
            "success": bool(peak_reward > COVERAGE_THRESHOLD),
        },
    }


def run_episode(env, policy, phase, mem, session_id=None):
    """执行单个 episode，返回 final coverage

    stop-at-peak 策略让 agent 在 coverage 下降时停推，
    因此 final coverage ≈ peak coverage（agent 停住，block 不再移动）。
    """
    # Phase C: recall 成功经验
    recalled = []
    if phase == "C":
        recalled = mem.recall(
            "high coverage push trajectory",
            n=RECALL_N,
            context_filter={"task.success": True},
        )

    # Phase C: 记忆驱动停推策略；Phase A/B: 基础策略
    if phase == "C" and recalled:
        active_policy = PushTMemoryPolicy(policy, recalled, MEMORY_WEIGHT)
    else:
        active_policy = policy

    obs_initial, _ = env.reset()
    obs = obs_initial
    actions = []
    final_reward = 0.0
    peak_reward = 0.0
    peak_step = 0

    # Phase C: 重置 episode 内状态
    if hasattr(active_policy, "reset_episode"):
        active_policy.reset_episode()

    for step in range(300):
        action = active_policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(list(action) if isinstance(action, np.ndarray) else action)
        final_reward = reward

        if reward > peak_reward:
            peak_reward = reward
            peak_step = step

        # Phase C: 让策略追踪 coverage，决定是否停推
        if hasattr(active_policy, "update_coverage"):
            active_policy.update_coverage(reward)

        if terminated or truncated:
            break

    # Phase B/C: learn 经验
    if phase in ("B", "C"):
        ctx = build_context(obs_initial, obs, actions, final_reward, peak_reward, peak_step)
        mem.learn(
            insight=f"PushT: coverage={final_reward:.3f}, peak={peak_reward:.3f}, {len(actions)} 步",
            context=ctx,
            session_id=session_id,
        )

    return final_reward


def run_phase(env, policy, phase, mem, episodes, session_id=None):
    """执行一个 Phase，返回平均 coverage"""
    coverages = []
    for ep in range(episodes):
        cov = run_episode(env, policy, phase, mem, session_id)
        coverages.append(cov)
        if (ep + 1) % 10 == 0:
            avg = np.mean(coverages)
            print(f"  Phase {phase} [{ep+1}/{episodes}] 平均 coverage: {avg:.3f}")
    return float(np.mean(coverages))


def parse_args():
    parser = argparse.ArgumentParser(
        description="robotmem PushT Demo — 记忆驱动推块（SDK 版）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现运行）")
    parser.add_argument(
        "--episodes", type=int, default=100, help="每阶段 episode 数（默认 100）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    episodes = args.episodes

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    print("=" * 50)
    print("robotmem PushT Demo (SDK)")
    print(f"每阶段 {episodes} episodes")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print(f"DB: {DB_PATH}")
    print("=" * 50)

    mem = RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx")
    env = gymnasium.make("gym_pusht/PushT-v0", obs_type="state")
    policy = PushTHeuristicPolicy(noise_scale=15.0)

    t0 = time.time()

    try:
        # Phase A: 基线（无记忆）
        print("\n--- Phase A: 基线（无记忆）---")
        cov_a = run_phase(env, policy, "A", mem, episodes)

        # Phase B+C: 记忆写入 + 轨迹回放（同一 session）
        with mem.session(context={"task": "pusht", "env": "PushT-v0"}) as sid:
            print("\n--- Phase B: 写入记忆（轨迹） ---")
            cov_b = run_phase(env, policy, "B", mem, episodes, sid)

            print("\n--- Phase C: 利用记忆（回放轨迹） ---")
            cov_c = run_phase(env, policy, "C", mem, episodes, sid)

        elapsed = time.time() - t0
        delta = cov_c - cov_a

        print(f"\n{'=' * 50}")
        print("演示结果")
        print(f"{'=' * 50}")
        print(f"  Phase A (基线):  {cov_a:.3f} ({cov_a:.1%})")
        print(f"  Phase B (写入):  {cov_b:.3f} ({cov_b:.1%})")
        print(f"  Phase C (回放):  {cov_c:.3f} ({cov_c:.1%})")
        print(f"  提升 (C - A):    {delta:+.3f} ({delta:+.1%})")
        print(f"  耗时: {elapsed:.0f}s")
        print(f"\n  注: 本示例为 API 用法教程。严格实验请参考 experiment.py。")
        print(f"\n数据存储于: {DB_DIR}")
    finally:
        env.close()
        mem.close()


if __name__ == "__main__":
    main()
