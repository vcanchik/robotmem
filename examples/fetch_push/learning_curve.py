"""robotmem 学习曲线 + 记忆命中分析

输出：
  1. Phase C 逐 episode 成功率曲线（PNG）
  2. recall 命中的 memory_id 频次分布
  3. 验证 recall 是否每次返回相同 top-5

运行:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src python learning_curve.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter

try:
    import gymnasium_robotics  # noqa: F401
    import gymnasium
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("需要安装: pip install gymnasium-robotics matplotlib")
    sys.exit(1)

from robotmem.sdk import RobotMemory

from policies import HeuristicPolicy, PhaseAwareMemoryPolicy

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-curve")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLLECTION = "curve"
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def build_context(obs, actions, success, steps, total_reward):
    """构建 context dict"""
    recent = actions[-10:] if len(actions) >= 10 else actions
    avg_action = np.mean(recent, axis=0) if recent else np.zeros(4)
    return {
        "params": {
            "approach_velocity": {"value": avg_action[0:3].tolist(), "type": "vector"},
            "grip_force": {"value": float(avg_action[3]), "type": "scalar"},
            "final_distance": {
                "value": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
                "unit": "m",
            },
        },
        "spatial": {
            "grip_position": obs["observation"][0:3].tolist(),
            "object_position": obs["observation"][3:6].tolist(),
            "target_position": obs["desired_goal"].tolist(),
        },
        "task": {
            "name": "push_to_target",
            "success": bool(success),
            "steps": steps,
            "total_reward": float(total_reward),
        },
    }


def run_phase_a(env, policy, episodes):
    """Phase A: 记录逐 episode 成功"""
    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        for _ in range(50):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        results.append(int(info.get("is_success", False)))
    return results


def run_phase_b(env, policy, mem, episodes, session_id):
    """Phase B: 写入记忆"""
    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        actions = []
        total_reward = 0.0
        for _ in range(50):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
            total_reward += reward
            if terminated or truncated:
                break

        success = info.get("is_success", False)
        results.append(int(success))

        ctx = build_context(obs, actions, success, len(actions), total_reward)
        dist = ctx["params"]["final_distance"]["value"]
        mem.learn(
            insight=f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
            context=ctx,
            session_id=session_id,
        )

        if (ep + 1) % 20 == 0:
            rate = sum(results) / len(results)
            print(f"    Phase B [{ep+1}/{episodes}] 成功率: {rate:.0%}")

    return results


def run_phase_c(env, policy, mem, episodes, session_id):
    """Phase C: 逐 episode 记录成功 + recall 命中分析"""
    results = []
    recall_hits = []  # 每次 recall 返回的 memory_id 列表
    recall_latencies = []

    for ep in range(episodes):
        # recall
        recalled = mem.recall(
            "push cube to target position",
            n=RECALL_N,
            context_filter={"task.success": True},
        )

        # 记录命中的 memory_id
        ids = [m.get("id", -1) for m in recalled]
        recall_hits.append(ids)

        # 记录延迟（从 _rrf_score 推断不了延迟，但 recall 返回的 dict 可能有 query_ms）
        # recall 返回 list[dict]，不直接包含 query_ms
        # 用时间戳测量
        t0 = time.monotonic()
        _ = mem.recall("push cube to target position", n=RECALL_N,
                       context_filter={"task.success": True})
        latency = (time.monotonic() - t0) * 1000
        recall_latencies.append(latency)

        active_policy = PhaseAwareMemoryPolicy(policy, recalled, MEMORY_WEIGHT) if recalled else policy

        obs, _ = env.reset()
        actions = []
        total_reward = 0.0
        for _ in range(50):
            action = active_policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
            total_reward += reward
            if terminated or truncated:
                break

        success = info.get("is_success", False)
        results.append(int(success))

        # learn
        ctx = build_context(obs, actions, success, len(actions), total_reward)
        dist = ctx["params"]["final_distance"]["value"]
        mem.learn(
            insight=f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
            context=ctx,
            session_id=session_id,
        )

        if (ep + 1) % 20 == 0:
            rate = sum(results) / len(results)
            print(f"    Phase C [{ep+1}/{episodes}] 成功率: {rate:.0%}")

    return results, recall_hits, recall_latencies


def cumulative_rate(results):
    """逐 episode 累计成功率"""
    rates = []
    total = 0
    for i, r in enumerate(results):
        total += r
        rates.append(total / (i + 1))
    return rates


def plot_learning_curve(a_results, c_results, seed, output_path):
    """画学习曲线"""
    a_rates = cumulative_rate(a_results)
    c_rates = cumulative_rate(c_results)
    episodes = range(1, len(a_results) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, [r * 100 for r in a_rates], label="Phase A (baseline)", color="#2196F3", linewidth=2)
    ax.plot(episodes, [r * 100 for r in c_rates], label="Phase C (with memory)", color="#FF5722", linewidth=2)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative Success Rate (%)", fontsize=12)
    ax.set_title(f"robotmem Learning Curve — FetchPush-v4 (seed={seed})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # 标注最终值
    final_a = a_rates[-1] * 100
    final_c = c_rates[-1] * 100
    ax.annotate(f"{final_a:.0f}%", xy=(len(a_results), final_a), fontsize=10, color="#2196F3")
    ax.annotate(f"{final_c:.0f}%", xy=(len(c_results), final_c), fontsize=10, color="#FF5722")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  学习曲线已保存: {output_path}")


def analyze_recall_hits(recall_hits, output_path):
    """分析 recall 命中分布"""
    all_ids = []
    for ids in recall_hits:
        all_ids.extend(ids)

    counter = Counter(all_ids)
    unique_ids = len(counter)
    total_recalls = len(recall_hits)

    # 是否每次返回相同 top-5？
    unique_sets = set()
    for ids in recall_hits:
        unique_sets.add(tuple(sorted(ids)))
    is_fixed = len(unique_sets) == 1

    lines = [
        "robotmem 记忆命中分析",
        "=" * 50,
        f"总 recall 次数: {total_recalls}",
        f"唯一 memory_id 数: {unique_ids}",
        f"唯一 ID 组合数: {len(unique_sets)}",
        f"是否固定偏置 (每次返回相同 top-5): {'是' if is_fixed else '否'}",
        "",
        "--- memory_id 命中频次 (top-20) ---",
    ]

    for mid, count in counter.most_common(20):
        pct = count / len(all_ids) * 100
        lines.append(f"  id={mid}: {count} 次 ({pct:.1f}%)")

    lines.append("")
    lines.append("--- 唯一 ID 组合 ---")
    for i, s in enumerate(sorted(unique_sets)):
        if i >= 10:
            lines.append(f"  ... 共 {len(unique_sets)} 个组合")
            break
        lines.append(f"  {list(s)}")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"  命中分析已保存: {output_path}")
    print(f"\n{text}")

    return {
        "unique_ids": unique_ids,
        "unique_sets": len(unique_sets),
        "is_fixed_bias": is_fixed,
        "top_5_ids": counter.most_common(5),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="robotmem 学习曲线 + 记忆命中分析")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=100, help="每阶段 episode 数")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    episodes = args.episodes

    random.seed(seed)
    np.random.seed(seed)

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem 学习曲线 + 记忆命中分析")
    print(f"seed={seed}, episodes={episodes}/phase")
    print("=" * 60)

    mem = RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx")
    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()

    t0 = time.time()

    try:
        # Phase A
        print("\n--- Phase A ---")
        a_results = run_phase_a(env, policy, episodes)
        print(f"  Phase A 成功率: {sum(a_results)/len(a_results):.0%}")

        # Phase B
        print("\n--- Phase B ---")
        with mem.session(context={"task": "push_to_target"}) as sid:
            b_results = run_phase_b(env, policy, mem, episodes, sid)

            # Phase C
            print("\n--- Phase C ---")
            c_results, recall_hits, recall_latencies = run_phase_c(env, policy, mem, episodes, sid)

        elapsed = time.time() - t0

        # 学习曲线图
        plot_learning_curve(
            a_results, c_results, seed,
            os.path.join(RESULTS_DIR, f"learning_curve_{seed}.png"),
        )

        # 命中分析
        hit_analysis = analyze_recall_hits(
            recall_hits,
            os.path.join(RESULTS_DIR, f"recall_hits_{seed}.txt"),
        )

        # recall 延迟统计
        latencies = np.array(recall_latencies)
        print(f"\n--- Recall 延迟统计 ({len(latencies)} 次) ---")
        print(f"  mean: {np.mean(latencies):.1f}ms")
        print(f"  p50:  {np.median(latencies):.1f}ms")
        print(f"  p95:  {np.percentile(latencies, 95):.1f}ms")
        print(f"  p99:  {np.percentile(latencies, 99):.1f}ms")
        print(f"  max:  {np.max(latencies):.1f}ms")

        # 保存综合结果
        summary = {
            "seed": seed,
            "episodes": episodes,
            "phase_a_rate": sum(a_results) / len(a_results),
            "phase_b_rate": sum(b_results) / len(b_results),
            "phase_c_rate": sum(c_results) / len(c_results),
            "delta_ca": sum(c_results) / len(c_results) - sum(a_results) / len(a_results),
            "recall_hits": hit_analysis,
            "recall_latency_ms": {
                "mean": round(float(np.mean(latencies)), 1),
                "p50": round(float(np.median(latencies)), 1),
                "p95": round(float(np.percentile(latencies, 95)), 1),
                "p99": round(float(np.percentile(latencies, 99)), 1),
                "max": round(float(np.max(latencies)), 1),
            },
            "elapsed_s": round(elapsed, 1),
        }
        summary_file = os.path.join(RESULTS_DIR, f"learning_curve_{seed}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n综合结果: {summary_file}")

    finally:
        env.close()
        mem.close()


if __name__ == "__main__":
    main()
