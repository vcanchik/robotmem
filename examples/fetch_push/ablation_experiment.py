"""robotmem 消融实验 — 分离记忆 vs 策略加权的贡献

3 个消融变体 + 基线 + 完整模型：
  1. baseline   — Phase A: HeuristicPolicy, 无记忆
  2. full       — Phase C: recall(success_filter) → PhaseAwareMemoryPolicy(0.3)
  3. no_filter  — recall 不加 context_filter（返回所有记忆，含失败）
  4. random     — 用随机偏置替代 recall 返回的 bias
  5. weight_N   — 不同 MEMORY_WEIGHT (0.1/0.3/0.5/0.7)

每个 variant 共享 Phase B 写入的记忆（同一 DB），只变 Phase C 的 recall/policy 行为。

运行:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src python ablation_experiment.py --seed 42
  PYTHONPATH=../../src python ablation_experiment.py --seed 42 --episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

try:
    import gymnasium_robotics  # noqa: F401
    import gymnasium
    import numpy as np
except ImportError:
    print("需要安装: pip install gymnasium-robotics")
    sys.exit(1)

from robotmem.sdk import RobotMemory

from policies import HeuristicPolicy, PhaseAwareMemoryPolicy, MemoryPolicy

# 数据隔离
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-ablation")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLLECTION = "ablation"
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


def run_phase_b(env, policy, mem, episodes, session_id):
    """Phase B: 写入记忆（所有 variant 共享）"""
    successes = 0
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
        successes += int(success)

        ctx = build_context(obs, actions, success, len(actions), total_reward)
        dist = ctx["params"]["final_distance"]["value"]
        mem.learn(
            insight=f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
            context=ctx,
            session_id=session_id,
        )

        if (ep + 1) % 10 == 0:
            print(f"    Phase B [{ep+1}/{episodes}] 成功率: {successes/(ep+1):.0%}")

    return successes / episodes


def run_phase_c_variant(env, policy, mem, episodes, variant, weight, session_id, rng):
    """Phase C: 按 variant 类型执行不同的 recall+policy 组合"""
    successes = 0
    for ep in range(episodes):
        # 构造 active_policy
        if variant == "baseline":
            active_policy = policy
        elif variant == "full":
            recalled = mem.recall("push cube to target position", n=RECALL_N,
                                  context_filter={"task.success": True})
            active_policy = PhaseAwareMemoryPolicy(policy, recalled, weight)
        elif variant == "no_filter":
            recalled = mem.recall("push cube to target position", n=RECALL_N)
            active_policy = PhaseAwareMemoryPolicy(policy, recalled, weight)
        elif variant == "random":
            # 随机偏置（和真实 bias 维度一致：4D）
            random_bias = rng.uniform(-0.5, 0.5, size=4)
            fake_memory = [{"params": {
                "approach_velocity": {"value": random_bias[:3].tolist(), "type": "vector"},
                "grip_force": {"value": float(random_bias[3]), "type": "scalar"},
            }, "task": {"success": True}}]
            active_policy = PhaseAwareMemoryPolicy(policy, fake_memory, weight)
        else:
            raise ValueError(f"未知 variant: {variant}")

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
        successes += int(success)

        if (ep + 1) % 10 == 0:
            print(f"    {variant}(w={weight}) [{ep+1}/{episodes}] 成功率: {successes/(ep+1):.0%}")

    return successes / episodes


def parse_args():
    parser = argparse.ArgumentParser(description="robotmem 消融实验")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=100, help="每阶段 episode 数")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    episodes = args.episodes

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 1000)  # 独立 RNG 给 random variant

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem 消融实验 (Ablation Study)")
    print(f"seed={seed}, episodes={episodes}/phase")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    mem = RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx")
    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()

    t0 = time.time()
    results = {}

    try:
        # Phase A: 基线
        print("\n--- Phase A: 基线（无记忆）---")
        np.random.seed(seed)
        rate_a = run_phase_c_variant(env, policy, mem, episodes, "baseline", 0.3, None, rng)
        results["baseline"] = rate_a
        print(f"  → baseline: {rate_a:.0%}")

        # Phase B: 写入记忆（一次，所有 variant 共享）
        print("\n--- Phase B: 写入记忆 ---")
        np.random.seed(seed)
        with mem.session(context={"task": "push_to_target", "env": "FetchPush-v4"}) as sid:
            rate_b = run_phase_b(env, policy, mem, episodes, sid)
        results["phase_b"] = rate_b
        print(f"  → Phase B: {rate_b:.0%}")

        # Phase C variants（每个都重置 seed 确保环境一致）
        variants = [
            ("full", 0.3),
            ("no_filter", 0.3),
            ("random", 0.3),
            ("full", 0.1),
            ("full", 0.5),
            ("full", 0.7),
        ]

        for variant, weight in variants:
            key = f"{variant}_w{weight}" if variant == "full" else variant
            print(f"\n--- Phase C: {key} ---")
            np.random.seed(seed)
            rate = run_phase_c_variant(env, policy, mem, episodes, variant, weight, None, rng)
            results[key] = rate
            delta = rate - rate_a
            print(f"  → {key}: {rate:.0%} (Δ={delta:+.0%})")

        elapsed = time.time() - t0

        # 汇总
        print(f"\n{'=' * 60}")
        print("消融实验汇总")
        print(f"{'=' * 60}")
        print(f"{'Variant':<20} {'成功率':<10} {'Delta':<10}")
        print("-" * 40)
        for key, rate in results.items():
            if key == "phase_b":
                continue
            delta = rate - rate_a if key != "baseline" else 0
            print(f"{key:<20} {rate:.0%}{'':<5} {delta:+.0%}")

        print(f"\n耗时: {elapsed:.0f}s")

        # 保存
        output = {
            "seed": seed,
            "episodes": episodes,
            "results": {k: round(v, 4) for k, v in results.items()},
            "elapsed_s": round(elapsed, 1),
        }
        result_file = os.path.join(RESULTS_DIR, f"ablation_{seed}.json")
        with open(result_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n结果已保存: {result_file}")

        # 文本版
        txt_file = os.path.join(RESULTS_DIR, f"ablation_{seed}.txt")
        with open(txt_file, "w") as f:
            f.write(f"robotmem 消融实验 — seed={seed}, {episodes} ep/phase\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'Variant':<20} {'成功率':<10} {'Delta':<10}\n")
            f.write("-" * 40 + "\n")
            for key, rate in results.items():
                if key == "phase_b":
                    continue
                delta = rate - rate_a if key != "baseline" else 0
                f.write(f"{key:<20} {rate:.0%}       {delta:+.0%}\n")
            f.write(f"\n耗时: {elapsed:.0f}s\n")
        print(f"文本版: {txt_file}")

    finally:
        env.close()
        mem.close()


if __name__ == "__main__":
    main()
