"""Consolidation 消融实验 — 巩固对 Phase C recall 质量的影响

对比两个 variant：
  1. no_consolidate: Phase B → Phase C（同一 session，不触发巩固）
  2. with_consolidate: Phase B → end_session(巩固) → Phase C

关键：当前实验中 consolidation 在 end_session 中运行，发生在 Phase C 之后。
本实验将 consolidation 移到 B→C 之间，验证巩固是否提升 recall 质量。

运行:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src python -u consolidation_ablation.py --seed 42
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

from policies import HeuristicPolicy, PhaseAwareMemoryPolicy

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def build_context(obs, actions, success, steps, total_reward):
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
    """Phase B: 写入记忆"""
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
    return successes / episodes


def run_phase_c(env, policy, mem, episodes):
    """Phase C: recall + 执行"""
    successes = 0
    for ep in range(episodes):
        recalled = mem.recall(
            "push cube to target position",
            n=RECALL_N,
            context_filter={"task.success": True},
        )
        active_policy = PhaseAwareMemoryPolicy(policy, recalled, MEMORY_WEIGHT) if recalled else policy

        obs, _ = env.reset()
        for _ in range(50):
            action = active_policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        successes += int(info.get("is_success", False))
    return successes / episodes


def count_active_memories(mem):
    """计算当前活跃记忆数"""
    row = mem._db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE status='active' AND collection=?",
        (mem._collection,),
    ).fetchone()
    return row[0] if row else 0


def run_variant(seed, episodes, consolidate_between):
    """运行单个 variant"""
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f".robotmem-consol-{'on' if consolidate_between else 'off'}")
    db_path = os.path.join(db_dir, "memory.db")

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    os.makedirs(db_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    mem = RobotMemory(db_path=db_path, collection="consol_exp", embed_backend="onnx")
    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()

    try:
        # Phase A: 基线
        np.random.seed(seed)
        rate_a = run_phase_c(env, policy, mem, episodes)

        # Phase B: 写入记忆
        np.random.seed(seed)
        with mem.session(context={"task": "push_to_target"}) as sid:
            rate_b = run_phase_b(env, policy, mem, episodes, sid)

            active_before = count_active_memories(mem)

            if consolidate_between:
                # 手动触发 end_session（包含 consolidation）
                # 不能直接调 end_session 因为 context manager 会重复调用
                # 所以在 session 内手动调 consolidate
                from robotmem.ops.memories import consolidate_session
                consolidated = consolidate_session(mem._db.conn, sid, mem._collection)
            else:
                consolidated = {"merged_groups": 0, "superseded_count": 0}

            active_after = count_active_memories(mem)

        # Phase C: recall + 执行（env seed 重置确保一致性）
        np.random.seed(seed)
        rate_c = run_phase_c(env, policy, mem, episodes)

        return {
            "rate_a": rate_a,
            "rate_b": rate_b,
            "rate_c": rate_c,
            "delta_ca": rate_c - rate_a,
            "active_before_consolidate": active_before,
            "active_after_consolidate": active_after,
            "consolidated": consolidated,
        }
    finally:
        env.close()
        mem.close()
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidation 消融实验")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=100, help="每阶段 episode 数")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    episodes = args.episodes
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Consolidation 消融实验")
    print(f"seed={seed}, episodes={episodes}/phase")
    print("=" * 60)

    t0 = time.time()

    # Variant 1: 不巩固
    print("\n--- Variant: NO consolidation ---")
    r_off = run_variant(seed, episodes, consolidate_between=False)
    print(f"  Phase A: {r_off['rate_a']:.0%} | Phase C: {r_off['rate_c']:.0%} | "
          f"Delta: {r_off['delta_ca']:+.0%}")
    print(f"  Active memories: {r_off['active_before_consolidate']} → {r_off['active_after_consolidate']}")

    # Variant 2: 巩固
    print("\n--- Variant: WITH consolidation ---")
    r_on = run_variant(seed, episodes, consolidate_between=True)
    print(f"  Phase A: {r_on['rate_a']:.0%} | Phase C: {r_on['rate_c']:.0%} | "
          f"Delta: {r_on['delta_ca']:+.0%}")
    print(f"  Active memories: {r_on['active_before_consolidate']} → {r_on['active_after_consolidate']}")
    print(f"  Consolidated: {r_on['consolidated']}")

    elapsed = time.time() - t0

    # 汇总
    print(f"\n{'=' * 60}")
    print("Consolidation 消融汇总")
    print(f"{'=' * 60}")
    print(f"{'Variant':<20} {'Phase A':<10} {'Phase C':<10} {'Delta':<10} {'Active':<10}")
    print("-" * 60)
    print(f"{'no_consolidate':<20} {r_off['rate_a']:.0%}{'':<5} {r_off['rate_c']:.0%}{'':<5} "
          f"{r_off['delta_ca']:+.0%}{'':<5} {r_off['active_after_consolidate']}")
    print(f"{'with_consolidate':<20} {r_on['rate_a']:.0%}{'':<5} {r_on['rate_c']:.0%}{'':<5} "
          f"{r_on['delta_ca']:+.0%}{'':<5} {r_on['active_after_consolidate']}")
    print(f"\n巩固效果: {r_on['delta_ca'] - r_off['delta_ca']:+.0%}")
    print(f"耗时: {elapsed:.0f}s")

    # 保存
    output = {
        "seed": seed,
        "episodes": episodes,
        "no_consolidate": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_off.items()},
        "with_consolidate": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_on.items()},
        "consolidation_effect": round(r_on["delta_ca"] - r_off["delta_ca"], 4),
        "elapsed_s": round(elapsed, 1),
    }
    result_file = os.path.join(RESULTS_DIR, f"consolidation_ablation_{seed}.json")
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    main()
