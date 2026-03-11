"""跨环境泛化 5-seed 实验 — FetchPush → FetchSlide

复用 cross_env.py 的逻辑，5 个 seed 批量运行。

运行:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src python -u cross_env_5seed.py
  PYTHONPATH=../../src python -u cross_env_5seed.py --episodes 100
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

from policies import HeuristicPolicy, SlidePolicy, PhaseAwareMemoryPolicy

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-crossenv5")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PUSH_COLLECTION = "push_exp"
SLIDE_COLLECTION = "slide_exp"
MEMORY_WEIGHT = 0.3
RECALL_N = 5
SEEDS = [42, 123, 456, 789, 2026]


def build_context(obs, actions, success, steps, total_reward, task_name):
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
            "name": task_name,
            "success": bool(success),
            "steps": steps,
            "total_reward": float(total_reward),
        },
    }


def run_episode(env, policy, mem, collection, recall_collection=None, session_id=None, task_name="push"):
    recalled = []
    if recall_collection:
        recalled = mem.recall(
            "push or slide object to target position",
            n=RECALL_N,
            context_filter={"task.success": True},
            collection=recall_collection,
        )
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
    if session_id:
        ctx = build_context(obs, actions, success, len(actions), total_reward, task_name)
        dist = ctx["params"]["final_distance"]["value"]
        mem.learn(
            insight=f"Fetch{task_name.capitalize()}: {'成功' if success else '失败'}, 距离 {dist:.3f}m",
            context=ctx,
            session_id=session_id,
            collection=collection,
        )
    return success


def run_phase(env, policy, mem, episodes, collection, recall_collection=None,
              session_id=None, task_name="push"):
    successes = 0
    for ep in range(episodes):
        ok = run_episode(env, policy, mem, collection, recall_collection, session_id, task_name)
        successes += int(ok)
    return successes / episodes


def run_single_seed(seed, episodes):
    """单 seed 完整实验"""
    random.seed(seed)
    np.random.seed(seed)

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    mem = RobotMemory(db_path=DB_PATH, embed_backend="onnx")

    try:
        # Phase 0: FetchPush 写入记忆
        push_env = gymnasium.make("FetchPush-v4")
        push_policy = HeuristicPolicy()
        try:
            with mem.session(context={"task": "push_to_target"}) as push_sid:
                rate_push = run_phase(push_env, push_policy, mem, episodes,
                                      collection=PUSH_COLLECTION, session_id=push_sid, task_name="push")
        finally:
            push_env.close()

        # FetchSlide
        slide_env = gymnasium.make("FetchSlide-v4")
        slide_policy = SlidePolicy()

        try:
            # Phase A: Slide 基线
            rate_a = run_phase(slide_env, slide_policy, mem, episodes,
                               collection=SLIDE_COLLECTION, task_name="slide")

            # Phase B: Slide + Push 记忆
            rate_b = run_phase(slide_env, slide_policy, mem, episodes,
                               collection=SLIDE_COLLECTION, recall_collection=PUSH_COLLECTION,
                               task_name="slide")
        finally:
            slide_env.close()

        return {"push": rate_push, "slide_a": rate_a, "slide_b": rate_b, "delta": rate_b - rate_a}
    finally:
        mem.close()


def parse_args():
    parser = argparse.ArgumentParser(description="跨环境泛化 5-seed 实验")
    parser.add_argument("--episodes", type=int, default=50, help="每阶段 episode 数")
    return parser.parse_args()


def main():
    args = parse_args()
    episodes = args.episodes
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem 跨环境泛化 5-seed 实验: FetchPush → FetchSlide")
    print(f"Seeds: {SEEDS}")
    print(f"每阶段 {episodes} episodes")
    print("=" * 60)

    all_results = {}
    t0 = time.time()

    for seed in SEEDS:
        print(f"\n{'─' * 40}")
        print(f"Seed {seed}")
        print(f"{'─' * 40}")
        result = run_single_seed(seed, episodes)
        all_results[seed] = result
        print(f"  Push: {result['push']:.0%} | Slide A: {result['slide_a']:.0%} | "
              f"Slide B: {result['slide_b']:.0%} | Delta: {result['delta']:+.0%}")

    elapsed = time.time() - t0

    # 汇总
    push_rates = [r["push"] for r in all_results.values()]
    a_rates = [r["slide_a"] for r in all_results.values()]
    b_rates = [r["slide_b"] for r in all_results.values()]
    deltas = [r["delta"] for r in all_results.values()]

    print(f"\n{'=' * 60}")
    print("5-seed 汇总")
    print(f"{'=' * 60}")
    print(f"{'seed':<8} {'Push':<8} {'Slide A':<10} {'Slide B':<10} {'Delta':<8}")
    print("-" * 44)
    for seed in SEEDS:
        r = all_results[seed]
        print(f"{seed:<8} {r['push']:.0%}{'':<3} {r['slide_a']:.0%}{'':<5} "
              f"{r['slide_b']:.0%}{'':<5} {r['delta']:+.0%}")
    print("-" * 44)
    print(f"{'均值':<8} {np.mean(push_rates):.0%}{'':<3} {np.mean(a_rates):.0%}{'':<5} "
          f"{np.mean(b_rates):.0%}{'':<5} {np.mean(deltas):+.0%}")
    print(f"{'标准差':<8} {np.std(push_rates, ddof=1):.1%}{'':<1} {np.std(a_rates, ddof=1):.1%}{'':<3} "
          f"{np.std(b_rates, ddof=1):.1%}{'':<3} {np.std(deltas, ddof=1):.1%}")
    print(f"\n全部正向: {all(d > 0 for d in deltas)}")
    print(f"耗时: {elapsed:.0f}s")

    # 保存
    output = {
        "seeds": SEEDS,
        "episodes": episodes,
        "results": {str(k): v for k, v in all_results.items()},
        "summary": {
            "push_mean": round(float(np.mean(push_rates)), 4),
            "slide_a_mean": round(float(np.mean(a_rates)), 4),
            "slide_b_mean": round(float(np.mean(b_rates)), 4),
            "delta_mean": round(float(np.mean(deltas)), 4),
            "delta_std": round(float(np.std(deltas, ddof=1)), 4),
            "all_positive": all(d > 0 for d in deltas),
        },
        "elapsed_s": round(elapsed, 1),
    }

    result_file = os.path.join(RESULTS_DIR, "cross_env_5seed.json")
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {result_file}")

    # 文本版
    txt_file = os.path.join(RESULTS_DIR, "cross_env_5seed.txt")
    with open(txt_file, "w") as f:
        f.write("robotmem 跨环境泛化 5-seed 实验: FetchPush → FetchSlide\n")
        f.write(f"每阶段 {episodes} episodes\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'seed':<8} {'Push':<8} {'Slide A':<10} {'Slide B':<10} {'Delta':<8}\n")
        f.write("-" * 44 + "\n")
        for seed in SEEDS:
            r = all_results[seed]
            f.write(f"{seed:<8} {r['push']:.0%}      {r['slide_a']:.0%}        "
                    f"{r['slide_b']:.0%}        {r['delta']:+.0%}\n")
        f.write("-" * 44 + "\n")
        f.write(f"均值     {np.mean(push_rates):.0%}      {np.mean(a_rates):.0%}        "
                f"{np.mean(b_rates):.0%}        {np.mean(deltas):+.0%}\n")
        f.write(f"标准差   {np.std(push_rates, ddof=1):.1%}    {np.std(a_rates, ddof=1):.1%}      "
                f"{np.std(b_rates, ddof=1):.1%}      {np.std(deltas, ddof=1):.1%}\n")
        f.write(f"\n全部正向: {all(d > 0 for d in deltas)}\n")
    print(f"文本版: {txt_file}")

    # 清理临时 DB
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)


if __name__ == "__main__":
    main()
