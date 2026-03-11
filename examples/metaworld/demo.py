"""Meta-World push-v3 跨实例空间记忆 Demo

验证: robotmem 空间回忆在 Meta-World push 上的效果。

机制:
  - 多个 task instance（不同 obj/target 位置）
  - Phase A: 启发式 + 默认参数 + 噪声（基线）
  - Phase B: 随机参数探索 + learn（积累经验）
  - Phase C: recall + 记忆驱动参数（空间回忆最近成功经验）

核心差异点（vs FetchPush）:
  - Meta-World 每个 instance 固定初始状态 → 参数探索更关键
  - spatial_sort 按 object 初始位置检索 → 相似布局用相似策略

运行:
  source .venv-pusht/bin/activate
  PYTHONPATH=src python examples/metaworld/demo.py [--seed 42] [--episodes 10]

注: 这是 API 教程。严格实验请参考 experiment.py。
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

import numpy as np

try:
    import metaworld
except ImportError:
    print("需要: pip install metaworld")
    sys.exit(1)

from robotmem.sdk import RobotMemory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import MetaWorldPushPolicy, MetaWorldMemoryPolicy

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-metaworld")
DB_PATH = os.path.join(DB_DIR, "memory.db")
COLLECTION = "metaworld-push"
MEMORY_WEIGHT = 0.3
RECALL_N = 3


def build_context(obs_initial, target_pos, total_reward, success,
                  approach_offset, push_speed, noise_scale):
    """构建 context — params/spatial/task 三区域"""
    return {
        "params": {
            "approach_offset": {"value": float(approach_offset), "type": "scalar"},
            "push_speed": {"value": float(push_speed), "type": "scalar"},
            "noise_scale": {"value": float(noise_scale), "type": "scalar"},
        },
        "spatial": {
            "object_position": [float(obs_initial[4]), float(obs_initial[5]), float(obs_initial[6])],
        },
        "task": {
            "name": "push-v3",
            "reward": float(total_reward),
            "success": bool(success),
            "target": [float(v) for v in target_pos],
        },
    }


def run_episode(env, policy, learn_mem=None, session_id=None, pre_reset_obs=None):
    """执行单个 episode，返回 (success, total_reward, obs_initial, target_pos)

    单次 reset — obs_initial/target_pos/obj_pos 来自同一次初始化。
    pre_reset_obs: 如果已经 reset 过，传入 obs 避免重复 reset。
    """
    if pre_reset_obs is not None:
        obs = pre_reset_obs
    else:
        obs, _ = env.reset()
    obs_initial = obs.copy()
    target_pos = env.unwrapped._target_pos.copy()
    total_reward = 0.0
    success = False

    for step in range(150):
        action = policy.act(obs, target_pos)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info.get("success", 0) > 0:
            success = True
        if terminated or truncated:
            break

    if learn_mem is not None:
        ctx = build_context(
            obs_initial, target_pos, total_reward, success,
            policy.approach_offset, policy.push_speed, policy.noise_scale,
        )
        learn_mem.learn(
            insight=f"push-v3: reward={total_reward:.1f}, success={success}",
            context=ctx,
            session_id=session_id,
        )

    return success, total_reward, obs_initial, target_pos


def run_phase(env_cls, tasks, episodes_per_instance, phase, mem=None, session_id=None):
    """在多个 task instance 上运行一个 Phase"""
    total_success = 0
    total_episodes = 0

    for task in tasks:
        env = env_cls()
        env.set_task(task)

        try:
            for ep in range(episodes_per_instance):
                pre_obs = None

                if phase == "C":
                    # Reset 一次，获取 obj_pos 用于 spatial recall
                    obs_c, _ = env.reset()
                    obj_pos = obs_c[4:7]
                    pre_obs = obs_c  # 传给 run_episode 避免重复 reset

                    recalled = mem.recall(
                        "successful push strategy",
                        n=RECALL_N,
                        context_filter={"task.success": True},
                        spatial_sort={
                            "field": "spatial.object_position",
                            "target": [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])],
                        },
                    )
                    policy = MetaWorldMemoryPolicy(
                        MetaWorldPushPolicy(noise_scale=0.3),
                        recalled,
                        MEMORY_WEIGHT,
                    )
                elif phase == "B":
                    # 随机参数探索
                    policy = MetaWorldPushPolicy(
                        noise_scale=np.random.uniform(0.1, 0.5),
                        approach_offset=np.random.uniform(0.02, 0.10),
                        push_speed=np.random.uniform(4.0, 12.0),
                    )
                else:
                    # Phase A: 默认参数
                    policy = MetaWorldPushPolicy(noise_scale=0.3)

                learn_mem = mem if phase == "B" else None
                success, reward, _, _ = run_episode(
                    env, policy, learn_mem=learn_mem,
                    session_id=session_id, pre_reset_obs=pre_obs,
                )
                total_success += int(success)
                total_episodes += 1
        finally:
            env.close()

    rate = total_success / total_episodes if total_episodes > 0 else 0
    return rate, total_success, total_episodes


def main():
    parser = argparse.ArgumentParser(description="Meta-World push-v3 空间记忆 Demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10, help="每个 instance 的 episode 数")
    parser.add_argument("--instances", type=int, default=20, help="使用的 task instance 数")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # 清空 DB
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    mem = RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx")

    # 获取 push-v3 task instances
    ml10 = metaworld.ML10(seed=seed)
    env_cls = ml10.train_classes["push-v3"]
    all_tasks = [t for t in ml10.train_tasks if t.env_name == "push-v3"]
    tasks = all_tasks[: args.instances]
    assert tasks, "未找到 push-v3 task instances，请检查 metaworld 版本"

    print("=" * 60)
    print("Meta-World push-v3 空间记忆 Demo")
    print(f"Seed: {seed}, Instances: {len(tasks)}, Episodes/instance: {args.episodes}")
    print("=" * 60)

    try:
        # Phase A: 基线
        print("\n--- Phase A: 基线（默认参数 + 噪声 0.3）---")
        random.seed(seed)
        np.random.seed(seed)
        rate_a, succ_a, total_a = run_phase(env_cls, tasks, args.episodes, "A")
        print(f"  Phase A: {succ_a}/{total_a} ({rate_a:.1%})")

        # Phase B: 参数探索 + 学习
        print("\n--- Phase B: 随机参数探索 + learn ---")
        random.seed(seed + 1000)
        np.random.seed(seed + 1000)
        with mem.session(context={"task": "push-v3"}) as sid:
            rate_b, succ_b, total_b = run_phase(
                env_cls, tasks, args.episodes, "B", mem=mem, session_id=sid,
            )
        print(f"  Phase B: {succ_b}/{total_b} ({rate_b:.1%})")

        # Phase C: 空间回忆
        print("\n--- Phase C: 空间回忆 + 记忆驱动参数 ---")
        random.seed(seed + 2000)
        np.random.seed(seed + 2000)
        rate_c, succ_c, total_c = run_phase(
            env_cls, tasks, args.episodes, "C", mem=mem,
        )
        print(f"  Phase C: {succ_c}/{total_c} ({rate_c:.1%})")

        # 结果
        delta = rate_c - rate_a
        print(f"\n{'=' * 60}")
        print("结果:")
        print(f"  Phase A (基线):     {rate_a:.1%}")
        print(f"  Phase B (探索):     {rate_b:.1%}")
        print(f"  Phase C (记忆回忆): {rate_c:.1%}")
        print(f"  Delta (C - A):      {delta:+.1%}")
        print(f"{'=' * 60}")

        if delta > 0.05:
            print(f"空间记忆提升 {delta:.1%} — 有效")
        elif delta > 0:
            print(f"空间记忆提升 {delta:.1%} — 小幅")
        else:
            print(f"空间记忆未提升 ({delta:.1%})")

        print("\n注: 这是 API 教程，严格实验请参考 experiment.py")

    finally:
        mem.close()


if __name__ == "__main__":
    main()
