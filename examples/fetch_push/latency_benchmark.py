"""robotmem recall 延迟 benchmark — 不同记忆量下的延迟

测试 recall 在 10/50/100/500/1000 条记忆下的延迟。

运行:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src python latency_benchmark.py
"""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from robotmem.sdk import RobotMemory

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-latency")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MEMORY_COUNTS = [10, 50, 100, 500, 1000]
RECALL_REPEATS = 50  # 每个量级 recall 50 次取统计


def generate_memories(mem, count):
    """批量写入指定数量的记忆"""
    rng = np.random.RandomState(42)
    for i in range(count):
        success = rng.random() > 0.5
        vel = rng.uniform(-1, 1, size=3)
        dist = rng.uniform(0, 0.2)
        ctx = {
            "params": {
                "approach_velocity": {"value": vel.tolist(), "type": "vector"},
                "grip_force": {"value": float(rng.uniform(-1, 1)), "type": "scalar"},
                "final_distance": {"value": float(dist), "unit": "m"},
            },
            "spatial": {
                "grip_position": rng.uniform(1.0, 1.5, size=3).tolist(),
                "object_position": rng.uniform(1.0, 1.5, size=3).tolist(),
                "target_position": rng.uniform(1.0, 1.5, size=3).tolist(),
            },
            "task": {
                "name": "push_to_target",
                "success": bool(success),
                "steps": int(rng.randint(10, 50)),
                "total_reward": float(rng.uniform(-50, 0)),
            },
        }
        mem.learn(
            insight=f"FetchPush ep{i}: {'成功' if success else '失败'}, 距离 {dist:.3f}m",
            context=ctx,
        )


def benchmark_recall(mem, repeats):
    """测量 recall 延迟"""
    latencies = []
    for _ in range(repeats):
        t0 = time.monotonic()
        mem.recall("push cube to target position", n=5, context_filter={"task.success": True})
        latency = (time.monotonic() - t0) * 1000
        latencies.append(latency)
    return np.array(latencies)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem recall 延迟 benchmark")
    print(f"记忆量: {MEMORY_COUNTS}")
    print(f"每个量级 recall {RECALL_REPEATS} 次")
    print("=" * 60)

    all_results = {}

    for count in MEMORY_COUNTS:
        # 每个量级新建 DB
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)

        mem = RobotMemory(db_path=DB_PATH, collection="latency", embed_backend="onnx")

        print(f"\n--- {count} 条记忆 ---")
        t0 = time.monotonic()
        generate_memories(mem, count)
        gen_time = (time.monotonic() - t0) * 1000
        print(f"  写入耗时: {gen_time:.0f}ms ({gen_time/count:.1f}ms/条)")

        # warmup
        for _ in range(3):
            mem.recall("push cube to target", n=5)

        # benchmark
        latencies = benchmark_recall(mem, RECALL_REPEATS)
        stats = {
            "count": count,
            "mean_ms": round(float(np.mean(latencies)), 2),
            "p50_ms": round(float(np.median(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "write_ms_per_record": round(gen_time / count, 2),
        }
        all_results[count] = stats

        print(f"  mean: {stats['mean_ms']:.1f}ms | p50: {stats['p50_ms']:.1f}ms | "
              f"p95: {stats['p95_ms']:.1f}ms | p99: {stats['p99_ms']:.1f}ms | max: {stats['max_ms']:.1f}ms")

        mem.close()

    # 汇总表
    print(f"\n{'=' * 60}")
    print("延迟汇总")
    print(f"{'=' * 60}")
    print(f"{'记忆数':<10} {'mean':<10} {'p50':<10} {'p95':<10} {'p99':<10} {'max':<10} {'写入/条':<10}")
    print("-" * 70)
    for count in MEMORY_COUNTS:
        s = all_results[count]
        print(f"{count:<10} {s['mean_ms']:<10.1f} {s['p50_ms']:<10.1f} {s['p95_ms']:<10.1f} "
              f"{s['p99_ms']:<10.1f} {s['max_ms']:<10.1f} {s['write_ms_per_record']:<10.1f}")

    # 保存
    result_file = os.path.join(RESULTS_DIR, "latency_benchmark.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {result_file}")

    # 文本版
    txt_file = os.path.join(RESULTS_DIR, "latency_benchmark.txt")
    with open(txt_file, "w") as f:
        f.write("robotmem recall 延迟 benchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'记忆数':<10} {'mean(ms)':<10} {'p50':<10} {'p95':<10} {'p99':<10} {'max':<10} {'写入/条(ms)':<12}\n")
        f.write("-" * 62 + "\n")
        for count in MEMORY_COUNTS:
            s = all_results[count]
            f.write(f"{count:<10} {s['mean_ms']:<10.1f} {s['p50_ms']:<10.1f} {s['p95_ms']:<10.1f} "
                    f"{s['p99_ms']:<10.1f} {s['max_ms']:<10.1f} {s['write_ms_per_record']:<12.1f}\n")
    print(f"文本版: {txt_file}")

    # 清理
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)


if __name__ == "__main__":
    main()
