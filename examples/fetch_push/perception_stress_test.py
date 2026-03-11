"""PerceptionBuffer 大数据量压测

测试 10K / 50K 条 perception 写入的性能和可靠性。

运行:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src python -u perception_stress_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

# Mock rclpy（必须在 import node.py 之前）
def _mock_module(name, attrs=None):
    mod = ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeNode:
    def __init__(self, name):
        self._name = name
        self._params = {}
        self._logger = MagicMock()

    def declare_parameter(self, name, default):
        self._params[name] = default

    def get_parameter(self, name):
        val = MagicMock()
        val.value = self._params.get(name, "")
        return val

    def get_logger(self):
        return self._logger

    def create_service(self, *a, **kw):
        pass

    def create_subscription(self, *a, **kw):
        pass

    def create_timer(self, *a, **kw):
        pass

    def create_publisher(self, *a, **kw):
        return MagicMock()

    def destroy_node(self):
        pass


class _FakePerceptionData:
    def __init__(self, **kwargs):
        self.header = None
        self.seq = 0
        self.perception_type = "visual"
        self.description = ""
        self.data = ""
        self.metadata = ""
        self.session_id = ""
        self.collection = ""
        for k, v in kwargs.items():
            setattr(self, k, v)


_mock_module("rclpy")
_mock_module("rclpy.callback_group", {
    "MutuallyExclusiveCallbackGroup": MagicMock,
    "ReentrantCallbackGroup": MagicMock,
})
_mock_module("rclpy.executors", {"MultiThreadedExecutor": MagicMock})
_mock_module("rclpy.node", {"Node": _FakeNode})
_mock_module("rclpy.qos", {
    "DurabilityPolicy": MagicMock(),
    "HistoryPolicy": MagicMock(),
    "QoSProfile": MagicMock,
    "ReliabilityPolicy": MagicMock(),
})
_mock_module("robotmem_msgs", {})
_mock_module("robotmem_msgs.msg", {
    "Memory": MagicMock,
    "NodeStatus": MagicMock,
    "PerceptionData": _FakePerceptionData,
})
_mock_module("robotmem_msgs.srv", {
    "EndSession": MagicMock, "Forget": MagicMock, "Learn": MagicMock,
    "Recall": MagicMock, "SavePerception": MagicMock,
    "StartSession": MagicMock, "Update": MagicMock,
})

import numpy as np
from robotmem.sdk import RobotMemory
from robotmem_ros.node import PerceptionBuffer, RobotMemNode

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-stress")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COUNTS = [10000, 50000]
BATCH_SIZE = 50


def create_buffer(db_path, collection, batch_size=BATCH_SIZE):
    """构造 PerceptionBuffer 实例"""
    mem = RobotMemory(db_path=db_path, collection=collection, embed_backend="none")
    buf = PerceptionBuffer(mem, batch_size=batch_size)
    return buf, mem


def run_stress_test(count, batch_size=BATCH_SIZE):
    """单个量级的压测"""
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    buf, mem = create_buffer(DB_PATH, "stress", batch_size)
    rng = np.random.RandomState(42)

    print(f"  写入 {count} 条 perception (batch_size={batch_size})...")
    t0 = time.monotonic()

    for i in range(count):
        msg = _FakePerceptionData(
            seq=i,
            perception_type="proprioceptive",
            description=f"grip pos [{rng.uniform(1,1.5):.3f}, {rng.uniform(1,1.5):.3f}, {rng.uniform(0.4,0.5):.3f}]",
            data=json.dumps({
                "grip": rng.uniform(1, 1.5, size=3).tolist(),
                "obj": rng.uniform(1, 1.5, size=3).tolist(),
                "target": rng.uniform(1, 1.5, size=3).tolist(),
                "action": rng.uniform(-1, 1, size=4).tolist(),
                "reward": float(rng.uniform(-1, 0)),
            }),
            metadata=json.dumps({"step": i, "episode": i // 50}),
            session_id="",
            collection="",
        )
        buf.add(msg)

    add_time = (time.monotonic() - t0) * 1000

    # flush 残留
    t1 = time.monotonic()
    buf.flush()
    flush_time = (time.monotonic() - t1) * 1000

    total_time = add_time + flush_time

    # 统计
    stats = buf.get_stats()

    # DB 大小
    db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    db_size_mb = db_size_bytes / (1024 * 1024)

    # 验证写入数量
    row = mem._db.conn.execute("SELECT COUNT(*) FROM memories WHERE collection='stress'").fetchone()
    actual_count = row[0] if row else 0

    result = {
        "count": count,
        "batch_size": batch_size,
        "total_time_ms": round(total_time, 1),
        "add_time_ms": round(add_time, 1),
        "flush_time_ms": round(flush_time, 1),
        "throughput_per_sec": round(count / (total_time / 1000), 0),
        "avg_per_record_ms": round(total_time / count, 3),
        "stats": stats,
        "actual_db_count": actual_count,
        "data_loss": count - actual_count,
        "db_size_mb": round(db_size_mb, 2),
    }

    mem.close()

    # 清理
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("PerceptionBuffer 大数据量压测")
    print(f"量级: {COUNTS}")
    print(f"batch_size: {BATCH_SIZE}")
    print("=" * 60)

    all_results = {}

    for count in COUNTS:
        print(f"\n--- {count:,} 条 ---")
        result = run_stress_test(count)
        all_results[count] = result

        print(f"  总时间: {result['total_time_ms']:.0f}ms")
        print(f"  吞吐量: {result['throughput_per_sec']:.0f} 条/秒")
        print(f"  每条: {result['avg_per_record_ms']:.3f}ms")
        print(f"  DB 大小: {result['db_size_mb']:.1f}MB")
        print(f"  丢失: {result['data_loss']} 条")
        print(f"  Stats: received={result['stats']['received']}, "
              f"written={result['stats']['written']}, "
              f"dropped={result['stats']['dropped']}, "
              f"failed={result['stats']['failed']}")

    # 汇总
    print(f"\n{'=' * 60}")
    print("压测汇总")
    print(f"{'=' * 60}")
    print(f"{'量级':<10} {'总时间':<12} {'吞吐量':<12} {'每条':<10} {'DB':<8} {'丢失':<6}")
    print("-" * 58)
    for count in COUNTS:
        r = all_results[count]
        print(f"{count:<10,} {r['total_time_ms']:<12.0f}ms {r['throughput_per_sec']:<12.0f}/s "
              f"{r['avg_per_record_ms']:<10.3f}ms {r['db_size_mb']:<8.1f}MB {r['data_loss']:<6}")

    # 保存
    result_file = os.path.join(RESULTS_DIR, "perception_stress_test.json")
    with open(result_file, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    main()
