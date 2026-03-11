"""robotmem 小时级耐久测试 — 长时间连续写入验证

验证目的: 长时间写入下 DB/WAL/FTS5 不退化（Issue #30 Task 5, P2）

运行方式:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src:../../ros/robotmem_ros python -u durability_test.py
  PYTHONPATH=../../src:../../ros/robotmem_ros python -u durability_test.py --duration 3600
  PYTHONPATH=../../src:../../ros/robotmem_ros python -u durability_test.py --duration 600 --fps 10

参数:
  --duration: 测试时长（秒，默认 3600 = 1 小时）
  --fps: 写入频率（条/秒，默认 30）
  --sample-interval: 采样间隔（秒，默认 600 = 10 分钟）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from types import ModuleType
from unittest.mock import MagicMock

# ── Mock ROS 2 模块 ──


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
from robotmem_ros.node import PerceptionBuffer

# 数据隔离
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-durability")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 退化检测：晚期 recall 延迟超过早期 N 倍视为退化
DEGRADATION_THRESHOLD = 2.0


def get_db_sizes(db_path):
    """获取 DB 文件 + WAL 大小"""
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    wal_path = db_path + "-wal"
    wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
    return db_size, wal_size


def measure_recall_latency(mem, query="perception data", n=5, iterations=10):
    """测量 recall 延迟（ms）"""
    latencies = []
    for _ in range(iterations):
        t0 = time.monotonic()
        mem.recall(query, n=n)
        latencies.append((time.monotonic() - t0) * 1000)
    return {
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.median(latencies), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
        "max_ms": round(max(latencies), 2),
    }


def measure_write_throughput(buf, rng, count=100):
    """测量写入吞吐量（条/秒）"""
    t0 = time.monotonic()
    for i in range(count):
        msg = _FakePerceptionData(
            seq=i + 100000,  # 不影响正常 seq
            perception_type="benchmark",
            description=f"throughput test {i}",
            data=json.dumps({"values": rng.uniform(-1, 1, size=6).tolist()}),
            session_id="",
            collection="",
        )
        buf.add(msg)
    buf.flush()
    elapsed = time.monotonic() - t0
    return round(count / elapsed, 0) if elapsed > 0 else 0


def parse_args():
    parser = argparse.ArgumentParser(description="robotmem 小时级耐久测试")
    parser.add_argument("--duration", type=int, default=3600, help="测试时长（秒，默认 3600）")
    parser.add_argument("--fps", type=int, default=30, help="写入频率（条/秒，默认 30）")
    parser.add_argument("--sample-interval", type=int, default=600, help="采样间隔（秒，默认 600）")
    return parser.parse_args()


def main():
    args = parse_args()
    duration = args.duration
    fps = args.fps
    sample_interval = args.sample_interval

    total_records = duration * fps

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem 小时级耐久测试 (Issue #30 Task 5)")
    print(f"时长: {duration}s ({duration/60:.0f}min)")
    print(f"写入频率: {fps} 条/秒")
    print(f"预计总量: {total_records:,} 条")
    print(f"采样间隔: {sample_interval}s")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    mem = RobotMemory(db_path=DB_PATH, collection="durability", embed_backend="none")
    buf = PerceptionBuffer(mem, batch_size=50)
    rng = np.random.RandomState(42)

    samples = []
    interval = 1.0 / fps  # 每条写入间隔
    last_sample_time = time.monotonic()
    records_written = 0

    t_start = time.monotonic()
    interrupted = False

    try:
        for i in range(total_records):
            t_iter = time.monotonic()

            # 写入一条 perception
            msg = _FakePerceptionData(
                seq=i,
                perception_type="proprioceptive",
                description=f"grip pos [{rng.uniform(1,1.5):.4f}, {rng.uniform(1,1.5):.4f}, {rng.uniform(0.4,0.5):.4f}]",
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
            records_written += 1

            # 采样
            now = time.monotonic()
            elapsed_total = now - t_start
            if now - last_sample_time >= sample_interval:
                buf.flush()  # 确保所有数据写入后采样

                db_size, wal_size = get_db_sizes(DB_PATH)
                recall_stats = measure_recall_latency(mem)
                throughput = measure_write_throughput(buf, rng)
                stats = buf.get_stats()

                sample = {
                    "elapsed_min": round(elapsed_total / 60, 1),
                    "records": records_written,
                    "db_size_mb": round(db_size / (1024 * 1024), 2),
                    "wal_size_mb": round(wal_size / (1024 * 1024), 2),
                    "recall_mean_ms": recall_stats["mean_ms"],
                    "recall_p99_ms": recall_stats["p99_ms"],
                    "write_throughput": throughput,
                    "buf_written": stats["written"],
                    "buf_failed": stats["failed"],
                }
                samples.append(sample)

                print(f"\n  [{elapsed_total/60:.0f}min] records={records_written:,} "
                      f"db={sample['db_size_mb']:.1f}MB wal={sample['wal_size_mb']:.1f}MB "
                      f"recall_mean={recall_stats['mean_ms']:.1f}ms "
                      f"recall_p99={recall_stats['p99_ms']:.1f}ms "
                      f"throughput={throughput:.0f}/s "
                      f"failed={stats['failed']}")

                last_sample_time = now

            # 进度显示
            if (i + 1) % (fps * 60) == 0:  # 每分钟
                elapsed_total = time.monotonic() - t_start
                pct = (i + 1) / total_records * 100
                print(f"  进度: {pct:.0f}% ({i+1:,}/{total_records:,}) "
                      f"已用 {elapsed_total/60:.0f}min")

            # 控制写入速率（简单 sleep，不精确但足够）
            elapsed_iter = time.monotonic() - t_iter
            sleep_time = interval - elapsed_iter
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 最终 flush
        buf.flush()

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n用户中断，保存已有结果...")
        buf.flush()

    finally:
        # 最终采样（mem.close() 前完成）
        elapsed_total = time.monotonic() - t_start
        final_stats = buf.get_stats()
        db_size, wal_size = get_db_sizes(DB_PATH)
        recall_stats = measure_recall_latency(mem)

        final_sample = {
            "elapsed_min": round(elapsed_total / 60, 1),
            "records": records_written,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "wal_size_mb": round(wal_size / (1024 * 1024), 2),
            "recall_mean_ms": recall_stats["mean_ms"],
            "recall_p99_ms": recall_stats["p99_ms"],
            "write_throughput": 0,
            "buf_written": final_stats["written"],
            "buf_failed": final_stats["failed"],
        }
        samples.append(final_sample)

        # 退化检测
        degradation = "无"
        if len(samples) >= 3:
            early_recall = np.mean([s["recall_mean_ms"] for s in samples[:2]])
            late_recall = np.mean([s["recall_mean_ms"] for s in samples[-2:]])
            if late_recall > early_recall * DEGRADATION_THRESHOLD:
                degradation = f"recall 延迟退化: 早期 {early_recall:.1f}ms → 晚期 {late_recall:.1f}ms"

        # 汇总
        status_label = "（用户中断）" if interrupted else ""
        result_lines = [
            f"{'=' * 60}",
            f"耐久测试结果 (Issue #30 Task 5){status_label}",
            f"{'=' * 60}",
            f"  实际时长: {elapsed_total/60:.1f}min",
            f"  写入总量: {records_written:,} 条",
            f"  DB 大小: {db_size/(1024*1024):.1f}MB",
            f"  WAL 大小: {wal_size/(1024*1024):.1f}MB",
            f"  写入失败: {final_stats['failed']}",
            f"  退化检测: {degradation}",
            "",
            "  采样时间线:",
            f"  {'时间':<10} {'记录数':<12} {'DB(MB)':<10} {'WAL(MB)':<10} {'recall(ms)':<12} {'吞吐(条/s)':<12}",
            "  " + "-" * 66,
        ]
        for s in samples:
            result_lines.append(
                f"  {s['elapsed_min']:<10.0f} {s['records']:<12,} {s['db_size_mb']:<10.1f} "
                f"{s['wal_size_mb']:<10.1f} {s['recall_mean_ms']:<12.1f} {s['write_throughput']:<12.0f}"
            )

        print("\n".join(result_lines))

        # 保存（文件名带时间戳，避免覆盖历史数据）
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = {
            "duration_s": round(elapsed_total, 1),
            "fps": fps,
            "total_records": records_written,
            "interrupted": interrupted,
            "final_db_size_mb": round(db_size / (1024 * 1024), 2),
            "final_wal_size_mb": round(wal_size / (1024 * 1024), 2),
            "failed": final_stats["failed"],
            "degradation": degradation,
            "samples": samples,
        }
        result_file = os.path.join(RESULTS_DIR, f"durability_{duration}s_{fps}fps_{ts}.json")
        with open(result_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_file}")

        txt_file = os.path.join(RESULTS_DIR, f"durability_{duration}s_{fps}fps_{ts}.txt")
        with open(txt_file, "w") as f:
            f.write("\n".join(result_lines))
        print(f"文本版: {txt_file}")

        mem.close()


if __name__ == "__main__":
    main()
