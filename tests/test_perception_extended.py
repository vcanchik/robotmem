"""PerceptionBuffer 扩展测试 — 并发写入 + 数据精度端到端

Issue #30 Task 3: 多 Buffer 并发写入（3 个实例, 3 线程, 各 1000 条）
Issue #30 Task 4: 数据精度端到端校验（numpy → JSON → SDK → SQLite → recall）
"""

from __future__ import annotations

import json
import sys
import threading
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Mock ROS 2 模块 ──

_ros_mocks = {}


def _mock_module(name, attrs=None):
    mod = ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    _ros_mocks[name] = mod
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


# ══════════════════════════════════════════════════════════════
# Task 3: 多 Buffer 并发写入
# ══════════════════════════════════════════════════════════════


class TestConcurrentPerceptionBuffer:
    """3 个 PerceptionBuffer 实例并发写入同一 SQLite DB，不丢数据"""

    ITEMS_PER_BUFFER = 1000
    NUM_BUFFERS = 3

    def test_concurrent_write_no_data_loss(self, tmp_path):
        """3 线程各写 1000 条到不同 collection，验证每个 collection written==1000"""
        db_path = str(tmp_path / "concurrent.db")
        collections = [f"coll_{i}" for i in range(self.NUM_BUFFERS)]

        # 创建 3 个 PerceptionBuffer，各自用不同 collection
        buffers = []
        mems = []
        for coll in collections:
            mem = RobotMemory(db_path=db_path, collection=coll, embed_backend="none")
            buf = PerceptionBuffer(mem, batch_size=50)
            buffers.append(buf)
            mems.append(mem)

        errors = []

        def writer(buf, coll, count):
            """线程工作函数：写入 count 条 perception"""
            try:
                for i in range(count):
                    msg = _FakePerceptionData(
                        seq=i,
                        perception_type="proprioceptive",
                        description=f"{coll} step {i}",
                        data=json.dumps({"value": [float(i), float(i) + 0.1, float(i) + 0.2]}),
                        metadata=json.dumps({"step": i, "collection": coll}),
                        session_id="",
                        collection="",  # 使用 buffer 绑定的 collection
                    )
                    buf.add(msg)
                buf.flush()
            except Exception as e:
                errors.append((coll, str(e)))

        # 3 个线程并发写入
        threads = []
        for buf, coll in zip(buffers, collections):
            t = threading.Thread(target=writer, args=(buf, coll, self.ITEMS_PER_BUFFER))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)
            if t.is_alive():
                errors.append(("timeout", f"线程 {t.name} 超时未结束"))

        # 验证无错误
        assert errors == [], f"并发写入出错: {errors}"

        # 验证每个 buffer 的 stats
        for i, (buf, coll) in enumerate(zip(buffers, collections)):
            stats = buf.get_stats()
            assert stats["received"] == self.ITEMS_PER_BUFFER, \
                f"{coll}: received={stats['received']}, expected={self.ITEMS_PER_BUFFER}"
            assert stats["written"] == self.ITEMS_PER_BUFFER, \
                f"{coll}: written={stats['written']}, expected={self.ITEMS_PER_BUFFER}"
            assert stats["failed"] == 0, \
                f"{coll}: failed={stats['failed']}"

        # 验证 DB 中每个 collection 的实际行数
        # 用第一个 mem 的连接查询（共享同一 DB 文件）
        conn = mems[0]._db.conn
        for coll in collections:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE collection=?",
                (coll,),
            ).fetchone()
            actual = row[0]
            assert actual == self.ITEMS_PER_BUFFER, \
                f"DB {coll}: actual={actual}, expected={self.ITEMS_PER_BUFFER}"

        # 验证总行数
        row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        total = row[0]
        assert total == self.NUM_BUFFERS * self.ITEMS_PER_BUFFER, \
            f"DB total={total}, expected={self.NUM_BUFFERS * self.ITEMS_PER_BUFFER}"

        # 清理
        for mem in mems:
            mem.close()

    def test_concurrent_write_with_different_batch_sizes(self, tmp_path):
        """不同 batch_size 的并发写入"""
        db_path = str(tmp_path / "concurrent_batch.db")
        batch_sizes = [10, 50, 100]
        items = 500

        buffers = []
        mems = []
        for i, bs in enumerate(batch_sizes):
            coll = f"batch_{bs}"
            mem = RobotMemory(db_path=db_path, collection=coll, embed_backend="none")
            buf = PerceptionBuffer(mem, batch_size=bs)
            buffers.append(buf)
            mems.append(mem)

        errors = []

        def writer(buf, coll, count):
            try:
                for i in range(count):
                    msg = _FakePerceptionData(
                        seq=i,
                        description=f"{coll} step {i}",
                        data=json.dumps({"x": float(i)}),
                        session_id="",
                        collection="",
                    )
                    buf.add(msg)
                buf.flush()
            except Exception as e:
                errors.append((coll, str(e)))

        threads = []
        for buf, bs in zip(buffers, batch_sizes):
            t = threading.Thread(target=writer, args=(buf, f"batch_{bs}", items))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)
            if t.is_alive():
                errors.append(("timeout", f"线程 {t.name} 超时未结束"))

        assert errors == [], f"并发写入出错: {errors}"

        for buf in buffers:
            stats = buf.get_stats()
            assert stats["written"] == items

        for mem in mems:
            mem.close()


# ══════════════════════════════════════════════════════════════
# Task 4: 数据精度端到端校验
# ══════════════════════════════════════════════════════════════


class TestDataPrecisionE2E:
    """numpy → JSON → SDK save_perception → SQLite → recall 链路无浮点精度损失"""

    def test_float64_precision_roundtrip(self, tmp_path):
        """float64 精度：原始 numpy → JSON → DB → 读出，误差 < 1e-10"""
        db_path = str(tmp_path / "precision.db")
        mem = RobotMemory(db_path=db_path, collection="precision", embed_backend="none")

        rng = np.random.RandomState(42)
        num_observations = 10

        # 录制原始 numpy array（float64）
        originals = []
        for i in range(num_observations):
            obs = {
                "grip_pos": rng.uniform(1.0, 1.5, size=3).astype(np.float64),
                "obj_pos": rng.uniform(1.0, 1.5, size=3).astype(np.float64),
                "target_pos": rng.uniform(1.0, 1.5, size=3).astype(np.float64),
                "action": rng.uniform(-1.0, 1.0, size=4).astype(np.float64),
                "reward": np.float64(rng.uniform(-1.0, 0.0)),
            }
            originals.append(obs)

            # 通过 SDK save_perception（和 Node 的 _save_perception_cb 等价）
            data_json = json.dumps({
                "grip_pos": obs["grip_pos"].tolist(),
                "obj_pos": obs["obj_pos"].tolist(),
                "target_pos": obs["target_pos"].tolist(),
                "action": obs["action"].tolist(),
                "reward": float(obs["reward"]),
            })
            mem.save_perception(
                description=f"observation step {i}",
                perception_type="proprioceptive",
                data=data_json,
                metadata=json.dumps({"step": i}),
            )

        # 从 DB 读出并验证精度
        conn = mem._db.conn
        rows = conn.execute(
            "SELECT perception_data FROM memories WHERE collection='precision' ORDER BY rowid",
        ).fetchall()

        assert len(rows) == num_observations, f"写入 {num_observations} 条，读出 {len(rows)} 条"

        for i, (row, original) in enumerate(zip(rows, originals)):
            restored = json.loads(row[0])

            for key in ["grip_pos", "obj_pos", "target_pos", "action"]:
                orig_arr = original[key]
                rest_arr = np.array(restored[key])
                max_err = np.max(np.abs(orig_arr - rest_arr))
                assert max_err < 1e-10, \
                    f"step {i}, {key}: max_err={max_err:.2e} > 1e-10"

            # scalar
            orig_reward = float(original["reward"])
            rest_reward = restored["reward"]
            err = abs(orig_reward - rest_reward)
            assert err < 1e-10, \
                f"step {i}, reward: err={err:.2e} > 1e-10"

        mem.close()

    def test_float32_to_float64_conversion(self, tmp_path):
        """float32 输入 → JSON（自动升级 float64）→ DB → 读出，验证 float32 精度保持"""
        db_path = str(tmp_path / "precision32.db")
        mem = RobotMemory(db_path=db_path, collection="precision32", embed_backend="none")

        rng = np.random.RandomState(123)

        # float32 原始数据（模拟 MuJoCo 的 obs，通常是 float64 但有些传感器返回 float32）
        original_f32 = rng.uniform(0.0, 1.0, size=10).astype(np.float32)

        data_json = json.dumps({"values": original_f32.tolist()})
        mem.save_perception(
            description="float32 precision test",
            perception_type="proprioceptive",
            data=data_json,
        )

        row = mem._db.conn.execute(
            "SELECT perception_data FROM memories WHERE collection=? AND perception_type=?",
            ("precision32", "proprioceptive"),
        ).fetchone()
        assert row is not None, "collection='precision32' 中未找到 perception 数据"
        restored = json.loads(row[0])
        restored_arr = np.array(restored["values"])

        # float32 → Python float → JSON → Python float → float64
        # 精度损失来自 float32 → float64 的转换，约 1e-7 量级
        for i in range(len(original_f32)):
            err = abs(float(original_f32[i]) - restored_arr[i])
            assert err < 1e-6, \
                f"index {i}: err={err:.2e} > 1e-6 (float32 precision)"

        mem.close()

    def test_perception_buffer_precision(self, tmp_path):
        """通过 PerceptionBuffer 的完整路径精度校验"""
        db_path = str(tmp_path / "buf_precision.db")
        mem = RobotMemory(db_path=db_path, collection="buf_prec", embed_backend="none")
        buf = PerceptionBuffer(mem, batch_size=5)

        rng = np.random.RandomState(456)
        num_items = 10

        originals = []
        for i in range(num_items):
            values = rng.uniform(-1.0, 1.0, size=7).astype(np.float64)
            originals.append(values)

            msg = _FakePerceptionData(
                seq=i,
                perception_type="proprioceptive",
                description=f"step {i}",
                data=json.dumps({"values": values.tolist()}),
                metadata=json.dumps({"step": i}),
                session_id="",
                collection="",
            )
            buf.add(msg)

        buf.flush()

        # 验证 stats
        stats = buf.get_stats()
        assert stats["written"] == num_items

        # 从 DB 读出
        rows = mem._db.conn.execute(
            "SELECT perception_data FROM memories WHERE collection='buf_prec' ORDER BY rowid",
        ).fetchall()

        assert len(rows) == num_items

        for i, (row, original) in enumerate(zip(rows, originals)):
            restored = json.loads(row[0])
            restored_arr = np.array(restored["values"])
            max_err = np.max(np.abs(original - restored_arr))
            assert max_err < 1e-10, \
                f"step {i}: max_err={max_err:.2e} > 1e-10"

        mem.close()

    def test_extreme_values_precision(self, tmp_path):
        """极值精度：非常大/小的浮点数"""
        db_path = str(tmp_path / "extreme.db")
        mem = RobotMemory(db_path=db_path, collection="extreme", embed_backend="none")

        extreme_values = [
            1e-15,
            1e15,
            -1e-15,
            -1e15,
            0.0,
            np.finfo(np.float64).tiny,  # 最小正 float64
            1.0 / 3.0,  # 无限循环小数
            np.pi,
            np.e,
            np.sqrt(2),
        ]

        data_json = json.dumps({"values": extreme_values})
        mem.save_perception(
            description="extreme values precision test",
            perception_type="proprioceptive",
            data=data_json,
        )

        row = mem._db.conn.execute(
            "SELECT perception_data FROM memories WHERE collection='extreme'",
        ).fetchone()
        restored = json.loads(row[0])

        for i, (orig, rest) in enumerate(zip(extreme_values, restored["values"])):
            if orig == 0.0:
                assert rest == 0.0
            else:
                rel_err = abs(orig - rest) / abs(orig)
                assert rel_err < 1e-10, \
                    f"index {i}: orig={orig}, restored={rest}, rel_err={rel_err:.2e}"

        mem.close()
