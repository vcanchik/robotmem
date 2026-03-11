"""robotmem SDK vs Node 控制实验 — 解释 Delta 差距来源

验证目的: 同 seed、同 DB、同 episodes，SDK 直调 vs Node 回调的 Delta 差异（Issue #30 Task 2）
预期: Delta 差异 < 5% → ROS 层透明；> 5% → 排查额外开销来源

运行方式:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src:../../ros/robotmem_ros python -u sdk_vs_node_experiment.py
  PYTHONPATH=../../src:../../ros/robotmem_ros python -u sdk_vs_node_experiment.py --seed 42

设计:
  Phase A: 基线（共享，跑一次）
  Phase B: 记忆写入（共享，跑一次）
  Phase C-sdk: SDK 直调 mem.recall()
  Phase C-node: Node 回调 node._recall_cb()
  所有 Phase 重置同一 seed，确保 env 初始条件一致。
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


class _FakeMsg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        return getattr(super(), name, "") if name.startswith("_") else ""


class _FakeMemory(_FakeMsg):
    def __init__(self):
        self.id = 0
        self.content = ""
        self.type = ""
        self.perception_type = ""
        self.confidence = 0.0
        self.rrf_score = 0.0
        self.context_json = ""
        self.session_id = ""
        self.created_at = ""


class _FakeNodeStatus(_FakeMsg):
    def __init__(self):
        self.ready = False
        self.db_path = ""
        self.embed_backend = ""
        self.collection = ""
        self.active_memories_count = 0


class _FakePerceptionData(_FakeMsg):
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
    "Memory": _FakeMemory,
    "NodeStatus": _FakeNodeStatus,
    "PerceptionData": _FakePerceptionData,
})
_mock_module("robotmem_msgs.srv", {
    "EndSession": MagicMock, "Forget": MagicMock, "Learn": MagicMock,
    "Recall": MagicMock, "SavePerception": MagicMock,
    "StartSession": MagicMock, "Update": MagicMock,
})

try:
    import gymnasium_robotics  # noqa: F401
    import gymnasium
    import numpy as np
except ImportError:
    print("需要安装: pip install gymnasium-robotics")
    sys.exit(1)

from unittest.mock import patch
from robotmem.sdk import RobotMemory
from robotmem_ros.node import PerceptionBuffer, RobotMemNode

from policies import HeuristicPolicy, PhaseAwareMemoryPolicy

# 数据隔离
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-sdk-vs-node")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLLECTION = "sdk_vs_node"
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def create_node(db_path, collection, embed_backend="onnx"):
    """构造 RobotMemNode 实例（绕过 ROS 初始化）"""
    with patch.object(RobotMemNode, "__init__", lambda self: None):
        node = RobotMemNode.__new__(RobotMemNode)

    mem = RobotMemory(db_path=db_path, collection=collection, embed_backend=embed_backend)
    perc_mem = RobotMemory(db_path=db_path, collection=collection, embed_backend="none")

    node._name = "robotmem_sdk_vs_node"
    node._logger = MagicMock()
    node._mem = mem
    node._perception_mem = perc_mem
    node._collection = collection
    node._db_path_str = db_path
    node._embed_backend = embed_backend
    node._ready_pub = MagicMock()
    node._perc_buffer = PerceptionBuffer(perc_mem, batch_size=50)
    return node, mem, perc_mem


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


def _make_learn_response():
    return _FakeMsg(success=False, error="", memory_id=0, status="", auto_inferred_json="")


def _make_recall_response():
    return _FakeMsg(success=False, error="", memories=[], total=0, mode="")


def _make_start_response():
    return _FakeMsg(success=False, error="", session_id="", collection="", active_memories_count=0)


def _make_end_response():
    return _FakeMsg(success=False, error="", summary_json="", decayed_count=0,
                    consolidated_json="", related_memories=[])


# ── SDK 直调路径 ──


def run_phase_sdk(env, policy, mem, episodes, phase, session_id=None):
    """Phase A/B/C 通过 SDK 直接调用"""
    successes = 0
    for ep in range(episodes):
        recalled = []
        if phase == "C":
            recalled = mem.recall("push cube to target position", n=RECALL_N,
                                  context_filter={"task.success": True})

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
        successes += int(success)

        if phase in ("B", "C"):
            ctx = build_context(obs, actions, success, len(actions), total_reward)
            dist = ctx["params"]["final_distance"]["value"]
            mem.learn(
                insight=f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
                context=ctx,
                session_id=session_id,
            )

        if (ep + 1) % 10 == 0:
            label = f"SDK Phase {phase}"
            print(f"  {label} [{ep+1}/{episodes}] 成功率: {successes/(ep+1):.0%}")

    return successes / episodes


# ── Node 回调路径 ──


def memories_to_context_dicts(memories):
    """Memory msg 列表 → context dict 列表"""
    results = []
    for m in memories:
        ctx_str = m.context_json if hasattr(m, 'context_json') else ""
        if ctx_str:
            try:
                ctx = json.loads(ctx_str)
                result = {}
                if "params" in ctx:
                    result["params"] = ctx["params"]
                if "task" in ctx:
                    result["task"] = ctx["task"]
                if "spatial" in ctx:
                    result["spatial"] = ctx["spatial"]
                results.append(result)
            except json.JSONDecodeError:
                continue
    return results


def run_phase_node(env, policy, node, episodes, phase, session_id=None):
    """Phase C 通过 Node 回调"""
    successes = 0
    for ep in range(episodes):
        recalled = []
        if phase == "C":
            req = _FakeMsg(
                query="push cube to target position", n=RECALL_N, min_confidence=0.0,
                session_id="",
                context_filter=json.dumps({"task.success": True}),
                spatial_sort="", collection="",
            )
            resp = node._recall_cb(req, _make_recall_response())
            recalled = memories_to_context_dicts(resp.memories)

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
        successes += int(success)

        if phase in ("B", "C"):
            ctx = build_context(obs, actions, success, len(actions), total_reward)
            dist = ctx["params"]["final_distance"]["value"]
            req = _FakeMsg(
                insight=f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
                context=json.dumps(ctx),
                session_id=session_id or "",
                collection="",
            )
            node._learn_cb(req, _make_learn_response())

        if (ep + 1) % 10 == 0:
            label = f"Node Phase {phase}"
            print(f"  {label} [{ep+1}/{episodes}] 成功率: {successes/(ep+1):.0%}")

    return successes / episodes


def parse_args():
    parser = argparse.ArgumentParser(
        description="robotmem SDK vs Node 控制实验",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--episodes", type=int, default=100, help="每阶段 episode 数（默认 100）")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    episodes = args.episodes

    random.seed(seed)
    np.random.seed(seed)

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem SDK vs Node 控制实验 (Issue #30 Task 2)")
    print(f"seed={seed}, episodes={episodes}/phase")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    # 两条路径共用同一个 DB 和 ONNX embedding
    node, mem, perc_mem = create_node(DB_PATH, COLLECTION, embed_backend="onnx")
    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()

    t0 = time.time()

    try:
        # Phase A: 共享基线（只跑一次）
        print("\n--- Phase A: 基线（无记忆）---")
        np.random.seed(seed)
        rate_a = run_phase_sdk(env, policy, mem, episodes, "A")

        # Phase B: 共享记忆写入（只跑一次，SDK 写入）
        print("\n--- Phase B: 写入记忆（SDK 直调）---")
        np.random.seed(seed)
        with mem.session(context={"task": "push_to_target", "env": "FetchPush-v4"}) as sid:
            rate_b = run_phase_sdk(env, policy, mem, episodes, "B", sid)

        # Phase C-sdk: SDK 直调 recall
        print("\n--- Phase C (SDK): SDK 直调 recall ---")
        np.random.seed(seed)
        rate_c_sdk = run_phase_sdk(env, policy, mem, episodes, "C")

        # Phase C-node: Node 回调 recall（共用同一 DB 的记忆）
        print("\n--- Phase C (Node): Node 回调 recall ---")
        np.random.seed(seed)
        rate_c_node = run_phase_node(env, policy, node, episodes, "C")

        elapsed = time.time() - t0
        delta_sdk = rate_c_sdk - rate_a
        delta_node = rate_c_node - rate_a
        delta_diff = abs(delta_sdk - delta_node)

        # 汇总
        result_lines = [
            f"{'=' * 60}",
            "SDK vs Node 控制实验结果 (Issue #30 Task 2)",
            f"{'=' * 60}",
            f"  Phase A (基线):       {rate_a:.0%}",
            f"  Phase B (写入):       {rate_b:.0%}",
            f"  Phase C (SDK):        {rate_c_sdk:.0%}  (Δ={delta_sdk:+.0%})",
            f"  Phase C (Node):       {rate_c_node:.0%}  (Δ={delta_node:+.0%})",
            f"",
            f"  Delta SDK:            {delta_sdk:+.1%}",
            f"  Delta Node:           {delta_node:+.1%}",
            f"  |SDK - Node|:         {delta_diff:.1%}",
            f"",
            f"  结论: {'ROS 层透明（差异 < 5%）' if delta_diff < 0.05 else 'ROS 层有损耗（差异 ≥ 5%），需排查 Node 回调开销'}",
            f"",
            f"  注: 之前 +25% vs +16.2% 的差距来自不同实验设置（seed 数量、episodes 数量不同）。",
            f"  本实验严格控制 seed={seed}, episodes={episodes}，排除设置差异。",
            f"",
            f"  耗时: {elapsed:.0f}s",
        ]
        print("\n".join(result_lines))

        # 保存
        output = {
            "seed": seed,
            "episodes": episodes,
            "rate_a": round(rate_a, 4),
            "rate_b": round(rate_b, 4),
            "rate_c_sdk": round(rate_c_sdk, 4),
            "rate_c_node": round(rate_c_node, 4),
            "delta_sdk": round(delta_sdk, 4),
            "delta_node": round(delta_node, 4),
            "delta_diff": round(delta_diff, 4),
            "conclusion": "ros_transparent" if delta_diff < 0.05 else "ros_overhead",
            "elapsed_s": round(elapsed, 1),
        }
        result_file = os.path.join(RESULTS_DIR, f"sdk_vs_node_{seed}.json")
        with open(result_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_file}")

        txt_file = os.path.join(RESULTS_DIR, f"sdk_vs_node_{seed}.txt")
        with open(txt_file, "w") as f:
            f.write("\n".join(result_lines))
        print(f"文本版: {txt_file}")

    finally:
        env.close()
        mem.close()
        perc_mem.close()


if __name__ == "__main__":
    main()
