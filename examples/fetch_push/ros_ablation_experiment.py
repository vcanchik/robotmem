"""robotmem ROS 2 Node 消融实验 — Node 回调 + random bias

验证目的: random bias 在 Node 回调路径下是否和真实记忆效果相当（Issue #30 Task 1）
预期: random Delta ≈ full Delta（差异 < 5%）→ "扰动效应"在 Node 层也成立

运行方式:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src:../../ros/robotmem_ros python -u ros_ablation_experiment.py
  PYTHONPATH=../../src:../../ros/robotmem_ros python -u ros_ablation_experiment.py --seed 42

三阶段 + 消融:
  Phase A: 基线（heuristic，无记忆）
  Phase B: 记忆写入（learn via Node callback）
  Phase C-full: 真实记忆（recall via Node callback）
  Phase C-random: 随机偏置（fake recall，随机 approach_velocity）
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

# ── Mock ROS 2 模块（必须在 import node.py 之前）──


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
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-ros-ablation")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLLECTION = "ros_ablation"
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def create_node(db_path, collection, embed_backend="onnx"):
    """构造 RobotMemNode 实例（绕过 ROS 初始化）"""
    with patch.object(RobotMemNode, "__init__", lambda self: None):
        node = RobotMemNode.__new__(RobotMemNode)

    mem = RobotMemory(db_path=db_path, collection=collection, embed_backend=embed_backend)
    perc_mem = RobotMemory(db_path=db_path, collection=collection, embed_backend="none")

    node._name = "robotmem_ros_ablation"
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


def learn_via_node(node, insight, context, session_id=None):
    """通过 Node 回调 learn"""
    req = _FakeMsg(
        insight=insight,
        context=json.dumps(context) if isinstance(context, dict) else context,
        session_id=session_id or "",
        collection="",
    )
    return node._learn_cb(req, _make_learn_response())


def recall_via_node(node, query, context_filter=None, n=5):
    """通过 Node 回调 recall"""
    req = _FakeMsg(
        query=query,
        n=n,
        min_confidence=0.0,
        session_id="",
        context_filter=json.dumps(context_filter) if context_filter else "",
        spatial_sort="",
        collection="",
    )
    return node._recall_cb(req, _make_recall_response())


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


def run_episode(env, policy, phase, node, session_id=None, variant="full", rng=None):
    """执行单个 episode — 支持 full/random/baseline variant"""
    if variant == "random" and rng is None:
        raise ValueError("variant='random' 时必须提供 rng 参数")

    if phase == "C" and variant == "full":
        resp = recall_via_node(node, "push cube to target position",
                               context_filter={"task.success": True}, n=RECALL_N)
        recalled = memories_to_context_dicts(resp.memories)
        active_policy = PhaseAwareMemoryPolicy(policy, recalled, MEMORY_WEIGHT) if recalled else policy
    elif phase == "C" and variant == "random":
        # 随机偏置（和真实 bias 维度一致：4D）
        random_bias = rng.uniform(-0.5, 0.5, size=4)
        fake_memory = [{"params": {
            "approach_velocity": {"value": random_bias[:3].tolist(), "type": "vector"},
            "grip_force": {"value": float(random_bias[3]), "type": "scalar"},
        }, "task": {"success": True}}]
        active_policy = PhaseAwareMemoryPolicy(policy, fake_memory, MEMORY_WEIGHT)
    else:
        active_policy = policy

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
    if phase in ("B", "C"):
        ctx = build_context(obs, actions, success, len(actions), total_reward)
        dist = ctx["params"]["final_distance"]["value"]
        learn_via_node(
            node,
            f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
            ctx,
            session_id=session_id,
        )

    return success


def run_phase(env, policy, phase, node, episodes, session_id=None, variant="full", rng=None):
    """执行一个 Phase"""
    successes = 0
    label = f"Phase {phase}" if phase != "C" else f"Phase C ({variant})"
    for ep in range(episodes):
        ok = run_episode(env, policy, phase, node, session_id, variant, rng)
        successes += int(ok)
        if (ep + 1) % 10 == 0:
            print(f"  {label} [{ep+1}/{episodes}] 成功率: {successes/(ep+1):.0%}")
    return successes / episodes


def parse_args():
    parser = argparse.ArgumentParser(
        description="robotmem ROS 2 Node 消融实验 — full vs random",
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
    rng = np.random.RandomState(seed + 1000)  # random variant 独立 RNG

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem ROS 2 Node 消融实验 (Issue #30 Task 1)")
    print(f"seed={seed}, episodes={episodes}/phase")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    node, mem, perc_mem = create_node(DB_PATH, COLLECTION, embed_backend="onnx")
    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()

    t0 = time.time()

    try:
        # Phase A: 基线（重置 seed 确保环境一致）
        print("\n--- Phase A: 基线（无记忆）---")
        np.random.seed(seed)
        rate_a = run_phase(env, policy, "A", node, episodes)

        # Phase B: 写入记忆（通过 Node callback）
        print("\n--- Phase B: 写入记忆（via Node callback）---")
        np.random.seed(seed)
        req = _FakeMsg(context='{"task": "push_to_target", "env": "FetchPush-v4"}', collection="")
        resp = node._start_session_cb(req, _make_start_response())
        sid = resp.session_id
        rate_b = run_phase(env, policy, "B", node, episodes, sid)

        # Phase C-full: 真实记忆（via Node callback）
        print("\n--- Phase C (full): 真实记忆 ---")
        np.random.seed(seed)
        rate_c_full = run_phase(env, policy, "C", node, episodes, sid, variant="full")

        # Phase C-random: 随机偏置
        print("\n--- Phase C (random): 随机偏置 ---")
        np.random.seed(seed)
        rate_c_random = run_phase(env, policy, "C", node, episodes, sid, variant="random", rng=rng)

        # end_session
        req = _FakeMsg(session_id=sid, has_outcome_score=True, outcome_score=float(rate_c_full))
        node._end_session_cb(req, _make_end_response())

        elapsed = time.time() - t0
        delta_full = rate_c_full - rate_a
        delta_random = rate_c_random - rate_a
        delta_diff = abs(delta_full - delta_random)

        # 汇总
        result_lines = [
            f"{'=' * 60}",
            "ROS 2 Node 消融实验结果 (Issue #30 Task 1)",
            f"{'=' * 60}",
            f"  Phase A (基线):       {rate_a:.0%}",
            f"  Phase B (写入):       {rate_b:.0%}",
            f"  Phase C (full):       {rate_c_full:.0%}  (Δ={delta_full:+.0%})",
            f"  Phase C (random):     {rate_c_random:.0%}  (Δ={delta_random:+.0%})",
            f"",
            f"  Delta full:           {delta_full:+.1%}",
            f"  Delta random:         {delta_random:+.1%}",
            f"  |full - random|:      {delta_diff:.1%}",
            f"",
            f"  结论: {'random ≈ full（差异 < 5%）→ 扰动效应在 Node 层也成立' if delta_diff < 0.05 else 'random ≠ full（差异 ≥ 5%）→ 记忆内容在 Node 层有额外作用'}",
            f"",
            f"  耗时: {elapsed:.0f}s",
            f"  seed: {seed}",
        ]
        print("\n".join(result_lines))

        # 保存结果
        output = {
            "seed": seed,
            "episodes": episodes,
            "rate_a": round(rate_a, 4),
            "rate_b": round(rate_b, 4),
            "rate_c_full": round(rate_c_full, 4),
            "rate_c_random": round(rate_c_random, 4),
            "delta_full": round(delta_full, 4),
            "delta_random": round(delta_random, 4),
            "delta_diff": round(delta_diff, 4),
            "conclusion": "perturbation_effect" if delta_diff < 0.05 else "memory_content_matters",
            "elapsed_s": round(elapsed, 1),
        }
        result_file = os.path.join(RESULTS_DIR, f"ros_ablation_{seed}.json")
        with open(result_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_file}")

        txt_file = os.path.join(RESULTS_DIR, f"ros_ablation_{seed}.txt")
        with open(txt_file, "w") as f:
            f.write("\n".join(result_lines))
        print(f"文本版: {txt_file}")

    finally:
        env.close()
        mem.close()
        perc_mem.close()


if __name__ == "__main__":
    main()
