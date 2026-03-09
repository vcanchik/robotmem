[English](README.md)

# robotmem — 让机器人从经验中学习

> 机器人跑了 1000 次实验，每次都从零开始。robotmem 把每次 episode 的经验存起来——参数、轨迹、成败——下次自动检索最相关的经验指导决策。

**FetchPush 实验验证**：+25% 成功率提升（42% → 67%），纯 CPU，5 分钟复现。

<p align="center">
  <img src="examples/demo.gif" alt="robotmem 30s demo: save → restart → recall" width="600">
</p>

## 快速开始

```bash
pip install robotmem
```

```python
from robotmem import learn, recall, save_perception, start_session, end_session

# 开始 episode
session = start_session(context='{"robot_id": "arm-01", "task": "push"}')

# 记录经验
learn(
    insight="grip_force=12.5N 时抓取成功率最高",
    context='{"params": {"grip_force": {"value": 12.5, "unit": "N"}}, "task": {"success": true}}'
)

# 检索经验（支持结构化过滤 + 空间近邻）
memories = recall(
    query="抓取力参数",
    context_filter='{"task.success": true}',
    spatial_sort='{"field": "spatial.position", "target": [1.3, 0.7, 0.42]}'
)

# 存感知数据
save_perception(
    description="抓取轨迹: 30步, 成功",
    perception_type="procedural",
    data='{"sampled_actions": [[0.1, -0.3, 0.05, 0.8], ...]}'
)

# 结束 episode（自动巩固 + proactive recall）
end_session(session_id=session["session_id"])
```

## 7 个 API

| API | 用途 |
|-----|------|
| `learn` | 记录物理经验（参数/策略/教训） |
| `recall` | 检索经验 — BM25 + 向量混合搜索，支持 `context_filter` 和 `spatial_sort` |
| `save_perception` | 存感知/轨迹/力矩（visual/tactile/proprioceptive/auditory/procedural） |
| `forget` | 删除错误记忆 |
| `update` | 修正记忆内容 |
| `start_session` | 开始 Episode |
| `end_session` | 结束 Episode（自动巩固 + proactive recall） |

## 核心能力

### 结构化经验检索

不只是向量搜索——robotmem 理解机器人经验的结构：

```python
# 只检索成功经验
recall(query="push to target", context_filter='{"task.success": true}')

# 找最近的空间场景
recall(query="grasp object", spatial_sort='{"field": "spatial.object_position", "target": [1.3, 0.7, 0.42]}')

# 组合: 成功 + 距离 < 0.05m
recall(
    query="push",
    context_filter='{"task.success": true, "params.final_distance.value": {"$lt": 0.05}}'
)
```

### context JSON 四分区

```json
{
    "params":  {"grip_force": {"value": 12.5, "unit": "N", "type": "scalar"}},
    "spatial": {"object_position": [1.3, 0.7, 0.42], "target_position": [1.25, 0.6, 0.42]},
    "robot":   {"id": "fetch-001", "type": "Fetch", "dof": 7},
    "task":    {"name": "push_to_target", "success": true, "steps": 38}
}
```

`recall` 返回的每条记忆自动提取 `params`/`spatial`/`robot`/`task` 为顶层字段。

### 记忆巩固 + Proactive Recall

`end_session` 自动触发：
- **巩固**：Jaccard 相似度 > 0.50 的同类记忆合并（保护 constraint/postmortem/高 confidence）
- **Proactive Recall**：返回历史相关记忆，供下个 episode 参考

## FetchPush Demo

```bash
cd examples/fetch_push
pip install gymnasium-robotics
PYTHONPATH=../../src python demo.py  # 90 episodes, ~2 分钟
```

三阶段实验：基线 → 记忆写入 → 记忆利用。预期 Phase C 成功率比 Phase A 高 10-20%。

## 架构

```
SQLite + FTS5 + vec0
├── BM25 全文搜索（jieba CJK 分词）
├── 向量搜索（FastEmbed ONNX，纯 CPU）
├── RRF 融合排序
├── 结构化过滤（context_filter）
└── 空间近邻排序（spatial_sort）
```

- 纯 CPU，无 GPU 依赖
- 单文件数据库 `~/.robotmem/memory.db`
- MCP Server（7 个工具）或直接 Python import
- Web 管理界面：`robotmem web`

## 竞品对比

| 维度 | MemoryVLA (学术) | Mem0 (产品) | **robotmem** |
|------|-----------------|-------------|-------------|
| 目标用户 | 特定 VLA 模型 | 文本 AI | **机器人 AI** |
| 记忆格式 | 向量（不可读） | 文本 | **自然语言 + 感知 + 参数** |
| 结构化过滤 | 不支持 | 不支持 | **支持（context_filter）** |
| 空间检索 | 不支持 | 不支持 | **支持（spatial_sort）** |
| 物理参数 | 不支持 | 不支持 | **支持（params 分区）** |
| 安装 | 论文代码编译 | pip install | **pip install** |
| 数据库 | 内嵌 | 云服务 | **本地 SQLite** |

## License

Apache-2.0
