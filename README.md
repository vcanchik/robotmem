[中文版](README_zh.md)

Fork note: this repository is maintained at `https://github.com/vcanchik/robotmem` and is based on the upstream project `https://github.com/robotmem/robotmem`. The upstream Apache-2.0 license is preserved in this fork.

# robotmem — Let Robots Learn from Experience

> Your robot ran 1000 experiments, starting from scratch every time. robotmem stores episode experiences — parameters, trajectories, outcomes — and retrieves the most relevant ones to guide future decisions.

**FetchPush experiment**: +25% success rate improvement (42% → 67%), CPU-only, reproducible in 5 minutes.

<p align="center">
  <img src="examples/demo.gif" alt="robotmem 30s demo: save → restart → recall" width="600">
</p>

## Quick Start

```bash
pip install robotmem
```

```python
from robotmem import learn, recall, save_perception, start_session, end_session

# Start an episode
session = start_session(context='{"robot_id": "arm-01", "task": "push"}')

# Record experience
learn(
    insight="grip_force=12.5N yields highest grasp success rate",
    context='{"params": {"grip_force": {"value": 12.5, "unit": "N"}}, "task": {"success": true}}'
)

# Retrieve experiences (structured filtering + spatial nearest-neighbor)
memories = recall(
    query="grip force parameters",
    context_filter='{"task.success": true}',
    spatial_sort='{"field": "spatial.position", "target": [1.3, 0.7, 0.42]}'
)

# Store perception data
save_perception(
    description="Grasp trajectory: 30 steps, success",
    perception_type="procedural",
    data='{"sampled_actions": [[0.1, -0.3, 0.05, 0.8], ...]}'
)

# End episode (auto-consolidation + proactive recall)
end_session(session_id=session["session_id"])
```

## 7 APIs

| API | Purpose |
|-----|---------|
| `learn` | Record physical experiences (parameters / strategies / lessons) |
| `recall` | Retrieve experiences — BM25 + vector hybrid search with `context_filter` and `spatial_sort` |
| `save_perception` | Store perception / trajectory / force data (visual / tactile / proprioceptive / auditory / procedural) |
| `forget` | Delete incorrect memories |
| `update` | Correct memory content |
| `start_session` | Begin an episode |
| `end_session` | End an episode (auto-consolidation + proactive recall) |

## Key Features

### Structured Experience Retrieval

Not just vector search — robotmem understands the structure of robot experiences:

```python
# Retrieve only successful experiences
recall(query="push to target", context_filter='{"task.success": true}')

# Find spatially nearest scenarios
recall(query="grasp object", spatial_sort='{"field": "spatial.object_position", "target": [1.3, 0.7, 0.42]}')

# Combine: success + distance < 0.05m
recall(
    query="push",
    context_filter='{"task.success": true, "params.final_distance.value": {"$lt": 0.05}}'
)
```

### Context JSON — 4 Sections

```json
{
    "params":  {"grip_force": {"value": 12.5, "unit": "N", "type": "scalar"}},
    "spatial": {"object_position": [1.3, 0.7, 0.42], "target_position": [1.25, 0.6, 0.42]},
    "robot":   {"id": "fetch-001", "type": "Fetch", "dof": 7},
    "task":    {"name": "push_to_target", "success": true, "steps": 38}
}
```

Each recalled memory automatically extracts `params` / `spatial` / `robot` / `task` as top-level fields.

### Memory Consolidation + Proactive Recall

`end_session` automatically triggers:
- **Consolidation**: Merges similar memories with Jaccard similarity > 0.50 (protects constraint / postmortem / high-confidence entries)
- **Proactive Recall**: Returns historically relevant memories for the next episode

## FetchPush Demo

```bash
cd examples/fetch_push
pip install gymnasium-robotics
PYTHONPATH=../../src python demo.py  # 90 episodes, ~2 min
```

Three-phase experiment: baseline → memory writing → memory utilization. Expected Phase C success rate 10-20% higher than Phase A.

## Architecture

```
SQLite + FTS5 + vec0
├── BM25 full-text search (jieba CJK tokenizer)
├── Vector search (FastEmbed ONNX, CPU-only)
├── RRF fusion ranking
├── Structured filtering (context_filter)
└── Spatial nearest-neighbor sorting (spatial_sort)
```

- CPU-only, no GPU required
- Single-file database `~/.robotmem/memory.db`
- MCP Server (7 tools) or direct Python import
- Web management UI: `robotmem web`

## Comparison

| Feature | MemoryVLA (Academic) | Mem0 (Product) | **robotmem** |
|---------|---------------------|----------------|-------------|
| Target users | Specific VLA models | Text AI | **Robotic AI** |
| Memory format | Vectors (opaque) | Text | **Natural language + perception + parameters** |
| Structured filtering | No | No | **Yes (`context_filter`)** |
| Spatial retrieval | No | No | **Yes (`spatial_sort`)** |
| Physical parameters | No | No | **Yes (`params` section)** |
| Installation | Compile from paper code | pip install | **pip install** |
| Database | Embedded | Cloud | **Local SQLite** |

## License

Apache-2.0
