"""robotmem convenience API — 同步 Python 接口

直接 Python 调用 robotmem，无需 MCP Context。
Lazy 初始化 DB + Embedder 单例，首次调用时自动连接。

用法:
    from robotmem import save_perception, recall

    save_perception(
        description="Grasped red cup: force=12.5N, 30 steps",
        perception_type="procedural",
        data='{"actions": [[0.1, -0.3, 0.05]], "force_peak": 12.5}',
    )

    memories = recall("how to grasp a cup")
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
import uuid

from .db import floats_to_blob

logger = logging.getLogger(__name__)

# ── 单例管理 ──

_lock = threading.Lock()
_db = None  # CogDatabase
_embedder = None  # Embedder
_config = None  # Config


def _ensure_init():
    """Lazy 初始化 DB + Embedder（线程安全）"""
    global _db, _embedder, _config
    if _db is not None:
        return
    with _lock:
        if _db is not None:
            return
        from .config import load_config
        from .db_cog import CogDatabase
        from .embed import create_embedder

        _config = load_config()
        _db = CogDatabase(_config)
        _ = _db.conn  # 触发 lazy connect

        _embedder = create_embedder(_config)
        try:
            _run_async(_embedder.check_availability())
        except Exception as e:
            logger.warning("robotmem: embedding 不可用 — %s", e)


def _run_async(coro):
    """同步执行 async 协程"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的 event loop — 直接用 asyncio.run
        return asyncio.run(coro)
    # 已有 event loop（Jupyter 等场景）— 在新线程中运行
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _resolve_collection(collection: str | None) -> str:
    _ensure_init()
    if collection and collection.strip():
        return collection.strip()
    return _config.default_collection


# ── Tool 1: save_perception ──


def save_perception(
    description: str,
    perception_type: str = "visual",
    data: str | None = None,
    metadata: str | None = None,
    collection: str | None = None,
    session_id: str | None = None,
) -> dict:
    """保存感知/轨迹/力矩（procedural memory）

    Args:
        description: 感知描述
        perception_type: 类型 (visual/tactile/auditory/proprioceptive/procedural)
        data: JSON 字符串，传感器数据
        metadata: JSON 字符串，元数据
        collection: 记忆集合名（默认 "default"）
        session_id: 会话 ID

    Returns:
        {"memory_id": int, "perception_type": str, "collection": str, "has_embedding": bool}
    """
    _ensure_init()
    from .ops.memories import insert_memory

    coll = _resolve_collection(collection)

    embedding = None
    if _embedder.available:
        try:
            emb_list = _run_async(_embedder.embed_one(description))
            embedding = floats_to_blob(emb_list)
        except Exception as e:
            logger.warning("save_perception embedding 失败: %s", e)

    memory_id = insert_memory(
        _db.conn,
        {
            "session_id": session_id,
            "collection": coll,
            "type": "perception",
            "content": description,
            "human_summary": description[:200],
            "perception_type": perception_type,
            "perception_data": data,
            "perception_metadata": metadata,
            "category": "observation",
            "confidence": 0.9,
            "source": "api",
            "scope": "project",
            "embedding": embedding,
        },
        vec_loaded=_db.vec_loaded,
    )

    if not memory_id:
        return {"error": "写入失败（可能重复）"}

    return {
        "memory_id": memory_id,
        "perception_type": perception_type,
        "collection": coll,
        "has_embedding": embedding is not None,
    }


# ── Tool 2: recall ──


def recall(
    query: str,
    collection: str | None = None,
    n: int = 5,
    min_confidence: float = 0.3,
    session_id: str | None = None,
    context_filter: dict | None = None,
    spatial_sort: dict | None = None,
) -> dict:
    """检索经验 — BM25 + Vec 混合搜索

    Args:
        query: 搜索查询
        collection: 记忆集合名
        n: 返回数量
        min_confidence: 最低 confidence 阈值
        session_id: 限定 session 范围
        context_filter: 结构化过滤条件 dict
        spatial_sort: 空间近邻排序 dict

    Returns:
        {"memories": [...], "total": int, "mode": str, "query_ms": float}
    """
    _ensure_init()
    from .search import recall as do_recall

    coll = _resolve_collection(collection)
    emb = _embedder if _embedder.available else None

    result = _run_async(
        do_recall(
            query=query,
            db=_db,
            embedder=emb,
            collection=coll,
            top_k=n,
            min_confidence=min_confidence,
            session_id=session_id,
            context_filter=context_filter,
            spatial_sort=spatial_sort,
        )
    )

    return {
        "memories": result.memories,
        "total": result.total,
        "mode": result.mode,
        "query_ms": round(result.query_ms, 1),
    }


# ── Tool 3: learn ──


def learn(
    insight: str,
    context: str = "",
    collection: str | None = None,
    session_id: str | None = None,
) -> dict:
    """记录物理经验（declarative memory）

    Args:
        insight: 经验/知识内容
        context: 上下文描述
        collection: 记忆集合名
        session_id: 会话 ID

    Returns:
        {"status": "created", "memory_id": int, "auto_inferred": {...}}
    """
    _ensure_init()
    from .auto_classify import (
        build_context_json,
        classify_category,
        classify_tags,
        estimate_confidence,
        extract_scope,
        normalize_scope_files,
    )
    from .dedup import check_duplicate
    from .ops.memories import insert_memory

    coll = _resolve_collection(collection)

    # auto_classify — 每步独立容错
    try:
        category = classify_category(insight)
    except Exception:
        category = "observation"

    try:
        confidence = estimate_confidence(insight, context)
    except Exception:
        confidence = 0.9

    try:
        scope = extract_scope(insight)
        scope_files = normalize_scope_files(scope.get("scope_files", []))
        scope_entities = scope.get("scope_entities", [])
    except Exception:
        scope_files, scope_entities = [], []

    try:
        inferred_tags = classify_tags(insight, context)
    except Exception:
        inferred_tags = []

    try:
        ctx_json = build_context_json(insight, context)
    except Exception:
        ctx_json = context

    # 去重
    try:
        dedup_result = check_duplicate(
            insight,
            coll,
            session_id,
            _db,
            _embedder if _embedder.available else None,
        )
        if dedup_result.is_dup:
            existing_id = (
                dedup_result.similar_facts[0].get("id")
                if dedup_result.similar_facts
                else None
            )
            return {
                "status": "duplicate",
                "method": dedup_result.method,
                "existing_id": existing_id,
                "similarity": dedup_result.similarity,
            }
    except Exception as e:
        logger.warning("learn 去重检查异常: %s", e)

    # Embedding
    embedding = None
    if _embedder.available:
        try:
            emb_list = _run_async(_embedder.embed_one(insight))
            embedding = floats_to_blob(emb_list)
        except Exception as e:
            logger.warning("learn embedding 失败: %s", e)

    memory_id = insert_memory(
        _db.conn,
        {
            "session_id": session_id,
            "collection": coll,
            "type": "fact",
            "content": insight,
            "human_summary": insight[:200],
            "context": ctx_json if isinstance(ctx_json, str) else json.dumps(ctx_json),
            "category": category,
            "confidence": confidence,
            "source": "api",
            "scope": "project",
            "scope_files": json.dumps(scope_files),
            "scope_entities": json.dumps(scope_entities),
            "embedding": embedding,
            "tags": inferred_tags,
            "tag_source": "auto",
        },
        vec_loaded=_db.vec_loaded,
    )

    if not memory_id:
        return {"error": "写入失败（可能重复）"}

    return {
        "status": "created",
        "memory_id": memory_id,
        "auto_inferred": {
            "category": category,
            "confidence": confidence,
            "tags": inferred_tags,
            "scope_files": scope_files,
        },
    }


# ── Tool 4: forget ──


def forget(memory_id: int, reason: str) -> dict:
    """删除错误记忆（软删除）

    Args:
        memory_id: 记忆 ID
        reason: 删除原因

    Returns:
        {"status": "forgotten", "memory_id": int, "content": str, "reason": str}
    """
    _ensure_init()
    from .ops.memories import get_memory, invalidate_memory

    mem = get_memory(_db.conn, memory_id)
    if not mem:
        return {"error": f"记忆 #{memory_id} 不存在"}
    if mem.get("status") != "active":
        return {"error": f"记忆 #{memory_id} 状态为 {mem.get('status')}，无法删除"}

    invalidate_memory(_db.conn, memory_id, reason)

    return {
        "status": "forgotten",
        "memory_id": memory_id,
        "content": (mem.get("content") or "")[:100],
        "reason": reason,
    }


# ── Tool 5: update ──


def update(memory_id: int, new_content: str, context: str = "") -> dict:
    """修正记忆内容

    Args:
        memory_id: 记忆 ID
        new_content: 新内容
        context: 上下文

    Returns:
        {"status": "updated", "memory_id": int, "old_content": str, "new_content": str}
    """
    _ensure_init()
    from .ops.memories import (
        get_memory,
        update_memory,
        update_memory_embedding,
    )
    from .ops.tags import add_tags

    mem = get_memory(_db.conn, memory_id)
    if not mem:
        return {"error": f"记忆 #{memory_id} 不存在"}
    if mem.get("status") != "active":
        return {"error": f"记忆 #{memory_id} 状态为 {mem.get('status')}，无法更新"}

    old_content = mem.get("content", "")

    # 重新分类
    try:
        from .auto_classify import classify_category, estimate_confidence

        category = classify_category(new_content)
        confidence = estimate_confidence(new_content, context)
    except Exception:
        category = mem.get("category", "observation")
        confidence = mem.get("confidence", 0.9)

    update_memory(
        _db.conn,
        memory_id,
        content=new_content,
        category=category,
        confidence=confidence,
    )

    # 重建 embedding
    if _embedder.available:
        try:
            new_emb = _run_async(_embedder.embed_one(new_content))
            update_memory_embedding(
                _db.conn,
                memory_id,
                new_emb,
                vec_loaded=_db.vec_loaded,
            )
        except Exception as e:
            logger.warning("update embedding 重建失败: %s", e)

    # 重建 tags
    try:
        from .auto_classify import classify_tags

        inferred_tags = classify_tags(new_content, context)
        if inferred_tags:
            add_tags(_db.conn, memory_id, inferred_tags, source="auto")
    except Exception as e:
        logger.warning("update tags 重建失败: %s", e)

    return {
        "status": "updated",
        "memory_id": memory_id,
        "old_content": old_content[:100],
        "new_content": new_content[:100],
        "auto_inferred": {
            "category": category,
            "confidence": confidence,
        },
    }


# ── Tool 6: start_session ──


def start_session(
    collection: str | None = None,
    context: str | None = None,
) -> dict:
    """开始新会话（episode）

    Args:
        collection: 记忆集合名
        context: JSON 字符串，会话上下文

    Returns:
        {"session_id": str, "collection": str, "active_memories_count": int}
    """
    _ensure_init()
    from .ops.sessions import get_or_create_session, update_session_context

    coll = _resolve_collection(collection)
    ext_id = str(uuid.uuid4())

    session = get_or_create_session(_db.conn, ext_id, coll)
    if not session:
        return {"error": "创建 session 失败"}

    if context:
        update_session_context(_db.conn, ext_id, context)

    try:
        active_count = _db.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE collection=? AND status='active'",
            (coll,),
        ).fetchone()[0]
    except Exception:
        active_count = 0

    return {
        "session_id": ext_id,
        "collection": coll,
        "active_memories_count": active_count,
    }


# ── Tool 7: end_session ──


def end_session(
    session_id: str,
    outcome_score: float | None = None,
) -> dict:
    """结束会话 — 标记结束 + 时间衰减 + 巩固

    Args:
        session_id: 会话 ID（start_session 返回的）
        outcome_score: 本次会话评分

    Returns:
        {"status": "ended", "session_id": str, "summary": {...}, ...}
    """
    _ensure_init()
    from .ops.memories import apply_time_decay, consolidate_session as do_consolidate
    from .ops.sessions import (
        get_session_summary,
        insert_session_outcome,
        mark_session_ended,
    )

    # 查询 collection
    try:
        row = _db.conn.execute(
            "SELECT collection FROM sessions WHERE external_id=?",
            (session_id,),
        ).fetchone()
        coll = row[0] if row else _config.default_collection
    except Exception:
        coll = _config.default_collection

    mark_session_ended(_db.conn, session_id)

    decayed = 0
    try:
        decayed = apply_time_decay(_db.conn)
    except Exception as e:
        logger.warning("end_session time_decay 失败: %s", e)

    consolidated = {"merged_groups": 0, "superseded_count": 0}
    try:
        consolidated = do_consolidate(_db.conn, session_id, coll)
    except Exception as e:
        logger.warning("consolidate_session 失败: %s", e)

    if outcome_score is not None:
        try:
            insert_session_outcome(_db.conn, session_id, outcome_score)
        except Exception as e:
            logger.warning("end_session outcome 写入失败: %s", e)

    summary = get_session_summary(_db.conn, session_id, coll)

    return {
        "status": "ended",
        "session_id": session_id,
        "summary": summary,
        "decayed_count": decayed,
        "consolidated": consolidated,
    }
