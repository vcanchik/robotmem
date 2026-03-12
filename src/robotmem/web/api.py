"""REST API — 12 个端点

宪法原则 2 落地：可见（列表/搜索）、可懂（详情/统计）、可控（删除/更新）。

三层防御：
- L1 事前：参数校验（类型/范围/非空）
- L2 事中：try-except 包裹 DB 操作
- L3 事后：统一 JSON 返回 + 错误码
"""

from __future__ import annotations

import logging
import sqlite3

from flask import Blueprint, current_app, jsonify, request

from ..db_cog import CogDatabase
from ..search import extract_context_fields
from ..ops.memories import (
    get_memory,
    invalidate_memory,
    update_memory,
)

import os

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _get_db() -> CogDatabase:
    return current_app.config["ROBOTMEM_DB"]


# ── GET /api/doctor ──

@api_bp.route("/doctor")
def doctor():
    """健康检查：FTS5/vec0 同步、零命中率、基础计数、DB 大小"""
    db = _get_db()
    config = current_app.config["ROBOTMEM_CONFIG"]

    try:
        conn = db.conn
        conn.execute("SELECT 1")  # 连接验活

        # 基础计数
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]

        by_type = {}
        for row in conn.execute(
            "SELECT type, COUNT(*) FROM memories "
            "WHERE status='active' GROUP BY type"
        ).fetchall():
            by_type[row[0]] = row[1]

        # Session 计数
        session_total = conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        session_by_status = {}
        for row in conn.execute(
            "SELECT status, COUNT(*) FROM sessions GROUP BY status"
        ).fetchall():
            session_by_status[row[0]] = row[1]

        # FTS5 同步检查
        fts_count = None
        fts_ok = None
        try:
            fts_count = conn.execute(
                "SELECT COUNT(*) FROM memories_fts"
            ).fetchone()[0]
            fts_ok = fts_count == total
        except sqlite3.OperationalError as e:
            logger.debug("FTS5 检查跳过: %s", e)

        # vec0 同步检查
        # 注意：不是所有记忆都有 embedding（embed_backend='none' 或 embedding 失败）
        # vec_count <= total 是正常的，vec_count > total 才是异常（孤儿 vec 条目）
        vec_count = None
        vec_ok = None
        try:
            vec_count = conn.execute(
                "SELECT COUNT(*) FROM memories_vec"
            ).fetchone()[0]
            vec_ok = vec_count <= total
        except sqlite3.OperationalError as e:
            logger.debug("vec0 检查跳过: %s", e)

        # 零命中率
        zero_hit = 0
        if total > 0:
            zero_hit = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE status='active' AND (access_count IS NULL OR access_count = 0)"
            ).fetchone()[0]

        zero_hit_rate = round(zero_hit / total * 100, 1) if total > 0 else 0.0

        # DB 文件大小
        db_path = config.db_path_resolved
        db_size = 0
        try:
            db_size = os.path.getsize(db_path)
        except OSError:
            pass

        return jsonify({
            "memories": {
                "total": total,
                "by_type": by_type,
            },
            "sessions": {
                "total": session_total,
                "by_status": session_by_status,
            },
            "fts5": {
                "indexed": fts_count,
                "expected": total,
                "ok": fts_ok,
            },
            "vec0": {
                "indexed": vec_count,
                "total_memories": total,
                "ok": vec_ok,
            },
            "zero_hit": {
                "count": zero_hit,
                "total": total,
                "rate": zero_hit_rate,
            },
            "db_size_bytes": db_size,
        })
    except Exception as e:
        logger.exception("doctor 检查失败")
        return jsonify({"error": "健康检查失败"}), 500


# ── GET /api/stats ──

@api_bp.route("/stats")
def stats():
    """统计信息：总数、类型分布、collection 列表"""
    db = _get_db()
    try:
        conn = db.conn

        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]

        by_type = {}
        for row in conn.execute(
            "SELECT type, COUNT(*) FROM memories "
            "WHERE status='active' GROUP BY type"
        ).fetchall():
            by_type[row[0]] = row[1]

        by_category = {}
        for row in conn.execute(
            "SELECT category, COUNT(*) FROM memories "
            "WHERE status='active' GROUP BY category"
        ).fetchall():
            by_category[row[0]] = row[1]

        collections = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT collection FROM memories ORDER BY collection"
            ).fetchall()
        ]

        recent = conn.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE status='active' AND created_at > datetime('now', '-24 hours')"
        ).fetchone()[0]

        return jsonify({
            "total": total,
            "by_type": by_type,
            "by_category": by_category,
            "collections": collections,
            "recent_24h": recent,
        })
    except Exception as e:
        logger.exception("stats 查询失败")
        return jsonify({"error": "查询失败"}), 500


# ── GET /api/recent-failures ──

@api_bp.route("/recent-failures")
def recent_failures():
    """最近失败记忆 — ?limit=5（category IN postmortem, gotcha）"""
    db = _get_db()

    # L1: 参数校验
    limit = request.args.get("limit", 5, type=int)
    limit = min(max(1, limit), 20)

    try:
        conn = db.conn
        rows = conn.execute("""
            SELECT id, collection, type, content, human_summary,
                   perception_type, category, confidence,
                   created_at
            FROM memories
            WHERE status = 'active'
              AND category IN ('postmortem', 'gotcha')
            ORDER BY created_at DESC
            LIMIT ?
        """, [limit]).fetchall()

        failures = [
            {
                "id": r[0], "collection": r[1], "type": r[2],
                "content": r[3], "human_summary": r[4],
                "perception_type": r[5], "category": r[6],
                "confidence": r[7], "created_at": r[8],
            }
            for r in rows
        ]

        return jsonify({"failures": failures, "total": len(failures)})
    except Exception as e:
        logger.exception("recent_failures 失败")
        return jsonify({"error": "查询失败"}), 500


# ── GET /api/memories ──

@api_bp.route("/memories")
def list_memories():
    """分页列表 — ?page=0&limit=30&collection=&type=&status=active"""
    db = _get_db()

    # L1: 参数解析
    page = request.args.get("page", 0, type=int)
    limit = request.args.get("limit", 30, type=int)
    limit = min(max(1, limit), 100)
    offset = page * limit

    collection = request.args.get("collection", "").strip()
    mem_type = request.args.get("type", "").strip()
    status = request.args.get("status", "active").strip()

    # P1 扩展参数
    category = request.args.get("category", "").strip()
    conf_min = request.args.get("confidence_min", type=float)
    conf_max = request.args.get("confidence_max", type=float)
    days = request.args.get("days", type=int)
    perception_type = request.args.get("perception_type", "").strip()

    try:
        conn = db.conn
        conditions = ["status = ?"]
        params: list = [status]

        if collection:
            conditions.append("collection = ?")
            params.append(collection)
        if mem_type:
            conditions.append("type = ?")
            params.append(mem_type)
        if category:
            cats = [c.strip() for c in category.split(",") if c.strip()]
            placeholders = ",".join("?" * len(cats))
            conditions.append(f"category IN ({placeholders})")
            params.extend(cats)
        if conf_min is not None:
            conditions.append("confidence >= ?")
            params.append(conf_min)
        if conf_max is not None:
            conditions.append("confidence <= ?")
            params.append(conf_max)
        if days and days > 0:
            conditions.append("created_at > datetime('now', ?)")
            params.append(f"-{days} days")
        if perception_type:
            conditions.append("perception_type = ?")
            params.append(perception_type)

        where = " AND ".join(conditions)

        # 总数
        total = conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {where}", params
        ).fetchone()[0]

        # 分页查询
        rows = conn.execute(f"""
            SELECT id, collection, type, content, human_summary,
                   perception_type, category, confidence, decay_rate,
                   source, scope, status, access_count,
                   created_at, updated_at
            FROM memories
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()

        memories = [
            {
                "id": r[0], "collection": r[1], "type": r[2],
                "content": r[3], "human_summary": r[4],
                "perception_type": r[5], "category": r[6],
                "confidence": r[7], "decay_rate": r[8],
                "source": r[9], "scope": r[10], "status": r[11],
                "access_count": r[12],
                "created_at": r[13], "updated_at": r[14],
            }
            for r in rows
        ]

        return jsonify({
            "memories": memories,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit if limit > 0 else 0,
        })
    except Exception as e:
        logger.exception("list_memories 失败")
        return jsonify({"error": "查询失败"}), 500


# ── GET /api/search ──

@api_bp.route("/search")
def search_memories():
    """搜索 — ?q=&collection=&top_k=10（FTS5 全文搜索）"""
    db = _get_db()

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "q 参数不能为空"}), 400

    collection = request.args.get("collection", "").strip()
    top_k = request.args.get("top_k", 10, type=int)
    top_k = min(max(1, top_k), 50)

    try:
        from ..ops.search import fts_search_memories
        conn = db.conn

        # collection 为空时传 None → 单次查询搜所有 collection
        coll_param = collection if collection else None
        results = fts_search_memories(conn, query, coll_param, limit=top_k)

        # context JSON 解析 — 提取 params/spatial/robot 便捷字段
        # perception_data 可能是大型 blob，搜索列表不返回（详情端点返回）
        for r in results:
            extract_context_fields(r)
            r.pop("perception_data", None)

        return jsonify({
            "results": results,
            "total": len(results),
            "query": query,
        })
    except Exception as e:
        logger.exception("search 失败")
        return jsonify({"error": "搜索失败"}), 500


# ── GET /api/memory/<id> ──

@api_bp.route("/memory/<int:memory_id>")
def get_memory_detail(memory_id: int):
    """单条详情"""
    db = _get_db()
    try:
        mem = get_memory(db.conn, memory_id)
        if not mem:
            return jsonify({"error": f"记忆 #{memory_id} 不存在"}), 404
        # 移除 binary 字段（embedding）
        mem.pop("embedding", None)
        return jsonify(mem)
    except Exception as e:
        logger.exception("get_memory_detail 失败")
        return jsonify({"error": "查询失败"}), 500


# ── DELETE /api/memory/<id> ──

@api_bp.route("/memory/<int:memory_id>", methods=["DELETE"])
def delete_memory(memory_id: int):
    """删除（软删除）"""
    db = _get_db()

    reason = (request.json or {}).get("reason", "Web UI 删除")
    if not reason:
        reason = "Web UI 删除"

    try:
        mem = get_memory(db.conn, memory_id)
        if not mem:
            return jsonify({"error": f"记忆 #{memory_id} 不存在"}), 404
        if mem.get("status") != "active":
            return jsonify({"error": f"记忆 #{memory_id} 已是 {mem.get('status')} 状态"}), 400

        invalidate_memory(db.conn, memory_id, reason)
        return jsonify({"status": "deleted", "memory_id": memory_id})
    except Exception as e:
        logger.exception("delete_memory 失败")
        return jsonify({"error": "删除失败"}), 500


# ── PUT /api/memory/<id> ──

@api_bp.route("/memory/<int:memory_id>", methods=["PUT"])
def update_memory_api(memory_id: int):
    """更新记忆字段"""
    db = _get_db()

    data = request.json
    if not data:
        return jsonify({"error": "请求体不能为空"}), 400

    try:
        mem = get_memory(db.conn, memory_id)
        if not mem:
            return jsonify({"error": f"记忆 #{memory_id} 不存在"}), 404

        # 只允许更新安全字段
        allowed = {
            "content", "human_summary", "category", "confidence",
            "decay_rate", "scope", "context",
        }
        updates = {k: v for k, v in data.items() if k in allowed}
        if not updates:
            return jsonify({"error": "没有可更新的字段"}), 400

        update_memory(db.conn, memory_id, **updates)
        return jsonify({"status": "updated", "memory_id": memory_id, "fields": list(updates.keys())})
    except Exception as e:
        logger.exception("update_memory_api 失败")
        return jsonify({"error": "更新失败"}), 500


# ── GET /api/sessions ──

@api_bp.route("/sessions")
def list_sessions():
    """会话列表"""
    db = _get_db()

    page = request.args.get("page", 0, type=int)
    limit = request.args.get("limit", 20, type=int)
    limit = min(max(1, limit), 50)
    offset = page * limit

    try:
        conn = db.conn

        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        rows = conn.execute("""
            SELECT s.id, s.external_id, s.collection, s.context,
                   s.session_count, s.status, s.created_at, s.updated_at,
                   COALESCE(mc.cnt, 0) as memory_count
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) as cnt
                FROM memories
                WHERE status = 'active'
                GROUP BY session_id
            ) mc ON mc.session_id = s.external_id
            ORDER BY s.created_at DESC
            LIMIT ? OFFSET ?
        """, [limit, offset]).fetchall()

        sessions = [
            {
                "id": r[0], "external_id": r[1], "collection": r[2],
                "context": r[3], "session_count": r[4], "status": r[5],
                "created_at": r[6], "updated_at": r[7],
                "memory_count": r[8],
            }
            for r in rows
        ]

        return jsonify({
            "sessions": sessions,
            "total": total,
            "page": page,
            "limit": limit,
        })
    except Exception as e:
        logger.exception("list_sessions 失败")
        return jsonify({"error": "查询失败"}), 500


# ── GET /api/collections ──

@api_bp.route("/collections")
def list_collections():
    """Collection 列表 + 每个的记忆数"""
    db = _get_db()
    try:
        conn = db.conn
        rows = conn.execute("""
            SELECT collection, COUNT(*) as cnt
            FROM memories
            WHERE status = 'active'
            GROUP BY collection
            ORDER BY cnt DESC
        """).fetchall()

        return jsonify({
            "collections": [
                {"name": r[0], "count": r[1]}
                for r in rows
            ],
        })
    except Exception as e:
        logger.exception("list_collections 失败")
        return jsonify({"error": "查询失败"}), 500


# ── GET /api/categories ──

@api_bp.route("/categories")
def list_categories():
    """数据库中实际存在的 category 列表（带计数）"""
    db = _get_db()
    try:
        rows = db.conn.execute("""
            SELECT category, COUNT(*) as cnt
            FROM memories
            WHERE status = 'active' AND category IS NOT NULL
            GROUP BY category
            ORDER BY cnt DESC
        """).fetchall()
        return jsonify({
            "categories": [
                {"name": r[0], "count": r[1]}
                for r in rows
            ],
        })
    except Exception as e:
        logger.exception("list_categories 失败")
        return jsonify({"error": "查询失败"}), 500


# ── GET /api/sessions/<external_id>/memories ──

@api_bp.route("/sessions/<external_id>/memories")
def session_memories(external_id: str):
    """Session 内记忆列表 — 按时间 ASC 排序（时间线视图）"""
    db = _get_db()

    limit = request.args.get("limit", 20, type=int)
    limit = min(max(1, limit), 100)

    try:
        conn = db.conn
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ? AND status = 'active'",
            [external_id],
        ).fetchone()[0]

        rows = conn.execute("""
            SELECT id, type, content, human_summary, perception_type,
                   category, confidence, created_at
            FROM memories
            WHERE session_id = ? AND status = 'active'
            ORDER BY created_at ASC
            LIMIT ?
        """, [external_id, limit]).fetchall()

        memories = [
            {
                "id": r[0], "type": r[1], "content": r[2],
                "human_summary": r[3], "perception_type": r[4],
                "category": r[5], "confidence": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]
        return jsonify({"memories": memories, "total": total})
    except Exception as e:
        logger.exception("session_memories 失败")
        return jsonify({"error": "查询失败"}), 500


# ── GET /api/outreach ──

@api_bp.route("/outreach")
def outreach():
    """Outreach 链接数据"""
    from robotmem.outreach import get_outreach_data
    return jsonify({"groups": get_outreach_data()})


# ── GET /api/outreach/check ──

@api_bp.route("/outreach/check", methods=["GET", "POST"])
def outreach_check():
    """检查所有 Outreach URL 可达性"""
    from robotmem.outreach import check_all_urls
    results = check_all_urls(timeout=5)
    return jsonify({"results": results})
