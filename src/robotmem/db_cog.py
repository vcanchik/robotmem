"""CogDatabase — 机器人认知数据库

连接管理 + schema 初始化 + tag_meta 同步。
数据操作方法由 ops/ 模块提供，本模块注册为 dispatch。
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from pathlib import Path

from .config import Config
from .db import floats_to_blob
from .resilience import safe_db_write, safe_db_transaction
from .schema import initialize_schema, initialize_vec
from .tag_tree import TAG_META_TREE

logger = logging.getLogger(__name__)


class CogDatabase:
    """机器人认知数据库 — 单一 memory.db 访问层

    初始化流程：
    1. 打开 SQLite 连接 + 4 条 PRAGMA
    2. 创建表/索引/触发器（schema.py）
    3. 加载 sqlite-vec 扩展 + 创建 vec0 表
    4. 同步 tag_meta 表
    """

    def __init__(self, config: Config):
        self._config = config
        self._db_path = config.db_path_resolved
        self._dim = config.effective_embedding_dim
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.RLock()
        self._vec_loaded = False
        self._closed = False

    @property
    def conn(self) -> sqlite3.Connection:
        """获取数据库连接（lazy init + 线程安全）"""
        with self._conn_lock:
            if self._closed:
                raise RuntimeError("CogDatabase 已关闭")
            if self._conn is None:
                self._connect()
            return self._conn  # type: ignore[return-value]

    def _connect(self) -> None:
        """创建连接 + 初始化 schema"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        # PRAGMA（产品架构圆桌 Linus 建议）
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")  # 8MB

        # schema
        initialize_schema(self._conn)

        # vec0
        self._vec_loaded = initialize_vec(self._conn, dim=self._dim)

        # tag_meta 同步
        self._ensure_tag_meta()

        logger.info(
            "CogDatabase 已连接: %s (vec=%s, dim=%d)",
            self._db_path, self._vec_loaded, self._dim,
        )

    def _ensure_tag_meta(self) -> None:
        """同步 tag_tree.py → tag_meta 表（幂等，原子事务）"""
        def _sync(c: sqlite3.Connection) -> None:
            c.executemany(
                "INSERT OR IGNORE INTO tag_meta (tag, parent, display_name) VALUES (?, ?, ?)",
                TAG_META_TREE,
            )

        ok, _ = safe_db_transaction(self.conn, _sync)
        if not ok:
            logger.warning("tag_meta 同步失败，下次启动时重试")

    @property
    def vec_loaded(self) -> bool:
        """sqlite-vec 扩展是否已加载"""
        return self._vec_loaded

    # ── 核心查询方法（dedup.py / conflict.py / search.py 依赖） ──

    def memory_exists(
        self, assertion: str, session_id: str | None, collection: str,
    ) -> bool:
        """精确匹配检查 — dedup Layer 1"""
        if session_id:
            row = self.conn.execute(
                "SELECT 1 FROM memories WHERE content = ? AND session_id = ? "
                "AND collection = ? AND status = 'active' LIMIT 1",
                (assertion, session_id, collection),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT 1 FROM memories WHERE content = ? "
                "AND collection = ? AND status = 'active' LIMIT 1",
                (assertion, collection),
            ).fetchone()
        return row is not None

    def fts_search_memories(
        self, query: str, collection: str, limit: int = 10,
    ) -> list[dict]:
        """FTS5 全文搜索 — dedup Layer 2"""
        from .db import tokenize_for_fts5

        fts_query = tokenize_for_fts5(query)
        if not fts_query:
            return []
        try:
            rows = self.conn.execute("""
                SELECT m.id, m.content, m.session_id, m.category, m.confidence
                FROM memories_fts f
                JOIN memories m ON m.id = f.rowid
                WHERE memories_fts MATCH ?
                  AND m.collection = ?
                  AND m.status = 'active'
                ORDER BY rank
                LIMIT ?
            """, (fts_query, collection, limit)).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("FTS5 搜索失败: %s", e)
            return []
        return [
            {"id": r[0], "content": r[1], "assertion": r[1],
             "session_id": r[2], "category": r[3], "confidence": r[4]}
            for r in rows
        ]

    def vec_search_memories(
        self, query_embedding: list[float], collection: str, limit: int = 10,
    ) -> list[dict]:
        """Vec0 向量搜索 — dedup Layer 3"""
        if not self._vec_loaded:
            return []
        blob = floats_to_blob(query_embedding)
        try:
            rows = self.conn.execute("""
                SELECT v.rowid, v.distance, m.content, m.session_id,
                       m.category, m.confidence
                FROM memories_vec v
                JOIN memories m ON m.id = v.rowid
                WHERE v.embedding MATCH ?
                  AND m.collection = ?
                  AND m.status = 'active'
                  AND k = ?
            """, (blob, collection, limit)).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("Vec 搜索失败: %s", e)
            return []
        return [
            {"id": r[0], "distance": r[1], "content": r[2], "assertion": r[2],
             "session_id": r[3], "category": r[4], "confidence": r[5]}
            for r in rows
        ]

    def supersede_memory(
        self, old_id: int, new_id: int, reason: str = "",
    ) -> None:
        """标记旧记忆被新记忆替代 — dedup / conflict"""
        safe_db_write(self.conn, """
            UPDATE memories
            SET status = 'superseded',
                superseded_by = ?,
                invalidated_reason = ?,
                invalidated_at = strftime('%Y-%m-%dT%H:%M:%f','now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%f','now')
            WHERE id = ?
        """, [new_id, reason, old_id])

    # ── 便利方法 ──

    @staticmethod
    def content_hash(text: str) -> str:
        """内容哈希 — 用于快速去重"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def close(self) -> None:
        """关闭数据库连接（关闭后不可重连）"""
        with self._conn_lock:
            self._closed = True
            if self._conn:
                try:
                    self._conn.close()
                except Exception as e:
                    logger.warning("CogDatabase 关闭异常: %s", e)
                finally:
                    self._conn = None
