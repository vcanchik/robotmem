"""ops/search.py 单元测试 — FTS5 + Vec 搜索"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from robotmem.ops.search import fts_search_memories, vec_search_memories
from robotmem.schema import initialize_schema
from robotmem.db import floats_to_blob


@pytest.fixture
def search_conn():
    """含测试数据的搜索用连接"""
    conn = sqlite3.connect(":memory:")
    initialize_schema(conn)

    # 插入测试记忆 + 手动 FTS5 同步（使用英文避免 CJK 分词差异）
    for i, content in enumerate([
        "robot grasping red block success",
        "gripper force 15N caused object sliding",
        "FetchPush push object to target position",
    ], start=1):
        conn.execute("""
            INSERT INTO memories (id, collection, type, content, human_summary,
                                 category, confidence, decay_rate, source, scope,
                                 status, scope_files, scope_entities)
            VALUES (?, 'test', 'fact', ?, ?, 'observation', 0.9, 0.01,
                    'tool', 'project', 'active', '[]', '[]')
        """, (i, content, content[:50]))
        # 手动 FTS5 同步
        conn.execute(
            "INSERT INTO memories_fts(rowid, content, human_summary, scope_files, scope_entities) VALUES (?, ?, ?, ?, ?)",
            (i, content, content[:50], "[]", "[]"),
        )

    # 插入一条非 active 记忆
    conn.execute("""
        INSERT INTO memories (id, collection, type, content, human_summary,
                             category, confidence, decay_rate, source, scope,
                             status, scope_files, scope_entities)
        VALUES (4, 'test', 'fact', 'deleted memory record', 'deleted', 'observation', 0.9, 0.01,
                'tool', 'project', 'superseded', '[]', '[]')
    """)
    conn.execute(
        "INSERT INTO memories_fts(rowid, content, human_summary, scope_files, scope_entities) VALUES (?, ?, ?, ?, ?)",
        (4, "deleted memory record", "deleted", "[]", "[]"),
    )

    conn.commit()
    yield conn
    conn.close()


class TestFtsSearchMemories:
    def test_basic_search(self, search_conn):
        results = fts_search_memories(search_conn, "robot grasping", "test")
        assert len(results) >= 1
        assert results[0]["content"] is not None

    def test_empty_query(self, search_conn):
        assert fts_search_memories(search_conn, "", "test") == []

    def test_none_query(self, search_conn):
        assert fts_search_memories(search_conn, None, "test") == []

    def test_whitespace_query(self, search_conn):
        assert fts_search_memories(search_conn, "   ", "test") == []

    def test_limit(self, search_conn):
        results = fts_search_memories(search_conn, "object", "test", limit=1)
        assert len(results) <= 1

    def test_limit_clamped_min(self, search_conn):
        """limit < 1 → clamp 到 1"""
        results = fts_search_memories(search_conn, "object", "test", limit=0)
        assert len(results) <= 1

    def test_limit_clamped_max(self, search_conn):
        """limit > 100 → clamp 到 100"""
        results = fts_search_memories(search_conn, "object", "test", limit=999)
        # 不崩溃
        assert isinstance(results, list)

    def test_wrong_collection(self, search_conn):
        results = fts_search_memories(search_conn, "robot", "nonexistent")
        assert results == []

    def test_returns_full_fields(self, search_conn):
        results = fts_search_memories(search_conn, "grasping", "test")
        if results:
            r = results[0]
            assert "id" in r
            assert "content" in r
            assert "type" in r
            assert "confidence" in r
            assert "perception_data" in r
            assert "bm25_score" in r

    def test_special_chars_cleaned(self, search_conn):
        """特殊字符不导致 FTS5 语法错误"""
        results = fts_search_memories(search_conn, 'test AND (OR) "hello', "test")
        assert isinstance(results, list)  # 不崩溃

    def test_fts5_keywords_cleaned(self, search_conn):
        """AND/OR/NOT 关键词被清理"""
        results = fts_search_memories(search_conn, "NOT AND OR NEAR", "test")
        # 全部被清理后可能是空 → 返回空列表
        assert isinstance(results, list)

    def test_single_char_tokens_filtered(self, search_conn):
        """单字符非 CJK token 被过滤"""
        results = fts_search_memories(search_conn, "a b c", "test")
        assert results == []  # 全部 < 2 字符 → 过滤

    def test_superseded_excluded(self, search_conn):
        """status != 'active' 的记忆不返回"""
        results = fts_search_memories(search_conn, "deleted memory", "test")
        for r in results:
            assert r["id"] != 4


class TestVecSearchMemories:
    def test_not_vec_loaded(self, search_conn):
        """vec_loaded=False → 空"""
        results = vec_search_memories(
            search_conn, [0.1, 0.2], "test", vec_loaded=False,
        )
        assert results == []

    def test_empty_embedding(self, search_conn):
        """空 embedding → 空"""
        results = vec_search_memories(
            search_conn, [], "test", vec_loaded=True,
        )
        assert results == []

    def test_none_embedding(self, search_conn):
        """None embedding → 空"""
        results = vec_search_memories(
            search_conn, None, "test", vec_loaded=True,
        )
        assert results == []

    def test_invalid_element_type(self, search_conn):
        """embedding 元素非数字 → 空"""
        results = vec_search_memories(
            search_conn, ["a", "b"], "test", vec_loaded=True,
        )
        assert results == []

    def test_limit_clamped(self, search_conn):
        """limit 超范围被 clamp"""
        results = vec_search_memories(
            search_conn, [0.1], "test", vec_loaded=True, limit=0,
        )
        assert isinstance(results, list)

    def test_vec_not_installed(self, search_conn):
        """vec0 表不存在时 → OperationalError → 空"""
        results = vec_search_memories(
            search_conn, [0.1, 0.2], "test", vec_loaded=True,
        )
        # memories_vec 表不存在 → OperationalError → 返回 []
        assert results == []
