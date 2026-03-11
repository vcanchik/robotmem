"""search.py recall 函数测试 — 覆盖 RRF、context_filter、spatial_sort、recall 主流程"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from robotmem.search import (
    RecallResult,
    extract_context_fields,
    rrf_merge,
    _apply_source_weight,
    _resolve_dotpath,
    _match_context_filter,
    _compute_spatial_distance,
    recall,
    recall_sync,
    _MISSING,
)


class TestExtractContextFields:
    def test_valid_json(self):
        m = {"context": json.dumps({"params": {"x": 1}, "task": {"success": True}})}
        extract_context_fields(m)
        assert m["params"] == {"x": 1}
        assert m["task"] == {"success": True}

    def test_empty_context(self):
        m = {"context": ""}
        extract_context_fields(m)
        assert "params" not in m

    def test_invalid_json(self):
        m = {"context": "not json"}
        extract_context_fields(m)
        assert "params" not in m

    def test_none_context(self):
        m = {"context": None}
        extract_context_fields(m)
        assert "params" not in m


class TestRRFMerge:
    def test_single_list(self):
        result = rrf_merge([{"id": 1}, {"id": 2}])
        assert len(result) == 2
        assert result[0]["id"] == 1  # rank 0 → higher score

    def test_two_lists_overlap(self):
        l1 = [{"id": 1}, {"id": 2}]
        l2 = [{"id": 2}, {"id": 3}]
        result = rrf_merge(l1, l2)
        # id=2 出现在两个列表中，分数更高
        assert result[0]["id"] == 2

    def test_empty_lists(self):
        assert rrf_merge([], []) == []

    def test_none_id(self):
        result = rrf_merge([{"x": 1}])  # 无 id
        assert result == []


class TestApplySourceWeight:
    def test_real_data_boosted(self):
        memories = [{
            "context": json.dumps({"env": {"sim_or_real": "real"}}),
            "_rrf_score": 1.0,
        }]
        _apply_source_weight(memories)
        assert memories[0]["_rrf_score"] == 1.5

    def test_sim_data_unchanged(self):
        memories = [{
            "context": json.dumps({"env": {"sim_or_real": "sim"}}),
            "_rrf_score": 1.0,
        }]
        _apply_source_weight(memories)
        assert memories[0]["_rrf_score"] == 1.0

    def test_no_context(self):
        memories = [{"_rrf_score": 1.0}]
        _apply_source_weight(memories)
        assert memories[0]["_rrf_score"] == 1.0

    def test_invalid_context(self):
        memories = [{"context": "not json", "_rrf_score": 1.0}]
        _apply_source_weight(memories)
        assert memories[0]["_rrf_score"] == 1.0


class TestResolveDotpath:
    def test_simple(self):
        assert _resolve_dotpath({"a": 1}, "a") == 1

    def test_nested(self):
        assert _resolve_dotpath({"a": {"b": 2}}, "a.b") == 2

    def test_missing(self):
        assert _resolve_dotpath({"a": 1}, "b") is _MISSING

    def test_not_dict(self):
        assert _resolve_dotpath({"a": [1, 2]}, "a.b") is _MISSING


class TestMatchContextFilter:
    def test_equal_match(self):
        assert _match_context_filter({"task": {"success": True}}, {"task.success": True})

    def test_equal_no_match(self):
        assert not _match_context_filter({"task": {"success": False}}, {"task.success": True})

    def test_lt(self):
        assert _match_context_filter({"params": {"dist": 0.03}}, {"params.dist": {"$lt": 0.05}})

    def test_lte(self):
        assert _match_context_filter({"x": 5}, {"x": {"$lte": 5}})

    def test_gt(self):
        assert _match_context_filter({"x": 10}, {"x": {"$gt": 5}})

    def test_gte(self):
        assert _match_context_filter({"x": 5}, {"x": {"$gte": 5}})

    def test_ne(self):
        assert _match_context_filter({"x": 5}, {"x": {"$ne": 3}})

    def test_missing_field(self):
        assert not _match_context_filter({}, {"x": 1})

    def test_type_error(self):
        assert not _match_context_filter({"x": "text"}, {"x": {"$lt": 5}})

    def test_multi_conditions(self):
        mem = {"task": {"success": True}, "params": {"dist": 0.03}}
        spec = {"task.success": True, "params.dist": {"$lt": 0.05}}
        assert _match_context_filter(mem, spec)


class TestComputeSpatialDistance:
    def test_exact_match(self):
        assert _compute_spatial_distance({"pos": [1, 2, 3]}, "pos", [1, 2, 3]) == 0.0

    def test_distance(self):
        d = _compute_spatial_distance({"pos": [0, 0]}, "pos", [3, 4])
        assert abs(d - 5.0) < 1e-6

    def test_missing_field(self):
        assert _compute_spatial_distance({}, "pos", [1, 2]) == float("inf")

    def test_dim_mismatch(self):
        assert _compute_spatial_distance({"pos": [1, 2]}, "pos", [1, 2, 3]) == float("inf")

    def test_not_list(self):
        assert _compute_spatial_distance({"pos": "text"}, "pos", [1]) == float("inf")


class TestRecall:
    """recall 主流程测试"""

    def _make_db_mock(self, fts_results=None, vec_results=None):
        db = MagicMock()
        conn = MagicMock()
        db.conn = conn
        db.vec_loaded = bool(vec_results)
        return db

    @pytest.mark.asyncio
    async def test_empty_query(self):
        db = self._make_db_mock()
        result = await recall("", db)
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_whitespace_query(self):
        db = self._make_db_mock()
        result = await recall("   ", db)
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_bm25_only(self):
        db = self._make_db_mock()
        fts_data = [
            {"id": 1, "content": "test", "confidence": 0.9, "context": "{}"},
            {"id": 2, "content": "test2", "confidence": 0.5, "context": "{}"},
        ]
        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall("test", db, collection="test")
                assert result.total == 2
                assert result.mode == "bm25_only"

    @pytest.mark.asyncio
    async def test_no_results(self):
        db = self._make_db_mock()
        with patch("robotmem.search.fts_search_memories", return_value=[]):
            result = await recall("test", db, collection="test")
            assert result.total == 0

    @pytest.mark.asyncio
    async def test_confidence_filter(self):
        db = self._make_db_mock()
        fts_data = [
            {"id": 1, "content": "test", "confidence": 0.1, "context": "{}"},
        ]
        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            result = await recall("test", db, collection="test", min_confidence=0.5)
            assert result.total == 0

    @pytest.mark.asyncio
    async def test_session_filter(self):
        db = self._make_db_mock()
        fts_data = [
            {"id": 1, "content": "test", "confidence": 0.9, "session_id": "s1", "context": "{}"},
            {"id": 2, "content": "test2", "confidence": 0.9, "session_id": "s2", "context": "{}"},
        ]
        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall("test", db, collection="test", session_id="s1")
                assert result.total == 1
                assert result.memories[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_context_filter(self):
        db = self._make_db_mock()
        fts_data = [
            {"id": 1, "content": "test", "confidence": 0.9,
             "context": json.dumps({"task": {"success": True}})},
            {"id": 2, "content": "test2", "confidence": 0.9,
             "context": json.dumps({"task": {"success": False}})},
        ]
        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall(
                    "test", db, collection="test",
                    context_filter={"task.success": True},
                )
                assert result.total == 1
                assert result.memories[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_spatial_sort(self):
        db = self._make_db_mock()
        fts_data = [
            {"id": 1, "content": "test", "confidence": 0.9,
             "context": json.dumps({"spatial": {"pos": [0, 0]}})},
            {"id": 2, "content": "test2", "confidence": 0.9,
             "context": json.dumps({"spatial": {"pos": [10, 10]}})},
        ]
        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall(
                    "test", db, collection="test",
                    spatial_sort={"field": "spatial.pos", "target": [0, 0]},
                )
                assert result.total == 2
                assert result.memories[0]["id"] == 1  # 更近

    @pytest.mark.asyncio
    async def test_spatial_sort_with_max_distance(self):
        db = self._make_db_mock()
        fts_data = [
            {"id": 1, "content": "test", "confidence": 0.9,
             "context": json.dumps({"spatial": {"pos": [0, 0]}})},
            {"id": 2, "content": "test2", "confidence": 0.9,
             "context": json.dumps({"spatial": {"pos": [100, 100]}})},
        ]
        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall(
                    "test", db, collection="test",
                    spatial_sort={"field": "spatial.pos", "target": [0, 0], "max_distance": 1.0},
                )
                assert result.total == 1

    @pytest.mark.asyncio
    async def test_spatial_sort_invalid(self):
        db = self._make_db_mock()
        fts_data = [{"id": 1, "content": "test", "confidence": 0.9, "context": "{}"}]
        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall(
                    "test", db, collection="test",
                    spatial_sort={"bad": "format"},
                )
                assert result.total == 1

    @pytest.mark.asyncio
    async def test_hybrid_mode(self):
        db = self._make_db_mock(vec_results=True)
        fts_data = [{"id": 1, "content": "test", "confidence": 0.9, "context": "{}"}]
        vec_data = [{"id": 2, "content": "test2", "confidence": 0.9, "context": "{}"}]

        embedder = AsyncMock()
        embedder.available = True
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])
        embedder.unavailable_reason = ""

        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.vec_search_memories", return_value=vec_data):
                with patch("robotmem.search.batch_touch_memories"):
                    result = await recall("test", db, embedder=embedder, collection="test")
                    assert result.mode == "hybrid"
                    assert result.total == 2

    @pytest.mark.asyncio
    async def test_vec_only_mode(self):
        db = self._make_db_mock(vec_results=True)
        vec_data = [{"id": 1, "content": "test", "confidence": 0.9, "context": "{}"}]

        embedder = AsyncMock()
        embedder.available = True
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])
        embedder.unavailable_reason = ""

        with patch("robotmem.search.fts_search_memories", return_value=[]):
            with patch("robotmem.search.vec_search_memories", return_value=vec_data):
                with patch("robotmem.search.batch_touch_memories"):
                    result = await recall("test", db, embedder=embedder, collection="test")
                    assert result.mode == "vec_only"

    @pytest.mark.asyncio
    async def test_embedder_not_available_raises(self):
        """embedder 不可用时 raise EmbeddingError（宪法 #4: 坏了就喊）"""
        from robotmem.exceptions import EmbeddingError

        db = self._make_db_mock()
        fts_data = [{"id": 1, "content": "test", "confidence": 0.9, "context": "{}"}]

        embedder = MagicMock()
        embedder.available = False
        embedder.unavailable_reason = "not installed"

        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with pytest.raises(EmbeddingError, match="Embedder 不可用"):
                await recall("test", db, embedder=embedder, collection="test")

    def test_embedder_not_available_raises_sync(self):
        """recall_sync: embedder 不可用时 raise EmbeddingError"""
        from robotmem.exceptions import EmbeddingError

        db = self._make_db_mock()
        fts_data = [{"id": 1, "content": "test", "confidence": 0.9, "context": "{}"}]

        embedder = MagicMock()
        embedder.available = False
        embedder.unavailable_reason = "not installed"

        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with pytest.raises(EmbeddingError, match="Embedder 不可用"):
                recall_sync("test", db, embedder=embedder, collection="test")

    @pytest.mark.asyncio
    async def test_bm25_exception(self):
        db = self._make_db_mock()
        with patch("robotmem.search.fts_search_memories", side_effect=Exception("db error")):
            result = await recall("test", db, collection="test")
            assert result.total == 0

    @pytest.mark.asyncio
    async def test_vec_exception(self):
        db = self._make_db_mock(vec_results=True)
        fts_data = [{"id": 1, "content": "test", "confidence": 0.9, "context": "{}"}]

        embedder = AsyncMock()
        embedder.available = True
        embedder.embed_one = AsyncMock(side_effect=Exception("embed error"))
        embedder.unavailable_reason = ""

        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall("test", db, embedder=embedder, collection="test")
                assert result.mode == "bm25_only"

    @pytest.mark.asyncio
    async def test_touch_exception(self):
        db = self._make_db_mock()
        fts_data = [{"id": 1, "content": "test", "confidence": 0.9, "context": "{}"}]

        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories", side_effect=Exception("touch fail")):
                result = await recall("test", db, collection="test")
                assert result.total == 1  # touch 失败不影响返回

    @pytest.mark.asyncio
    async def test_top_k_clamped(self):
        db = self._make_db_mock()
        fts_data = [{"id": i, "content": f"test{i}", "confidence": 0.9, "context": "{}"} for i in range(20)]

        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall("test", db, collection="test", top_k=3)
                assert result.total == 3

    @pytest.mark.asyncio
    async def test_empty_session_id_normalized(self):
        """空字符串 session_id 统一为 None"""
        db = self._make_db_mock()
        fts_data = [{"id": 1, "content": "test", "confidence": 0.9, "session_id": "s1", "context": "{}"}]

        with patch("robotmem.search.fts_search_memories", return_value=fts_data):
            with patch("robotmem.search.batch_touch_memories"):
                result = await recall("test", db, collection="test", session_id="")
                assert result.total == 1  # 空 session_id 不过滤
