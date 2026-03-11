"""Web API 测试 — Flask test client"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from robotmem.config import Config
from robotmem.ops.memories import insert_memory
from robotmem.schema import initialize_schema
from robotmem.web import create_app


@pytest.fixture
def app(tmp_path):
    """创建测试用 Flask app + 临时数据库"""
    db_path = tmp_path / "test.db"
    config = Config(db_path=str(db_path))
    app = create_app(config)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def seeded_client(app):
    """带初始数据的 test client — 走 insert_memory 完整链路（FTS5 同步）"""
    db = app.config["ROBOTMEM_DB"]
    conn = db.conn

    for i in range(5):
        insert_memory(conn, {
            "collection": "test-coll",
            "type": "fact" if i < 3 else "perception",
            "content": f"Test memory content {i}",
            "human_summary": f"Test summary {i}",
            "category": "observation",
            "confidence": 0.9 - i * 0.1,
        })

    return app.test_client()


class TestIndex:
    """首页"""

    def test_index_returns_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"robotmem" in res.data


class TestStats:
    """GET /api/stats"""

    def test_empty_db(self, client):
        res = client.get("/api/stats")
        assert res.status_code == 200
        data = res.get_json()
        assert data["total"] == 0

    def test_with_data(self, seeded_client):
        res = seeded_client.get("/api/stats")
        data = res.get_json()
        assert data["total"] == 5
        assert data["by_type"]["fact"] == 3
        assert data["by_type"]["perception"] == 2
        assert "test-coll" in data["collections"]


class TestListMemories:
    """GET /api/memories"""

    def test_empty(self, client):
        res = client.get("/api/memories")
        data = res.get_json()
        assert data["total"] == 0
        assert data["memories"] == []

    def test_paginated(self, seeded_client):
        res = seeded_client.get("/api/memories?page=0&limit=2")
        data = res.get_json()
        assert data["total"] == 5
        assert len(data["memories"]) == 2
        assert data["pages"] == 3

    def test_filter_type(self, seeded_client):
        res = seeded_client.get("/api/memories?type=perception")
        data = res.get_json()
        assert data["total"] == 2

    def test_filter_collection(self, seeded_client):
        res = seeded_client.get("/api/memories?collection=test-coll")
        data = res.get_json()
        assert data["total"] == 5

    def test_filter_nonexistent_collection(self, seeded_client):
        res = seeded_client.get("/api/memories?collection=nope")
        data = res.get_json()
        assert data["total"] == 0


class TestGetMemory:
    """GET /api/memory/<id>"""

    def test_exists(self, seeded_client):
        res = seeded_client.get("/api/memory/1")
        assert res.status_code == 200
        data = res.get_json()
        assert data["id"] == 1
        assert "content" in data

    def test_not_found(self, client):
        res = client.get("/api/memory/9999")
        assert res.status_code == 404

    def test_no_embedding_in_response(self, seeded_client):
        res = seeded_client.get("/api/memory/1")
        data = res.get_json()
        assert "embedding" not in data


class TestDeleteMemory:
    """DELETE /api/memory/<id>"""

    def test_delete(self, seeded_client):
        res = seeded_client.delete(
            "/api/memory/1",
            data=json.dumps({"reason": "test"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "deleted"

        # 确认已标记为 invalidated
        res2 = seeded_client.get("/api/memory/1")
        assert res2.get_json()["status"] == "invalidated"

    def test_delete_not_found(self, client):
        res = client.delete(
            "/api/memory/9999",
            data=json.dumps({"reason": "test"}),
            content_type="application/json",
        )
        assert res.status_code == 404

    def test_delete_already_invalidated(self, seeded_client):
        # 先删除
        seeded_client.delete(
            "/api/memory/1",
            data=json.dumps({"reason": "test"}),
            content_type="application/json",
        )
        # 再次删除
        res = seeded_client.delete(
            "/api/memory/1",
            data=json.dumps({"reason": "test again"}),
            content_type="application/json",
        )
        assert res.status_code == 400


class TestUpdateMemory:
    """PUT /api/memory/<id>"""

    def test_update_content(self, seeded_client):
        res = seeded_client.put(
            "/api/memory/1",
            data=json.dumps({"content": "Updated content"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "updated"
        assert "content" in data["fields"]

    def test_update_not_found(self, client):
        res = client.put(
            "/api/memory/9999",
            data=json.dumps({"content": "X"}),
            content_type="application/json",
        )
        assert res.status_code == 404

    def test_update_no_body(self, seeded_client):
        res = seeded_client.put(
            "/api/memory/1",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert res.status_code == 400

    def test_update_disallowed_field(self, seeded_client):
        res = seeded_client.put(
            "/api/memory/1",
            data=json.dumps({"id": 999}),
            content_type="application/json",
        )
        assert res.status_code == 400


class TestSearch:
    """GET /api/search"""

    def test_empty_query(self, client):
        res = client.get("/api/search")
        assert res.status_code == 400

    def test_search_with_results(self, seeded_client):
        res = seeded_client.get("/api/search?q=memory content")
        data = res.get_json()
        assert data["total"] > 0

    def test_search_no_match(self, seeded_client):
        res = seeded_client.get("/api/search?q=xyznonexistent")
        data = res.get_json()
        assert data["total"] == 0

    def test_search_extracts_context_params(self, app):
        """搜索返回含 context JSON 的记忆时，自动提取 params 字段"""
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        ctx = json.dumps({"params": {"grip_force": {"value": 12.5, "unit": "N"}}})
        insert_memory(conn, {
            "collection": "test-coll",
            "type": "fact",
            "content": "grip force calibration result",
            "category": "observation",
            "confidence": 0.8,
            "context": ctx,
        })
        client = app.test_client()
        res = client.get("/api/search?q=grip force calibration")
        data = res.get_json()
        assert data["total"] > 0
        hit = data["results"][0]
        assert hit["params"] == {"grip_force": {"value": 12.5, "unit": "N"}}

    def test_search_empty_context_no_extra_fields(self, seeded_client):
        """搜索返回 context 为空的记忆时，不添加额外字段"""
        res = seeded_client.get("/api/search?q=memory content")
        data = res.get_json()
        if data["total"] > 0:
            hit = data["results"][0]
            assert "params" not in hit
            assert "spatial" not in hit
            assert "robot" not in hit


class TestRecentFailures:
    """GET /api/recent-failures"""

    def test_empty(self, client):
        res = client.get("/api/recent-failures")
        data = res.get_json()
        assert data["total"] == 0
        assert data["failures"] == []

    def test_with_failures(self, app):
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        # postmortem 记忆
        insert_memory(conn, {
            "collection": "test-coll",
            "type": "fact",
            "content": "Gripper failed on smooth surface",
            "human_summary": "Gripper failure on smooth surface",
            "category": "postmortem",
            "confidence": 0.8,
        })
        # gotcha 记忆
        insert_memory(conn, {
            "collection": "test-coll",
            "type": "fact",
            "content": "Force sensor saturated at 20N",
            "category": "gotcha",
            "confidence": 0.7,
        })
        # 普通记忆（不应出现）
        insert_memory(conn, {
            "collection": "test-coll",
            "type": "fact",
            "content": "Normal observation",
            "category": "observation",
            "confidence": 0.9,
        })

        client = app.test_client()
        res = client.get("/api/recent-failures")
        data = res.get_json()
        assert data["total"] == 2
        categories = {f["category"] for f in data["failures"]}
        assert categories == {"postmortem", "gotcha"}

    def test_limit(self, app):
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        for i in range(5):
            insert_memory(conn, {
                "collection": "test-coll",
                "type": "fact",
                "content": f"Failure {i}",
                "category": "postmortem",
                "confidence": 0.5,
            })
        client = app.test_client()
        res = client.get("/api/recent-failures?limit=3")
        data = res.get_json()
        assert len(data["failures"]) == 3


class TestSessions:
    """GET /api/sessions"""

    def test_empty(self, client):
        res = client.get("/api/sessions")
        data = res.get_json()
        assert data["total"] == 0

    def test_with_data(self, app, seeded_client):
        # 插入 session
        db = app.config["ROBOTMEM_DB"]
        db.conn.execute(
            "INSERT INTO sessions (external_id, collection, status) VALUES (?, ?, 'active')",
            ["test-session-1", "test-coll"],
        )
        db.conn.commit()

        res = seeded_client.get("/api/sessions")
        data = res.get_json()
        assert data["total"] == 1
        assert data["sessions"][0]["external_id"] == "test-session-1"


class TestCollections:
    """GET /api/collections"""

    def test_empty(self, client):
        res = client.get("/api/collections")
        data = res.get_json()
        assert data["collections"] == []

    def test_with_data(self, seeded_client):
        res = seeded_client.get("/api/collections")
        data = res.get_json()
        assert len(data["collections"]) == 1
        assert data["collections"][0]["name"] == "test-coll"
        assert data["collections"][0]["count"] == 5


class TestCategories:
    """GET /api/categories"""

    def test_empty(self, client):
        res = client.get("/api/categories")
        data = res.get_json()
        assert data["categories"] == []

    def test_with_data(self, seeded_client):
        res = seeded_client.get("/api/categories")
        data = res.get_json()
        assert len(data["categories"]) > 0
        names = [c["name"] for c in data["categories"]]
        assert "observation" in names
        for c in data["categories"]:
            assert c["count"] > 0


class TestDoctor:
    """GET /api/doctor"""

    def test_empty_db(self, client):
        res = client.get("/api/doctor")
        assert res.status_code == 200
        data = res.get_json()
        assert data["memories"]["total"] == 0
        assert data["sessions"]["total"] == 0
        assert data["zero_hit"]["rate"] == 0.0
        assert data["db_size_bytes"] >= 0

    def test_with_data(self, seeded_client):
        res = seeded_client.get("/api/doctor")
        data = res.get_json()
        assert data["memories"]["total"] == 5
        assert data["memories"]["by_type"]["fact"] == 3
        assert data["memories"]["by_type"]["perception"] == 2

    def test_fts5_sync(self, seeded_client):
        """insert_memory 走完整链路，FTS5 应同步"""
        res = seeded_client.get("/api/doctor")
        data = res.get_json()
        assert data["fts5"]["ok"] is True
        assert data["fts5"]["indexed"] == data["fts5"]["expected"]

    def test_zero_hit_rate(self, seeded_client):
        """新插入记忆未被 recall，零命中率 100%"""
        res = seeded_client.get("/api/doctor")
        data = res.get_json()
        assert data["zero_hit"]["count"] == 5
        assert data["zero_hit"]["rate"] == 100.0

    def test_zero_hit_after_access(self, app):
        """有 access_count > 0 的记忆，零命中率降低"""
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        for i in range(4):
            insert_memory(conn, {
                "collection": "test-coll",
                "type": "fact",
                "content": f"Doctor test {i}",
                "category": "observation",
                "confidence": 0.8,
            })
        # 模拟 2 条被召回（取前 2 条 ID，避免硬编码）
        conn.execute(
            "UPDATE memories SET access_count = 3 "
            "WHERE id IN (SELECT id FROM memories ORDER BY id LIMIT 2)"
        )
        conn.commit()

        client = app.test_client()
        res = client.get("/api/doctor")
        data = res.get_json()
        assert data["zero_hit"]["count"] == 2
        assert data["zero_hit"]["rate"] == 50.0

    def test_session_counts(self, app):
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        conn.execute(
            "INSERT INTO sessions (external_id, collection, status) VALUES (?, ?, 'active')",
            ["doc-sess-1", "test"],
        )
        conn.execute(
            "INSERT INTO sessions (external_id, collection, status) VALUES (?, ?, 'ended')",
            ["doc-sess-2", "test"],
        )
        conn.commit()

        client = app.test_client()
        res = client.get("/api/doctor")
        data = res.get_json()
        assert data["sessions"]["total"] == 2
        assert data["sessions"]["by_status"]["active"] == 1
        assert data["sessions"]["by_status"]["ended"] == 1

    def test_vec0_ok_when_partial_embeddings(self, app):
        """vec_count < total 时 vec0.ok 仍为 True（部分记忆无 embedding 是正常的）"""
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        # 插入记忆但不插入 vec0 条目（模拟 embed_backend='none'）
        conn.execute("""
            INSERT INTO memories (id, collection, type, content, human_summary,
                                 category, confidence, decay_rate, source, scope,
                                 status, scope_files, scope_entities)
            VALUES (100, 'test', 'fact', 'no embedding memory', 'no emb',
                    'observation', 0.9, 0.01, 'sdk', 'project', 'active', '[]', '[]')
        """)
        conn.commit()

        client = app.test_client()
        res = client.get("/api/doctor")
        data = res.get_json()
        # vec_count=0 < total=1 → vec_ok 应为 True（不是误报）
        assert data["vec0"]["ok"] is True
        assert data["vec0"]["total_memories"] >= 1


class TestMemoriesExtendedFilters:
    """GET /api/memories 扩展参数"""

    def test_filter_category(self, seeded_client):
        res = seeded_client.get("/api/memories?category=observation")
        data = res.get_json()
        assert data["total"] == 5

    def test_filter_category_multi(self, seeded_client):
        res = seeded_client.get("/api/memories?category=observation,postmortem")
        data = res.get_json()
        assert data["total"] == 5  # seeded_client 只有 observation

    def test_filter_confidence_min(self, seeded_client):
        # confidence: 0.9, 0.8, 0.7, 0.6, 0.5
        res = seeded_client.get("/api/memories?confidence_min=0.7")
        data = res.get_json()
        assert data["total"] == 3

    def test_filter_confidence_max(self, seeded_client):
        res = seeded_client.get("/api/memories?confidence_max=0.6")
        data = res.get_json()
        assert data["total"] == 2

    def test_filter_days(self, seeded_client):
        # 所有数据刚插入，都在 1 天内
        res = seeded_client.get("/api/memories?days=1")
        data = res.get_json()
        assert data["total"] == 5

    def test_filter_perception_type(self, seeded_client):
        res = seeded_client.get("/api/memories?perception_type=visual")
        data = res.get_json()
        assert data["total"] == 0


class TestSessionMemories:
    """GET /api/sessions/<external_id>/memories"""

    def test_no_session(self, client):
        res = client.get("/api/sessions/nonexistent/memories")
        data = res.get_json()
        assert data["total"] == 0
        assert data["memories"] == []

    def test_with_data(self, app):
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        conn.execute(
            "INSERT INTO sessions (external_id, collection, status) VALUES (?, ?, 'active')",
            ["test-timeline-sess", "test-coll"],
        )
        conn.commit()
        for i in range(3):
            insert_memory(conn, {
                "collection": "test-coll",
                "type": "fact" if i < 2 else "perception",
                "content": f"Timeline memory {i}",
                "category": "observation",
                "confidence": 0.8,
                "session_id": "test-timeline-sess",
            })

        client = app.test_client()
        res = client.get("/api/sessions/test-timeline-sess/memories")
        data = res.get_json()
        assert data["total"] == 3
        assert len(data["memories"]) == 3
        # ASC 排序验证
        times = [m["created_at"] for m in data["memories"]]
        assert times == sorted(times)

    def test_limit(self, app):
        db = app.config["ROBOTMEM_DB"]
        conn = db.conn
        conn.execute(
            "INSERT INTO sessions (external_id, collection, status) VALUES (?, ?, 'active')",
            ["test-limit-sess", "test-coll"],
        )
        conn.commit()
        for i in range(5):
            insert_memory(conn, {
                "collection": "test-coll",
                "type": "fact",
                "content": f"Limit test {i}",
                "category": "observation",
                "confidence": 0.5,
                "session_id": "test-limit-sess",
            })
        client = app.test_client()
        res = client.get("/api/sessions/test-limit-sess/memories?limit=2")
        data = res.get_json()
        assert len(data["memories"]) == 2
        assert data["total"] == 5
