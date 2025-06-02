import pytest
from src.agents.note_taker import NoteTaker
import mongomock
from unittest.mock import patch
import datetime
import os

@pytest.fixture
def note_taker():
    with patch('src.agents.note_taker.MongoClient', new=mongomock.MongoClient):
        return NoteTaker(mongo_uri="mongodb://localhost:27017", db_name="test_logs", collection_name="logs")

def test_initialization(note_taker):
    assert note_taker.db.name == "test_logs"
    assert note_taker.collection.name == "logs"
    # Indexes should exist
    index_names = note_taker.collection.index_information().keys()
    assert any("type" in str(name) for name in index_names)
    assert any("user" in str(name) for name in index_names)
    assert any("session_id" in str(name) for name in index_names)
    assert any("timestamp" in str(name) for name in index_names)

def test_log_session_start_end(note_taker):
    note_taker.log_session_start(user="alice", session_id="sess1", agent="agentA")
    note_taker.log_session_end(user="alice", session_id="sess1", agent="agentA")
    starts = note_taker.get_logs(log_type="session_start", user="alice", session_id="sess1")
    ends = note_taker.get_logs(log_type="session_end", user="alice", session_id="sess1")
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0]["agent"] == "agentA"
    assert ends[0]["agent"] == "agentA"

def test_get_sessions(note_taker):
    note_taker.log_session_start(user="alice", session_id="sess1")
    note_taker.log_session_start(user="alice", session_id="sess2")
    sessions = note_taker.get_sessions(user="alice")
    assert set(sessions) == {"sess1", "sess2"}

def test_log_retention_and_archiving(tmp_path, note_taker):
    # Insert logs with old timestamps
    old_time = datetime.datetime.utcnow() - datetime.timedelta(days=40)
    note_taker.collection.insert_one({"type": "query", "content": {"query": "old"}, "timestamp": old_time})
    note_taker.collection.insert_one({"type": "query", "content": {"query": "recent"}, "timestamp": datetime.datetime.utcnow()})
    # Delete old logs
    deleted = note_taker.delete_old_logs(days=30)
    assert deleted == 1
    # Archive old logs
    note_taker.collection.insert_one({"type": "query", "content": {"query": "old2"}, "timestamp": old_time})
    archive_path = os.path.join(tmp_path, "archive.json")
    archived = note_taker.archive_old_logs(days=30, file_path=archive_path)
    assert archived == 1
    with open(archive_path, "r", encoding="utf-8") as f:
        data = f.read()
        assert "old2" in data

def test_audit_trail_edit_and_delete(note_taker):
    # Insert a log
    note_taker.log_query("to edit", user="alice", agent="agentA")
    log = note_taker.collection.find_one({"content.query": "to edit"})
    log_id = log["_id"]
    # Edit the log
    note_taker.edit_log(log_id, {"query": "edited"}, user="bob", agent="agentB")
    edited = note_taker.collection.find_one({"_id": log_id})
    assert edited["content"]["query"] == "edited"
    assert "provenance" in edited
    assert edited["provenance"][-1]["edited_by"] == "bob"
    # Delete the log
    note_taker.delete_log(log_id, user="carol", agent="agentC")
    deleted = note_taker.collection.find_one({"_id": log_id})
    # Should be None (deleted)
    assert deleted is None

def test_log_query_with_agent_and_provenance(note_taker):
    note_taker.log_query("query with agent", user="alice", session_id="sess1", agent="agentA")
    logs = note_taker.get_logs(log_type="query", user="alice", session_id="sess1")
    assert logs[0]["agent"] == "agentA"

def test_log_query(note_taker):
    note_taker.log_query("What is a transformer?")
    logs = note_taker.get_logs(log_type="query")
    assert len(logs) == 1
    assert logs[0]["type"] == "query"
    assert logs[0]["content"]["query"] == "What is a transformer?"

def test_log_selected_papers(note_taker):
    note_taker.log_selected_papers(["123", "456"], user="alice")
    logs = note_taker.get_logs(log_type="selected_papers", user="alice")
    assert len(logs) == 1
    assert logs[0]["content"]["paper_ids"] == ["123", "456"]
    assert logs[0]["user"] == "alice"

def test_log_hypothesis(note_taker):
    note_taker.log_hypothesis("Initial hypothesis", refined=False, session_id="sess1")
    note_taker.log_hypothesis("Refined hypothesis", refined=True, session_id="sess1")
    logs = note_taker.get_logs(log_type="hypothesis", session_id="sess1")
    assert len(logs) == 2
    assert logs[0]["content"]["refined"] is False
    assert logs[1]["content"]["refined"] is True

def test_log_feedback(note_taker):
    note_taker.log_feedback("Good result", reason="Relevant papers", user="bob")
    logs = note_taker.get_logs(log_type="feedback", user="bob")
    assert len(logs) == 1
    assert logs[0]["content"]["feedback"] == "Good result"
    assert logs[0]["content"]["reason"] == "Relevant papers"

def test_log_insight(note_taker):
    note_taker.log_insight("Transformer usage increased in 2020.")
    logs = note_taker.get_logs(log_type="insight")
    assert len(logs) == 1
    assert "Transformer usage" in logs[0]["content"]["insight"]

def test_log_visualization(note_taker):
    note_taker.log_visualization("line_plot", {"color": "blue"}, session_id="sess2")
    logs = note_taker.get_logs(log_type="visualization", session_id="sess2")
    assert len(logs) == 1
    assert logs[0]["content"]["viz_type"] == "line_plot"
    assert logs[0]["content"]["customization"]["color"] == "blue"

def test_log_regeneration(note_taker):
    note_taker.log_regeneration("hypothesis", "User requested new hypothesis.")
    logs = note_taker.get_logs(log_type="regeneration")
    assert len(logs) == 1
    assert logs[0]["content"]["stage"] == "hypothesis"
    assert "User requested" in logs[0]["content"]["reason"]

def test_get_logs_filtering(note_taker):
    note_taker.log_query("Q1", user="alice", session_id="s1")
    note_taker.log_query("Q2", user="bob", session_id="s2")
    logs_alice = note_taker.get_logs(user="alice")
    logs_s2 = note_taker.get_logs(session_id="s2")
    assert all(l["user"] == "alice" for l in logs_alice)
    assert all(l["session_id"] == "s2" for l in logs_s2)

def test_empty_logs(note_taker):
    logs = note_taker.get_logs(log_type="nonexistent")
    assert logs == []

def test_log_special_characters(note_taker):
    special = "特殊字符!@#"
    note_taker.log_query(special)
    logs = note_taker.get_logs(log_type="query")
    assert special in logs[0]["content"]["query"]

def test_log_large_content(note_taker):
    large_text = "A" * 10000
    note_taker.log_feedback(large_text)
    logs = note_taker.get_logs(log_type="feedback")
    assert logs[0]["content"]["feedback"] == large_text 