import pytest
import tempfile
from pathlib import Path
from memory.chat_history import ChatHistory

@pytest.fixture
def db(tmp_path):
    return ChatHistory(db_path=tmp_path / "test.db")

def test_log_and_retrieve(db):
    db.log("user", "Hello Jarvis")
    db.log("assistant", "Hello! How can I help?")
    history = db.recent(count=5)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

def test_recent_respects_count(db):
    for i in range(10):
        db.log("user", f"message {i}")
    history = db.recent(count=3)
    assert len(history) == 3

def test_session_isolation(db):
    db.log("user", "session A msg", session_id="a")
    db.log("user", "session B msg", session_id="b")
    history_a = db.recent(count=5, session_id="a")
    assert len(history_a) == 1
    assert history_a[0]["content"] == "session A msg"
