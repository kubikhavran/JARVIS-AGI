from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

class UserFacts:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def upsert(self, key: str, value: str, confidence: float = 1.0) -> None:
        existing = self._conn.execute(
            "SELECT id FROM user_facts WHERE key=?", (key,)
        ).fetchone()
        now = datetime.now(timezone.utc).isoformat()
        if existing:
            self._conn.execute(
                "UPDATE user_facts SET value=?, confidence=?, updated_at=? WHERE key=?",
                (value, confidence, now, key),
            )
        else:
            self._conn.execute(
                "INSERT INTO user_facts (key, value, confidence, updated_at) VALUES (?,?,?,?)",
                (key, value, confidence, now),
            )
        self._conn.commit()

    def get_all(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM user_facts").fetchall()
        return {k: v for k, v in rows}
