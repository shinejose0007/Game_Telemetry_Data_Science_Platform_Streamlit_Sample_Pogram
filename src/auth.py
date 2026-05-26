import hashlib
import sqlite3
from datetime import datetime
from typing import Optional, Dict, List

DB_PATH = "data/game_telemetry.db"


def _hash_password(password: str) -> str:
    """Simple SHA-256 password hashing for a local demo app."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_auth_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'analyst',
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]
    if count == 0:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            ("admin", _hash_password("admin123"), "admin", datetime.utcnow().isoformat()),
        )
    conn.commit()
    conn.close()


def register_user(username: str, password: str, role: str = "analyst", db_path: str = DB_PATH) -> bool:
    if not username or not password:
        return False
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username.strip(), _hash_password(password), role, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate(username: str, password: str, db_path: str = DB_PATH) -> Optional[Dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?",
        (username.strip(), _hash_password(password)),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "role": row[2]}
    return None


def list_users(db_path: str = DB_PATH) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "username": r[1], "role": r[2], "created_at": r[3]}
        for r in rows
    ]
