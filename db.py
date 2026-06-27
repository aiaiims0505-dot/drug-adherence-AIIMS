import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "data.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('patient','doctor')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            name TEXT NOT NULL,
            drug_class TEXT DEFAULT 'None',
            dose_mg REAL,
            times_per_day INTEGER NOT NULL DEFAULT 1,
            reminder_times TEXT,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS dose_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            medication_id INTEGER NOT NULL REFERENCES medications(id),
            scheduled_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('taken','missed','pending')),
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(medication_id, scheduled_date)
        );
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            p_nonadherent REAL NOT NULL,
            is_nonadherent INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS lifestyle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            date TEXT NOT NULL,
            on_diet INTEGER DEFAULT 0,
            exercised INTEGER DEFAULT 0,
            slept_adequately INTEGER DEFAULT 0,
            glucose_reading REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date)
        );
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            endpoint TEXT NOT NULL,
            p256dh_key TEXT NOT NULL,
            auth_key TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sms_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            phone TEXT NOT NULL,
            message TEXT NOT NULL,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)


# ── Query helpers ──

def get_user(user_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def get_active_medications(user_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM medications WHERE user_id=? AND active=1 ORDER BY created_at",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_today_logs(user_id: int, today: str) -> dict:
    """Returns {medication_id: status} for today."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT medication_id, status FROM dose_logs WHERE user_id=? AND scheduled_date=?",
            (user_id, today)
        ).fetchall()
    return {r["medication_id"]: r["status"] for r in rows}


def get_weekly_stats(user_id: int, week_start: str) -> tuple[int, int]:
    """Returns (taken, total_expected) for the week."""
    with get_db() as conn:
        taken = conn.execute(
            "SELECT COUNT(*) FROM dose_logs WHERE user_id=? AND scheduled_date>=? AND status='taken'",
            (user_id, week_start)
        ).fetchone()[0]
        meds = conn.execute(
            "SELECT times_per_day FROM medications WHERE user_id=? AND active=1",
            (user_id,)
        ).fetchall()
    total = sum(m["times_per_day"] for m in meds) * 7
    return taken, max(total, 1)


def get_latest_assessment(user_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_assessments(user_id: int, limit: int = 5) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM assessments WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def get_lifestyle_log(user_id: int, date: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM lifestyle_logs WHERE user_id=? AND date=?",
            (user_id, date)
        ).fetchone()
    return dict(row) if row else None


def get_lifestyle_history(user_id: int, days: int = 7) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM lifestyle_logs WHERE user_id=? ORDER BY date DESC LIMIT ?",
            (user_id, days)
        ).fetchall()
    return [dict(r) for r in rows]


def get_sms_log(user_id: int, limit: int = 20) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sms_log WHERE user_id=? ORDER BY sent_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_patients() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE role='patient' ORDER BY created_at",
        ).fetchall()
    return [dict(r) for r in rows]


def get_dose_grid(user_id: int, start_date: str, end_date: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM dose_logs WHERE user_id=? AND scheduled_date>=? AND scheduled_date<=? ORDER BY scheduled_date",
            (user_id, start_date, end_date)
        ).fetchall()
    return [dict(r) for r in rows]


def get_lifestyle_range(user_id: int, start_date: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM lifestyle_logs WHERE user_id=? AND date>=? ORDER BY date DESC",
            (user_id, start_date)
        ).fetchall()
    return [dict(r) for r in rows]
