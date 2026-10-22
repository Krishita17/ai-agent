"""SQLite storage. One file, no ORM, no migrations framework."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT DEFAULT '',
  emoji TEXT DEFAULT '',
  accent TEXT DEFAULT '#F5B93B',
  system_prompt TEXT DEFAULT '',
  provider TEXT DEFAULT 'ollama',
  model TEXT DEFAULT '',
  temperature REAL DEFAULT 0.7,
  max_steps INTEGER DEFAULT 18,
  grants TEXT DEFAULT '{}',
  scopes TEXT DEFAULT '{}',
  status TEXT DEFAULT 'idle',
  archived INTEGER DEFAULT 0,
  created_at REAL,
  updated_at REAL
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  brief TEXT DEFAULT '',
  agent_id TEXT,
  parent_task_id TEXT,
  status TEXT DEFAULT 'queued',
  priority TEXT DEFAULT 'normal',
  result TEXT DEFAULT '',
  error TEXT DEFAULT '',
  created_at REAL,
  started_at REAL,
  finished_at REAL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  task_id TEXT,
  agent_id TEXT,
  status TEXT DEFAULT 'running',
  steps INTEGER DEFAULT 0,
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0,
  cost REAL DEFAULT 0,
  latency_ms INTEGER DEFAULT 0,
  provider TEXT DEFAULT '',
  model TEXT DEFAULT '',
  transcript TEXT DEFAULT '[]',
  started_at REAL,
  finished_at REAL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  task_id TEXT,
  agent_id TEXT,
  kind TEXT,
  payload TEXT,
  ts REAL
);
CREATE TABLE IF NOT EXISTS evals (
  id TEXT PRIMARY KEY,
  run_id TEXT,
  task_id TEXT,
  agent_id TEXT,
  kind TEXT DEFAULT 'auto',
  score REAL,
  rubric TEXT DEFAULT '{}',
  notes TEXT DEFAULT '',
  judge TEXT DEFAULT '',
  created_at REAL
);
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  agent_id TEXT,
  key TEXT,
  value TEXT,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY,
  run_id TEXT,
  agent_id TEXT,
  tool TEXT,
  args TEXT,
  state TEXT DEFAULT 'pending',
  created_at REAL,
  decided_at REAL
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE TABLE IF NOT EXISTS duties (
  id TEXT PRIMARY KEY,
  agent_id TEXT,
  title TEXT NOT NULL,
  brief TEXT DEFAULT '',
  cadence_minutes INTEGER DEFAULT 1440,
  next_run_at REAL,
  active INTEGER DEFAULT 1,
  last_task_id TEXT,
  runs INTEGER DEFAULT 0,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS escalations (
  id TEXT PRIMARY KEY,
  task_id TEXT,
  run_id TEXT,
  agent_id TEXT,
  reason TEXT,
  question TEXT,
  answer TEXT DEFAULT '',
  state TEXT DEFAULT 'open',
  created_at REAL,
  resolved_at REAL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id);
"""


def conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        config.ensure_dirs()
        c = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        # Under WAL, NORMAL is the recommended durability level: it fsyncs at
        # checkpoints rather than on every commit, which is markedly faster for
        # the many small writes an event-heavy run produces. The only thing at
        # risk is the last transaction on an OS crash or power loss — never
        # corruption — which is an acceptable trade for a local console.
        c.execute("PRAGMA synchronous=NORMAL")
        # Spend the busy window retrying inside SQLite instead of surfacing
        # "database is locked" the moment two threads commit at once.
        c.execute("PRAGMA busy_timeout=30000")
        _local.conn = c
    return _local.conn


MIGRATIONS = [
    ("agents", "autonomy", "TEXT DEFAULT 'supervised'"),
    ("agents", "shift", "TEXT DEFAULT 'always'"),
    ("tasks", "attempts", "INTEGER DEFAULT 0"),
    ("tasks", "max_attempts", "INTEGER DEFAULT 2"),
    ("tasks", "duty_id", "TEXT"),
    ("tasks", "source", "TEXT DEFAULT 'operator'"),
    # Which process owns a run, so an interrupted one can be told from a live one.
    ("runs", "pid", "INTEGER"),
    # A one-off scheduled task — a reminder — fires once and then retires itself
    # rather than recurring on its cadence like a standing duty.
    ("duties", "once", "INTEGER DEFAULT 0"),
]


def _migrate(c: sqlite3.Connection) -> None:
    """Additive column adds. SQLite has no IF NOT EXISTS for columns."""
    for table, column, decl in MIGRATIONS:
        cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    c.commit()


def init() -> None:
    c = conn()
    c.executescript(SCHEMA)
    c.commit()
    _migrate(c)
    for k, v in config.DEFAULTS.items():
        c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))
    c.commit()
    # The audit chain's schema lives in security.py (which imports this module,
    # hence the late import). It is created here so that every entry point gets
    # a complete database — `hermes run` on a fresh install used to reach its
    # first audited action and fail on a missing table.
    from . import security
    security.init_audit()
    # Tools shipped after an agent was created would otherwise be invisible to
    # it forever, since a grant dict is a closed list.
    from .runtime import tools
    tools.backfill_grants()
    prune_events()


# The live feed is fed from the bus in memory; the events table behind it is
# written on every emit and read by nothing. At roughly 15 KB per run that is
# ~127 MB a year from a single hourly standing duty, growing forever. Keep a
# window of it for forensics — the tamper-evident record is the audit chain,
# which is never pruned — and let the rest go.
EVENT_ROW_CEILING = 50_000


def prune_events() -> int:
    """Drop events past the retention window. Returns how many went."""
    c = conn()
    try:
        days = float(setting("retention.event_days", "14") or 14)
    except ValueError:
        days = 14
    removed = 0
    if days > 0:
        cur = c.execute("DELETE FROM events WHERE ts < ?", (now() - days * 86400,))
        removed += cur.rowcount or 0
    # A burst inside the window can still run away, so cap the row count too.
    cur = c.execute(
        "DELETE FROM events WHERE rowid NOT IN "
        "(SELECT rowid FROM events ORDER BY ts DESC LIMIT ?)", (EVENT_ROW_CEILING,))
    removed += cur.rowcount or 0
    if removed:
        c.commit()
    return removed


def nid() -> str:
    return uuid.uuid4().hex[:12]


def now() -> float:
    return time.time()


def q(sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn().execute(sql, args).fetchall()]


def q1(sql: str, args: tuple = ()) -> dict | None:
    # fetchone stops at the first matching row rather than materialising the
    # whole result set as q() would — this is the hot path behind setting(),
    # get_key() and every single-row lookup.
    row = conn().execute(sql, args).fetchone()
    return dict(row) if row else None


def ex(sql: str, args: tuple = ()) -> None:
    c = conn()
    c.execute(sql, args)
    c.commit()


def setting(key: str, default: str = "") -> str:
    row = q1("SELECT value FROM settings WHERE key=?", (key,))
    return row["value"] if row else config.DEFAULTS.get(key, default)


def settings_all() -> dict[str, str]:
    """Every known setting in one query, defaults filling the gaps.

    The bootstrap payload and the settings route both need the whole set; doing
    it per key is one round-trip per default, on every console load.
    """
    stored = {r["key"]: r["value"] for r in q("SELECT key,value FROM settings")}
    return {k: stored.get(k, default) for k, default in config.DEFAULTS.items()}


def set_setting(key: str, value: str) -> None:
    ex("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
       (key, value))


def get_key(provider: str) -> str:
    """Vault key, falling back to the environment."""
    stored = setting(f"key.{provider}", "")
    if stored:
        return config.decrypt(stored)
    return config.bootstrap_env().get(provider, "")


def set_key(provider: str, raw: str) -> None:
    set_setting(f"key.{provider}", config.encrypt(raw) if raw else "")


def jload(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(value) if value else fallback
    except (TypeError, ValueError):
        return fallback
