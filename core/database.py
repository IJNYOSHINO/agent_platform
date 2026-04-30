from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import get_settings


class Base(DeclarativeBase):
    pass


def _create_engine():
    cfg = get_settings()
    return create_engine(
        cfg.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


_SCHEMA_MIGRATIONS: list[tuple[str, tuple[str, ...]]] = [
    (
        "2026_04_30_01_task_payload_columns",
        (
            "ALTER TABLE IF EXISTS tasks ADD COLUMN IF NOT EXISTS instruction TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE IF EXISTS tasks ADD COLUMN IF NOT EXISTS media_content JSON NOT NULL DEFAULT '[]'",
        ),
    ),
    (
        "2026_04_30_02_task_owner_index",
        (
            "ALTER TABLE IF EXISTS tasks ADD COLUMN IF NOT EXISTS owner_id INTEGER",
            "CREATE INDEX IF NOT EXISTS ix_tasks_owner_id ON tasks (owner_id)",
        ),
    ),
    (
        "2026_04_30_03_long_term_owner_index",
        (
            "ALTER TABLE IF EXISTS long_term_memory ADD COLUMN IF NOT EXISTS owner_id INTEGER",
            "CREATE INDEX IF NOT EXISTS ix_long_term_memory_owner_id ON long_term_memory (owner_id)",
        ),
    ),
]


def _ensure_migration_table() -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                migration_id TEXT PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _applied_migrations() -> set[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT migration_id FROM schema_migrations")).scalars().all()
    return set(rows)


def _record_migration(migration_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO schema_migrations (migration_id, applied_at) VALUES (:migration_id, :applied_at)"),
            {"migration_id": migration_id, "applied_at": datetime.now(timezone.utc)},
        )


def ensure_db_schema() -> None:
    """Create tables and apply lightweight, ordered schema migrations."""
    import core.models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    _ensure_migration_table()
    applied = _applied_migrations()
    for migration_id, statements in _SCHEMA_MIGRATIONS:
        if migration_id in applied:
            continue
        with engine.begin() as conn:
            for statement in statements:
                conn.exec_driver_sql(statement)
        _record_migration(migration_id)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    ensure_db_schema()
