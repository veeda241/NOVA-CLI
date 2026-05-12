from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env", override=True)


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./data/wayne.db")
    if "[YOUR-PASSWORD]" in url or "your_supabase_password" in url:
        raise RuntimeError(
            "DATABASE_URL still contains a Supabase password placeholder. "
            "Set backend/.env DATABASE_URL to your real Supabase pooled connection string."
        )
    if url.startswith("postgres://"):
        url = "postgresql://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://") and "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    if url.startswith("sqlite:///./"):
        db_path = BACKEND_DIR / url.replace("sqlite:///./", "", 1)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"
    if url.startswith("sqlite:///"):
        db_path = Path(url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return url


DATABASE_URL = _database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from models import (  # noqa: F401
        ConnectionLog,
        ConversationHistory,
        DeviceCommandLog,
        DeviceStatus,
        ExperienceBuffer,
        FileAccessLog,
        FileIndex,
        FileOperationLog,
        GoldenResponse,
        Habit,
        ImprovementLog,
        Interaction,
        KnownContact,
        LanguagePattern,
        PCOperationLog,
        ResponseCache,
        SpeedMetric,
        StartupProgram,
        Task,
        TopicTracker,
        UserState,
        UserPreference,
        WakeEvent,
        WatchedFolder,
    )

    Base.metadata.create_all(bind=engine)
    ensure_schema()


def ensure_schema() -> None:
    """Add RL columns to existing SQLite/Postgres databases without dropping data."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    is_postgres = DATABASE_URL.startswith("postgresql")
    json_type = "JSONB" if is_postgres else "JSON"
    timestamp_type = "TIMESTAMPTZ" if is_postgres else "DATETIME"
    now_default = "NOW()" if is_postgres else "CURRENT_TIMESTAMP"

    additions: dict[str, dict[str, str]] = {
        "tasks": {
            "completed_at": timestamp_type,
            "completion_time_hrs": "FLOAT",
        },
        "interactions": {
            "timestamp": f"{timestamp_type} DEFAULT {now_default}",
            "action_type": "TEXT",
            "state_snapshot": json_type,
            "emotion_score": "FLOAT",
            "was_interrupted": "BOOLEAN DEFAULT FALSE",
            "follow_up_count": "INTEGER DEFAULT 0",
        },
        "user_preferences": {
            "reward_sum": "FLOAT DEFAULT 0.0",
        },
        "golden_responses": {
            "state_context": json_type,
        },
        "topic_tracker": {
            "avg_reward": "FLOAT DEFAULT 0.5",
        },
        "improvement_log": {
            "improvement_type": "TEXT",
            "old_behavior": "TEXT",
            "new_behavior": "TEXT",
            "trigger_reason": "TEXT",
            "reward_delta": "FLOAT",
        },
        "wake_events": {
            "event_type": "TEXT DEFAULT 'wake'",
        },
        "file_index": {
            "file_size": "BIGINT DEFAULT 0",
            "tags": f"{json_type} DEFAULT '[]'",
        },
        "pc_operations_log": {
            "bytes_freed": "BIGINT DEFAULT 0",
        },
    }

    with engine.begin() as connection:
        for table, columns in additions.items():
            if table not in tables:
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for column, ddl in columns.items():
                if column not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))

        if is_postgres:
            realtime_tables = ("connection_log",)
            secured_tables = ("connection_log", "response_cache", "speed_metrics")
            for table in secured_tables:
                if table in tables:
                    connection.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
                    connection.execute(
                        text(
                            """
                            DO $$
                            BEGIN
                                IF NOT EXISTS (
                                    SELECT 1 FROM pg_policies
                                    WHERE schemaname = 'public'
                                      AND tablename = :table_name
                                      AND policyname = 'full_access'
                                ) THEN
                                    EXECUTE format('CREATE POLICY full_access ON public.%I FOR ALL USING (true)', :table_name);
                                END IF;
                            END $$;
                            """
                        ),
                        {"table_name": table},
                    )
            for table in realtime_tables:
                if table in tables:
                    connection.execute(
                        text(
                            """
                            DO $$
                            BEGIN
                                IF EXISTS (
                                    SELECT 1 FROM pg_publication
                                    WHERE pubname = 'supabase_realtime'
                                ) AND NOT EXISTS (
                                    SELECT 1 FROM pg_publication_tables
                                    WHERE pubname = 'supabase_realtime'
                                      AND schemaname = 'public'
                                      AND tablename = :table_name
                                ) THEN
                                    EXECUTE format('ALTER PUBLICATION supabase_realtime ADD TABLE public.%I', :table_name);
                                END IF;
                            END $$;
                            """
                        ),
                        {"table_name": table},
                    )


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
