from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    priority: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completion_time_hrs: Mapped[float | None] = mapped_column(Float, nullable=True)


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class DeviceStatus(Base):
    __tablename__ = "device_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    device_name: Mapped[str] = mapped_column(String(255), nullable=False)
    device_type: Mapped[str] = mapped_column(String(40), nullable=False)
    battery_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cpu_percent: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    ram_percent: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    disk_percent: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    is_online: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(80), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    push_token: Mapped[str | None] = mapped_column(Text, nullable=True)


class DeviceCommandLog(Base):
    __tablename__ = "device_command_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    command: Mapped[str] = mapped_column(String(80), nullable=False)
    issued_by: Mapped[str] = mapped_column(String(120), default="wayne", nullable=False)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    status: Mapped[str] = mapped_column(String(80), default="queued", nullable=False)


class WakeEvent(Base):
    __tablename__ = "wake_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(80), default="wake_word", nullable=False)
    session_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), default="wake", nullable=False)
    fired_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    wayne_response: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    tool_used: Mapped[str | None] = mapped_column(String(120), nullable=True)
    action_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    state_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    response_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    explicit_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    implicit_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    emotion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_reward: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    was_interrupted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    follow_up_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    preference_key: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    preference_value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reward_sum: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class GoldenResponse(Base):
    __tablename__ = "golden_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    wayne_response: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    state_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reward_score: Mapped[float] = mapped_column(Float, nullable=False)
    use_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class TopicTracker(Base):
    __tablename__ = "topic_tracker"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_asked: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    avg_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    avg_reward: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)


class ImprovementLog(Base):
    __tablename__ = "improvement_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    improvement: Mapped[str | None] = mapped_column(Text, nullable=True)
    improvement_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    old_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    trigger_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    triggered_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reward_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class UserState(Base):
    __tablename__ = "user_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    active_hour: Mapped[int | None] = mapped_column(Integer, nullable=True)
    day_of_week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    session_length_mins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dominant_intent: Mapped[str | None] = mapped_column(String(80), nullable=True)
    emotion: Mapped[str | None] = mapped_column(String(80), nullable=True)
    location_context: Mapped[str | None] = mapped_column(String(120), nullable=True)
    device_active: Mapped[str | None] = mapped_column(String(120), nullable=True)
    recent_topics: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    stress_level: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    focus_level: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    productivity_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)


class ExperienceBuffer(Base):
    __tablename__ = "experience_buffer"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    state: Mapped[dict] = mapped_column(JSON, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    reward: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    next_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    done: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class Habit(Base):
    __tablename__ = "habits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    habit_type: Mapped[str] = mapped_column(String(120), nullable=False)
    pattern: Mapped[str] = mapped_column(Text, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    last_observed: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)


class LanguagePattern(Base):
    __tablename__ = "language_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_type: Mapped[str] = mapped_column(String(120), nullable=False)
    pattern_value: Mapped[str] = mapped_column(Text, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class KnownContact(Base):
    __tablename__ = "known_contacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    relationship: Mapped[str | None] = mapped_column(String(120), nullable=True)
    last_mentioned: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class FileAccessLog(Base):
    __tablename__ = "file_access_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    access_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class FileIndex(Base):
    __tablename__ = "file_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(Text, unique=True, index=True, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    file_ext: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    file_size: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    file_type: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    is_restricted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_system_file: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    parent_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    tags: Mapped[Any | None] = mapped_column(JSON, default=list, nullable=True)
    content_preview: Mapped[str | None] = mapped_column(Text, nullable=True)


class FileOperationLog(Base):
    __tablename__ = "file_operations_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation: Mapped[str] = mapped_column(String(120), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    performed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class PCOperationLog(Base):
    __tablename__ = "pc_operations_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation: Mapped[str] = mapped_column(String(120), nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    bytes_freed: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    performed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class WatchedFolder(Base):
    __tablename__ = "watched_folders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    folder_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    watch_type: Mapped[str] = mapped_column(String(80), default="all", nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    event_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class StartupProgram(Base):
    __tablename__ = "startup_programs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    program_name: Mapped[str] = mapped_column(String(255), nullable=False)
    program_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    category: Mapped[str] = mapped_column(String(80), default="unknown", nullable=False)
    impact: Mapped[str] = mapped_column(String(40), default="medium", nullable=False)
    last_modified: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
