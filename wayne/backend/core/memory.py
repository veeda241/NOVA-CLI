from __future__ import annotations

from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import ConversationHistory


def add_message(db: Session, session_id: str, role: str, content: str) -> ConversationHistory:
    message = ConversationHistory(session_id=session_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_history(db: Session, session_id: str, limit: int = 20) -> list[dict[str, str]]:
    stmt = (
        select(ConversationHistory)
        .where(ConversationHistory.session_id == session_id)
        .order_by(ConversationHistory.timestamp.desc())
        .limit(limit)
    )
    rows: Iterable[ConversationHistory] = db.scalars(stmt).all()
    return [{"role": row.role, "content": row.content} for row in reversed(list(rows))]
