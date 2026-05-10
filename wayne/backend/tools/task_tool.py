from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from database import SessionLocal
from models import Task


VALID_PRIORITIES = {"low", "medium", "high"}


def serialize_task(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "priority": task.priority,
        "completed": task.completed,
        "created_at": task.created_at.isoformat(timespec="seconds"),
    }


def list_tasks(db: Session) -> list[dict[str, Any]]:
    stmt = select(Task).order_by(Task.completed.asc(), Task.created_at.desc())
    return [serialize_task(task) for task in db.scalars(stmt).all()]


def create_task(db: Session, title: str, priority: str = "medium") -> dict[str, Any]:
    priority = priority if priority in VALID_PRIORITIES else "medium"
    task = Task(title=title.strip(), priority=priority)
    db.add(task)
    db.commit()
    db.refresh(task)
    return serialize_task(task)


def toggle_task(db: Session, task_id: int) -> dict[str, Any]:
    task = db.get(Task, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")
    task.completed = not task.completed
    db.commit()
    db.refresh(task)
    return serialize_task(task)


def delete_task(db: Session, task_id: int) -> dict[str, Any]:
    task = db.get(Task, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")
    deleted = serialize_task(task)
    db.delete(task)
    db.commit()
    return deleted


def manage_tasks(
    action: str,
    title: str | None = None,
    task_id: int | None = None,
    priority: str = "medium",
    db: Session | None = None,
) -> dict[str, Any]:
    owns_session = db is None
    session = db or SessionLocal()
    try:
        if action == "create":
            if not title:
                raise ValueError("A title is required to create a task.")
            create_task(session, title, priority)
        elif action == "complete":
            if task_id is None:
                raise ValueError("task_id is required to complete a task.")
            toggle_task(session, task_id)
        elif action == "delete":
            if task_id is None:
                raise ValueError("task_id is required to delete a task.")
            delete_task(session, task_id)
        elif action != "list":
            raise ValueError(f"Unsupported task action: {action}")
        return {"status": "ok", "tasks": list_tasks(session)}
    finally:
        if owns_session:
            session.close()
