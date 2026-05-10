from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from database import SessionLocal
from models import FileOperationLog, WatchedFolder

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover
    FileSystemEventHandler = object  # type: ignore
    Observer = None  # type: ignore


class WAYNEFileHandler(FileSystemEventHandler):
    def on_created(self, event):  # type: ignore[no-untyped-def]
        self._log("created", event.src_path)

    def on_modified(self, event):  # type: ignore[no-untyped-def]
        self._log("modified", event.src_path)

    def on_deleted(self, event):  # type: ignore[no-untyped-def]
        self._log("deleted", event.src_path)

    def on_moved(self, event):  # type: ignore[no-untyped-def]
        self._log("moved", event.src_path, {"destination": event.dest_path})

    def _log(self, event_type: str, path: str, details: dict | None = None) -> None:
        with SessionLocal() as db:
            db.add(FileOperationLog(operation=f"watched_{event_type}", file_path=path, details=details or {}, success=True))
            folder = db.scalar(select(WatchedFolder).where(WatchedFolder.folder_path == path))
            if folder:
                folder.event_count += 1
            db.commit()


class FileWatcher:
    def __init__(self) -> None:
        self.observer = Observer() if Observer else None
        self.watched_paths: list[str] = []

    def watch(self, path: str) -> dict:
        if not self.observer:
            return {"success": False, "error": "watchdog is not installed"}
        handler = WAYNEFileHandler()
        self.observer.schedule(handler, path, recursive=True)
        if not self.observer.is_alive():
            self.observer.start()
        if path not in self.watched_paths:
            self.watched_paths.append(path)
        with SessionLocal() as db:
            row = db.scalar(select(WatchedFolder).where(WatchedFolder.folder_path == path))
            if row:
                row.watch_type = "all"
                row.added_at = datetime.now()
            else:
                db.add(WatchedFolder(folder_path=path, watch_type="all"))
            db.commit()
        return {"success": True, "watching": path}

    def unwatch(self, path: str) -> dict:
        with SessionLocal() as db:
            row = db.scalar(select(WatchedFolder).where(WatchedFolder.folder_path == path))
            if row:
                db.delete(row)
                db.commit()
        self.watched_paths = [item for item in self.watched_paths if item != path]
        return {"success": True, "unwatched": path}

    def get_watched(self) -> list[dict]:
        with SessionLocal() as db:
            rows = db.scalars(select(WatchedFolder).order_by(WatchedFolder.added_at.desc())).all()
            return [
                {"id": row.id, "folder_path": row.folder_path, "watch_type": row.watch_type, "event_count": row.event_count, "added_at": row.added_at.isoformat()}
                for row in rows
            ]


file_watcher = FileWatcher()
