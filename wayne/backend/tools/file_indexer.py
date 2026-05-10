from __future__ import annotations

import mimetypes
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from database import SessionLocal
from models import FileIndex
from tools.privilege_manager import privilege_manager

if sys.platform == "win32":
    ROOT_DIRS = [drive for drive in ("C:\\", "D:\\", "E:\\") if os.path.exists(drive)]
    SKIP_DIRS = {"$Recycle.Bin", "System Volume Information", "Windows\\WinSxS", "Windows\\Installer", "node_modules", ".git", ".venv"}
elif sys.platform == "darwin":
    ROOT_DIRS = ["/"]
    SKIP_DIRS = {".Spotlight-V100", ".fseventsd", ".Trashes", "private/var/vm", "dev", "proc", "node_modules", ".git"}
else:
    ROOT_DIRS = ["/"]
    SKIP_DIRS = {"proc", "sys", "dev", "run", "snap", ".cache/thumbnails", "node_modules", ".git"}


class FileIndexer:
    def __init__(self) -> None:
        self.indexed_count = 0
        self.is_running = False
        self.last_error: str | None = None

    def start_background_index(self, roots: list[str] | None = None) -> dict:
        if self.is_running:
            return {"status": "already_running", "indexed": self.indexed_count}
        thread = threading.Thread(target=self._index_all, args=(roots or ROOT_DIRS,), daemon=True)
        thread.start()
        return {"status": "indexing_started", "roots": roots or ROOT_DIRS}

    def _index_all(self, roots: list[str]) -> None:
        self.is_running = True
        self.indexed_count = 0
        try:
            for root in roots:
                if os.path.exists(root):
                    self._walk_directory(root)
        finally:
            self.is_running = False

    def _walk_directory(self, root: str) -> None:
        try:
            for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                dirnames[:] = [name for name in dirnames if not self._should_skip(os.path.join(dirpath, name))]
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    try:
                        self.index_file(path)
                    except (PermissionError, OSError):
                        self.index_restricted(path)
                    except SQLAlchemyError as exc:
                        self.last_error = str(exc).splitlines()[0]
                    except Exception as exc:
                        self.last_error = str(exc)
        except (PermissionError, OSError) as exc:
            self.last_error = str(exc)

    def _should_skip(self, path: str) -> bool:
        lowered = path.lower()
        return any(skip.lower() in lowered for skip in SKIP_DIRS)

    def index_file(self, path: str) -> None:
        stat = os.stat(path)
        name = os.path.basename(path)
        ext = Path(path).suffix.lower()
        mime, _ = mimetypes.guess_type(path)
        file_type = self.categorize(ext, mime)
        preview = ""
        if file_type in {"text", "code", "document", "data"} and stat.st_size < 1024 * 1024:
            try:
                with open(path, "r", errors="ignore", encoding="utf-8") as handle:
                    preview = handle.read(500)
            except Exception:
                preview = ""
        preview = preview.replace("\x00", "")
        row_data = {
            "file_path": path,
            "file_name": name,
            "file_ext": ext,
            "file_size": stat.st_size,
            "file_type": file_type,
            "is_restricted": privilege_manager.is_restricted(path),
            "is_system_file": self.is_system(path),
            "parent_dir": os.path.dirname(path),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "indexed_at": datetime.now(),
            "content_preview": preview,
        }
        with SessionLocal() as db:
            try:
                row = db.scalar(select(FileIndex).where(FileIndex.file_path == path))
                if row:
                    for key, value in row_data.items():
                        setattr(row, key, value)
                else:
                    db.add(FileIndex(**row_data))
                db.commit()
                self.indexed_count += 1
            except SQLAlchemyError:
                db.rollback()
                raise

    def index_restricted(self, path: str) -> None:
        name = os.path.basename(path)
        with SessionLocal() as db:
            row = db.scalar(select(FileIndex).where(FileIndex.file_path == path))
            data = {
                "file_path": path,
                "file_name": name,
                "file_ext": Path(path).suffix.lower(),
                "file_size": 0,
                "file_type": "restricted",
                "is_restricted": True,
                "is_system_file": True,
                "parent_dir": os.path.dirname(path),
                "indexed_at": datetime.now(),
            }
            if row:
                for key, value in data.items():
                    setattr(row, key, value)
            else:
                db.add(FileIndex(**data))
            db.commit()

    def categorize(self, ext: str, mime: str | None) -> str:
        categories = {
            "document": {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".md"},
            "spreadsheet": {".xls", ".xlsx", ".csv", ".ods"},
            "presentation": {".ppt", ".pptx", ".odp"},
            "code": {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt"},
            "image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico", ".tiff"},
            "video": {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v"},
            "audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"},
            "archive": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
            "executable": {".exe", ".msi", ".app", ".dmg", ".deb", ".rpm", ".sh", ".bat", ".cmd"},
            "system": {".dll", ".sys", ".so", ".dylib", ".ini", ".reg", ".plist"},
            "data": {".json", ".xml", ".yaml", ".yml", ".toml", ".db", ".sqlite", ".sql"},
        }
        for category, extensions in categories.items():
            if ext in extensions:
                return category
        if mime and mime.startswith("text"):
            return "text"
        return "other"

    def is_system(self, path: str) -> bool:
        indicators = ["System32", "SysWOW64", "/System/", "/usr/lib", "/usr/bin", "/bin", "/sbin", "Windows\\", "Program Files"]
        return any(indicator in path for indicator in indicators)


file_indexer = FileIndexer()
