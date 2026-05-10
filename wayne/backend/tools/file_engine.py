from __future__ import annotations

import mimetypes
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

from sqlalchemy import or_, select

from database import SessionLocal
from models import FileIndex, FileOperationLog
from tools.file_indexer import file_indexer
from tools.file_opener import FileOpener
from tools.file_reader import FileReader
from tools.file_writer import FileWriter
from tools.privilege_manager import privilege_manager

reader = FileReader()
writer = FileWriter()
opener = FileOpener()


class FileEngine:
    async def search(self, query: str, file_type: str | None = None, directory: str | None = None, max_results: int = 20, include_restricted: bool = False) -> dict:
        query = (query or "").strip()
        with SessionLocal() as db:
            stmt = select(FileIndex).where(
                or_(FileIndex.file_name.ilike(f"%{query}%"), FileIndex.content_preview.ilike(f"%{query}%"))
            )
            if file_type:
                stmt = stmt.where(FileIndex.file_type == file_type)
            if directory:
                stmt = stmt.where(FileIndex.file_path.ilike(f"{os.path.expanduser(directory)}%"))
            if not include_restricted:
                stmt = stmt.where(FileIndex.is_restricted.is_(False))
            rows = db.scalars(stmt.order_by(FileIndex.access_count.desc(), FileIndex.modified_at.desc()).limit(max_results)).all()
            results = [self._serialize_index(row) for row in rows]
        if not results:
            deep = await self.deep_search(query, directory)
            results = deep.get("results", [])[:max_results]
        return {"results": results, "count": len(results), "query": query}

    async def deep_search(self, query: str, directory: str | None = None) -> dict:
        root = os.path.expanduser(directory or ("C:\\" if sys.platform == "win32" else "/"))
        results: list[dict] = []
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [item for item in dirnames if item not in {"node_modules", ".git", ".venv", "$Recycle.Bin"}]
                for name in filenames:
                    if query.lower() in name.lower():
                        path = os.path.join(dirpath, name)
                        try:
                            stat = os.stat(path)
                            results.append({"file_path": path, "file_name": name, "file_size": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()})
                        except OSError:
                            pass
                        if len(results) >= 50:
                            return {"results": results, "count": len(results)}
        except Exception as exc:
            return {"results": results, "count": len(results), "error": str(exc)}
        return {"results": results, "count": len(results)}

    async def read(self, path: str, summarize: bool = False, force_restricted: bool = False) -> dict:
        path = await self._resolve_path(path)
        if not path:
            return {"success": False, "error": "File not found"}
        if privilege_manager.is_restricted(path) and not force_restricted:
            result = privilege_manager.read_restricted_file(path)
        else:
            result = await reader.read(path)
        self._touch_index(path)
        self._log("read", path, result, result.get("success", True))
        return result

    async def write(self, path: str, content: str, mode: str = "overwrite") -> dict:
        path = os.path.expanduser(path)
        result = privilege_manager.write_restricted_file(path, content) if privilege_manager.is_restricted(path) else await writer.write(path, content, mode)
        if result.get("success") and os.path.exists(path):
            try:
                file_indexer.index_file(path)
            except Exception:
                pass
        self._log("write", path, {"mode": mode, **result}, result.get("success", False), result.get("error"))
        return result

    async def open(self, path: str) -> dict:
        resolved = await self._resolve_path(path)
        if not resolved:
            return {"success": False, "error": "File not found"}
        result = opener.open(resolved)
        self._touch_index(resolved)
        self._log("open", resolved, result, result.get("success", False), result.get("error"))
        return result

    async def create(self, path: str, content: str = "", is_directory: bool = False) -> dict:
        try:
            path = os.path.expanduser(path)
            if is_directory:
                os.makedirs(path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                file_indexer.index_file(path)
            result = {"success": True, "created": path, "type": "directory" if is_directory else "file"}
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        self._log("create", path, result, result.get("success", False), result.get("error"))
        return result

    async def delete(self, path: str, permanent: bool = False, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": f"Confirm delete {path}? Set confirmed=true to proceed."}
        path = os.path.expanduser(path)
        try:
            if permanent:
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            elif sys.platform == "win32":
                subprocess.run(["powershell", "-NoProfile", "-Command", f"Remove-Item -LiteralPath '{path}' -Confirm:$false"], timeout=20)
            elif sys.platform == "darwin":
                subprocess.run(["osascript", "-e", f'tell app "Finder" to delete POSIX file "{path}"'], timeout=20)
            else:
                subprocess.run(["gio", "trash", path], timeout=20)
            result = {"success": True, "deleted": path, "permanent": permanent}
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        self._log("delete", path, result, result.get("success", False), result.get("error"))
        return result

    async def copy(self, source: str, destination: str) -> dict:
        try:
            source = os.path.expanduser(source)
            destination = os.path.expanduser(destination)
            shutil.copytree(source, destination) if os.path.isdir(source) else shutil.copy2(source, destination)
            result = {"success": True, "copied_to": destination}
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        self._log("copy", source, {"destination": destination, **result}, result.get("success", False), result.get("error"))
        return result

    async def move(self, source: str, destination: str) -> dict:
        try:
            source = os.path.expanduser(source)
            destination = os.path.expanduser(destination)
            shutil.move(source, destination)
            result = {"success": True, "moved_to": destination}
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        self._log("move", source, {"destination": destination, **result}, result.get("success", False), result.get("error"))
        return result

    async def rename(self, path: str, new_name: str) -> dict:
        return await self.move(path, os.path.join(os.path.dirname(os.path.expanduser(path)), new_name))

    async def list_directory(self, path: str | None = None, show_hidden: bool = False) -> dict:
        path = os.path.expanduser(path or os.getcwd())
        try:
            entries = []
            for item in os.scandir(path):
                if not show_hidden and item.name.startswith("."):
                    continue
                stat = item.stat()
                entries.append({"name": item.name, "path": item.path, "is_dir": item.is_dir(), "size": stat.st_size, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(), "type": self._type_for(item.name)})
            entries.sort(key=lambda row: (not row["is_dir"], row["name"].lower()))
            return {"success": True, "entries": entries, "count": len(entries), "path": path}
        except Exception as exc:
            return {"success": False, "error": str(exc), "path": path}

    async def extract_archive(self, path: str, destination: str | None = None) -> dict:
        path = os.path.expanduser(path)
        destination = os.path.expanduser(destination or os.path.dirname(path))
        try:
            ext = Path(path).suffix.lower()
            if ext == ".zip":
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(destination)
            elif ext in {".tar", ".gz", ".bz2", ".xz"}:
                with tarfile.open(path) as archive:
                    archive.extractall(destination)
            elif ext in {".rar", ".7z"}:
                subprocess.run(["7z", "x", path, f"-o{destination}"], timeout=120)
            else:
                return {"success": False, "error": f"Unsupported archive type: {ext}"}
            return {"success": True, "extracted_to": destination}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def get_info(self, path: str) -> dict:
        path = os.path.expanduser(path)
        try:
            stat = os.stat(path)
            mime, _ = mimetypes.guess_type(path)
            return {"success": True, "path": path, "name": os.path.basename(path), "size": stat.st_size, "size_human": self._human_size(stat.st_size), "created": datetime.fromtimestamp(stat.st_ctime).isoformat(), "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(), "is_directory": os.path.isdir(path), "mime_type": mime, "permissions": oct(stat.st_mode)[-3:], "is_restricted": privilege_manager.is_restricted(path)}
        except Exception as exc:
            return {"success": False, "error": str(exc), "path": path}

    async def _resolve_path(self, path: str) -> str | None:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded
        search = await self.search(os.path.basename(path), max_results=1, include_restricted=True)
        return search["results"][0]["file_path"] if search["results"] else None

    def _serialize_index(self, row: FileIndex) -> dict:
        return {"file_path": row.file_path, "file_name": row.file_name, "file_ext": row.file_ext, "file_size": row.file_size, "file_type": row.file_type, "is_restricted": row.is_restricted, "is_system_file": row.is_system_file, "parent_dir": row.parent_dir, "modified_at": row.modified_at.isoformat() if row.modified_at else None, "access_count": row.access_count, "content_preview": row.content_preview}

    def _touch_index(self, path: str) -> None:
        with SessionLocal() as db:
            row = db.scalar(select(FileIndex).where(FileIndex.file_path == path))
            if row:
                row.access_count += 1
                row.last_accessed = datetime.now()
                db.commit()

    def _log(self, operation: str, path: str, details: dict, success: bool = True, error: str | None = None) -> None:
        with SessionLocal() as db:
            db.add(FileOperationLog(operation=operation, file_path=path, details=details, success=success, error_message=error))
            db.commit()

    def _type_for(self, name: str) -> str:
        return file_indexer.categorize(Path(name).suffix.lower(), mimetypes.guess_type(name)[0])

    def _human_size(self, size: int) -> str:
        value = float(size)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if value < 1024:
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} PB"


file_engine = FileEngine()
