from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional until dependencies are installed
    pdfplumber = None


BASE_FILE_DIRECTORY = Path(os.getenv("BASE_FILE_DIRECTORY", str(Path.home()))).expanduser().resolve()


def _safe_path(path: str | None = None) -> Path:
    candidate = BASE_FILE_DIRECTORY if not path else Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = BASE_FILE_DIRECTORY / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(BASE_FILE_DIRECTORY)
    except ValueError as exc:
        raise ValueError(f"Path is outside BASE_FILE_DIRECTORY: {BASE_FILE_DIRECTORY}") from exc
    return resolved


def list_files(directory: str | None = None, extension_filter: str | None = None) -> dict[str, Any]:
    path = _safe_path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries: list[dict[str, Any]] = []
    for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if extension_filter and item.is_file() and item.suffix.lower() != extension_filter.lower():
            continue
        stat = item.stat()
        entries.append(
            {
                "name": item.name,
                "path": str(item),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return {"base_directory": str(BASE_FILE_DIRECTORY), "directory": str(path), "files": entries, "count": len(entries)}


def open_file(path: str, summarize: bool = False) -> dict[str, Any]:
    file_path = _safe_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is a directory: {file_path}")

    stat = file_path.stat()
    text = ""
    if file_path.suffix.lower() == ".pdf":
        if pdfplumber is None:
            text = "pdfplumber is not installed. Install backend requirements to read PDFs."
        else:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    return {
        "path": str(file_path),
        "name": file_path.name,
        "extension": file_path.suffix,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "summarize": summarize,
        "content": text[:3000],
        "truncated": len(text) > 3000,
    }


def search_files(directory: str | None = None, query: str = "", extension_filter: str | None = None) -> dict[str, Any]:
    path = _safe_path(directory)
    matches: list[dict[str, Any]] = []
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            if extension_filter and file_path.suffix.lower() != extension_filter.lower():
                continue
            if query.lower() in filename.lower():
                stat = file_path.stat()
                matches.append({"name": filename, "path": str(file_path), "size": stat.st_size})
    return {"directory": str(path), "query": query, "matches": matches[:100]}
