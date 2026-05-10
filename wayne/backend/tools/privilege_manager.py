from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


class PrivilegeManager:
    RESTRICTED_HINTS = {
        "win32": ["C:\\Windows\\System32", "C:\\Windows\\SysWOW64", "C:\\ProgramData\\Microsoft", "C:\\System Volume Information"],
        "darwin": ["/System", "/private/etc", "/usr/standalone"],
        "linux": ["/etc/shadow", "/etc/sudoers", "/root", "/boot", "/sys", "/proc"],
    }

    def is_admin(self) -> bool:
        try:
            if sys.platform == "win32":
                return bool(ctypes.windll.shell32.IsUserAnAdmin())
            return os.geteuid() == 0
        except Exception:
            return False

    def is_restricted(self, path: str) -> bool:
        normalized = str(Path(path).expanduser())
        return any(normalized.lower().startswith(item.lower()) for item in self.RESTRICTED_HINTS.get(sys.platform, []))

    def read_restricted_file(self, path: str) -> dict:
        if not self.is_admin():
            return {
                "success": False,
                "requires_elevation": True,
                "error": "Restricted file access requires running W.A.Y.N.E as administrator.",
                "path": path,
            }
        try:
            with open(path, "r", errors="ignore", encoding="utf-8") as handle:
                return {"success": True, "content": handle.read(50000), "path": path}
        except Exception as exc:
            return {"success": False, "error": str(exc), "path": path}

    def write_restricted_file(self, path: str, content: str) -> dict:
        if not self.is_admin():
            return {
                "success": False,
                "requires_elevation": True,
                "error": "Restricted write requires running W.A.Y.N.E as administrator.",
                "path": path,
            }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            return {"success": True, "path": path, "bytes_written": len(content.encode("utf-8"))}
        except Exception as exc:
            return {"success": False, "error": str(exc), "path": path}


privilege_manager = PrivilegeManager()
