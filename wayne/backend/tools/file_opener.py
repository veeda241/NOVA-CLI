from __future__ import annotations

import os
import subprocess
import sys


class FileOpener:
    def open(self, path: str) -> dict:
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            return {"success": True, "opened": path}
        except Exception as exc:
            return {"success": False, "error": str(exc), "path": path}

    def open_with(self, path: str, app: str) -> dict:
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-a", app, path])
            else:
                subprocess.Popen([app, path])
            return {"success": True, "opened": path, "app": app}
        except Exception as exc:
            return {"success": False, "error": str(exc), "path": path, "app": app}

    def open_in_editor(self, path: str) -> dict:
        editors = ["code", "notepad"] if sys.platform == "win32" else ["code", "nano", "vim"]
        for editor in editors:
            try:
                subprocess.Popen([editor, path])
                return {"success": True, "editor": editor, "path": path}
            except FileNotFoundError:
                continue
            except Exception as exc:
                return {"success": False, "error": str(exc), "path": path}
        return {"success": False, "error": "No editor found", "path": path}
