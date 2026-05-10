from __future__ import annotations

import os

from tools.privilege_manager import privilege_manager


class FileWriter:
    async def write(self, path: str, content: str, mode: str = "overwrite") -> dict:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            file_mode = "a" if mode == "append" else "w"
            with open(path, file_mode, encoding="utf-8") as handle:
                handle.write(content)
            return {"success": True, "path": path, "bytes_written": len(content.encode("utf-8")), "mode": mode}
        except PermissionError:
            return privilege_manager.write_restricted_file(path, content)
        except Exception as exc:
            return {"success": False, "path": path, "error": str(exc)}
