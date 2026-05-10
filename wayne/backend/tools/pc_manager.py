from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import psutil

from database import SessionLocal
from models import PCOperationLog, StartupProgram


class PCManager:
    async def clear_all_cache(self, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": "Confirm clear all cache? This may close cached app data."}
        results = {
            "temp": await self.clear_temp_files(True),
            "browser": await self.clear_browser_cache(True),
            "dns": await self.flush_dns(),
            "thumbnail": await self.clear_thumbnail_cache(True),
        }
        total = sum(item.get("bytes_freed", 0) for item in results.values() if isinstance(item, dict))
        self._log("clear_all_cache", "cache", results, total, True)
        return {"success": True, "total_freed": total, "total_freed_human": self._human_size(total), "details": results}

    async def clear_temp_files(self, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": "Confirm clear temporary files?"}
        paths = [tempfile.gettempdir(), os.environ.get("TEMP", ""), os.environ.get("TMP", "")]
        if sys.platform == "win32":
            paths.append(os.path.expanduser("~\\AppData\\Local\\Temp"))
        else:
            paths.append(os.path.expanduser("~/.cache"))
        freed = sum(self._clear_directory(path) for path in set(paths) if path and os.path.exists(path))
        self._log("clear_temp_files", "cache", {"paths": paths}, freed, True)
        return {"success": True, "bytes_freed": freed, "freed_human": self._human_size(freed)}

    async def clear_browser_cache(self, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": "Confirm clear browser cache?"}
        paths = self._browser_cache_paths()
        cleared: dict[str, str] = {}
        freed = 0
        for browser, path in paths.items():
            if os.path.exists(path):
                size = self._get_dir_size(path)
                self._clear_directory(path)
                freed += size
                cleared[browser] = self._human_size(size)
        self._log("clear_browser_cache", "cache", cleared, freed, True)
        return {"success": True, "bytes_freed": freed, "freed_human": self._human_size(freed), "browsers_cleared": cleared}

    async def clear_thumbnail_cache(self, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": "Confirm clear thumbnail cache?"}
        if sys.platform == "win32":
            path = os.path.expanduser("~\\AppData\\Local\\Microsoft\\Windows\\Explorer")
            freed = self._clear_by_pattern(path, "thumbcache_*.db")
        else:
            path = os.path.expanduser("~/Library/Caches/com.apple.QuickLook.thumbnailcache" if sys.platform == "darwin" else "~/.cache/thumbnails")
            freed = self._get_dir_size(path) if os.path.exists(path) else 0
            shutil.rmtree(path, ignore_errors=True)
        return {"success": True, "bytes_freed": freed, "freed_human": self._human_size(freed)}

    async def flush_dns(self) -> dict:
        try:
            command = ["ipconfig", "/flushdns"] if sys.platform == "win32" else ["dscacheutil", "-flushcache"] if sys.platform == "darwin" else ["systemd-resolve", "--flush-caches"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            return {"success": result.returncode == 0, "output": result.stdout or result.stderr}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def get_memory_info(self) -> dict:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {"total": mem.total, "available": mem.available, "used": mem.used, "percent": mem.percent, "total_human": self._human_size(mem.total), "available_human": self._human_size(mem.available), "used_human": self._human_size(mem.used), "swap_total": self._human_size(swap.total), "swap_used": self._human_size(swap.used), "swap_percent": swap.percent}

    async def optimize_memory(self, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": "Confirm memory optimization?"}
        before = psutil.virtual_memory().percent
        try:
            if sys.platform == "win32":
                subprocess.run(["powershell", "-NoProfile", "-Command", "[GC]::Collect()"], capture_output=True, timeout=10)
            elif sys.platform == "darwin":
                subprocess.run(["purge"], capture_output=True, timeout=15)
            else:
                subprocess.run(["sync"], capture_output=True, timeout=10)
        except Exception:
            pass
        after = psutil.virtual_memory().percent
        return {"success": True, "before_percent": before, "after_percent": after}

    async def list_processes(self, sort_by: str = "cpu") -> dict:
        psutil.cpu_percent(interval=0.1)
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status", "username"]):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        key = "memory_percent" if sort_by == "memory" else "name" if sort_by == "name" else "cpu_percent"
        processes.sort(key=lambda item: item.get(key) or 0, reverse=key != "name")
        return {"processes": processes[:100], "total": len(processes)}

    async def kill_process(self, name: str | None = None, pid: int | None = None, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": f"Confirm kill process {pid or name}?"}
        killed = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                matches = (pid and proc.info["pid"] == pid) or (name and name.lower() in (proc.info["name"] or "").lower())
                if matches:
                    proc.terminate()
                    killed.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {"success": True, "killed": killed, "count": len(killed)}

    async def get_startup_programs(self) -> dict:
        programs = []
        if sys.platform == "win32":
            programs.extend(self._windows_startup_programs())
        else:
            for folder in [os.path.expanduser("~/.config/autostart"), os.path.expanduser("~/Library/LaunchAgents")]:
                if os.path.exists(folder):
                    programs.extend({"name": item, "path": os.path.join(folder, item), "location": folder, "is_enabled": not item.endswith(".disabled")} for item in os.listdir(folder))
        with SessionLocal() as db:
            for program in programs:
                db.add(StartupProgram(program_name=program["name"], program_path=program.get("path"), is_enabled=program.get("is_enabled", True), category=program.get("location", "startup")))
            db.commit()
        return {"programs": programs, "count": len(programs)}

    async def disable_startup_program(self, name: str, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": f"Confirm disable startup program {name}?"}
        return {"success": False, "error": "Startup disable is listed but not modified automatically. Use Task Manager Startup tab for final control.", "name": name}

    async def get_disk_info(self) -> dict:
        partitions = []
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                partitions.append({"device": part.device, "mountpoint": part.mountpoint, "fstype": part.fstype, "total": self._human_size(usage.total), "used": self._human_size(usage.used), "free": self._human_size(usage.free), "percent": usage.percent})
            except Exception:
                pass
        return {"partitions": partitions}

    async def disk_cleanup(self, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": "Confirm disk cleanup?"}
        temp = await self.clear_temp_files(True)
        trash = await self.empty_trash(True)
        freed = temp.get("bytes_freed", 0) + trash.get("bytes_freed", 0)
        self._log("disk_cleanup", "disk", {"temp": temp, "trash": trash}, freed, True)
        return {"success": True, "total_freed": freed, "total_freed_human": self._human_size(freed)}

    async def empty_trash(self, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": "Confirm empty trash/recycle bin?"}
        freed = 0
        try:
            if sys.platform == "win32":
                subprocess.run(["powershell", "-NoProfile", "-Command", "Clear-RecycleBin -Force -ErrorAction SilentlyContinue"], capture_output=True, timeout=30)
            else:
                trash = os.path.expanduser("~/.Trash" if sys.platform == "darwin" else "~/.local/share/Trash")
                freed = self._get_dir_size(trash)
                shutil.rmtree(trash, ignore_errors=True)
            return {"success": True, "bytes_freed": freed, "freed_human": self._human_size(freed)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def get_network_info(self) -> dict:
        return {"interfaces": {name: {"is_up": stat.isup, "speed": stat.speed} for name, stat in psutil.net_if_stats().items()}, "io": psutil.net_io_counters()._asdict()}

    async def network_speed_test(self) -> dict:
        try:
            import speedtest

            test = speedtest.Speedtest()
            test.get_best_server()
            return {"success": True, "download_mbps": round(test.download() / 1_000_000, 2), "upload_mbps": round(test.upload() / 1_000_000, 2), "ping_ms": round(test.results.ping, 1)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def set_dns(self, primary: str, secondary: str = "8.8.4.4", confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": f"Confirm DNS change to {primary}, {secondary}?"}
        return {"success": False, "error": "DNS changes require adapter selection and admin approval. Use Windows Network Settings for final apply.", "primary": primary, "secondary": secondary}

    async def get_system_status(self) -> dict:
        return {"cpu_percent": psutil.cpu_percent(interval=1), "cpu_count": psutil.cpu_count(), "memory": await self.get_memory_info(), "disk": await self.get_disk_info(), "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()}

    async def set_performance_mode(self, mode: str, confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": f"Confirm switch to {mode} performance mode?"}
        try:
            if sys.platform == "win32":
                modes = {"performance": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c", "balanced": "381b4222-f694-41f0-9685-ff5bb260df2e", "powersaver": "a1841308-3541-4fab-bc81-f71556f20b4a"}
                subprocess.run(["powercfg", "/setactive", modes.get(mode, modes["balanced"])], capture_output=True, timeout=10)
            return {"success": True, "mode": mode}
        except Exception as exc:
            return {"success": False, "error": str(exc), "mode": mode}

    async def read_registry(self, key_path: str, value_name: str | None = None) -> dict:
        if sys.platform != "win32":
            return {"success": False, "error": "Registry is Windows only."}
        try:
            import winreg

            hive_name, subkey = key_path.split("\\", 1)
            hive = winreg.HKEY_CURRENT_USER if hive_name.upper() in {"HKCU", "HKEY_CURRENT_USER"} else winreg.HKEY_LOCAL_MACHINE
            key = winreg.OpenKey(hive, subkey)
            if value_name:
                value, _ = winreg.QueryValueEx(key, value_name)
                return {"success": True, "value": value}
            values = {}
            index = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, index)
                    values[name] = str(value)
                    index += 1
                except OSError:
                    break
            return {"success": True, "values": values}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def write_registry(self, key_path: str, value_name: str, value_data: str, value_type: str = "REG_SZ", confirmed: bool = False) -> dict:
        if not confirmed:
            return {"success": False, "needs_confirmation": True, "message": f"Confirm registry write {key_path}\\{value_name}?"}
        return {"success": False, "error": "Registry writes are intentionally blocked unless implemented with a signed admin helper.", "key_path": key_path}

    def _windows_startup_programs(self) -> list[dict]:
        programs = []
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run")
            index = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, index)
                    programs.append({"name": name, "path": value, "location": "registry", "is_enabled": True})
                    index += 1
                except OSError:
                    break
        except Exception:
            pass
        folder = os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup")
        if os.path.exists(folder):
            programs.extend({"name": item, "path": os.path.join(folder, item), "location": "startup_folder", "is_enabled": True} for item in os.listdir(folder))
        return programs

    def _browser_cache_paths(self) -> dict[str, str]:
        if sys.platform == "win32":
            return {"chrome": os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Cache"), "edge": os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Cache"), "firefox": os.path.expanduser("~\\AppData\\Local\\Mozilla\\Firefox\\Profiles")}
        if sys.platform == "darwin":
            return {"chrome": os.path.expanduser("~/Library/Caches/Google/Chrome"), "safari": os.path.expanduser("~/Library/Caches/com.apple.Safari"), "firefox": os.path.expanduser("~/Library/Caches/Firefox")}
        return {"chrome": os.path.expanduser("~/.cache/google-chrome"), "firefox": os.path.expanduser("~/.cache/mozilla/firefox")}

    def _clear_directory(self, path: str) -> int:
        freed = 0
        try:
            for item in os.scandir(path):
                try:
                    size = self._get_item_size(item.path)
                    shutil.rmtree(item.path, ignore_errors=True) if item.is_dir() else os.remove(item.path)
                    freed += size
                except Exception:
                    pass
        except Exception:
            pass
        return freed

    def _clear_by_pattern(self, directory: str, pattern: str) -> int:
        freed = 0
        for path in glob.glob(os.path.join(directory, pattern)):
            try:
                size = os.path.getsize(path)
                os.remove(path)
                freed += size
            except Exception:
                pass
        return freed

    def _get_dir_size(self, path: str) -> int:
        total = 0
        try:
            for root, _, files in os.walk(path):
                for file_name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, file_name))
                    except Exception:
                        pass
        except Exception:
            pass
        return total

    def _get_item_size(self, path: str) -> int:
        return os.path.getsize(path) if os.path.isfile(path) else self._get_dir_size(path)

    def _human_size(self, size: int | float) -> str:
        value = float(size)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if value < 1024:
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} PB"

    def _log(self, operation: str, category: str, details: dict, bytes_freed: int = 0, success: bool = True) -> None:
        with SessionLocal() as db:
            db.add(PCOperationLog(operation=operation, category=category, details=details, bytes_freed=bytes_freed, success=success, result="ok" if success else "failed"))
            db.commit()


pc_manager = PCManager()
