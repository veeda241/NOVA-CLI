"""
NOVA Configuration Loader
===========================

Unified configuration system inspired by:
  - Aider: AI pair programming (model settings, code edit modes)
  - Crush: Terminal AI agent (JSON config with providers, MCP, LSP, permissions)
  - OpenJarvis: Modular AI assistant (TOML config with 5 pillars)

Load priority (highest wins):
  1. Environment variables (NOVA_* prefix)
  2. Project-local config: ./nova_config.toml
  3. User-level config:    ~/.nova/config.toml
  4. Built-in defaults

Usage:
    from nova.config import nova_config
    
    # Access any config value
    provider = nova_config.get("intelligence.default_provider")
    models = nova_config.get("intelligence.models")
    threshold = nova_config.get("agent.system.confidence_threshold")
    
    # Get full section
    ui_cfg = nova_config.section("ui")
    
    # Resolve env var references ($VAR_NAME -> actual value)
    api_key = nova_config.resolve("intelligence.providers.huggingface.api_key")
"""

import os
import sys
import json
import copy
from typing import Any, Dict, List, Optional
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# TOML Parser — Uses tomllib (3.11+) or fallback
# ═══════════════════════════════════════════════════════════════

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli for <3.11
    except ImportError:
        tomllib = None


def _parse_toml(filepath: str) -> Dict:
    """Parse a TOML file and return as a dict."""
    if tomllib is None:
        # Minimal fallback parser for basic TOML
        return _fallback_toml_parse(filepath)
    
    with open(filepath, "rb") as f:
        return tomllib.load(f)


def _fallback_toml_parse(filepath: str) -> Dict:
    """
    Minimal TOML parser fallback for Python <3.11 without tomli.
    Handles basic key=value, [sections], [[arrays]], strings, ints, floats, bools, arrays.
    """
    data = {}
    current_section = data
    current_path = []
    array_mode = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        
        # Remove inline comments (but not inside strings)
        if " #" in line and not line.startswith('"') and not line.startswith("'"):
            # Simple heuristic: split on ' #' if not within a quoted value
            in_string = False
            comment_pos = -1
            for i, ch in enumerate(line):
                if ch in ('"', "'") and (i == 0 or line[i-1] != "\\"):
                    in_string = not in_string
                if ch == '#' and not in_string and i > 0 and line[i-1] == ' ':
                    comment_pos = i
                    break
            if comment_pos > 0:
                line = line[:comment_pos].strip()
        
        # Array of tables [[section]]
        if line.startswith("[[") and line.endswith("]]"):
            section_name = line[2:-2].strip()
            parts = section_name.split(".")
            parent = data
            for part in parts[:-1]:
                if part not in parent:
                    parent[part] = {}
                parent = parent[part]
            key = parts[-1]
            if key not in parent:
                parent[key] = []
            new_item = {}
            parent[key].append(new_item)
            current_section = new_item
            current_path = parts
            continue
        
        # Table [section]
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            parts = section_name.split(".")
            current_section = data
            current_path = parts
            for part in parts:
                if part not in current_section:
                    current_section[part] = {}
                current_section = current_section[part]
            continue
        
        # Key = Value
        if "=" in line:
            eq_pos = line.index("=")
            key = line[:eq_pos].strip()
            value = line[eq_pos + 1:].strip()
            
            # Parse value
            current_section[key] = _parse_toml_value(value)
    
    return data


def _parse_toml_value(value: str) -> Any:
    """Parse a single TOML value."""
    # Multi-line strings
    if value.startswith('"""'):
        return value.strip('"""')
    
    # Strings
    if (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    # Booleans
    if value == "true":
        return True
    if value == "false":
        return False
    
    # Arrays
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        # Simple array parsing (doesn't handle nested arrays)
        items = []
        for item in inner.split(","):
            item = item.strip()
            if item:
                items.append(_parse_toml_value(item))
        return items
    
    # Numbers
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


# ═══════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION — Built-in fallback values
# ═══════════════════════════════════════════════════════════════

DEFAULTS = {
    "nova": {
        "name": "NOVA",
        "version": "0.1.0",
        "description": "Neural Observation & Virtual Assistant",
        "data_directory": ".nova",
        "context_paths": ["AGENTS.md", "NOVA.md"],
    },
    "intelligence": {
        "default_provider": "huggingface",
        "preferred_engine": "api",
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.0,
        "auto_fallback": True,
        "fallback_order": ["huggingface", "ollama"],
        "providers": {
            "huggingface": {
                "enabled": True,
                "type": "openai-compat",
                "base_url": "https://router.huggingface.co/v1",
                "api_key": "$HF_API_TOKEN",
                "default_model": "google/gemma-2-2b-it",
            },
            "ollama": {
                "enabled": True,
                "type": "openai-compat",
                "base_url": "http://localhost:11434",
                "default_model": "tinyllama",
            },
        },
        "models": [
            {"id": "google/gemma-2-2b-it", "name": "Gemma 2B", "provider": "huggingface"},
            {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5", "provider": "huggingface"},
            {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "name": "DeepSeek R1", "provider": "huggingface"},
            {"id": "tinyllama", "name": "TinyLlama (Local)", "provider": "ollama"},
        ],
    },
    "agent": {
        "default_mode": "chat",
        "max_turns": 10,
        "system_prompt_prefix": "",
        "chat": {
            "system_prompt": (
                "You are NOVA, a Neural Interaction Engine running on a Windows PC. "
                "You have a built-in Neural Intent Engine (NIE) that handles system commands "
                "INSTANTLY without needing you. Be concise, helpful, and friendly."
            ),
        },
        "code": {
            "enabled": True,
            "edit_format": "diff",
            "auto_commit": False,
            "lint_after_edit": True,
            "repo_map": True,
            "watch_files": False,
        },
        "system": {
            "confidence_threshold": 0.35,
            "enable_os_bridge": True,
            "enable_screenshots": True,
            "enable_lock": True,
            "enable_shutdown": True,
        },
    },
    "tools": {
        "disabled_tools": [],
        "storage": {
            "default_backend": "sqlite",
            "db_path": "~/.nova/memory.db",
        },
        "mcp": {"enabled": False},
        "lsp": {"auto_lsp": True},
    },
    "engine": {
        "nie_enabled": True,
        "embed_dim": 64,
        "ff_dim": 128,
        "model_weights_dir": "nie_weights",
        "ollama": {"host": "http://localhost:11434"},
    },
    "consciousness": {
        "enable_personality": True,
        "enable_emotions": True,
        "enable_learning": True,
        "enable_meta_cognition": True,
        "personality_defaults": {
            "humor": 0.5,
            "formality": 0.5,
            "enthusiasm": 0.5,
            "curiosity": 0.5,
            "verbosity": 0.5,
            "creativity": 0.5,
            "empathy": 0.5,
            "directness": 0.5,
            "confidence": 0.5,
        },
    },
    "learning": {
        "enabled": True,
        "store_preferences": True,
        "track_intents": True,
        "track_time_patterns": True,
        "metrics": {
            "accuracy_weight": 0.6,
            "latency_weight": 0.2,
            "cost_weight": 0.1,
            "efficiency_weight": 0.1,
        },
    },
    "permissions": {
        "auto_approve_read": True,
        "auto_approve_modify": True,
        "require_confirm_execute": False,
        "require_confirm_shutdown": True,
        "allowed_tools": ["system_status", "screenshot"],
        "app_whitelist": {
            "browsers": ["chrome", "brave", "firefox", "edge"],
            "productivity": ["notepad", "calculator", "paint", "vscode", "word", "excel"],
            "system": ["explorer", "task manager", "settings", "terminal", "cmd", "powershell"],
            "media": ["spotify"],
            "web": ["google", "youtube", "github"],
        },
    },
    "ui": {
        "theme": "default",
        "show_header": True,
        "show_system_status": True,
        "show_workspace_files": True,
        "show_nie_status": True,
        "show_intent_classification": True,
        "compact_mode": False,
    },
    "telemetry": {
        "enabled": False,
        "db_path": "~/.nova/telemetry.db",
    },
    "logging": {
        "enabled": True,
        "log_dir": "logs",
        "log_format": "jsonl",
        "max_entries": 10000,
    },
}


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════════

class NovaConfig:
    """
    Unified NOVA configuration with layered loading.
    
    Merges config from:
      1. Built-in defaults
      2. User-level config (~/.nova/config.toml)
      3. Project-level config (./nova_config.toml)
      4. Environment variables (NOVA_* prefix overrides)
    """

    # Search paths for config files (in load order, later wins)
    CONFIG_FILENAMES = [
        "nova_config.toml",
        "nova.toml",
        ".nova.toml",
    ]

    def __init__(self):
        self._config: Dict = {}
        self._loaded_from: List[str] = []
        self._load()

    def _load(self):
        """Load configuration from all sources."""
        # 1. Start with defaults
        self._config = copy.deepcopy(DEFAULTS)

        # 2. User-level config
        user_config_path = os.path.expanduser("~/.nova/config.toml")
        if os.path.exists(user_config_path):
            try:
                user_cfg = _parse_toml(user_config_path)
                self._deep_merge(self._config, user_cfg)
                self._loaded_from.append(user_config_path)
            except Exception:
                pass

        # 3. Project-level config (search upward from CWD)
        project_config = self._find_project_config()
        if project_config:
            try:
                proj_cfg = _parse_toml(project_config)
                self._deep_merge(self._config, proj_cfg)
                self._loaded_from.append(project_config)
            except Exception:
                pass

        # 4. Environment variable overrides (NOVA_* prefix)
        self._apply_env_overrides()

    def _find_project_config(self) -> Optional[str]:
        """Find project config by searching CWD and parent dirs."""
        search_dir = os.getcwd()
        
        # Also check the NOVA-CLI project root
        nova_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        search_dirs = [search_dir, nova_root]
        
        for directory in search_dirs:
            for filename in self.CONFIG_FILENAMES:
                config_path = os.path.join(directory, filename)
                if os.path.exists(config_path):
                    return config_path
        
        return None

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base (in-place)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self):
        """Apply NOVA_* env vars as config overrides.
        
        Convention: NOVA_SECTION__KEY=value
        Example:    NOVA_INTELLIGENCE__DEFAULT_PROVIDER=ollama
                    NOVA_UI__COMPACT_MODE=true
        """
        prefix = "NOVA_"
        for key, value in os.environ.items():
            if key.startswith(prefix) and len(key) > len(prefix):
                config_path = key[len(prefix):].lower().replace("__", ".")
                parts = config_path.split(".")
                
                # Navigate to the right dict level
                target = self._config
                for part in parts[:-1]:
                    if part in target and isinstance(target[part], dict):
                        target = target[part]
                    else:
                        target = None
                        break
                
                if target is not None and parts:
                    # Parse the value
                    final_key = parts[-1]
                    if value.lower() == "true":
                        target[final_key] = True
                    elif value.lower() == "false":
                        target[final_key] = False
                    else:
                        try:
                            if "." in value:
                                target[final_key] = float(value)
                            else:
                                target[final_key] = int(value)
                        except ValueError:
                            target[final_key] = value

    # ─── Public API ───

    def get(self, dotpath: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated path.
        
        Examples:
            config.get("intelligence.default_provider")  -> "huggingface"
            config.get("agent.system.confidence_threshold")  -> 0.35
            config.get("ui.show_header")  -> True
        """
        parts = dotpath.split(".")
        current = self._config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current

    def resolve(self, dotpath: str, default: Any = None) -> Any:
        """
        Get a config value, resolving $ENV_VAR references to actual values.
        
        Example:
            config.resolve("intelligence.providers.huggingface.api_key")
            # If value is "$HF_API_TOKEN", returns os.environ["HF_API_TOKEN"]
        """
        value = self.get(dotpath, default)
        
        if isinstance(value, str) and value.startswith("$"):
            env_var = value[1:]
            return os.environ.get(env_var, default)
        
        return value

    def section(self, name: str) -> Dict:
        """Get a full config section as a dict."""
        result = self.get(name)
        if isinstance(result, dict):
            return copy.deepcopy(result)
        return {}

    def get_provider_config(self, provider_name: str = None) -> Dict:
        """
        Get the configuration for a specific AI provider.
        If no name given, returns the default provider config.
        """
        if provider_name is None:
            provider_name = self.get("intelligence.default_provider", "huggingface")
        
        return self.section(f"intelligence.providers.{provider_name}")

    def get_available_models(self) -> List[Dict]:
        """Get the list of available models for quick-switching."""
        return self.get("intelligence.models", [])

    def get_active_provider(self) -> str:
        """Get the name of the currently active provider."""
        return self.get("intelligence.default_provider", "huggingface")

    def get_nie_config(self) -> Dict:
        """Get the NIE (Neural Intent Engine) configuration."""
        return self.section("engine")

    def get_consciousness_config(self) -> Dict:
        """Get the consciousness layer configuration."""
        return self.section("consciousness")

    def get_permissions_config(self) -> Dict:
        """Get the permissions configuration."""
        return self.section("permissions")

    def get_ui_config(self) -> Dict:
        """Get the UI configuration."""
        return self.section("ui")

    @property
    def loaded_from(self) -> List[str]:
        """Return the list of config files that were loaded."""
        return list(self._loaded_from)

    def to_dict(self) -> Dict:
        """Return the full config as a dict (for debugging)."""
        return copy.deepcopy(self._config)

    def __repr__(self) -> str:
        sources = ", ".join(self._loaded_from) if self._loaded_from else "defaults only"
        return f"<NovaConfig loaded_from=[{sources}]>"


# ═══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════

nova_config = NovaConfig()
