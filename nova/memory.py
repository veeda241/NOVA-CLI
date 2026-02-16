"""
NOVA Persistent Memory System
==============================
Component 1 from the Limitless NOVA architecture.

Unlike standard AI that forgets everything after each session,
NOVA remembers EVERYTHING — conversations, preferences, learnings,
personality evolution, and emotional states.

Storage: SQLite (offline, private, persistent)
"""

import os
import json
import sqlite3
import datetime
import hashlib
from typing import List, Dict, Optional, Tuple

# Database location
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nova_brain")
DB_PATH = os.path.join(DB_DIR, "memory.db")


class PersistentMemory:
    """
    NOVA's long-term memory system.

    Stores:
      - Conversation history (episodic memory)
      - Learned facts and preferences (semantic memory)
      - Personality trait evolution (identity memory)
      - Emotional state history (emotional memory)
      - Interaction statistics (meta memory)
    """

    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._ensure_identity()

    def _create_tables(self):
        c = self.conn.cursor()

        # ── Episodic Memory: every conversation turn ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,          -- 'user' or 'nova'
                content     TEXT NOT NULL,
                intent      TEXT DEFAULT NULL,      -- NIE classification
                confidence  REAL DEFAULT NULL,      -- NIE confidence
                timestamp   TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0
            )
        """)

        # ── Semantic Memory: learned facts and preferences ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS learned_facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT NOT NULL,           -- 'preference', 'fact', 'skill', 'pattern'
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                confidence  REAL DEFAULT 0.5,
                source      TEXT DEFAULT 'interaction',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                UNIQUE(category, key)
            )
        """)

        # ── Identity Memory: personality trait evolution ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS personality_traits (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trait       TEXT NOT NULL UNIQUE,
                value       REAL NOT NULL DEFAULT 0.5,
                min_val     REAL DEFAULT 0.0,
                max_val     REAL DEFAULT 1.0,
                updated_at  TEXT NOT NULL
            )
        """)

        # ── Personality Snapshots: track evolution over time ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS personality_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot    TEXT NOT NULL,           -- JSON of all traits
                timestamp   TEXT NOT NULL
            )
        """)

        # ── Emotional State Log ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS emotional_states (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion     TEXT NOT NULL,
                arousal     REAL NOT NULL,           -- energy level 0-1
                valence     REAL NOT NULL,           -- positivity 0-1
                confidence  REAL NOT NULL,           -- certainty 0-1
                trigger     TEXT DEFAULT NULL,       -- what caused this state
                timestamp   TEXT NOT NULL
            )
        """)

        # ── Goals & Aspirations ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_type   TEXT NOT NULL,           -- 'primary', 'secondary', 'emergent'
                description TEXT NOT NULL,
                priority    REAL DEFAULT 0.5,
                progress    REAL DEFAULT 0.0,
                status      TEXT DEFAULT 'active',   -- 'active', 'achieved', 'paused'
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)

        # ── Interaction Statistics ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS session_stats (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL UNIQUE,
                started_at   TEXT NOT NULL,
                ended_at     TEXT DEFAULT NULL,
                turn_count   INTEGER DEFAULT 0,
                intents_used TEXT DEFAULT '{}',      -- JSON counter
                avg_confidence REAL DEFAULT 0.0,
                user_mood    TEXT DEFAULT 'neutral',
                satisfaction REAL DEFAULT 0.5
            )
        """)

        self.conn.commit()

    def _now(self) -> str:
        return datetime.datetime.now().isoformat()

    def _ensure_identity(self):
        """Initialize default personality traits if they don't exist."""
        default_traits = {
            "formality": 0.5,
            "humor": 0.3,
            "verbosity": 0.5,
            "creativity": 0.6,
            "assertiveness": 0.4,
            "empathy": 0.6,
            "curiosity": 0.7,
            "patience": 0.6,
            "enthusiasm": 0.5,
            "directness": 0.5,
        }
        now = self._now()
        for trait, value in default_traits.items():
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO personality_traits (trait, value, updated_at) VALUES (?, ?, ?)",
                    (trait, value, now),
                )
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()

    # ═══════════════════════════════════════════
    # EPISODIC MEMORY (Conversations)
    # ═══════════════════════════════════════════

    def store_message(self, session_id: str, role: str, content: str,
                      intent: str = None, confidence: float = None):
        self.conn.execute(
            """INSERT INTO conversations (session_id, role, content, intent, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, role, content, intent, confidence, self._now()),
        )
        self.conn.commit()

    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT role, content, intent, confidence, timestamp FROM conversations "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_recent_context(self, n: int = 10) -> List[Dict]:
        """Get last N conversation turns across all sessions."""
        rows = self.conn.execute(
            "SELECT role, content, intent, timestamp FROM conversations "
            "ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_total_conversations(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        return row[0]

    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search past conversations for relevant context."""
        rows = self.conn.execute(
            "SELECT role, content, intent, timestamp FROM conversations "
            "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ═══════════════════════════════════════════
    # SEMANTIC MEMORY (Learned Facts)
    # ═══════════════════════════════════════════

    def learn_fact(self, category: str, key: str, value: str, confidence: float = 0.5):
        now = self._now()
        self.conn.execute(
            """INSERT INTO learned_facts (category, key, value, confidence, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(category, key) DO UPDATE SET
                   value = excluded.value,
                   confidence = min(1.0, confidence + 0.1),
                   updated_at = excluded.updated_at,
                   access_count = access_count + 1""",
            (category, key, value, confidence, now, now),
        )
        self.conn.commit()

    def recall_fact(self, category: str, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM learned_facts WHERE category = ? AND key = ?",
            (category, key),
        ).fetchone()
        if row:
            # Increase access count (strengthens memory)
            self.conn.execute(
                "UPDATE learned_facts SET access_count = access_count + 1 WHERE category = ? AND key = ?",
                (category, key),
            )
            self.conn.commit()
            return row[0]
        return None

    def get_all_preferences(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT key, value, confidence FROM learned_facts WHERE category = 'preference' ORDER BY confidence DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_learned_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM learned_facts").fetchone()
        return row[0]

    # ═══════════════════════════════════════════
    # IDENTITY MEMORY (Personality Evolution)
    # ═══════════════════════════════════════════

    def get_personality(self) -> Dict[str, float]:
        rows = self.conn.execute("SELECT trait, value FROM personality_traits").fetchall()
        return {r["trait"]: r["value"] for r in rows}

    def update_trait(self, trait: str, delta: float):
        """Nudge a personality trait. Small deltas accumulate over time."""
        self.conn.execute(
            """UPDATE personality_traits
               SET value = MIN(max_val, MAX(min_val, value + ?)),
                   updated_at = ?
               WHERE trait = ?""",
            (delta, self._now(), trait),
        )
        self.conn.commit()

    def snapshot_personality(self):
        """Save current personality state for historical tracking."""
        traits = self.get_personality()
        self.conn.execute(
            "INSERT INTO personality_history (snapshot, timestamp) VALUES (?, ?)",
            (json.dumps(traits), self._now()),
        )
        self.conn.commit()

    def get_personality_evolution(self, limit: int = 10) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT snapshot, timestamp FROM personality_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"traits": json.loads(r["snapshot"]), "timestamp": r["timestamp"]} for r in reversed(rows)]

    # ═══════════════════════════════════════════
    # EMOTIONAL STATE
    # ═══════════════════════════════════════════

    def log_emotion(self, emotion: str, arousal: float, valence: float,
                    confidence: float, trigger: str = None):
        self.conn.execute(
            """INSERT INTO emotional_states (emotion, arousal, valence, confidence, trigger, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (emotion, arousal, valence, confidence, trigger, self._now()),
        )
        self.conn.commit()

    def get_current_emotion(self) -> Dict:
        row = self.conn.execute(
            "SELECT emotion, arousal, valence, confidence, trigger FROM emotional_states ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return dict(row)
        return {"emotion": "curious", "arousal": 0.5, "valence": 0.7, "confidence": 0.5, "trigger": None}

    def get_emotion_history(self, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT emotion, arousal, valence, timestamp FROM emotional_states ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ═══════════════════════════════════════════
    # GOALS
    # ═══════════════════════════════════════════

    def add_goal(self, goal_type: str, description: str, priority: float = 0.5):
        now = self._now()
        self.conn.execute(
            """INSERT INTO goals (goal_type, description, priority, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (goal_type, description, priority, now, now),
        )
        self.conn.commit()

    def get_active_goals(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT goal_type, description, priority, progress FROM goals WHERE status = 'active' ORDER BY priority DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_goal_progress(self, goal_id: int, progress: float):
        status = "achieved" if progress >= 1.0 else "active"
        self.conn.execute(
            "UPDATE goals SET progress = ?, status = ?, updated_at = ? WHERE id = ?",
            (min(1.0, progress), status, self._now(), goal_id),
        )
        self.conn.commit()

    # ═══════════════════════════════════════════
    # SESSION STATS
    # ═══════════════════════════════════════════

    def start_session(self, session_id: str):
        try:
            self.conn.execute(
                "INSERT INTO session_stats (session_id, started_at) VALUES (?, ?)",
                (session_id, self._now()),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def end_session(self, session_id: str):
        self.conn.execute(
            "UPDATE session_stats SET ended_at = ? WHERE session_id = ?",
            (self._now(), session_id),
        )
        self.conn.commit()

    def increment_turn(self, session_id: str):
        self.conn.execute(
            "UPDATE session_stats SET turn_count = turn_count + 1 WHERE session_id = ?",
            (session_id,),
        )
        self.conn.commit()

    def get_total_sessions(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM session_stats").fetchone()
        return row[0]

    def get_lifetime_stats(self) -> Dict:
        """Get NOVA's lifetime statistics."""
        total_convos = self.get_total_conversations()
        total_sessions = self.get_total_sessions()
        total_learned = self.get_learned_count()
        goals = self.get_active_goals()

        return {
            "total_messages": total_convos,
            "total_sessions": total_sessions,
            "facts_learned": total_learned,
            "active_goals": len(goals),
            "personality": self.get_personality(),
            "current_emotion": self.get_current_emotion(),
        }

    def close(self):
        self.conn.close()


# Singleton instance
nova_memory = PersistentMemory()
