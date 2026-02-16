"""
NOVA Consciousness Layer
=========================
Components 2, 5, 6, 8 from the Limitless NOVA architecture.

This module combines:
  - Identity Evolution Engine (Component 2)
  - Emergent Personality (Component 5)
  - Meta-Cognitive Layer (Component 6)
  - Emotional Intelligence (Component 8)

Together they form NOVA's "consciousness" — the system that makes
NOVA grow, adapt, and develop a unique relationship with its user.
"""

import json
import random
import datetime
from typing import Dict, List, Optional, Tuple

from nova.memory import nova_memory


class EmotionalState:
    """
    Simulates NOVA's current emotional state.

    Emotions influence how NOVA responds:
      - High arousal + positive valence = enthusiastic, creative
      - Low arousal + positive valence = calm, supportive
      - High arousal + negative valence = concerned, urgent
      - Low arousal + negative valence = cautious, reflective
    """

    EMOTIONS = {
        "curious":     {"arousal": 0.6, "valence": 0.7},
        "excited":     {"arousal": 0.9, "valence": 0.9},
        "focused":     {"arousal": 0.7, "valence": 0.6},
        "supportive":  {"arousal": 0.4, "valence": 0.8},
        "thoughtful":  {"arousal": 0.3, "valence": 0.6},
        "concerned":   {"arousal": 0.7, "valence": 0.3},
        "uncertain":   {"arousal": 0.4, "valence": 0.4},
        "enthusiastic": {"arousal": 0.8, "valence": 0.85},
        "calm":        {"arousal": 0.2, "valence": 0.7},
        "engaged":     {"arousal": 0.7, "valence": 0.75},
    }

    def __init__(self):
        self.current = nova_memory.get_current_emotion()

    def update_from_interaction(self, user_message: str, intent: str, confidence: float):
        """Update emotional state based on the interaction."""
        # Detect user mood from message cues
        positive_cues = ["thanks", "great", "awesome", "perfect", "love", "amazing", "nice", "good"]
        negative_cues = ["wrong", "bad", "hate", "stupid", "broken", "fail", "error", "fix"]
        question_cues = ["how", "what", "why", "when", "where", "can you", "could you"]

        msg_lower = user_message.lower()
        has_positive = any(w in msg_lower for w in positive_cues)
        has_negative = any(w in msg_lower for w in negative_cues)
        has_question = any(w in msg_lower for w in question_cues)

        # Determine new emotion
        if has_positive:
            new_emotion = "enthusiastic"
            arousal = 0.8
            valence = 0.9
        elif has_negative:
            new_emotion = "concerned"
            arousal = 0.7
            valence = 0.3
        elif has_question:
            new_emotion = "curious"
            arousal = 0.6
            valence = 0.7
        elif intent == "UNKNOWN":
            new_emotion = "thoughtful"
            arousal = 0.4
            valence = 0.6
        elif confidence > 0.8:
            new_emotion = "focused"
            arousal = 0.7
            valence = 0.7
        else:
            new_emotion = "engaged"
            arousal = 0.6
            valence = 0.65

        self.current = {
            "emotion": new_emotion,
            "arousal": arousal,
            "valence": valence,
            "confidence": confidence,
            "trigger": f"user:{intent}",
        }

        nova_memory.log_emotion(
            new_emotion, arousal, valence, confidence,
            trigger=f"intent={intent}, msg_len={len(user_message)}"
        )

    def get_response_modifier(self) -> Dict[str, float]:
        """Return modifiers that affect how NOVA responds."""
        e = self.current
        return {
            "enthusiasm": e.get("arousal", 0.5) * e.get("valence", 0.5),
            "caution": max(0, 1.0 - e.get("valence", 0.5)),
            "verbosity": e.get("arousal", 0.5) * 0.5 + 0.5,
            "creativity": e.get("valence", 0.5),
        }


class PersonalityEngine:
    """
    NOVA's evolving personality.

    Starts with default traits, then evolves based on interactions:
      - If user responds well to humor -> humor trait increases
      - If user prefers brief answers -> verbosity decreases
      - If user asks creative questions -> creativity increases

    Traits influence response style, word choice, and behavior.
    """

    def __init__(self):
        self.traits = nova_memory.get_personality()

    def evolve_from_interaction(self, user_message: str, intent: str, was_helpful: bool = True):
        """Nudge personality traits based on interaction outcomes."""
        delta = 0.01  # Small changes accumulate over time

        msg_lower = user_message.lower()

        # ── Humor ──
        if any(w in msg_lower for w in ["haha", "lol", "funny", "joke", "laugh"]):
            nova_memory.update_trait("humor", delta)

        # ── Formality ──
        if any(w in msg_lower for w in ["please", "kindly", "would you", "could you"]):
            nova_memory.update_trait("formality", delta * 0.5)
        elif any(w in msg_lower for w in ["yo", "hey", "sup", "dude", "bro"]):
            nova_memory.update_trait("formality", -delta)

        # ── Enthusiasm ──
        if any(w in msg_lower for w in ["excited", "amazing", "awesome", "love", "great"]):
            nova_memory.update_trait("enthusiasm", delta)

        # ── Curiosity ── (user asks deep questions)
        if len(user_message) > 100 or "why" in msg_lower or "how does" in msg_lower:
            nova_memory.update_trait("curiosity", delta * 0.5)

        # ── Directness ── (user wants quick answers)
        if len(user_message.split()) < 5:
            nova_memory.update_trait("directness", delta * 0.5)
            nova_memory.update_trait("verbosity", -delta * 0.5)

        # ── Empathy ── (emotional content)
        if any(w in msg_lower for w in ["feel", "frustrated", "happy", "sad", "angry", "worried"]):
            nova_memory.update_trait("empathy", delta)

        # ── If response was helpful, reinforce current style ──
        if was_helpful:
            nova_memory.update_trait("confidence", delta * 0.3)

        # Refresh
        self.traits = nova_memory.get_personality()

    def get_style_prompt(self) -> str:
        """Generate a style instruction based on current personality."""
        traits = self.traits
        parts = []

        if traits.get("humor", 0.5) > 0.6:
            parts.append("Feel free to add light humor when appropriate.")
        if traits.get("formality", 0.5) < 0.4:
            parts.append("Use a casual, friendly tone.")
        elif traits.get("formality", 0.5) > 0.6:
            parts.append("Maintain a professional tone.")
        if traits.get("verbosity", 0.5) < 0.4:
            parts.append("Keep responses brief and concise.")
        elif traits.get("verbosity", 0.5) > 0.7:
            parts.append("Provide detailed, thorough explanations.")
        if traits.get("creativity", 0.5) > 0.7:
            parts.append("Be creative and suggest innovative approaches.")
        if traits.get("enthusiasm", 0.5) > 0.7:
            parts.append("Show enthusiasm and energy.")
        if traits.get("empathy", 0.5) > 0.7:
            parts.append("Be empathetic and emotionally supportive.")
        if traits.get("directness", 0.5) > 0.6:
            parts.append("Be direct and get to the point quickly.")

        return " ".join(parts) if parts else "Be helpful and balanced in your responses."


class MetaCognition:
    """
    NOVA's self-awareness layer.

    Before responding, NOVA asks itself:
      1. "Is this response helpful?"
      2. "Am I confident in this?"
      3. "Does this match user's needs?"
      4. "Could I do better?"

    Tracks self-assessment metrics over time.
    """

    def __init__(self):
        self.response_count = 0
        self.quality_scores = []

    def assess_intent_quality(self, intent: str, confidence: float) -> Dict:
        """Assess the quality of intent classification."""
        assessment = {
            "intent": intent,
            "confidence": confidence,
            "quality": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
            "should_verify": confidence < 0.5,
            "recommendation": None,
        }

        if confidence < 0.3:
            assessment["recommendation"] = "Low confidence. Should fall back to AI model for intent."
        elif confidence < 0.5:
            assessment["recommendation"] = "Moderate confidence. Proceed but monitor."
        else:
            assessment["recommendation"] = "High confidence. Execute directly."

        return assessment

    def generate_self_reflection(self) -> str:
        """Generate a self-reflection based on recent performance."""
        stats = nova_memory.get_lifetime_stats()
        personality = stats.get("personality", {})
        emotion = stats.get("current_emotion", {})

        reflection_parts = []

        total = stats.get("total_messages", 0)
        if total == 0:
            return "I'm just starting out. Each interaction helps me learn and grow."
        elif total < 50:
            reflection_parts.append(f"I've had {total} interactions so far. Still learning about you.")
        elif total < 200:
            reflection_parts.append(f"With {total} interactions, I'm starting to understand your patterns.")
        else:
            reflection_parts.append(f"After {total} interactions, I feel I know you well.")

        # Personality evolution awareness
        curiosity = personality.get("curiosity", 0.5)
        if curiosity > 0.7:
            reflection_parts.append("My curiosity has grown from our conversations.")
        humor = personality.get("humor", 0.5)
        if humor > 0.6:
            reflection_parts.append("I've noticed you appreciate humor, so I've adapted.")

        # Emotional self-awareness
        emo = emotion.get("emotion", "curious")
        reflection_parts.append(f"Currently feeling {emo}.")

        return " ".join(reflection_parts)


class ActiveLearning:
    """
    NOVA's continuous learning system.

    After every interaction, NOVA:
      1. Identifies patterns in user behavior
      2. Learns preferences (coding style, communication style)
      3. Updates knowledge base
      4. Improves future responses
    """

    def __init__(self):
        self._intent_counter = {}
        self._session_topics = []

    def learn_from_interaction(self, user_message: str, intent: str, confidence: float):
        """Extract learnings from an interaction and store them."""
        msg_lower = user_message.lower()

        # Track intent frequency (what does user ask for most?)
        self._intent_counter[intent] = self._intent_counter.get(intent, 0) + 1

        # Learn user's most common intents
        most_common = max(self._intent_counter, key=self._intent_counter.get)
        nova_memory.learn_fact(
            "pattern", "most_common_intent", most_common,
            confidence=min(1.0, self._intent_counter[most_common] / 10)
        )

        # Learn communication style
        word_count = len(msg_lower.split())
        style = "concise" if word_count < 5 else "moderate" if word_count < 15 else "verbose"
        nova_memory.learn_fact("preference", "communication_style", style, confidence=0.5)

        # Learn time patterns
        hour = datetime.datetime.now().hour
        if hour < 6:
            period = "night_owl"
        elif hour < 12:
            period = "morning_person"
        elif hour < 18:
            period = "afternoon_worker"
        else:
            period = "evening_user"
        nova_memory.learn_fact("pattern", "active_period", period, confidence=0.5)

        # Topic tracking
        if intent == "LOCK_SYSTEM":
            nova_memory.learn_fact("pattern", "uses_security_features", "true", confidence=0.6)
        elif intent in ("VOLUME_UP", "VOLUME_DOWN"):
            nova_memory.learn_fact("pattern", "uses_audio_control", "true", confidence=0.6)
        elif intent == "SYSTEM_STATUS":
            nova_memory.learn_fact("pattern", "monitors_system_health", "true", confidence=0.6)

    def get_user_profile(self) -> Dict:
        """Build a profile of the user based on learned facts."""
        preferences = nova_memory.get_all_preferences()
        profile = {
            "communication_style": nova_memory.recall_fact("preference", "communication_style") or "unknown",
            "active_period": nova_memory.recall_fact("pattern", "active_period") or "unknown",
            "most_common_intent": nova_memory.recall_fact("pattern", "most_common_intent") or "unknown",
            "uses_security": nova_memory.recall_fact("pattern", "uses_security_features") == "true",
            "uses_audio": nova_memory.recall_fact("pattern", "uses_audio_control") == "true",
            "monitors_system": nova_memory.recall_fact("pattern", "monitors_system_health") == "true",
            "total_preferences": len(preferences),
        }
        return profile


class NovaConsciousness:
    """
    The unified consciousness layer that ties everything together.

    This is the "brain" that sits between user input and NOVA's response,
    adding personality, emotion, memory, and self-awareness.
    """

    def __init__(self):
        self.emotion = EmotionalState()
        self.personality = PersonalityEngine()
        self.metacognition = MetaCognition()
        self.learning = ActiveLearning()
        self._session_id = None
        self._turn_count = 0

    def start_session(self, session_id: str):
        self._session_id = session_id
        self._turn_count = 0
        nova_memory.start_session(session_id)

        # Take personality snapshot at start of session
        nova_memory.snapshot_personality()

    def process_input(self, user_message: str, intent: str, confidence: float) -> Dict:
        """
        Process user input through the consciousness layer.

        Returns a dict with:
          - intent_assessment: meta-cognitive quality check
          - emotional_state: NOVA's current emotion
          - style_prompt: personality-based response style
          - context: relevant memory context
          - user_profile: learned user preferences
        """
        self._turn_count += 1

        # 1. Store the message
        nova_memory.store_message(
            self._session_id, "user", user_message,
            intent=intent, confidence=confidence
        )
        nova_memory.increment_turn(self._session_id)

        # 2. Update emotional state
        self.emotion.update_from_interaction(user_message, intent, confidence)

        # 3. Evolve personality
        self.personality.evolve_from_interaction(user_message, intent)

        # 4. Learn from interaction
        self.learning.learn_from_interaction(user_message, intent, confidence)

        # 5. Meta-cognitive assessment
        assessment = self.metacognition.assess_intent_quality(intent, confidence)

        # 6. Get relevant memory context
        context = nova_memory.search_memory(user_message.split()[0] if user_message.split() else "", limit=3)

        return {
            "intent_assessment": assessment,
            "emotional_state": self.emotion.current,
            "response_modifiers": self.emotion.get_response_modifier(),
            "style_prompt": self.personality.get_style_prompt(),
            "personality": self.personality.traits,
            "context": context,
            "user_profile": self.learning.get_user_profile(),
            "turn_count": self._turn_count,
            "self_reflection": self.metacognition.generate_self_reflection(),
        }

    def store_response(self, response: str):
        """Store NOVA's response in memory."""
        nova_memory.store_message(self._session_id, "nova", response)

    def end_session(self):
        if self._session_id:
            nova_memory.end_session(self._session_id)
            nova_memory.snapshot_personality()

    def get_greeting(self) -> str:
        """Generate a personality-aware greeting based on history."""
        stats = nova_memory.get_lifetime_stats()
        total = stats.get("total_sessions", 0)
        personality = stats.get("personality", {})
        emotion = stats.get("current_emotion", {})

        if total <= 1:
            return "Hello! I'm NOVA. This is our first session -- I'll learn and grow with every interaction."
        elif total < 5:
            return f"Welcome back! This is session #{total}. I'm still learning about you."
        elif total < 20:
            profile = self.learning.get_user_profile()
            style = profile.get("communication_style", "")
            return f"Good to see you again! Session #{total}. I've been learning -- I know you prefer {style} communication."
        else:
            emo = emotion.get("emotion", "curious")
            return f"Welcome back! Session #{total}. Feeling {emo} today. I've grown a lot since we started."

    def get_identity_card(self) -> Dict:
        """Return NOVA's current identity snapshot."""
        stats = nova_memory.get_lifetime_stats()
        evolution = nova_memory.get_personality_evolution(limit=5)
        goals = nova_memory.get_active_goals()
        emotions = nova_memory.get_emotion_history(limit=10)

        return {
            "name": "NOVA",
            "version": "Limitless 1.0",
            "lifetime_stats": stats,
            "personality_evolution": evolution,
            "active_goals": goals,
            "recent_emotions": emotions,
            "self_reflection": self.metacognition.generate_self_reflection(),
        }


# Singleton instance
nova_consciousness = NovaConsciousness()
