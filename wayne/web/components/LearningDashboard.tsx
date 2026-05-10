"use client";

import { Brain, Download, RefreshCw, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

type Preference = { key: string; value: string; confidence: number; sample_count: number };
type Topic = { topic: string; frequency: number; avg_score: number };
type Interaction = { id: number; created_at: string; user_message: string; intent: string | null; final_reward: number };
type Habit = { id: number; pattern: string; habit_type: string; frequency: number; confidence: number };
type Contact = { id: number; name: string; mention_count: number; relationship?: string | null };
type Improvement = { id: number; improvement?: string | null; improvement_type?: string | null; new_behavior?: string | null; trigger_reason?: string | null; applied_at: string };
type Stats = {
  total_interactions: number;
  average_reward: number;
  golden_responses_stored: number;
  golden_responses_count?: number;
  habit_count?: number;
  known_contacts_count?: number;
  improvement_count?: number;
  emotion_distribution?: Record<string, number>;
  interaction_heatmap?: number[][];
  top_files?: { file_name: string; access_count: number; file_type: string }[];
  task_completion_rate?: number;
  top_topics: Topic[];
  learned_preferences: Preference[];
  learning_score: number;
};

export function LearningDashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [history, setHistory] = useState<Interaction[]>([]);
  const [habits, setHabits] = useState<Habit[]>([]);
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [improvements, setImprovements] = useState<Improvement[]>([]);

  async function load() {
    const [statsResponse, historyResponse, habitsResponse, contactsResponse, improvementsResponse] = await Promise.all([
      fetch("/api/learning/stats", { cache: "no-store" }),
      fetch("/api/learning/history", { cache: "no-store" }),
      fetch("/api/learning/habits", { cache: "no-store" }),
      fetch("/api/learning/contacts", { cache: "no-store" }),
      fetch("/api/learning/improvements", { cache: "no-store" })
    ]);
    setStats(await statsResponse.json());
    setHistory((await historyResponse.json()).interactions || []);
    setHabits((await habitsResponse.json()).habits || []);
    setContacts((await contactsResponse.json()).contacts || []);
    setImprovements((await improvementsResponse.json()).improvements || []);
  }

  useEffect(() => {
    load();
    const id = window.setInterval(load, 30000);
    return () => window.clearInterval(id);
  }, []);

  async function updatePreference(key: string, current: string) {
    const value = window.prompt(`Set ${key}`, current);
    if (!value) return;
    await fetch("/api/learning/preferences", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, value })
    });
    load();
  }

  const score = stats?.learning_score || 0;
  const maxHeat = Math.max(1, ...(stats?.interaction_heatmap || []).flat());
  const prefs = stats?.learned_preferences || [];
  const traitText = (key: string, fallback: string) => {
    const pref = prefs.find((item) => item.key === key);
    return pref ? `${fallback}: ${pref.value} (${Math.round(pref.confidence * 100)}%)` : fallback;
  };

  async function resetLearning() {
    if (!window.confirm("Reset W.A.Y.N.E learning data? Conversation history is preserved.")) return;
    await fetch("/api/learning/reset", { method: "DELETE" });
    load();
  }

  function exportLearning() {
    const blob = new Blob([JSON.stringify({ stats, history, habits, contacts, improvements }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "wayne-learning-export.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="wayne-border min-h-0 flex-1 overflow-y-auto bg-surface/80 p-4">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2 font-heading text-xl text-cyan">
          <Brain size={20} />
          Learning
        </div>
        <button onClick={load} className="wayne-border bg-panel p-2 text-cyan" title="Refresh learning stats">
          <RefreshCw size={15} />
        </button>
      </div>

      <div className="wayne-border mb-4 bg-panel p-4">
        <div className="text-xs uppercase text-cyan/60">W.A.Y.N.E Learning Score</div>
        <div className="mt-2 text-4xl text-cyan">{score.toFixed(1)} / 100</div>
        <div className="mt-3 h-2 bg-background">
          <div className="h-full bg-cyan" style={{ width: `${Math.min(100, score)}%` }} />
        </div>
        <div className="mt-2 text-xs text-cyan/50">{stats?.total_interactions || 0} interactions learned</div>
        <div className="mt-3 grid gap-2 text-xs text-cyan/70 sm:grid-cols-4">
          <span>{stats?.habit_count || 0} habits</span>
          <span>{stats?.known_contacts_count || 0} contacts</span>
          <span>{stats?.golden_responses_count || stats?.golden_responses_stored || 0} golden</span>
          <span>{Math.round((stats?.task_completion_rate || 0) * 100)}% task completion</span>
        </div>
      </div>

      <div className="wayne-border mb-4 bg-panel p-4">
        <h2 className="mb-3 text-cyan">Personality Model</h2>
        <div className="grid gap-2 md:grid-cols-2">
          {[traitText("response_length", "Response length"), traitText("tone", "Tone"), traitText("format", "Format"), traitText("preferred_wake_hour", "Peak wake hour"), traitText("language_complexity", "Language complexity"), traitText("proactivity_level", "Proactivity")].map((trait) => (
            <div key={trait} className="wayne-border bg-background p-2 text-xs text-cyan/80">{trait}</div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Learned Preferences</h2>
          <div className="space-y-2">
            {(stats?.learned_preferences || []).map((pref) => (
              <button key={pref.key} onClick={() => updatePreference(pref.key, pref.value)} className="wayne-border grid w-full grid-cols-[1fr_90px_70px] gap-2 bg-background p-2 text-left text-xs">
                <span className="text-cyan/70">{pref.key}</span>
                <span className="text-amber">{pref.value}</span>
                <span className="text-success">{Math.round(pref.confidence * 100)}%</span>
              </button>
            ))}
          </div>
        </div>

        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Top Topics</h2>
          <div className="space-y-3">
            {(stats?.top_topics || []).map((topic) => (
              <div key={topic.topic}>
                <div className="mb-1 flex justify-between text-xs">
                  <span className="text-cyan">{topic.topic}</span>
                  <span className="text-amber">{topic.frequency} · {Math.round(topic.avg_score * 100)}%</span>
                </div>
                <div className="h-2 bg-background">
                  <div className="h-full bg-success" style={{ width: `${Math.min(100, topic.frequency * 12)}%` }} />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5 text-sm text-cyan/70">Golden responses stored: <span className="text-amber">{stats?.golden_responses_stored || 0}</span></div>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Emotion History</h2>
          <div className="space-y-2">
            {Object.entries(stats?.emotion_distribution || { neutral: 0 }).map(([emotion, count]) => (
              <div key={emotion}>
                <div className="mb-1 flex justify-between text-xs text-cyan/70"><span>{emotion}</span><span>{count}</span></div>
                <div className="h-2 bg-background"><div className="h-full bg-cyan" style={{ width: `${Math.min(100, count * 15)}%` }} /></div>
              </div>
            ))}
          </div>
        </div>
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Activity Heatmap</h2>
          <div className="grid gap-1">
            {(stats?.interaction_heatmap || Array.from({ length: 7 }, () => Array(24).fill(0))).map((row, day) => (
              <div key={day} className="grid gap-1" style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
                {row.map((count, hour) => (
                  <div key={`${day}-${hour}`} title={`day ${day + 1}, ${hour}:00 - ${count}`} className="h-3" style={{ background: `rgba(0,212,255,${0.08 + (count / maxHeat) * 0.75})` }} />
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Detected Habits</h2>
          <div className="space-y-2 text-xs">
            {habits.slice(0, 8).map((habit) => (
              <div key={habit.id} className="wayne-border bg-background p-2">
                <div className="text-cyan/80">{habit.pattern.replaceAll("_", " ")}</div>
                <div className="text-amber">{Math.round(habit.confidence * 100)}% confident · {habit.frequency}x</div>
              </div>
            ))}
            {!habits.length && <div className="text-cyan/50">No habits learned yet.</div>}
          </div>
        </div>
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Known People</h2>
          <div className="grid gap-2 text-xs">
            {contacts.slice(0, 8).map((contact) => (
              <div key={contact.id} className="wayne-border bg-background p-2 text-cyan/80">{contact.name}<span className="float-right text-amber">{contact.mention_count}</span></div>
            ))}
            {!contacts.length && <div className="text-cyan/50">No contacts learned yet.</div>}
          </div>
        </div>
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Recent Improvements</h2>
          <div className="space-y-2 text-xs">
            {improvements.slice(0, 6).map((item) => (
              <div key={item.id} className="border-l border-cyan/40 pl-3 text-cyan/80">
                <div>{item.improvement || item.new_behavior || item.improvement_type}</div>
                <div className="text-cyan/40">{new Date(item.applied_at).toLocaleString()}</div>
              </div>
            ))}
            {!improvements.length && <div className="text-cyan/50">No autonomous improvements logged yet.</div>}
          </div>
        </div>
      </div>

      <div className="wayne-border mt-4 bg-panel p-4">
        <h2 className="mb-3 text-cyan">Interaction History</h2>
        <div className="space-y-2">
          {history.map((item) => (
            <div key={item.id} className="grid grid-cols-[120px_1fr_80px_70px] gap-3 border-b border-cyan/10 pb-2 text-xs">
              <span className="text-cyan/50">{new Date(item.created_at).toLocaleTimeString()}</span>
              <span className="truncate text-cyan/80">{item.user_message}</span>
              <span className="text-amber">{item.intent || "general"}</span>
              <span className={item.final_reward > 0.7 ? "text-success" : item.final_reward < 0.4 ? "text-danger" : "text-amber"}>
                {Math.round(item.final_reward * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="wayne-border mt-4 flex flex-wrap items-center gap-3 bg-panel p-4">
        <button onClick={exportLearning} className="wayne-border flex items-center gap-2 bg-background px-3 py-2 text-xs text-cyan">
          <Download size={14} /> Export Learning JSON
        </button>
        <button onClick={resetLearning} className="wayne-border flex items-center gap-2 bg-background px-3 py-2 text-xs text-danger">
          <RotateCcw size={14} /> Reset Learning
        </button>
        <span className="text-xs text-cyan/50">Experience buffer and preferences update after every interaction.</span>
      </div>
    </section>
  );
}
