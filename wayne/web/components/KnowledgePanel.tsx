"use client";

import { Search } from "lucide-react";
import { FormEvent, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

export function KnowledgePanel() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [source, setSource] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setAnswer("");
    try {
      const response = await fetch(`${API_URL}/knowledge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, source: "auto" }),
      });
      const data = await response.json();
      setAnswer(data.answer || "No answer found.");
      setSource(data.source || "knowledge");
    } catch {
      setAnswer("Knowledge service is offline.");
      setSource("offline");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="wayne-border bg-panel p-3">
      <div className="mb-2 flex items-center gap-2 text-xs uppercase text-cyan/70">
        <Search size={14} />
        Knowledge
      </div>
      <form onSubmit={submit} className="flex gap-2">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ask a fact..."
          className="wayne-border h-9 min-w-0 flex-1 bg-background px-2 text-xs text-white outline-none focus:border-cyan"
        />
        <button className="wayne-border h-9 w-9 bg-cyan/10 text-cyan disabled:opacity-50" disabled={loading} title="Search">
          <Search size={14} className={loading ? "animate-pulse" : ""} />
        </button>
      </form>
      {answer && (
        <div className="mt-3 border-l border-cyan/30 pl-3 text-xs leading-5 text-cyan/80">
          <div className="mb-1 inline-block rounded border border-amber/30 px-1.5 py-0.5 text-[10px] uppercase text-amber">{source}</div>
          <p>{answer}</p>
        </div>
      )}
    </section>
  );
}
