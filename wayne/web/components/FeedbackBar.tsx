"use client";

import { useState } from "react";

export function FeedbackBar({ interactionId }: { interactionId: number }) {
  const [submitted, setSubmitted] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);

  async function submit(score: number) {
    setSelected(score);
    setSubmitted(true);
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interaction_id: interactionId, score })
    });
  }

  if (submitted) {
    return <div className="pt-2 text-[11px] text-amber">{"★".repeat(selected || 0)}{"☆".repeat(5 - (selected || 0))} Noted.</div>;
  }

  return (
    <div className="flex items-center gap-2 pt-2 text-[11px]">
      <span className="text-cyan/50">Rate:</span>
      {[1, 2, 3, 4, 5].map((score) => (
        <button key={score} onClick={() => submit(score)} className="text-cyan/50 transition hover:text-amber" title={`${score} stars`}>
          ★
        </button>
      ))}
      <button onClick={() => submit(1)} className="wayne-border px-2 py-0.5 text-danger">
        Wrong
      </button>
      <button onClick={() => submit(5)} className="wayne-border px-2 py-0.5 text-success">
        Perfect
      </button>
    </div>
  );
}
