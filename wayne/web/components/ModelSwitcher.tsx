"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

export function ModelSwitcher() {
  const [models, setModels] = useState<string[]>([]);
  const [current, setCurrent] = useState("qwen2.5:1.5b");
  const [status, setStatus] = useState("Local");

  useEffect(() => {
    fetch(`${API_URL}/models`, { cache: "no-store" })
      .then((response) => response.json())
      .then((data) => {
        setModels(data.models || []);
        setCurrent(data.current || "qwen2.5:1.5b");
        setStatus(data.error ? "Ollama offline" : "Local AI");
      })
      .catch(() => setStatus("Ollama offline"));
  }, []);

  async function switchModel(model: string) {
    setCurrent(model);
    const response = await fetch(`${API_URL}/models/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model })
    });
    const data = await response.json();
    setCurrent(data.model || model);
  }

  return (
    <div className="space-y-2">
      <div className="text-xs uppercase text-cyan/60">Model</div>
      <select
        value={current}
        onChange={(event) => switchModel(event.target.value)}
        className="wayne-border w-full bg-background px-2 py-2 text-xs text-cyan outline-none"
      >
        {models.length === 0 && <option value={current}>{current}</option>}
        {models.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
      <div className="text-[10px] text-cyan/50">{status}</div>
    </div>
  );
}
