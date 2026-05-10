"use client";

import { useCallback, useMemo, useState } from "react";

export type WayneMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolUsed?: string[] | string | null;
  interactionId?: number | null;
};

const seed: WayneMessage[] = [
  { id: "seed", role: "assistant", content: "[AI RESPONSE] W.A.Y.N.E online. Wireless Artificial Yielding Network Engine ready." }
];

export function useWayne() {
  const [messages, setMessages] = useState<WayneMessage[]>(seed);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [lastInteractionId, setLastInteractionId] = useState<number | null>(null);

  const apiMessages = useMemo(
    () => messages.map((message) => ({ role: message.role, content: message.content })),
    [messages]
  );

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;
      const userMessage: WayneMessage = { id: crypto.randomUUID(), role: "user", content: trimmed };
      setMessages((current) => [...current, userMessage]);
      setLoading(true);
      setToast(null);
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 45000);
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: apiMessages, query: trimmed, session_id: "web", prev_interaction_id: lastInteractionId }),
          signal: controller.signal
        });
        const data = await response.json();
        const assistant: WayneMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.reply || "[AI RESPONSE] No response received.",
          toolUsed: data.tool_used,
          interactionId: data.interaction_id ?? null
        };
        if (data.interaction_id) setLastInteractionId(data.interaction_id);
        setMessages((current) => [...current, assistant]);
      } catch (error) {
        const message =
          error instanceof DOMException && error.name === "AbortError"
            ? "W.A.Y.N.E response timed out. The local model is busy; try a shorter command."
            : "Connection to W.A.Y.N.E backend failed.";
        setToast(message);
        setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: `[OFFLINE MODE] [AI RESPONSE] ${message}` }]);
      } finally {
        window.clearTimeout(timeout);
        setLoading(false);
      }
    },
    [apiMessages, lastInteractionId, loading]
  );

  return { messages, loading, toast, sendMessage, setToast };
}
