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
  const [streaming, setStreaming] = useState(false);
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
      const timeout = window.setTimeout(() => controller.abort(), 120000);
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: apiMessages, query: trimmed, session_id: "web", prev_interaction_id: lastInteractionId, stream: true }),
          signal: controller.signal
        });
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("text/event-stream") && response.body) {
          const assistantId = crypto.randomUUID();
          let assistant: WayneMessage = { id: assistantId, role: "assistant", content: "" };
          setMessages((current) => [...current, assistant]);
          setLoading(false);
          setStreaming(true);

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop() || "";
            for (const event of events) {
              const line = event.split("\n").find((item) => item.startsWith("data: "));
              if (!line) continue;
              const data = JSON.parse(line.slice(6));
              if (data.type === "token") {
                assistant = { ...assistant, content: assistant.content + data.token };
                setMessages((current) => current.map((message) => (message.id === assistantId ? assistant : message)));
              } else if (data.type === "done") {
                assistant = { ...assistant, interactionId: data.interaction_id ?? null };
                if (data.interaction_id) setLastInteractionId(data.interaction_id);
                setMessages((current) => current.map((message) => (message.id === assistantId ? assistant : message)));
                setStreaming(false);
              } else if (data.type === "error") {
                throw new Error(data.message || "Streaming failed");
              }
            }
          }
        } else {
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
        }
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
        setStreaming(false);
      }
    },
    [apiMessages, lastInteractionId, loading]
  );

  return { messages, loading, streaming, toast, sendMessage, setToast };
}
