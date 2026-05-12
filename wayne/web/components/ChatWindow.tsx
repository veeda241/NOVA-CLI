"use client";

import { useEffect, useRef } from "react";
import type { WayneMessage } from "../hooks/useWayne";
import { FeedbackBar } from "./FeedbackBar";

function tagOf(content: string) {
  const match = content.match(/^\[([^\]]+)\]/);
  return match?.[1] || "AI RESPONSE";
}

export function ChatWindow({ messages, loading, streaming }: { messages: WayneMessage[]; loading: boolean; streaming?: boolean }) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <section className="wayne-border min-h-[180px] flex-1 overflow-y-auto bg-surface/80 p-4">
      <div className="flex flex-col gap-3">
        {messages.map((message) => {
          const isUser = message.role === "user";
          const tag = tagOf(message.content);
          return (
            <article key={message.id} className={`animate-fade max-w-[82%] ${isUser ? "self-end" : "self-start"}`}>
              {!isUser && <div className="mb-1 text-xs text-cyan">{`[${tag}]`}</div>}
              <div
                className={`wayne-border px-4 py-3 text-sm leading-6 ${
                  isUser ? "bg-cyan/15 text-white" : "bg-panel text-cyan-50"
                } ${streaming && !isUser && message.id === messages[messages.length - 1]?.id ? "animate-pulse border-cyan" : ""}`}
              >
                <p className="whitespace-pre-wrap">
                  {message.content.replace(/^\[[^\]]+\]\s*/, "")}
                  {streaming && !isUser && message.id === messages[messages.length - 1]?.id && <span className="ml-1 animate-pulse text-cyan">|</span>}
                </p>
                {!isUser && message.toolUsed && (
                  <div className="mt-3 grid grid-cols-[90px_1fr] gap-2 border-t border-cyan/10 pt-2 text-xs">
                    <span className="text-cyan">Tool</span>
                    <span className="text-amber">{Array.isArray(message.toolUsed) ? message.toolUsed.join(", ") : message.toolUsed}</span>
                  </div>
                )}
                {!isUser && message.interactionId && <FeedbackBar interactionId={message.interactionId} />}
              </div>
            </article>
          );
        })}
        {(loading || streaming) && (
          <div className="wayne-border flex w-fit items-center gap-2 bg-panel p-3 text-xs text-cyan">
            <span>{streaming ? "Streaming..." : "W.A.Y.N.E is processing..."}</span>
            {[0, 1, 2].map((dot) => (
              <span key={dot} className="h-2 w-2 animate-bounceDot rounded-full bg-cyan" style={{ animationDelay: `${dot * 120}ms` }} />
            ))}
          </div>
        )}
        <div ref={endRef} />
      </div>
    </section>
  );
}
