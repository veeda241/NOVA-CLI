"use client";

import { Activity, RefreshCw, Server, WifiOff } from "lucide-react";
import type { ConnectionStatus as Status } from "../hooks/useConnection";

export function ConnectionStatus({
  status,
  health,
  reconnectAttempt,
  onRetry
}: {
  status: Status;
  health: any;
  reconnectAttempt: number;
  onRetry: () => void;
}) {
  const speed = health?.speed ?? {};
  const connected = status === "connected" && health?.status !== "offline";
  const color = connected ? "text-success border-success/40" : status === "reconnecting" ? "text-amber border-amber/50" : "text-red border-red/50";
  const label = connected
    ? "W.A.Y.N.E ONLINE"
    : status === "reconnecting"
      ? `Reconnecting... attempt ${reconnectAttempt}`
      : status === "offline"
        ? "OFFLINE MODE"
        : "CONNECTION LOST";

  return (
    <div className={`wayne-border flex h-9 items-center gap-3 bg-surface px-4 text-[11px] ${color}`}>
      <span className={`h-2 w-2 rounded-full ${connected ? "animate-pulse bg-success" : status === "reconnecting" ? "animate-pulse bg-amber" : "bg-red"}`} />
      <span className="font-heading tracking-[2px]">{label}</span>
      <span className="ml-auto flex items-center gap-2 text-cyan/70">
        <Server size={13} />
        Backend {health?.status === "offline" ? "offline" : "online"}
      </span>
      <span className="flex items-center gap-2 text-cyan/70">
        <Activity size={13} />
        {health?.model ?? "qwen"} | cache {speed.hit_rate ?? 0}% | first token {speed.avg_first_token_ms ?? 0}ms
      </span>
      {!connected && (
        <button onClick={onRetry} className="wayne-border flex items-center gap-1 bg-panel px-2 py-1 text-cyan hover:border-cyan">
          {status === "offline" ? <WifiOff size={13} /> : <RefreshCw size={13} />}
          Retry
        </button>
      )}
    </div>
  );
}
