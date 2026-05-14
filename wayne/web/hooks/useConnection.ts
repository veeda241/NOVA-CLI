"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createConnectionManager } from "../lib/connectionManager";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "reconnecting" | "offline";

export function useConnection(sessionId: string) {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [health, setHealth] = useState<any>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const managerRef = useRef<ReturnType<typeof createConnectionManager> | null>(null);

  useEffect(() => {
    let active = true;
    const manager = createConnectionManager(sessionId);
    managerRef.current = manager;
    manager.on("connected", () => active && setStatus("connected"));
    manager.on("disconnected", () => active && setStatus("disconnected"));
    manager.on("reconnecting", ({ attempt }: any) => {
      if (!active) return;
      setStatus("reconnecting");
      setReconnectAttempt(attempt);
    });
    manager.on("offline", () => active && setStatus("offline"));
    manager.on("health", (data: any) => active && setHealth(data));
    manager.refreshHealth();
    manager.connect(`/ws/chat/${sessionId}`).catch(() => active && setStatus("disconnected"));
    return () => {
      active = false;
      manager.disconnect();
    };
  }, [sessionId]);

  const send = useCallback((data: any) => managerRef.current?.send(data) ?? false, []);
  const retry = useCallback(() => managerRef.current?.reconnect(), []);

  return { status, health, reconnectAttempt, send, retry, manager: managerRef.current };
}
