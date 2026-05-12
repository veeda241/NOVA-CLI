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
    const manager = createConnectionManager(sessionId);
    managerRef.current = manager;
    manager.on("connected", () => setStatus("connected"));
    manager.on("disconnected", () => setStatus("disconnected"));
    manager.on("reconnecting", ({ attempt }: any) => {
      setStatus("reconnecting");
      setReconnectAttempt(attempt);
    });
    manager.on("offline", () => setStatus("offline"));
    manager.on("health", (data: any) => setHealth(data));
    manager.connect(`/ws/chat/${sessionId}`).catch(() => setStatus("disconnected"));
    return () => manager.disconnect();
  }, [sessionId]);

  const send = useCallback((data: any) => managerRef.current?.send(data) ?? false, []);
  const retry = useCallback(() => managerRef.current?.reconnect(), []);

  return { status, health, reconnectAttempt, send, retry, manager: managerRef.current };
}
