"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import BrowserWakeWord from "./BrowserWakeWord";
import WakeBootScreen from "./WakeBootScreen";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace("http://", "ws://").replace("https://", "wss://");

type WayneEvent = {
  id?: number;
  event_type?: "wake" | "sleep";
  fired_at?: string;
};

export default function WakeListener() {
  const [screen, setScreen] = useState<"wake" | "sleep" | null>(null);
  const lastEventRef = useRef<string | number | null>(null);

  const triggerWake = useCallback(async () => {
    setScreen("wake");
    window.setTimeout(() => {
      setScreen(null);
      window.dispatchEvent(new CustomEvent("wayne:start-voice"));
    }, 4400);
    try {
      await fetch(`${API_URL}/wayne/wake`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "browser_wake_word" })
      });
    } catch {
      // Browser wake still shows locally if the backend is momentarily unavailable.
    }
  }, []);

  const handleEvent = useCallback((event: WayneEvent | null) => {
    if (!event) return;
    const key = event.id ?? event.fired_at ?? `${event.event_type}-${Date.now()}`;
    if (lastEventRef.current === key) return;
    lastEventRef.current = key;
    if (event.event_type === "sleep") {
      setScreen("sleep");
      window.setTimeout(() => setScreen(null), 2500);
      return;
    }
    setScreen("wake");
    window.setTimeout(() => {
      setScreen(null);
      window.dispatchEvent(new CustomEvent("wayne:start-voice"));
    }, 4400);
  }, []);

  useEffect(() => {
    let closed = false;
    let socket: WebSocket | null = null;

    const connect = () => {
      socket = new WebSocket(`${WS_URL}/ws/wayne/events`);
      socket.onmessage = (message) => {
        const payload = JSON.parse(message.data);
        if (payload.type === "wayne_event") handleEvent(payload.event);
      };
      socket.onclose = () => {
        if (!closed) window.setTimeout(connect, 3000);
      };
    };
    connect();

    const poll = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/wayne/status`);
        const data = await response.json();
        handleEvent(data.last_wake);
      } catch {
        // WebSocket reconnect handles normal operation; polling is a fallback.
      }
    }, 5000);

    return () => {
      closed = true;
      window.clearInterval(poll);
      socket?.close();
    };
  }, [handleEvent]);

  return (
    <>
      <BrowserWakeWord onWake={triggerWake} />
      {screen && <WakeBootScreen mode={screen} />}
    </>
  );
}
