"use client";

import { useEffect, useState } from "react";

type EventItem = { id?: string; title: string; start: string; end: string };
const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";
const colors = ["border-cyan", "border-amber", "border-success"];

export function CalendarPanel() {
  const [events, setEvents] = useState<EventItem[]>([]);

  useEffect(() => {
    fetch(`${API_URL}/events/today`, { cache: "no-store" })
      .then((response) => response.json())
      .then(setEvents)
      .catch(() => setEvents([]));
  }, []);

  return (
    <section>
      <h2 className="mb-2 font-heading text-sm text-cyan">Today</h2>
      <div className="space-y-2">
        {events.length === 0 && <div className="wayne-border bg-panel p-2 text-xs text-cyan/50">No events detected</div>}
        {events.slice(0, 5).map((event, index) => (
          <div key={event.id || `${event.start}-${index}`} className={`border-l-2 ${colors[index % colors.length]} bg-panel p-2 text-xs`}>
            <div className="text-cyan/70">{event.start?.slice(11, 16) || "All day"}</div>
            <div className="truncate text-white">{event.title}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
