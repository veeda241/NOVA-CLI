"use client";

import { CalendarDays, Clock3, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

type DatePayload = {
  time_12h: string;
  day_name: string;
  month_name: string;
  day_number: number;
  year: number;
  week_number: number;
  day_of_year: number;
};

type SpecialDay = {
  name: string;
  formatted_date?: string;
  days_away?: number;
};

export function DateTimeWidget() {
  const [clock, setClock] = useState(new Date());
  const [dateInfo, setDateInfo] = useState<DatePayload | null>(null);
  const [today, setToday] = useState<SpecialDay[]>([]);
  const [upcoming, setUpcoming] = useState<SpecialDay | null>(null);

  useEffect(() => {
    const timer = window.setInterval(() => setClock(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const [dateRes, todayRes, upcomingRes] = await Promise.all([
          fetch(`${API_URL}/datetime`),
          fetch(`${API_URL}/special-days/today?country=IN`),
          fetch(`${API_URL}/special-days/upcoming?days=7&country=IN`),
        ]);
        if (!active) return;
        const dateJson = await dateRes.json();
        const todayJson = await todayRes.json();
        const upcomingJson = await upcomingRes.json();
        setDateInfo(dateJson);
        setToday(todayJson.special_days || []);
        setUpcoming(Array.isArray(upcomingJson) && upcomingJson.length ? upcomingJson[0] : null);
      } catch {
        setDateInfo(null);
      }
    }
    load();
    const timer = window.setInterval(load, 60_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const fallbackDate = clock.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" });

  return (
    <section className="wayne-border bg-panel p-3">
      <div className="mb-2 flex items-center gap-2 text-xs uppercase text-cyan/70">
        <Clock3 size={14} />
        Live Clock
      </div>
      <div className="font-heading text-xl text-cyan">{dateInfo?.time_12h || clock.toLocaleTimeString()}</div>
      <div className="mt-1 flex items-start gap-2 text-xs leading-5 text-amber">
        <CalendarDays size={14} className="mt-0.5 shrink-0" />
        <span>
          {dateInfo ? `${dateInfo.day_name}, ${dateInfo.month_name} ${dateInfo.day_number}, ${dateInfo.year}` : fallbackDate}
        </span>
      </div>
      {dateInfo && <div className="mt-2 text-[11px] text-cyan/50">Week {dateInfo.week_number} | Day {dateInfo.day_of_year}</div>}
      {today.length > 0 && (
        <div className="mt-3 rounded border border-cyan/30 bg-cyan/10 px-2 py-1 text-[11px] text-cyan">
          <Sparkles size={12} className="mr-1 inline" />
          {today.map((item) => item.name).join(", ")}
        </div>
      )}
      {upcoming && (
        <div className="mt-2 text-[11px] text-cyan/55">
          Next: {upcoming.name} in {upcoming.days_away} day{upcoming.days_away === 1 ? "" : "s"}
        </div>
      )}
    </section>
  );
}
