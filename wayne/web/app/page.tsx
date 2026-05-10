"use client";

import { Activity, CalendarPlus, Clock3, Cpu, FileSearch, ListTodo, MessageSquare, SunMedium } from "lucide-react";
import { useEffect, useState } from "react";
import { CalendarPanel } from "../components/CalendarPanel";
import { ChatWindow } from "../components/ChatWindow";
import { FilePanel } from "../components/FilePanel";
import { InputBar } from "../components/InputBar";
import { Sidebar } from "../components/Sidebar";
import { TaskPanel } from "../components/TaskPanel";
import { DevicePanel } from "../components/DevicePanel";
import { TrackingView } from "../components/TrackingView";
import { LearningDashboard } from "../components/LearningDashboard";
import { FileManager } from "../components/FileManager";
import { PCManagerPanel } from "../components/PCManagerPanel";
import { useWayne } from "../hooks/useWayne";

const quickActions = [
  { label: "Ask W.A.Y.N.E", icon: MessageSquare },
  { label: "Schedule Meeting", icon: CalendarPlus },
  { label: "Open File", icon: FileSearch },
  { label: "View Tasks", icon: ListTodo },
  { label: "System Status", icon: Cpu },
  { label: "Summarize Day", icon: SunMedium }
];

const tickers = ["W.A.Y.N.E: NOMINAL", "QWEN 2.5: RUNNING LOCAL", "NETWORK: SECURED", "DEVICES: TRACKED", "MEMORY: INDEXED"];

function Header() {
  const [clock, setClock] = useState("");
  useEffect(() => {
    const update = () => setClock(new Date().toLocaleTimeString());
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="wayne-border flex h-16 items-center justify-between bg-surface px-5">
      <div>
        <div className="font-heading text-2xl tracking-[4px] text-cyan">W.A.Y.N.E</div>
        <div className="mt-0.5 text-[10px] text-amber">Wireless Artificial Yielding Network Engine</div>
      </div>
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2 text-cyan/80">
          <Clock3 size={16} />
          {clock}
        </div>
        <div className="flex items-center gap-2 text-success">
          <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-success" />
          ONLINE
        </div>
      </div>
    </header>
  );
}

function Footer() {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setIndex((current) => (current + 1) % tickers.length), 3000);
    return () => clearInterval(id);
  }, []);
  return (
    <footer className="wayne-border flex h-10 items-center justify-between bg-surface px-5 text-xs text-cyan/60">
      <span>W.A.Y.N.E v1.0 — Wireless Artificial Yielding Network Engine</span>
      <span className="text-cyan">{tickers[index]}</span>
      <span>QWEN 2.5 LOCAL ENGINE</span>
    </footer>
  );
}

export default function Home() {
  const { messages, loading, toast, sendMessage, setToast } = useWayne();
  const [activeView, setActiveView] = useState("Chat");

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 3500);
    return () => clearTimeout(id);
  }, [toast, setToast]);

  return (
    <main className="h-screen overflow-hidden bg-background p-3 text-cyan-50">
      <div className="grid h-full grid-rows-[64px_1fr_40px] gap-3">
        <Header />
        <div className="grid min-h-0 grid-cols-[220px_minmax(0,1fr)_200px] gap-3">
          <Sidebar active={activeView} onSelect={setActiveView} />
          <section className="flex min-h-0 flex-col gap-3">
            {activeView === "Learning" ? (
              <LearningDashboard />
            ) : activeView === "Files" ? (
              <FileManager />
            ) : activeView === "PC Manager" ? (
              <PCManagerPanel />
            ) : (
              <>
                <div className="wayne-border flex flex-wrap items-center gap-2 bg-surface p-3">
                  {quickActions.map((action) => {
                    const Icon = action.icon;
                    return (
                      <button key={action.label} onClick={() => sendMessage(action.label)} className="wayne-border flex items-center gap-2 bg-panel px-3 py-2 text-xs text-cyan hover:border-cyan">
                        <Icon size={15} />
                        {action.label}
                      </button>
                    );
                  })}
                  <div className="ml-auto flex items-center gap-2 text-xs text-success">
                    <Activity size={15} />
                    Shared backend active
                  </div>
                </div>
                <ChatWindow messages={messages} loading={loading} />
                <TrackingView />
                <InputBar onSend={sendMessage} loading={loading} />
              </>
            )}
          </section>
          <aside className="wayne-border flex min-h-0 flex-col gap-5 overflow-y-auto bg-surface p-3">
            <TaskPanel />
            <CalendarPanel />
            <FilePanel onOpenFile={(path) => sendMessage(`Open file: ${path}`)} />
            <DevicePanel />
          </aside>
        </div>
        <Footer />
      </div>
      {toast && (
        <div className="wayne-border fixed bottom-16 right-5 bg-panel px-4 py-3 text-sm text-amber shadow-lg shadow-cyan/10">
          {toast}
        </div>
      )}
    </main>
  );
}
