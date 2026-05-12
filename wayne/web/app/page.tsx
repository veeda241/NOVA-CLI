"use client";

import { Activity, CalendarPlus, Clock3, Cpu, FileSearch, ListTodo, MessageSquare, SunMedium } from "lucide-react";
import { useEffect, useState } from "react";
import { CalendarPanel } from "../components/CalendarPanel";
import { ChatWindow } from "../components/ChatWindow";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { DateTimeWidget } from "../components/DateTimeWidget";
import { FilePanel } from "../components/FilePanel";
import { InputBar } from "../components/InputBar";
import { KnowledgePanel } from "../components/KnowledgePanel";
import { Sidebar } from "../components/Sidebar";
import { TaskPanel } from "../components/TaskPanel";
import { DevicePanel } from "../components/DevicePanel";
import { TrackingView } from "../components/TrackingView";
import { LearningDashboard } from "../components/LearningDashboard";
import { FileManager } from "../components/FileManager";
import { PCManagerPanel } from "../components/PCManagerPanel";
import { useWayne } from "../hooks/useWayne";
import { useConnection } from "../hooks/useConnection";

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
    <header className="wayne-border flex h-16 shrink-0 items-center justify-between bg-surface px-5">
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
    <footer className="wayne-border flex h-10 shrink-0 items-center justify-between gap-4 bg-surface px-5 text-xs text-cyan/60">
      <span>W.A.Y.N.E v1.0 — Wireless Artificial Yielding Network Engine</span>
      <span className="text-cyan">{tickers[index]}</span>
      <span>QWEN 2.5 LOCAL ENGINE</span>
    </footer>
  );
}

export default function Home() {
  const { messages, loading, streaming, toast, sendMessage, setToast } = useWayne();
  const connection = useConnection("web");
  const [activeView, setActiveView] = useState("Chat");

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 3500);
    return () => clearTimeout(id);
  }, [toast, setToast]);

  return (
    <main className="h-dvh min-h-0 overflow-hidden bg-background p-3 text-cyan-50">
      <div className="grid h-full min-h-0 grid-rows-[36px_64px_minmax(0,1fr)_40px] gap-3">
        <ConnectionStatus status={connection.status} health={connection.health} reconnectAttempt={connection.reconnectAttempt} onRetry={connection.retry} />
        <Header />
        <div className="grid min-h-0 grid-cols-[240px_minmax(520px,1fr)_260px] gap-3 overflow-hidden">
          <Sidebar active={activeView} onSelect={setActiveView} />
          <section className="flex min-h-0 min-w-0 flex-col gap-3 overflow-hidden">
            {activeView === "Learning" ? (
              <LearningDashboard />
            ) : activeView === "Files" ? (
              <FileManager />
            ) : activeView === "PC Manager" ? (
              <PCManagerPanel />
            ) : (
              <>
                <div className="wayne-border grid shrink-0 grid-cols-[repeat(auto-fit,minmax(150px,1fr))] items-center gap-2 bg-surface p-3">
                  {quickActions.map((action) => {
                    const Icon = action.icon;
                    return (
                      <button key={action.label} onClick={() => sendMessage(action.label)} className="wayne-border flex h-10 items-center justify-center gap-2 bg-panel px-3 text-xs text-cyan hover:border-cyan">
                        <Icon size={15} />
                        {action.label}
                      </button>
                    );
                  })}
                  <div className="flex h-10 items-center justify-center gap-2 text-xs text-success">
                    <Activity size={15} />
                    Shared backend active
                  </div>
                </div>
                <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
                  <ChatWindow messages={messages} loading={loading} streaming={streaming} />
                  <TrackingView />
                </div>
                <InputBar onSend={sendMessage} loading={loading} />
              </>
            )}
          </section>
          <aside className="wayne-border flex min-h-0 min-w-0 flex-col gap-4 overflow-y-auto bg-surface p-3">
            <DateTimeWidget />
            <KnowledgePanel />
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
