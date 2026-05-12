"use client";

import { Brain, CalendarDays, Cpu, FileSearch, Folder, GraduationCap, ListTodo, MessageSquare, Mic, Search, SlidersHorizontal } from "lucide-react";
import { LanguageSelector } from "./LanguageSelector";
import { ModelSwitcher } from "./ModelSwitcher";

const nav = [
  { label: "Chat", icon: MessageSquare },
  { label: "Schedule", icon: CalendarDays },
  { label: "Files", icon: Folder },
  { label: "PC Manager", icon: SlidersHorizontal },
  { label: "Tasks", icon: ListTodo },
  { label: "System", icon: Cpu },
  { label: "Learning", icon: GraduationCap }
];

const capabilities = [
  { label: "Memory", icon: Brain },
  { label: "File search", icon: Search },
  { label: "Calendar", icon: CalendarDays },
  { label: "Microphone", icon: Mic }
];

export function Sidebar({ active, onSelect }: { active: string; onSelect: (label: string) => void }) {
  return (
    <aside className="wayne-border flex min-h-0 min-w-0 flex-col overflow-hidden bg-surface">
      <div className="shrink-0 border-b border-cyan/20 p-4 font-heading text-xl tracking-wider text-cyan">
        W.A.Y.N.E
        <div className="mt-1 font-body text-[10px] leading-4 tracking-normal text-amber">Wireless Artificial Yielding Network Engine</div>
      </div>
      <nav className="min-h-0 flex-1 overflow-y-auto py-3">
        {nav.map((item, index) => {
          const Icon = item.icon;
          const isActive = item.label === active;
          return (
            <button
              key={item.label}
              onClick={() => onSelect(item.label)}
              className={`flex w-full items-center gap-3 border-l-2 px-4 py-3 text-left text-sm ${
                isActive ? "border-cyan bg-panel text-cyan" : "border-transparent text-cyan/70 hover:bg-panel/70"
              }`}
            >
              <Icon size={17} />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="shrink-0 border-t border-cyan/20 p-4">
        <ModelSwitcher />
        <div className="mt-3">
          <LanguageSelector />
        </div>
        <div className="my-4 h-px bg-cyan/10" />
        <div className="mb-3 text-xs uppercase text-cyan/60">Capabilities</div>
        <div className="space-y-2">
          {capabilities.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.label} className="flex items-center gap-2 text-xs text-cyan/80">
                <Icon size={14} />
                {item.label}
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
