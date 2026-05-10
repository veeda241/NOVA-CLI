"use client";

import { Cpu, HardDrive, Network, RefreshCw, Trash2, Zap } from "lucide-react";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

type ProcessRow = { pid: number; name: string; cpu_percent: number; memory_percent: number; status: string };

export function PCManagerPanel() {
  const [status, setStatus] = useState<any>(null);
  const [processes, setProcesses] = useState<ProcessRow[]>([]);
  const [startup, setStartup] = useState<any[]>([]);
  const [message, setMessage] = useState("");

  async function load() {
    const [pc, ps, startupRes] = await Promise.all([
      fetch(`${API_URL}/pc/status`).then((r) => r.json()),
      fetch(`${API_URL}/pc/processes?sort_by=memory`).then((r) => r.json()),
      fetch(`${API_URL}/pc/startup`).then((r) => r.json())
    ]);
    setStatus(pc);
    setProcesses(ps.processes || []);
    setStartup(startupRes.programs || []);
  }

  async function post(path: string, body: any = { confirmed: true }) {
    const res = await fetch(`${API_URL}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    setMessage(data.message || data.error || data.total_freed_human || data.freed_human || (data.success ? "Done" : "Action returned"));
    load();
  }

  async function killProcess(row: ProcessRow) {
    if (!window.confirm(`Kill ${row.name} (${row.pid})?`)) return;
    await post("/pc/processes/kill", { pid: row.pid, confirmed: true });
  }

  useEffect(() => {
    load().catch(() => setMessage("PC manager unavailable"));
  }, []);

  const memory = status?.memory;
  const disks = status?.disk?.partitions || [];

  return (
    <section className="wayne-border min-h-0 flex-1 overflow-y-auto bg-surface/80 p-4">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="font-heading text-xl text-cyan">PC Manager</h1>
        <button onClick={load} className="wayne-border bg-panel p-2 text-cyan"><RefreshCw size={16} /></button>
      </div>
      {message && <div className="wayne-border mb-4 bg-panel p-3 text-sm text-amber">{message}</div>}
      <div className="grid gap-4 lg:grid-cols-4">
        <button onClick={() => post("/pc/cache/clear")} className="wayne-border bg-panel p-4 text-left">
          <Trash2 className="mb-2 text-success" size={20} />
          <div className="text-cyan">Clear Cache</div>
          <div className="text-xs text-cyan/50">Temp, browser, DNS, thumbnails</div>
        </button>
        <button onClick={() => post("/pc/memory/optimize")} className="wayne-border bg-panel p-4 text-left">
          <Cpu className="mb-2 text-cyan" size={20} />
          <div className="text-cyan">Memory {memory?.percent ?? 0}%</div>
          <div className="text-xs text-cyan/50">{memory?.available_human || "-"} available</div>
        </button>
        <button onClick={() => post("/pc/disk/cleanup")} className="wayne-border bg-panel p-4 text-left">
          <HardDrive className="mb-2 text-amber" size={20} />
          <div className="text-cyan">Disk Cleanup</div>
          <div className="text-xs text-cyan/50">{disks[0]?.free || "-"} free</div>
        </button>
        <button onClick={() => post("/pc/dns/flush", {})} className="wayne-border bg-panel p-4 text-left">
          <Network className="mb-2 text-success" size={20} />
          <div className="text-cyan">Flush DNS</div>
          <div className="text-xs text-cyan/50">Network cache</div>
        </button>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Disks</h2>
          <div className="space-y-3">
            {disks.map((disk: any) => (
              <div key={disk.mountpoint}>
                <div className="mb-1 flex justify-between text-xs"><span>{disk.device || disk.mountpoint}</span><span>{disk.percent}%</span></div>
                <div className="h-2 bg-background"><div className="h-full bg-cyan" style={{ width: `${disk.percent}%` }} /></div>
              </div>
            ))}
          </div>
        </div>
        <div className="wayne-border bg-panel p-4">
          <h2 className="mb-3 text-cyan">Performance Mode</h2>
          <div className="flex gap-2">
            {["performance", "balanced", "powersaver"].map((mode) => (
              <button key={mode} onClick={() => post("/pc/performance", { mode, confirmed: true })} className="wayne-border flex items-center gap-2 bg-background px-3 py-2 text-xs text-cyan"><Zap size={13} />{mode}</button>
            ))}
          </div>
        </div>
      </div>
      <div className="wayne-border mt-4 bg-panel p-4">
        <h2 className="mb-3 text-cyan">Top Processes</h2>
        <div className="max-h-72 overflow-y-auto">
          {processes.slice(0, 30).map((row) => (
            <div key={row.pid} className="grid grid-cols-[70px_1fr_70px_70px_70px] gap-2 border-b border-cyan/10 py-2 text-xs">
              <span className="text-cyan/60">{row.pid}</span>
              <span className="truncate text-cyan">{row.name}</span>
              <span>{Math.round(row.cpu_percent || 0)}% CPU</span>
              <span>{Math.round(row.memory_percent || 0)}% RAM</span>
              <button onClick={() => killProcess(row)} className="wayne-border text-danger">Kill</button>
            </div>
          ))}
        </div>
      </div>
      <div className="wayne-border mt-4 bg-panel p-4">
        <h2 className="mb-3 text-cyan">Startup Programs</h2>
        <div className="grid gap-2 md:grid-cols-2">
          {startup.slice(0, 20).map((item) => (
            <div key={`${item.name}-${item.path}`} className="wayne-border bg-background p-2 text-xs">
              <div className="truncate text-cyan">{item.name}</div>
              <div className="truncate text-cyan/50">{item.path}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
