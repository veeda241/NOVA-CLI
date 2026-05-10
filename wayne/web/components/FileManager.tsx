"use client";

import { File, Folder, Info, Play, RefreshCw, Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

type Entry = { name?: string; path?: string; is_dir?: boolean; size?: number; file_size?: number; modified?: string; type?: string; file_path?: string; file_name?: string; file_type?: string; is_restricted?: boolean };

export function FileManager() {
  const [path, setPath] = useState("C:\\");
  const [query, setQuery] = useState("");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [preview, setPreview] = useState<string>("");
  const [status, setStatus] = useState("");
  const [indexStatus, setIndexStatus] = useState<{ indexed: number; is_running: boolean } | null>(null);

  async function loadDirectory(target = path) {
    setStatus("Loading folder...");
    const res = await fetch(`${API_URL}/files/list?path=${encodeURIComponent(target)}`);
    const data = await res.json();
    setEntries(data.entries || []);
    setPath(data.path || target);
    setStatus(data.error || `${data.count || 0} entries`);
  }

  async function searchFiles() {
    if (!query.trim()) return loadDirectory();
    setStatus("Searching index...");
    const res = await fetch(`${API_URL}/files/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, max_results: 50 })
    });
    const data = await res.json();
    setEntries(data.results || []);
    setStatus(`${data.count || 0} search results`);
  }

  async function readFile(item: Entry) {
    const filePath = item.path || item.file_path || "";
    if (item.is_dir) return loadDirectory(filePath);
    setPreview("Reading...");
    const res = await fetch(`${API_URL}/files/read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: filePath })
    });
    const data = await res.json();
    setPreview(data.content || data.error || JSON.stringify(data, null, 2));
  }

  async function openFile(item: Entry) {
    const filePath = item.path || item.file_path || "";
    await fetch(`${API_URL}/files/open`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: filePath }) });
  }

  async function deleteFile(item: Entry) {
    const filePath = item.path || item.file_path || "";
    if (!window.confirm(`Delete ${filePath}?`)) return;
    const res = await fetch(`${API_URL}/files/operation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operation: "delete", path: filePath, confirmed: true })
    });
    const data = await res.json();
    setStatus(data.success ? "Deleted" : data.error || data.message || "Delete failed");
    loadDirectory();
  }

  async function startIndex() {
    await fetch(`${API_URL}/indexer/start`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ roots: [path] }) });
    refreshIndex();
  }

  async function refreshIndex() {
    const res = await fetch(`${API_URL}/indexer/status`);
    setIndexStatus(await res.json());
  }

  useEffect(() => {
    loadDirectory().catch(() => setStatus("Folder unavailable"));
    refreshIndex().catch(() => null);
  }, []);

  return (
    <section className="wayne-border grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_360px] gap-3 bg-surface/80 p-4">
      <div className="min-h-0">
        <div className="mb-3 flex flex-wrap gap-2">
          <input value={path} onChange={(e) => setPath(e.target.value)} className="wayne-border min-w-0 flex-1 bg-background px-3 py-2 text-sm text-cyan outline-none" />
          <button onClick={() => loadDirectory()} className="wayne-border bg-panel px-3 text-cyan"><RefreshCw size={16} /></button>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search all indexed files..." className="wayne-border min-w-[220px] bg-background px-3 py-2 text-sm text-cyan outline-none" />
          <button onClick={searchFiles} className="wayne-border flex items-center gap-2 bg-cyan/10 px-3 text-cyan"><Search size={16} /> Search</button>
          <button onClick={startIndex} className="wayne-border bg-panel px-3 text-xs text-amber">Index This Folder</button>
        </div>
        <div className="mb-2 flex justify-between text-xs text-cyan/60">
          <span>{status}</span>
          <span>Indexed {indexStatus?.indexed ?? 0} this run {indexStatus?.is_running ? "· running" : ""}</span>
        </div>
        <div className="wayne-border max-h-[calc(100vh-280px)] overflow-y-auto bg-background">
          {entries.map((item) => {
            const isDir = item.is_dir;
            const filePath = item.path || item.file_path || "";
            const name = item.name || item.file_name || filePath;
            const Icon = isDir ? Folder : File;
            return (
              <div key={filePath} className="grid grid-cols-[24px_1fr_90px_120px_120px] items-center gap-2 border-b border-cyan/10 px-3 py-2 text-xs hover:bg-panel">
                <Icon size={15} className={isDir ? "text-amber" : item.is_restricted ? "text-danger" : "text-cyan"} />
                <button onClick={() => readFile(item)} className="truncate text-left text-cyan">{name}</button>
                <span className="text-cyan/50">{item.type || item.file_type || "-"}</span>
                <span className="text-cyan/50">{item.size || item.file_size || 0} B</span>
                <div className="flex gap-1 justify-end">
                  <button onClick={() => openFile(item)} title="Open" className="wayne-border p-1 text-success"><Play size={12} /></button>
                  <button onClick={() => readFile(item)} title="Info/Preview" className="wayne-border p-1 text-cyan"><Info size={12} /></button>
                  <button onClick={() => deleteFile(item)} title="Delete" className="wayne-border p-1 text-danger"><Trash2 size={12} /></button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <aside className="wayne-border min-h-0 overflow-y-auto bg-background p-3">
        <h2 className="mb-3 font-heading text-cyan">Preview</h2>
        <pre className="whitespace-pre-wrap break-words text-xs leading-5 text-cyan/80">{preview || "Select a file to preview its contents."}</pre>
      </aside>
    </section>
  );
}
