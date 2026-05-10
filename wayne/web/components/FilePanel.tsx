"use client";

import { File, FileCode2, FileText } from "lucide-react";
import { useEffect, useState } from "react";

type FileItem = { name: string; path: string; type: string; size: number };
const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

function iconFor(name: string) {
  const ext = name.split(".").pop()?.toLowerCase();
  if (ext === "pdf" || ext === "md") return FileText;
  if (["py", "ts", "tsx", "js"].includes(ext || "")) return FileCode2;
  return File;
}

function colorFor(name: string) {
  const ext = name.split(".").pop()?.toLowerCase();
  if (ext === "pdf") return "text-danger";
  if (ext === "py") return "text-success";
  if (ext === "ts" || ext === "tsx") return "text-cyan";
  if (ext === "md") return "text-amber";
  return "text-cyan/50";
}

export function FilePanel({ onOpenFile }: { onOpenFile: (path: string) => void }) {
  const [files, setFiles] = useState<FileItem[]>([]);

  useEffect(() => {
    fetch(`${API_URL}/files`, { cache: "no-store" })
      .then((response) => response.json())
      .then((data) => setFiles(data.files || []))
      .catch(() => setFiles([]));
  }, []);

  return (
    <section>
      <h2 className="mb-2 font-heading text-sm text-cyan">Files</h2>
      <div className="space-y-1">
        {files.slice(0, 6).map((file) => {
          const Icon = iconFor(file.name);
          return (
            <button key={file.path} onClick={() => onOpenFile(file.path)} className="flex w-full items-center gap-2 bg-panel p-2 text-left text-xs hover:bg-cyan/10">
              <Icon size={14} className={colorFor(file.name)} />
              <span className="min-w-0 flex-1 truncate">{file.name}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
