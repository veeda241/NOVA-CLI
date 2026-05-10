"use client";

import { Plus, Trash2 } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";

type Task = { id: number; title: string; priority: "low" | "medium" | "high"; completed: boolean };
const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

export function TaskPanel() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [title, setTitle] = useState("");

  async function fetchTasks() {
    const response = await fetch(`${API_URL}/tasks`, { cache: "no-store" });
    setTasks(await response.json());
  }

  useEffect(() => {
    fetchTasks().catch(() => setTasks([]));
  }, []);

  async function addTask(event: FormEvent) {
    event.preventDefault();
    if (!title.trim()) return;
    await fetch(`${API_URL}/tasks`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title, priority: "medium" }) });
    setTitle("");
    await fetchTasks();
  }

  async function toggle(id: number) {
    await fetch(`${API_URL}/tasks/${id}`, { method: "PATCH" });
    await fetchTasks();
  }

  async function remove(id: number) {
    await fetch(`${API_URL}/tasks/${id}`, { method: "DELETE" });
    await fetchTasks();
  }

  return (
    <section className="min-h-0">
      <h2 className="mb-2 font-heading text-sm text-cyan">Tasks</h2>
      <div className="space-y-2">
        {tasks.slice(0, 6).map((task) => (
          <div key={task.id} className="wayne-border flex items-center gap-2 bg-panel p-2 text-xs">
            <input type="checkbox" checked={task.completed} onChange={() => toggle(task.id)} className="accent-cyan" />
            <span className={`min-w-0 flex-1 truncate ${task.completed ? "text-cyan/40 line-through" : ""}`}>{task.title}</span>
            <span className={`${task.priority === "high" ? "text-danger" : task.priority === "medium" ? "text-amber" : "text-success"}`}>{task.priority}</span>
            <button onClick={() => remove(task.id)} className="text-cyan/60 hover:text-danger" title="Delete task">
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
      <form onSubmit={addTask} className="mt-2 flex gap-2">
        <input value={title} onChange={(event) => setTitle(event.target.value)} className="wayne-border min-w-0 flex-1 bg-background px-2 py-2 text-xs outline-none focus:border-cyan" placeholder="New task" />
        <button className="wayne-border bg-cyan/10 px-2 text-cyan" title="Add task">
          <Plus size={16} />
        </button>
      </form>
    </section>
  );
}
