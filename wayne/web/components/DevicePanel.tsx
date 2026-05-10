"use client";

import { Laptop, Moon, Power, RefreshCw, Smartphone, VolumeX } from "lucide-react";
import { useEffect, useState } from "react";

type Device = {
  device_id: string;
  device_name: string;
  device_type: string;
  battery_level: number;
  is_online: boolean;
  last_seen: string;
};

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

export function DevicePanel() {
  const [devices, setDevices] = useState<Device[]>([]);
  const [pending, setPending] = useState<{ device: Device; command: string } | null>(null);

  async function fetchDevices() {
    const response = await fetch(`${API_URL}/device/status`, { cache: "no-store" });
    const data = await response.json();
    setDevices(data.devices || []);
  }

  useEffect(() => {
    fetchDevices().catch(() => setDevices([]));
    const id = setInterval(() => fetchDevices().catch(() => setDevices([])), 10000);
    return () => clearInterval(id);
  }, []);

  async function confirmCommand() {
    if (!pending) return;
    await fetch(`${API_URL}/device/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: pending.device.device_id, command: pending.command, confirmed: true, issued_by: "web" })
    });
    setPending(null);
    await fetchDevices();
  }

  const commands = [
    { label: "Sleep", value: "sleep", icon: Moon },
    { label: "Lock", value: "lock", icon: Power },
    { label: "Restart", value: "restart", icon: RefreshCw },
    { label: "Shutdown", value: "shutdown", icon: Power }
  ];

  return (
    <section>
      <h2 className="mb-2 font-heading text-sm text-cyan">Devices</h2>
      <div className="space-y-2">
        {devices.map((device) => {
          const Icon = device.device_type === "iphone" ? Smartphone : Laptop;
          return (
            <div key={device.device_id} className="wayne-border bg-panel p-3 text-xs">
              <div className="flex items-center gap-2">
                <Icon size={16} className="text-cyan" />
                <span className="min-w-0 flex-1 truncate text-white">{device.device_name}</span>
                <span className={`h-2 w-2 rounded-full ${device.is_online ? "bg-success" : "bg-danger"}`} />
              </div>
              <div className="mt-2 h-2 bg-background">
                <div className="h-full bg-cyan" style={{ width: `${Math.max(0, Math.min(device.battery_level, 100))}%` }} />
              </div>
              <div className="mt-1 text-cyan/60">{device.battery_level}% | {device.last_seen}</div>
              <div className="mt-2 grid grid-cols-2 gap-1">
                {(device.device_type === "iphone" ? [{ label: "Lock", value: "lock", icon: Power }, { label: "Mute", value: "mute", icon: VolumeX }] : commands).map((command) => {
                  const CommandIcon = command.icon;
                  return (
                    <button key={command.value} onClick={() => setPending({ device, command: command.value })} className="wayne-border flex items-center justify-center gap-1 bg-background py-1 text-cyan hover:border-cyan">
                      <CommandIcon size={12} />
                      {command.label}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      {pending && (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/60">
          <div className="wayne-border w-80 bg-surface p-5">
            <h3 className="font-heading text-lg text-cyan">Confirm Command</h3>
            <p className="mt-3 text-sm text-cyan/80">Are you sure you want to {pending.command} your {pending.device.device_name}?</p>
            <div className="mt-5 flex justify-end gap-2">
              <button onClick={() => setPending(null)} className="wayne-border px-3 py-2 text-sm text-cyan/70">Cancel</button>
              <button onClick={confirmCommand} className="wayne-border bg-danger/20 px-3 py-2 text-sm text-danger">Confirm</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
