"use client";

import { useEffect, useState } from "react";

type Device = {
  device_id: string;
  device_name: string;
  battery_level: number;
  cpu_percent: number;
  ram_percent: number;
  is_online: boolean;
  last_seen: string;
  ip_address?: string;
  latitude?: number;
  longitude?: number;
};

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace("http://", "ws://").replace("https://", "wss://");

export function TrackingView() {
  const [devices, setDevices] = useState<Record<string, Device>>({});

  useEffect(() => {
    fetch(`${API_URL}/device/status`)
      .then((response) => response.json())
      .then((data) => {
        const mapped: Record<string, Device> = {};
        for (const device of data.devices || []) mapped[device.device_id] = device;
        setDevices(mapped);
      })
      .catch(() => undefined);

    const socket = new WebSocket(`${WS_URL}/ws/track/web-dashboard`);
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.device) setDevices((current) => ({ ...current, [data.device.device_id]: data.device }));
    };
    return () => socket.close();
  }, []);

  return (
    <section className="wayne-border bg-surface p-3">
      <h2 className="mb-2 font-heading text-sm text-cyan">Tracking</h2>
      <table className="w-full text-left text-xs">
        <thead className="text-cyan/60">
          <tr>
            <th className="py-1">Device</th>
            <th>Battery</th>
            <th>CPU</th>
            <th>RAM</th>
            <th>Online</th>
            <th>Last Seen</th>
            <th>Location</th>
          </tr>
        </thead>
        <tbody>
          {Object.values(devices).map((device) => (
            <tr key={device.device_id} className="border-t border-cyan/10">
              <td className="py-1 text-white">{device.device_name}</td>
              <td>{device.battery_level}%</td>
              <td>{device.cpu_percent}%</td>
              <td>{device.ram_percent}%</td>
              <td className={device.is_online ? "text-success" : "text-danger"}>{device.is_online ? "Online" : "Offline"}</td>
              <td>{device.last_seen}</td>
              <td>{device.latitude && device.longitude ? `${device.latitude}, ${device.longitude}` : device.ip_address || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
