"use client";

type Listener = (data: any) => void;

class WAYNEConnectionManager {
  private ws: WebSocket | null = null;
  private endpoint = "";
  private reconnectAttempts = 0;
  private reconnectDelay = 1000;
  private readonly maxReconnectDelay = 30000;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private healthInterval: ReturnType<typeof setInterval> | null = null;
  private messageQueue: any[] = [];
  private listeners = new Map<string, Listener[]>();
  private online = false;

  constructor(private sessionId: string) {
    if (typeof window !== "undefined") {
      window.addEventListener("online", () => this.reconnect());
      window.addEventListener("offline", () => this.handleOffline());
      this.startHealthCheck();
    }
  }

  connect(endpoint: string): Promise<void> {
    this.endpoint = endpoint;
    return new Promise((resolve, reject) => {
      const url = `ws://localhost:8000${endpoint}`;
      const socket = new WebSocket(url);
      this.ws = socket;

      socket.onopen = () => {
        this.online = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.startPing();
        this.flushQueue();
        this.emit("connected", { sessionId: this.sessionId });
        resolve();
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "ping") {
            this.send({ type: "pong", timestamp: Date.now() });
            return;
          }
          this.emit(data.type, data);
        } catch {
          this.emit("message", event.data);
        }
      };

      socket.onclose = (event) => {
        this.online = false;
        this.stopPing();
        this.emit("disconnected", { code: event.code });
        if (!event.wasClean) this.scheduleReconnect();
      };

      socket.onerror = () => {
        this.online = false;
        this.emit("error", {});
        reject(new Error("WebSocket connection failed"));
      };
    });
  }

  send(data: any): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      return true;
    }
    this.messageQueue.push(data);
    return false;
  }

  on(event: string, callback: Listener) {
    this.listeners.set(event, [...(this.listeners.get(event) ?? []), callback]);
  }

  disconnect() {
    this.stopPing();
    if (this.healthInterval) clearInterval(this.healthInterval);
    this.ws?.close(1000, "Client disconnecting");
  }

  reconnect() {
    if (!this.endpoint) return;
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) return;
    void this.connect(this.endpoint).catch(() => this.scheduleReconnect());
  }

  getStatus() {
    return {
      connected: this.online,
      attempts: this.reconnectAttempts,
      queued: this.messageQueue.length,
      readyState: this.ws?.readyState
    };
  }

  private scheduleReconnect() {
    this.reconnectAttempts += 1;
    const delay = Math.min(this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1), this.maxReconnectDelay);
    this.emit("reconnecting", { attempt: this.reconnectAttempts, delay });
    window.setTimeout(() => this.reconnect(), delay);
  }

  private handleOffline() {
    this.online = false;
    this.emit("offline", {});
  }

  private startPing() {
    this.stopPing();
    this.pingInterval = setInterval(() => {
      this.send({ type: "ping", timestamp: Date.now() });
    }, 5000);
  }

  private stopPing() {
    if (this.pingInterval) clearInterval(this.pingInterval);
    this.pingInterval = null;
  }

  private startHealthCheck() {
    this.healthInterval = setInterval(async () => {
      try {
        const response = await fetch("http://localhost:8000/health", {
          cache: "no-store",
          signal: AbortSignal.timeout(3000)
        });
        const data = await response.json();
        this.emit("health", data);
        if (!this.online) this.reconnect();
      } catch {
        this.emit("health", { status: "offline" });
      }
    }, 3000);
  }

  private flushQueue() {
    while (this.messageQueue.length > 0) {
      const next = this.messageQueue.shift();
      this.send(next);
    }
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach((callback) => callback(data));
  }
}

export const createConnectionManager = (sessionId: string) => new WAYNEConnectionManager(sessionId);
