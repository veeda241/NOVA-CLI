import Combine
import Foundation
import Network

final class ConnectionService: ObservableObject {
    enum ConnectionStatus {
        case connecting, connected, disconnected, reconnecting, offline

        var label: String {
            switch self {
            case .connecting: return "Connecting..."
            case .connected: return "W.A.Y.N.E ONLINE"
            case .disconnected: return "Connection Lost"
            case .reconnecting: return "Reconnecting..."
            case .offline: return "Offline Mode"
            }
        }

        var color: String {
            switch self {
            case .connected: return "00ff88"
            case .reconnecting: return "ffaa00"
            case .connecting: return "00d4ff"
            default: return "ff4455"
            }
        }
    }

    @Published var isConnected = false
    @Published var status: ConnectionStatus = .connecting
    @Published var reconnectAttempt = 0
    @Published var latencyMs: Double = 0
    @Published var cacheHitRate: Double = 0

    private let monitor = NWPathMonitor()
    private var healthTimer: Timer?
    private var reconnectTimer: Timer?
    private var reconnectDelay: TimeInterval = 1
    private let maxDelay: TimeInterval = 30
    private var sockets: [String: URLSessionWebSocketTask] = [:]
    private var queues: [String: [[String: Any]]] = [:]

    init() {
        setupNetworkMonitor()
        startHealthCheck()
    }

    func setupNetworkMonitor() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                if path.status == .satisfied {
                    self?.reconnect()
                } else {
                    self?.isConnected = false
                    self?.status = .offline
                }
            }
        }
        monitor.start(queue: DispatchQueue.global(qos: .background))
    }

    func startHealthCheck() {
        healthTimer?.invalidate()
        healthTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            Task { await self?.checkHealth() }
        }
        Task { await checkHealth() }
    }

    @MainActor
    func reconnect() {
        reconnectAttempt = 0
        reconnectDelay = 1
        status = .connecting
        Task { await checkHealth() }
    }

    func checkHealth() async {
        let start = Date()
        do {
            let (data, response) = try await URLSession.shared.data(from: Config.backendURL.appendingPathComponent("health"))
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                await handleUnhealthy()
                return
            }
            let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let speed = json?["speed"] as? [String: Any]
            await MainActor.run {
                self.isConnected = true
                self.status = .connected
                self.reconnectAttempt = 0
                self.reconnectDelay = 1
                self.latencyMs = Date().timeIntervalSince(start) * 1000
                self.cacheHitRate = speed?["hit_rate"] as? Double ?? 0
            }
        } catch {
            await handleUnhealthy()
        }
    }

    func connectWebSocket(endpoint: String, identifier: String, onMessage: @escaping (Data) -> Void) {
        let url = Config.backendWebSocketURL.appendingPathComponent(endpoint)
        let task = URLSession.shared.webSocketTask(with: url)
        sockets[identifier] = task
        task.resume()
        receive(task: task, endpoint: endpoint, identifier: identifier, onMessage: onMessage)
        flush(identifier: identifier)
    }

    func sendWebSocket(identifier: String, data: [String: Any]) {
        guard let task = sockets[identifier], task.state == .running,
              let jsonData = try? JSONSerialization.data(withJSONObject: data),
              let jsonString = String(data: jsonData, encoding: .utf8)
        else {
            queues[identifier, default: []].append(data)
            return
        }
        task.send(.string(jsonString)) { _ in }
    }

    private func handleUnhealthy() async {
        await MainActor.run {
            isConnected = false
            status = reconnectAttempt > 0 ? .reconnecting : .disconnected
            reconnectAttempt += 1
            scheduleReconnect()
        }
    }

    private func scheduleReconnect() {
        let delay = min(reconnectDelay * pow(1.5, Double(max(0, reconnectAttempt - 1))), maxDelay)
        reconnectTimer?.invalidate()
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            Task { await self?.checkHealth() }
        }
    }

    private func receive(task: URLSessionWebSocketTask, endpoint: String, identifier: String, onMessage: @escaping (Data) -> Void) {
        task.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .data(let data):
                    onMessage(data)
                case .string(let text):
                    if text.contains("\"type\":\"ping\"") || text.contains("\"type\": \"ping\"") {
                        self?.sendWebSocket(identifier: identifier, data: ["type": "pong", "timestamp": Date().timeIntervalSince1970])
                    }
                    if let data = text.data(using: .utf8) {
                        onMessage(data)
                    }
                @unknown default:
                    break
                }
                self?.receive(task: task, endpoint: endpoint, identifier: identifier, onMessage: onMessage)
            case .failure:
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    self?.connectWebSocket(endpoint: endpoint, identifier: identifier, onMessage: onMessage)
                }
            }
        }
    }

    private func flush(identifier: String) {
        for message in queues[identifier] ?? [] {
            sendWebSocket(identifier: identifier, data: message)
        }
        queues[identifier] = []
    }
}
