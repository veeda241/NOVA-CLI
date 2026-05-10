import Foundation

@MainActor
final class WebSocketService: ObservableObject {
    @Published var deviceUpdates: [String: DeviceStatus] = [:]
    private var trackTask: URLSessionWebSocketTask?
    private var commandTask: URLSessionWebSocketTask?

    func connect(deviceId: String = Config.phoneDeviceId) {
        connectTracking(deviceId: deviceId)
        connectCommands(deviceId: deviceId)
    }

    private func connectTracking(deviceId: String) {
        trackTask?.cancel(with: .goingAway, reason: nil)
        trackTask = URLSession.shared.webSocketTask(with: Config.backendWebSocketURL.appendingPathComponent("ws/track/\(deviceId)"))
        trackTask?.resume()
        receiveTracking()
    }

    private func connectCommands(deviceId: String) {
        commandTask?.cancel(with: .goingAway, reason: nil)
        commandTask = URLSession.shared.webSocketTask(with: Config.backendWebSocketURL.appendingPathComponent("ws/commands/\(deviceId)"))
        commandTask?.resume()
        receiveCommands(deviceId: deviceId)
    }

    private func receiveTracking() {
        trackTask?.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                if case .success(let message) = result, case .string(let text) = message, let data = text.data(using: .utf8) {
                    if let envelope = try? JSONDecoder().decode(DeviceEnvelope.self, from: data) {
                        self.deviceUpdates[envelope.device.deviceId] = envelope.device
                    }
                }
                self.receiveTracking()
            }
        }
    }

    private func receiveCommands(deviceId: String) {
        commandTask?.receive { [weak self] _ in
            Task { @MainActor in
                self?.receiveCommands(deviceId: deviceId)
            }
        }
    }

    func sendPhoneStatus(battery: Int = 100) {
        let payload: [String: Any] = ["battery": battery, "online": true, "type": "iphone"]
        guard let data = try? JSONSerialization.data(withJSONObject: payload), let text = String(data: data, encoding: .utf8) else { return }
        trackTask?.send(.string(text)) { _ in }
    }

    func disconnect() {
        trackTask?.cancel(with: .goingAway, reason: nil)
        commandTask?.cancel(with: .goingAway, reason: nil)
    }
}

private struct DeviceEnvelope: Decodable {
    let type: String
    let device: DeviceStatus
}
