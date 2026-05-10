import Foundation

enum Config {
    static let ollamaModel = "qwen2.5:1.5b"
    static let backendURL = URL(string: "http://192.168.31.141:8000")!
    static let backendWebSocketURL = URL(string: "ws://192.168.31.141:8000")!
    static let phoneDeviceId = "iphone-001"
}
