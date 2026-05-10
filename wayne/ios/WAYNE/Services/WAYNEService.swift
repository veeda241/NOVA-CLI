import Foundation
#if os(iOS)
import UIKit
#endif

final class WAYNEService {
    static let shared = WAYNEService()
    private let decoder = JSONDecoder()

    private init() {}

    private func request<T: Decodable>(_ path: String, method: String = "GET", body: Encodable? = nil) async throws -> T {
        var request = URLRequest(url: Config.backendURL.appendingPathComponent(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body {
            request.httpBody = try JSONEncoder().encode(AnyEncodable(body))
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try decoder.decode(T.self, from: data)
    }

    struct ChatResult {
        let reply: String
        let interactionId: Int?
    }

    func sendMessageResult(query: String, messages: [[String: String]], previousInteractionId: Int? = nil) async -> ChatResult {
        struct RequestBody: Encodable { let query: String; let messages: [[String: String]]; let sessionId: String
            let previousInteractionId: Int?
            enum CodingKeys: String, CodingKey { case query, messages; case sessionId = "session_id"; case previousInteractionId = "prev_interaction_id" }
        }
        struct ResponseBody: Decodable { let reply: String; let interactionId: Int?
            enum CodingKeys: String, CodingKey { case reply; case interactionId = "interaction_id" }
        }
        do {
            let response: ResponseBody = try await request("chat", method: "POST", body: RequestBody(query: query, messages: messages, sessionId: "ios", previousInteractionId: previousInteractionId))
            return ChatResult(reply: response.reply, interactionId: response.interactionId)
        } catch {
            return ChatResult(reply: "[OFFLINE MODE] [AI RESPONSE] W.A.Y.N.E backend unavailable.", interactionId: nil)
        }
    }

    func sendMessage(query: String, messages: [[String: String]]) async -> String {
        await sendMessageResult(query: query, messages: messages).reply
    }

    func submitFeedback(interactionId: Int, score: Int) async {
        struct Body: Encodable { let interactionId: Int; let score: Int
            enum CodingKeys: String, CodingKey { case score; case interactionId = "interaction_id" }
        }
        let _: EmptyResponse? = try? await request("feedback", method: "POST", body: Body(interactionId: interactionId, score: score))
    }

    func getLearningStats() async -> LearningStats {
        (try? await request("learning/stats")) ?? LearningStats.empty
    }

    func getTasks() async -> [TaskItem] {
        (try? await request("tasks")) ?? []
    }

    func createTask(title: String, priority: String) async {
        struct Body: Encodable { let title: String; let priority: String }
        let _: EmptyResponse? = try? await request("tasks", method: "POST", body: Body(title: title, priority: priority))
    }

    func toggleTask(id: Int) async {
        let _: EmptyResponse? = try? await request("tasks/\(id)", method: "PATCH")
    }

    func deleteTask(id: Int) async {
        let _: EmptyResponse? = try? await request("tasks/\(id)", method: "DELETE")
    }

    func getEvents() async -> [CalendarEvent] {
        (try? await request("events/today")) ?? []
    }

    func getDeviceStatus() async -> [DeviceStatus] {
        struct Response: Decodable { let devices: [DeviceStatus] }
        let response: Response? = try? await request("device/status")
        return response?.devices ?? []
    }

    func sendDeviceCommand(deviceId: String, command: String) async {
        struct Body: Encodable { let deviceId: String; let command: String; let confirmed: Bool; let issuedBy: String
            enum CodingKeys: String, CodingKey { case command, confirmed; case deviceId = "device_id"; case issuedBy = "issued_by" }
        }
        let _: EmptyResponse? = try? await request("device/command", method: "POST", body: Body(deviceId: deviceId, command: command, confirmed: true, issuedBy: "ios"))
    }

    func getFiles(dir: String) async -> [FileItem] {
        guard var components = URLComponents(url: Config.backendURL.appendingPathComponent("files/list"), resolvingAgainstBaseURL: false) else { return [] }
        components.queryItems = [URLQueryItem(name: "path", value: dir)]
        guard let url = components.url else { return [] }
        struct Response: Decodable { let entries: [FileItem] }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            return try decoder.decode(Response.self, from: data).entries
        } catch {
            return []
        }
    }

    func searchFiles(query: String, directory: String? = nil) async -> [FileItem] {
        struct Body: Encodable { let query: String; let directory: String?; let maxResults: Int
            enum CodingKeys: String, CodingKey { case query, directory; case maxResults = "max_results" }
        }
        struct Response: Decodable { let results: [FileItem] }
        let response: Response? = try? await request("files/search", method: "POST", body: Body(query: query, directory: directory, maxResults: 30))
        return response?.results ?? []
    }

    func readFile(path: String) async -> String {
        struct Body: Encodable { let path: String; let summarize: Bool; let forceRestricted: Bool
            enum CodingKeys: String, CodingKey { case path, summarize; case forceRestricted = "force_restricted" }
        }
        struct Response: Decodable { let content: String?; let error: String? }
        let response: Response? = try? await request("files/read", method: "POST", body: Body(path: path, summarize: false, forceRestricted: false))
        return response?.content ?? response?.error ?? "No preview available."
    }

    func openFile(path: String) async {
        struct Body: Encodable { let path: String }
        let _: EmptyResponse? = try? await request("files/open", method: "POST", body: Body(path: path))
    }

    func deleteFile(path: String) async {
        struct Body: Encodable { let operation: String; let path: String; let permanent: Bool; let confirmed: Bool }
        let _: EmptyResponse? = try? await request("files/operation", method: "POST", body: Body(operation: "delete", path: path, permanent: false, confirmed: true))
    }

    func getSystemStatus() async -> SystemStatus {
        (try? await request("pc/status")) ?? SystemStatus.empty
    }

    func getProcesses() async -> [ProcessItem] {
        struct Response: Decodable { let processes: [ProcessItem] }
        let response: Response? = try? await request("pc/processes")
        return response?.processes ?? []
    }

    func clearCache() async -> String {
        struct Body: Encodable { let confirmed: Bool }
        struct Response: Decodable { let totalFreedHuman: String?; let freedHuman: String?; let message: String?
            enum CodingKeys: String, CodingKey { case message; case totalFreedHuman = "total_freed_human"; case freedHuman = "freed_human" }
        }
        let response: Response? = try? await request("pc/cache/clear", method: "POST", body: Body(confirmed: true))
        return response?.totalFreedHuman ?? response?.freedHuman ?? response?.message ?? "Cache clear requested."
    }

    func optimizeMemory() async -> String {
        struct Body: Encodable { let confirmed: Bool }
        struct Response: Decodable { let freedHuman: String?; let message: String?
            enum CodingKeys: String, CodingKey { case message; case freedHuman = "freed_human" }
        }
        let response: Response? = try? await request("pc/memory/optimize", method: "POST", body: Body(confirmed: true))
        return response?.freedHuman ?? response?.message ?? "Memory optimization requested."
    }

    func killProcess(name: String) async {
        struct Body: Encodable { let name: String; let pid: Int?; let confirmed: Bool }
        let _: EmptyResponse? = try? await request("pc/processes/kill", method: "POST", body: Body(name: name, pid: nil, confirmed: true))
    }

    func registerPhone(pushToken: String?) async {
        struct Body: Encodable { let deviceId: String; let type: String; let name: String; let pushToken: String?
            enum CodingKeys: String, CodingKey { case type, name; case deviceId = "device_id"; case pushToken = "push_token" }
        }
        let body = Body(deviceId: Config.phoneDeviceId, type: "iphone", name: UIDeviceName.current, pushToken: pushToken)
        let _: EmptyResponse? = try? await request("device/register", method: "POST", body: body)
    }
}

struct EmptyResponse: Decodable {}

struct AnyEncodable: Encodable {
    private let encodeValue: (Encoder) throws -> Void
    init(_ value: Encodable) { self.encodeValue = value.encode }
    func encode(to encoder: Encoder) throws { try encodeValue(encoder) }
}

enum UIDeviceName {
    static var current: String {
        #if os(iOS)
        return UIDevice.current.name
        #else
        return "iPhone"
        #endif
    }
}
