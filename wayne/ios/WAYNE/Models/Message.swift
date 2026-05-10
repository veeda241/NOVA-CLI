import Foundation

struct Message: Identifiable, Codable, Hashable {
    let id: UUID
    let role: String
    let content: String
    let interactionId: Int?

    init(id: UUID = UUID(), role: String, content: String, interactionId: Int? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.interactionId = interactionId
    }

    var tag: String {
        guard content.first == "[", let end = content.firstIndex(of: "]") else { return "AI RESPONSE" }
        return String(content[content.index(after: content.startIndex)..<end])
    }

    var displayContent: String {
        content.replacingOccurrences(of: #"^\[[^\]]+\]\s*"#, with: "", options: .regularExpression)
    }
}
