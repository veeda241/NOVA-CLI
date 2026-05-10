import Foundation

struct TaskItem: Identifiable, Codable, Hashable {
    let id: Int
    let title: String
    let priority: String
    let completed: Bool
    let createdAt: String?

    enum CodingKeys: String, CodingKey {
        case id, title, priority, completed
        case createdAt = "created_at"
    }
}
