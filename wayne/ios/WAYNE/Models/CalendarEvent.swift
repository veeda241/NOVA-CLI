import Foundation

struct CalendarEvent: Identifiable, Codable, Hashable {
    let id: String?
    let title: String
    let start: String
    let end: String
}
