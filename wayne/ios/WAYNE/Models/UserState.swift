import Foundation

struct UserState: Codable, Hashable {
    let hour: Int?
    let dayOfWeek: Int?
    let intent: String?
    let emotion: String?
    let preferredLength: String?
    let preferredTone: String?

    enum CodingKeys: String, CodingKey {
        case hour, intent, emotion
        case dayOfWeek = "day_of_week"
        case preferredLength = "preferred_length"
        case preferredTone = "preferred_tone"
    }
}

struct SystemStatus: Codable {
    let cpuPercent: Double
    let memory: MemoryStatus?
    let diskPercent: Double?
    let batteryPercent: Double?
    let batteryPlugged: Bool?

    enum CodingKeys: String, CodingKey {
        case memory
        case cpuPercent = "cpu_percent"
        case diskPercent = "disk_percent"
        case batteryPercent = "battery_percent"
        case batteryPlugged = "battery_plugged"
    }

    static let empty = SystemStatus(cpuPercent: 0, memory: nil, diskPercent: 0, batteryPercent: nil, batteryPlugged: nil)
}

struct MemoryStatus: Codable {
    let percent: Double
    let usedHuman: String?
    let totalHuman: String?

    enum CodingKeys: String, CodingKey {
        case percent
        case usedHuman = "used_human"
        case totalHuman = "total_human"
    }
}

struct ProcessItem: Identifiable, Codable, Hashable {
    var id: Int { pid }
    let pid: Int
    let name: String
    let cpuPercent: Double?
    let memoryPercent: Double?
    let status: String?

    enum CodingKeys: String, CodingKey {
        case pid, name, status
        case cpuPercent = "cpu_percent"
        case memoryPercent = "memory_percent"
    }
}
