import Foundation
import CoreLocation

struct DeviceStatus: Identifiable, Codable, Hashable {
    let id: Int?
    let deviceId: String
    let deviceName: String
    let deviceType: String
    let batteryLevel: Int
    let cpuPercent: Double
    let ramPercent: Double
    let diskPercent: Double
    let isOnline: Bool
    let lastSeen: String
    let ipAddress: String?
    let latitude: Double?
    let longitude: Double?

    enum CodingKeys: String, CodingKey {
        case id
        case deviceId = "device_id"
        case deviceName = "device_name"
        case deviceType = "device_type"
        case batteryLevel = "battery_level"
        case cpuPercent = "cpu_percent"
        case ramPercent = "ram_percent"
        case diskPercent = "disk_percent"
        case isOnline = "is_online"
        case lastSeen = "last_seen"
        case ipAddress = "ip_address"
        case latitude, longitude
    }

    var coordinate: CLLocationCoordinate2D? {
        guard let latitude, let longitude else { return nil }
        return CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }
}
