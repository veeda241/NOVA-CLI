import SwiftUI

struct DeviceCard: View {
    let device: DeviceStatus

    private var batteryColor: Color {
        if device.batteryLevel > 50 { return .wayneGreen }
        if device.batteryLevel >= 20 { return .wayneAmber }
        return .wayneRed
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: device.deviceType == "iphone" ? "iphone" : "laptopcomputer")
                    .foregroundStyle(Color.wayneAccent)
                VStack(alignment: .leading) {
                    Text(device.deviceName).font(.headline.monospaced())
                    Text(device.deviceType).font(.caption.monospaced()).foregroundStyle(Color.wayneAccent.opacity(0.65))
                }
                Spacer()
                StatusBadge(online: device.isOnline)
            }
            ProgressView(value: Double(device.batteryLevel), total: 100)
                .tint(batteryColor)
            Text("Battery \(device.batteryLevel)% | CPU \(Int(device.cpuPercent))% | RAM \(Int(device.ramPercent))%")
                .font(.caption.monospaced())
                .foregroundStyle(Color.white.opacity(0.75))
            Text("Last seen \(device.lastSeen)")
                .font(.caption2.monospaced())
                .foregroundStyle(Color.wayneAccent.opacity(0.7))
        }
        .padding()
        .background(Color.waynePanel)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.wayneAccent.opacity(0.3), lineWidth: 0.5))
    }
}
