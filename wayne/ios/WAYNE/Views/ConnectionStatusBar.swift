import SwiftUI

struct ConnectionStatusBar: View {
    @ObservedObject var connection: ConnectionService

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color(hex: connection.status.color))
                .frame(width: 7, height: 7)
                .opacity(connection.status == .connected ? 1 : 0.75)

            Text(connection.status.label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Color(hex: connection.status.color))

            if connection.status == .reconnecting {
                Text("attempt \(connection.reconnectAttempt)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Color.wayneMuted)
            }

            Spacer()

            if connection.status == .connected {
                Text("\(Int(connection.latencyMs))ms")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Color.wayneMuted)
                Text("cache \(Int(connection.cacheHitRate))%")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Color.wayneMuted)
            } else if connection.status == .disconnected {
                Button("Retry") {
                    connection.reconnect()
                }
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Color.wayneAccent)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Color.wayneSurface)
        .overlay(Rectangle().frame(height: 0.5).foregroundStyle(Color(hex: connection.status.color).opacity(0.4)), alignment: .bottom)
    }
}
