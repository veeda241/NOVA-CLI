import MapKit
import SwiftUI

struct TrackingView: View {
    @StateObject private var sockets = WebSocketService()
    @State private var devices: [DeviceStatus] = []

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 14) {
                ForEach(currentDevices) { device in
                    VStack(alignment: .leading, spacing: 8) {
                        DeviceCard(device: device)
                        Text("IP \(device.ipAddress ?? "-") | Disk \(Int(device.diskPercent))%")
                            .font(.caption.monospaced())
                            .foregroundStyle(Color.wayneAccent.opacity(0.75))
                        if let coordinate = device.coordinate {
                            Map(initialPosition: .region(MKCoordinateRegion(center: coordinate, span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)))) {
                                Marker(device.deviceName, coordinate: coordinate)
                            }
                            .frame(height: 180)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                    }
                }
            }
            .padding()
        }
        .background(Color.wayneBackground.ignoresSafeArea())
        .navigationTitle("Tracking")
        .task {
            devices = await WAYNEService.shared.getDeviceStatus()
            sockets.connect()
        }
        .onDisappear { sockets.disconnect() }
    }

    private var currentDevices: [DeviceStatus] {
        let updates = Array(sockets.deviceUpdates.values)
        return updates.isEmpty ? devices : updates
    }
}
