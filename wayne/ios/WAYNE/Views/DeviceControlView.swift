import SwiftUI

struct DeviceControlView: View {
    @State private var devices: [DeviceStatus] = []

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                ForEach(devices) { device in
                    VStack(spacing: 12) {
                        DeviceCard(device: device)
                        let laptop = device.deviceType == "laptop"
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                            if laptop {
                                CommandButton(icon: "moon.fill", label: "Sleep", device: device, command: "sleep")
                                CommandButton(icon: "lock.fill", label: "Lock", device: device, command: "lock")
                                CommandButton(icon: "arrow.clockwise", label: "Restart", device: device, command: "restart")
                                CommandButton(icon: "power", label: "Shutdown", device: device, command: "shutdown")
                            } else {
                                CommandButton(icon: "lock.fill", label: "Lock Screen", device: device, command: "lock")
                                CommandButton(icon: "moon.zzz.fill", label: "Do Not Disturb", device: device, command: "do_not_disturb_on")
                                CommandButton(icon: "speaker.slash.fill", label: "Mute", device: device, command: "mute")
                            }
                        }
                    }
                }
            }
            .padding()
        }
        .background(Color.wayneBackground.ignoresSafeArea())
        .navigationTitle("W.A.Y.N.E Device Control")
        .task { await refresh() }
        .refreshable { await refresh() }
    }

    private func refresh() async {
        devices = await WAYNEService.shared.getDeviceStatus()
    }
}
