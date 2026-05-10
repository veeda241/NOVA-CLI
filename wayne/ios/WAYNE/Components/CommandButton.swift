import SwiftUI

struct CommandButton: View {
    let icon: String
    let label: String
    let device: DeviceStatus
    let command: String
    @State private var showingConfirm = false

    var body: some View {
        Button {
            showingConfirm = true
        } label: {
            Label(label, systemImage: icon)
                .font(.caption.monospaced())
                .frame(maxWidth: .infinity)
                .padding(8)
                .background(Color.wayneBackground)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.wayneAccent.opacity(0.3), lineWidth: 0.5))
        }
        .foregroundStyle(Color.wayneAccent)
        .alert("Confirm Command", isPresented: $showingConfirm) {
            Button("Cancel", role: .cancel) {}
            Button("Confirm", role: .destructive) {
                Task { await WAYNEService.shared.sendDeviceCommand(deviceId: device.deviceId, command: command) }
            }
        } message: {
            Text("W.A.Y.N.E will \(command) your \(device.deviceName). Confirm?")
        }
    }
}
