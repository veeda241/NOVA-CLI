import SwiftUI

struct StatusBadge: View {
    let online: Bool

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(online ? Color.wayneGreen : Color.wayneRed)
                .frame(width: 8, height: 8)
            Text(online ? "Online" : "Offline")
                .font(.caption.monospaced())
        }
        .foregroundStyle(online ? Color.wayneGreen : Color.wayneRed)
    }
}
