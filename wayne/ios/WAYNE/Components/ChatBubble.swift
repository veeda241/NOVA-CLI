import SwiftUI

struct ChatBubble: View {
    let message: Message

    var body: some View {
        VStack(alignment: message.role == "user" ? .trailing : .leading, spacing: 4) {
            if message.role != "user" {
                HStack(spacing: 6) {
                    Text("W.A.Y.N.E")
                        .font(.caption2.monospaced())
                        .foregroundStyle(Color.wayneAmber)
                    if let badge = sourceBadge(for: message.content) {
                        Text(badge)
                            .font(.caption2.monospaced())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .foregroundStyle(Color.wayneBackground)
                            .background(badgeColor(for: badge))
                            .clipShape(Capsule())
                    }
                }
            }
            HStack {
                if message.role == "user" { Spacer(minLength: 40) }
                if message.role != "user" {
                    Image(systemName: "cpu").foregroundStyle(Color.wayneAccent)
                }
                Text(message.displayContent)
                    .font(.body.monospaced())
                    .padding(12)
                    .foregroundStyle(.white)
                    .background(message.role == "user" ? Color.wayneAccent.opacity(0.18) : Color.waynePanel)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.wayneAccent.opacity(0.35), lineWidth: 0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                if message.role != "user" { Spacer(minLength: 40) }
            }
        }
    }

    private func sourceBadge(for content: String) -> String? {
        let lowered = content.lowercased()
        if lowered.contains("source: system clock") { return "CLOCK" }
        if lowered.contains("source: special days") { return "SPECIAL" }
        if lowered.contains("source: wikipedia") { return "WIKI" }
        if lowered.contains("source: current web") { return "WEB" }
        return nil
    }

    private func badgeColor(for badge: String) -> Color {
        switch badge {
        case "CLOCK": return Color.wayneAccent
        case "SPECIAL": return Color.wayneAmber
        case "WIKI": return Color.blue
        case "WEB": return Color.wayneGreen
        default: return Color.wayneAccent
        }
    }
}
