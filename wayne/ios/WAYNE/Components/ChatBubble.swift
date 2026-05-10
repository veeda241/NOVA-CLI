import SwiftUI

struct ChatBubble: View {
    let message: Message

    var body: some View {
        VStack(alignment: message.role == "user" ? .trailing : .leading, spacing: 4) {
            if message.role != "user" {
                Text("W.A.Y.N.E")
                    .font(.caption2.monospaced())
                    .foregroundStyle(Color.wayneAmber)
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
}
