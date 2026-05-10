import SwiftUI

struct VoiceView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var voice = VoiceService()
    @State private var pulse = false
    @State private var rotation = 0.0

    var onDismissTranscript: (String) -> Void

    var body: some View {
        ZStack {
            Color.wayneBackground.ignoresSafeArea()
            VStack(spacing: 28) {
                Spacer()
                orb
                Text(statusText)
                    .font(.title3.monospaced())
                    .foregroundStyle(statusColor)
                    .opacity(voice.isListening ? (pulse ? 0.55 : 1) : 1)
                transcriptView
                Spacer()
                Button {
                    let final = voice.liveTranscript
                    voice.stopDuplexSession()
                    onDismissTranscript(final)
                    dismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(.title2.weight(.bold))
                        .foregroundStyle(Color.wayneRed)
                        .frame(width: 64, height: 64)
                        .overlay(Circle().stroke(Color.wayneRed, lineWidth: 1))
                }
                .padding(.bottom, 36)
            }
        }
        .onAppear {
            voice.startDuplexSession()
            withAnimation(.easeInOut(duration: 2).repeatForever(autoreverses: true)) { pulse = true }
            withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) { rotation = 360 }
        }
        .onDisappear { voice.stopDuplexSession() }
    }

    private var orb: some View {
        ZStack {
            if voice.isListening {
                ForEach(0..<3) { index in
                    Circle()
                        .stroke(Color.wayneAccent.opacity(0.4), lineWidth: 1)
                        .frame(width: 200, height: 200)
                        .scaleEffect(pulse ? 1.8 : 1.0)
                        .opacity(pulse ? 0 : 1)
                        .animation(.easeOut(duration: 1.4).repeatForever(autoreverses: false).delay(Double(index) * 0.4), value: pulse)
                }
            }
            if voice.isThinking {
                Circle()
                    .trim(from: 0, to: 0.75)
                    .stroke(Color.wayneAmber, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                    .frame(width: 230, height: 230)
                    .rotationEffect(.degrees(rotation))
            }
            if voice.isSpeaking {
                ForEach(0..<12) { index in
                    Capsule()
                        .fill(LinearGradient(colors: [.wayneAccent.opacity(0.2), .wayneAccent], startPoint: .bottom, endPoint: .top))
                        .frame(width: 5, height: pulse ? CGFloat(24 + (index % 4) * 8) : CGFloat(12 + (index % 3) * 6))
                        .offset(y: -130)
                        .rotationEffect(.degrees(Double(index) * 30))
                        .animation(.easeInOut(duration: 0.45).repeatForever(autoreverses: true).delay(Double(index) * 0.04), value: pulse)
                }
            }
            Circle()
                .fill(voice.interruptDetected ? Color.wayneRed.opacity(0.45) : Color.waynePanel)
                .frame(width: 200, height: 200)
                .shadow(color: (voice.interruptDetected ? Color.wayneRed : Color.wayneAccent).opacity(0.8), radius: pulse ? 32 : 18)
                .scaleEffect(pulse && voice.isListening ? 1.05 : 1.0)
            Text("W.A.Y.N.E")
                .font(.title.monospaced().weight(.bold))
                .foregroundStyle(Color.wayneAccent)
        }
    }

    private var transcriptView: some View {
        VStack(spacing: 10) {
            wordLine(text: voice.liveTranscript, color: .wayneAccent)
            wordLine(text: voice.aiResponse, color: .wayneAmber)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 28)
        .lineLimit(3)
    }

    private func wordLine(text: String, color: Color) -> some View {
        Text(text.split(separator: " ").suffix(32).joined(separator: " "))
            .font(.body.monospaced())
            .foregroundStyle(color)
            .multilineTextAlignment(.center)
            .transition(.opacity.combined(with: .move(edge: .bottom)))
            .animation(.easeIn(duration: 0.1), value: text)
    }

    private var statusText: String {
        if voice.interruptDetected { return "Interrupted" }
        if voice.isThinking { return "W.A.Y.N.E Processing..." }
        if voice.isSpeaking { return "W.A.Y.N.E Speaking..." }
        return "W.A.Y.N.E Listening..."
    }

    private var statusColor: Color {
        if voice.interruptDetected { return .wayneRed }
        if voice.isThinking { return .wayneAmber }
        if voice.isSpeaking { return .wayneGreen }
        return .wayneAccent
    }
}
