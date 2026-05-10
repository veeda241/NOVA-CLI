import SwiftUI

struct BootScreenView: View {
    @State private var completedSteps: [String] = []
    @State private var isDone = false
    @Binding var isShowing: Bool

    private let bootSteps = [
        "Initializing neural core...",
        "Loading Gemma local model...",
        "Checking local backend link...",
        "Syncing task database...",
        "Activating device control...",
        "Warming up voice synthesis...",
        "All systems nominal."
    ]

    var body: some View {
        ZStack {
            Color(hex: "060c14").ignoresSafeArea()
            VStack(spacing: 24) {
                Text("W.A.Y.N.E")
                    .font(.system(size: 36, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(hex: "00d4ff"))
                    .shadow(color: Color(hex: "00d4ff").opacity(0.5), radius: 10)

                Text("Wireless Artificial Yielding Network Engine")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(Color(hex: "ffaa00"))
                    .multilineTextAlignment(.center)

                VStack(alignment: .leading, spacing: 8) {
                    ForEach(completedSteps, id: \.self) { step in
                        HStack(spacing: 10) {
                            Image(systemName: "checkmark")
                                .foregroundColor(Color(hex: "00ff88"))
                                .font(.system(size: 11))
                            Text(step)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(Color(hex: "4a7a9b"))
                        }
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 32)

                if isDone {
                    VStack(spacing: 8) {
                        Text("W.A.Y.N.E ONLINE")
                            .font(.system(size: 18, weight: .medium, design: .monospaced))
                            .foregroundColor(Color(hex: "00ff88"))
                        Text("How can I assist you, Sir?")
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundColor(Color(hex: "4a7a9b"))
                    }
                    .padding(20)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color(hex: "00d4ff").opacity(0.4), lineWidth: 0.5)
                    )
                    .transition(.opacity)
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                            withAnimation { isShowing = false }
                        }
                    }
                }
            }
            .padding(24)
        }
        .onAppear { runBootSequence() }
    }

    private func runBootSequence() {
        completedSteps = []
        isDone = false
        for (index, step) in bootSteps.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * 0.4) {
                withAnimation { completedSteps.append(step) }
                if index == bootSteps.count - 1 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        withAnimation { isDone = true }
                    }
                }
            }
        }
    }
}
