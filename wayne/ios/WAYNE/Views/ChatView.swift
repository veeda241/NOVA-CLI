import SwiftUI

struct ChatView: View {
    @State private var messages: [Message] = [Message(role: "assistant", content: "[AI RESPONSE] W.A.Y.N.E online.\nWireless Artificial Yielding Network Engine\nHow can I assist you?")]
    @State private var input = ""
    @State private var loading = false
    @State private var showingVoice = false
    @State private var lastInteractionId: Int?

    var body: some View {
        VStack {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(messages) { message in
                            VStack(alignment: message.role == "user" ? .trailing : .leading, spacing: 4) {
                                ChatBubble(message: message).id(message.id)
                                if message.role != "user", let interactionId = message.interactionId {
                                    HStack(spacing: 8) {
                                        ForEach(1...5, id: \.self) { score in
                                            Button("★") {
                                                Task { await WAYNEService.shared.submitFeedback(interactionId: interactionId, score: score) }
                                            }
                                            .foregroundStyle(Color.wayneAmber)
                                        }
                                    }
                                    .font(.caption.monospaced())
                                }
                            }
                        }
                        if loading { ProgressView().tint(Color.wayneAccent) }
                    }
                    .padding()
                }
                .onChange(of: messages.count) { _ in
                    if let last = messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
            HStack {
                TextField("Issue a command...", text: $input)
                    .textFieldStyle(.plain)
                    .font(.body.monospaced())
                    .padding(12)
                    .background(Color.waynePanel)
                    .foregroundStyle(.white)
                Button {
                    showingVoice = true
                } label: {
                    Image(systemName: "mic.fill")
                        .foregroundStyle(Color.wayneAccent)
                }
                Button("Send") { send() }
                    .font(.body.monospaced())
                    .foregroundStyle(Color.wayneAccent)
            }
            .padding()
            .background(Color.wayneSurface)
        }
        .background(Color.wayneBackground.ignoresSafeArea())
        .navigationTitle("W.A.Y.N.E")
        .fullScreenCover(isPresented: $showingVoice) {
            VoiceView { transcript in
                let final = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                if !final.isEmpty {
                    input = final
                    send()
                }
            }
        }
    }

    private func send() {
        let query = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty, !loading else { return }
        input = ""
        let user = Message(role: "user", content: query)
        messages.append(user)
        loading = true
        Task {
            let apiMessages = messages.map { ["role": $0.role, "content": $0.content] }
            let result = await WAYNEService.shared.sendMessageResult(query: query, messages: apiMessages, previousInteractionId: lastInteractionId)
            lastInteractionId = result.interactionId
            messages.append(Message(role: "assistant", content: result.reply, interactionId: result.interactionId))
            loading = false
        }
    }
}
