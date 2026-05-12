import AVFoundation
import Foundation

@MainActor
final class VoiceService: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published var isListening = false
    @Published var isSpeaking = false
    @Published var isThinking = false
    @Published var liveTranscript = ""
    @Published var aiResponse = ""
    @Published var interruptDetected = false
    @Published var detectedLanguage = "auto"
    @Published var confidence: Double = 0

    private var audioEngine = AVAudioEngine()
    private var synthesizer = AVSpeechSynthesizer()
    private var webSocketTask: URLSessionWebSocketTask?
    private var interruptionThreshold: Float = 0.02
    private var consecutiveInterruptFrames = 0
    private var hasSpeech = false
    private var silenceFrames = 0
    private var lastSpeechEnd = Date.distantPast
    private var ttsBuffer = ""
    private let sessionId = UUID().uuidString

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func startDuplexSession() {
        AVAudioSession.sharedInstance().requestRecordPermission { _ in }
        liveTranscript = ""
        aiResponse = ""
        interruptDetected = false
        connectWebSocket()
        startAudioCapture()
        isListening = true
    }

    func stopDuplexSession() {
        sendJSON(["type": "stop_voice"])
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        synthesizer.stopSpeaking(at: .immediate)
        speak(text: "W.A.Y.N.E standing by.")
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        isListening = false
        isSpeaking = false
        isThinking = false
        consecutiveInterruptFrames = 0
        ttsBuffer = ""
    }

    private func connectWebSocket() {
        let url = Config.backendWebSocketURL.appendingPathComponent("ws/voice/\(sessionId)")
        webSocketTask = URLSession.shared.webSocketTask(with: url)
        webSocketTask?.resume()
        receiveMessage()
    }

    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                if case .success(let message) = result {
                    switch message {
                    case .string(let text):
                        self.handleWebSocketMessage(text)
                    case .data(let data):
                        if let text = String(data: data, encoding: .utf8) { self.handleWebSocketMessage(text) }
                    @unknown default:
                        break
                    }
                }
                if self.webSocketTask != nil { self.receiveMessage() }
            }
        }
    }

    private func handleWebSocketMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let message = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = message["type"] as? String else { return }
        switch type {
        case "ready":
            isListening = true
            aiResponse = message["message"] as? String ?? "W.A.Y.N.E voice ready."
        case "transcribing":
            isThinking = true
            isListening = false
        case "transcript":
            liveTranscript = message["text"] as? String ?? liveTranscript
            detectedLanguage = message["language"] as? String ?? detectedLanguage
            confidence = message["confidence"] as? Double ?? confidence
            isThinking = true
            isListening = false
        case "low_confidence":
            aiResponse = message["message"] as? String ?? "Please repeat that once."
            confidence = message["confidence"] as? Double ?? confidence
            isThinking = false
            isListening = true
        case "ai_token":
            let token = message["token"] as? String ?? ""
            aiResponse += token
            speakIncremental(token: token)
        case "ai_done":
            flushSpeechBuffer()
            isThinking = false
            isListening = true
        case "interrupted":
            synthesizer.stopSpeaking(at: .immediate)
            isSpeaking = false
            isListening = true
            isThinking = false
        default:
            break
        }
    }

    private func startAudioCapture() {
        try? AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth, .duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            Task { @MainActor in
                guard let self else { return }
                self.monitorInterruption(buffer: buffer)
                self.monitorSpeechEnd(buffer: buffer)
                if let pcm = self.pcmData(from: buffer) {
                    self.sendJSON(["type": "audio_chunk", "data": pcm.base64EncodedString()])
                }
            }
        }
        audioEngine.prepare()
        try? audioEngine.start()
    }

    private func pcmData(from buffer: AVAudioPCMBuffer) -> Data? {
        guard let channel = buffer.floatChannelData?[0] else { return nil }
        let count = Int(buffer.frameLength)
        var samples = [Int16]()
        samples.reserveCapacity(count)
        for index in 0..<count {
            let value = max(-1, min(1, channel[index]))
            samples.append(Int16(value * Float(Int16.max)))
        }
        return Data(bytes: samples, count: samples.count * MemoryLayout<Int16>.size)
    }

    private func monitorInterruption(buffer: AVAudioPCMBuffer) {
        guard isSpeaking, let channel = buffer.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }
        var sum: Float = 0
        for index in 0..<count { sum += channel[index] * channel[index] }
        let rms = sqrt(sum / Float(count))
        if rms > interruptionThreshold {
            consecutiveInterruptFrames += 1
        } else {
            consecutiveInterruptFrames = 0
        }
        if consecutiveInterruptFrames >= 3 {
            interrupt()
        }
    }

    private func monitorSpeechEnd(buffer: AVAudioPCMBuffer) {
        guard !isSpeaking, let channel = buffer.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }
        var sum: Float = 0
        for index in 0..<count { sum += channel[index] * channel[index] }
        let rms = sqrt(sum / Float(count))
        if rms > 0.018 {
            hasSpeech = true
            silenceFrames = 0
        } else if hasSpeech {
            silenceFrames += 1
        }
        if hasSpeech && silenceFrames >= 10 && Date().timeIntervalSince(lastSpeechEnd) > 1.4 {
            hasSpeech = false
            silenceFrames = 0
            lastSpeechEnd = Date()
            isThinking = true
            isListening = false
            sendJSON(["type": "speech_end"])
        }
    }

    func speakIncremental(token: String) {
        ttsBuffer += token
        let words = ttsBuffer.split(separator: " ").count
        if ttsBuffer.contains(".") || ttsBuffer.contains("?") || ttsBuffer.contains("!") || words >= 12 {
            speak(text: ttsBuffer)
            ttsBuffer = ""
        }
    }

    private func flushSpeechBuffer() {
        if !ttsBuffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            speak(text: ttsBuffer)
            ttsBuffer = ""
        }
    }

    private func speak(text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: ttsLanguage(for: detectedLanguage))
        utterance.rate = 0.52
        utterance.pitchMultiplier = 0.9
        synthesizer.speak(utterance)
        isSpeaking = true
        isListening = false
        isThinking = false
    }

    func setLanguage(_ language: String) {
        detectedLanguage = language
        sendJSON(["type": "set_language", "language": language])
    }

    private func ttsLanguage(for code: String) -> String {
        switch code {
        case "ta": return "ta-IN"
        case "hi": return "hi-IN"
        case "te": return "te-IN"
        case "kn": return "kn-IN"
        case "ml": return "ml-IN"
        case "fr": return "fr-FR"
        case "es": return "es-ES"
        case "de": return "de-DE"
        case "ja": return "ja-JP"
        case "zh": return "zh-CN"
        case "ar": return "ar-SA"
        case "pt": return "pt-BR"
        case "ru": return "ru-RU"
        case "ko": return "ko-KR"
        default: return "en-US"
        }
    }

    func interrupt() {
        synthesizer.stopSpeaking(at: .immediate)
        sendJSON(["type": "interrupt"])
        isSpeaking = false
        isListening = true
        interruptDetected = true
        consecutiveInterruptFrames = 0
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.interruptDetected = false }
    }

    private func sendJSON(_ payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else { return }
        webSocketTask?.send(.string(text)) { _ in }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            self.isListening = true
        }
    }
}
