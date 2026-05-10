import AVFoundation
import Speech
import SwiftUI

final class WakeWordService: ObservableObject {
    @Published var isListening = false
    @Published var wakeDetected = false

    private let audioEngine = AVAudioEngine()
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var isRestarting = false

    func startPassiveListening() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else { return }
            DispatchQueue.main.async { self?.beginListening() }
        }
    }

    func stopPassiveListening() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        isListening = false
    }

    private func beginListening() {
        guard !audioEngine.isRunning else { return }
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: [.duckOthers, .allowBluetooth])
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let request = recognitionRequest else { return }
        request.shouldReportPartialResults = true

        let inputNode = audioEngine.inputNode
        recognitionTask = recognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            if let result {
                let text = result.bestTranscription.formattedString.lowercased()
                if text.contains("wayne") || text.contains("hey wayne") {
                    DispatchQueue.main.async {
                        self.wakeDetected = true
                        self.restartListening()
                    }
                }
            }
            if error != nil {
                self.restartListening()
            }
        }

        let format = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }

        do {
            try audioEngine.start()
            isListening = true
        } catch {
            restartListening()
        }
    }

    private func restartListening() {
        guard !isRestarting else { return }
        isRestarting = true
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        isListening = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.isRestarting = false
            self.beginListening()
        }
    }
}
