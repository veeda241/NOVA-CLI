"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { LanguageSelector } from "./LanguageSelector";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace("http://", "ws://").replace("https://", "wss://");

export function VoiceOrb({ onClose }: { onClose: (finalTranscript: string) => void }) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [interrupted, setInterrupted] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [aiResponse, setAiResponse] = useState("");
  const [language, setLanguage] = useState("auto");
  const [languageName, setLanguageName] = useState("Auto Detect");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [lowConfidenceMessage, setLowConfidenceMessage] = useState("");

  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ttsBufferRef = useRef("");
  const monitorTimerRef = useRef<number | null>(null);
  const hasSpeechRef = useRef(false);
  const silenceFramesRef = useRef(0);
  const speakingRef = useRef(false);
  const lastSentAtRef = useRef(0);

  useEffect(() => {
    startDuplexSession();
    return () => stopDuplexSession(false);
  }, []);

  useEffect(() => {
    let frame = 0;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (!canvas || !context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (!isSpeaking) return;
      const center = canvas.width / 2;
      for (let index = 0; index < 12; index += 1) {
        const angle = (Math.PI * 2 * index) / 12;
        const height = 18 + Math.sin(Date.now() / 120 + index) * 12;
        const x = center + Math.cos(angle) * 100;
        const y = center + Math.sin(angle) * 100;
        context.save();
        context.translate(x, y);
        context.rotate(angle + Math.PI / 2);
        context.fillStyle = "#00d4ff";
        context.fillRect(-2, -height / 2, 4, height);
        context.restore();
      }
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [isSpeaking]);

  async function startDuplexSession() {
    try {
      const socket = new WebSocket(`${WS_URL}/ws/voice/web-${crypto.randomUUID()}`);
      socketRef.current = socket;
      socket.onmessage = handleWebSocketMessage;
      socket.onopen = () => setIsListening(true);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      streamRef.current = stream;
      const AudioCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioContext = new AudioCtor({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = async (event) => {
        if (!event.data.size || speakingRef.current) return;
        const buffer = await event.data.arrayBuffer();
        sendSocketMessage({ type: "audio_chunk", data: arrayBufferToBase64(buffer) });
      };
      recorder.start(250);
      recorderRef.current = recorder;
      startAudioMonitor();
    } catch (error) {
      setIsListening(false);
      setIsThinking(false);
      setAiResponse(error instanceof Error ? error.message : "Voice session could not start.");
    }
  }

  function startAudioMonitor() {
    const data = new Uint8Array(2048);
    const tick = () => {
      const analyser = analyserRef.current;
      if (analyser) {
        analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let index = 0; index < data.length; index += 1) {
          const normalized = (data[index] - 128) / 128;
          sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / data.length);
        if (speechSynthesis.speaking && rms > 0.02) {
          interrupt();
        } else if (!speakingRef.current && rms > 0.018) {
          hasSpeechRef.current = true;
          silenceFramesRef.current = 0;
          setIsListening(true);
        } else if (hasSpeechRef.current) {
          silenceFramesRef.current += 1;
          if (silenceFramesRef.current >= 10 && Date.now() - lastSentAtRef.current > 1400) {
            lastSentAtRef.current = Date.now();
            hasSpeechRef.current = false;
            silenceFramesRef.current = 0;
            setIsThinking(true);
            setIsListening(false);
            setLowConfidenceMessage("");
            sendSocketMessage({ type: "speech_end" });
          }
        }
      }
      monitorTimerRef.current = window.setTimeout(tick, 100);
    };
    tick();
  }

  function handleWebSocketMessage(event: MessageEvent) {
    const message = JSON.parse(event.data);
    switch (message.type) {
      case "ready":
        setLanguage(message.language || "auto");
        setIsListening(true);
        speak(message.message || "Wayne voice ready.");
        break;
      case "language_set":
        setLanguage(message.language || "auto");
        speak(message.greeting || "Language set.");
        break;
      case "transcribing":
        setIsThinking(true);
        setIsListening(false);
        break;
      case "transcript":
        setLiveTranscript(message.text || "");
        setLanguage(message.language || language);
        setLanguageName(message.language_name || message.language || languageName);
        setConfidence(typeof message.confidence === "number" ? message.confidence : null);
        setIsThinking(true);
        break;
      case "low_confidence":
        setLowConfidenceMessage(message.message || "Please repeat that once.");
        setIsThinking(false);
        setIsListening(true);
        speak(message.message || "Please repeat that once.");
        break;
      case "ai_token":
        setAiResponse((current) => current + message.token);
        speakIncremental(message.token);
        break;
      case "ai_done":
        flushSpeech();
        setIsThinking(false);
        if (!speechSynthesis.speaking) setIsListening(true);
        break;
      case "interrupted":
        speechSynthesis.cancel();
        speakingRef.current = false;
        setIsSpeaking(false);
        setIsListening(true);
        setInterrupted(true);
        setTimeout(() => setInterrupted(false), 500);
        break;
      case "error":
        setAiResponse(message.message || "Voice error.");
        setIsThinking(false);
        setIsListening(true);
        break;
      default:
        break;
    }
  }

  function speakIncremental(token: string) {
    ttsBufferRef.current += token;
    const words = ttsBufferRef.current.trim().split(/\s+/).filter(Boolean);
    if (/[.?!।؟]\s*$/.test(ttsBufferRef.current) || words.length >= 10) {
      speak(ttsBufferRef.current);
      ttsBufferRef.current = "";
    }
  }

  function flushSpeech() {
    if (ttsBufferRef.current.trim()) {
      speak(ttsBufferRef.current);
      ttsBufferRef.current = "";
    }
  }

  function speak(text: string) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = browserTtsLanguage(language);
    utterance.voice = speechSynthesis.getVoices().find((voice) => voice.lang.toLowerCase().startsWith(utterance.lang.slice(0, 2))) ?? speechSynthesis.getVoices()[0] ?? null;
    utterance.rate = 0.95;
    utterance.pitch = 0.9;
    utterance.onstart = () => {
      speakingRef.current = true;
      setIsSpeaking(true);
      setIsListening(false);
      setIsThinking(false);
    };
    utterance.onend = () => {
      speakingRef.current = false;
      setIsSpeaking(false);
      setIsListening(true);
    };
    speechSynthesis.speak(utterance);
  }

  function interrupt() {
    speechSynthesis.cancel();
    sendSocketMessage({ type: "interrupt" });
    speakingRef.current = false;
    setIsSpeaking(false);
    setIsListening(true);
    setInterrupted(true);
    setTimeout(() => setInterrupted(false), 500);
  }

  function setVoiceLanguage(nextLanguage: string) {
    setLanguage(nextLanguage);
    sendSocketMessage({ type: "set_language", language: nextLanguage });
  }

  function sendSocketMessage(payload: Record<string, unknown>) {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
  }

  function stopDuplexSession(closeOverlay = true) {
    sendSocketMessage({ type: "stop_voice" });
    if (socketRef.current && socketRef.current.readyState !== WebSocket.CLOSED) socketRef.current.close();
    if (recorderRef.current?.state !== "inactive") recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (audioContextRef.current && audioContextRef.current.state !== "closed") void audioContextRef.current.close().catch(() => undefined);
    if (monitorTimerRef.current) window.clearTimeout(monitorTimerRef.current);
    socketRef.current = null;
    recorderRef.current = null;
    streamRef.current = null;
    audioContextRef.current = null;
    analyserRef.current = null;
    speechSynthesis.cancel();
    setIsListening(false);
    setIsSpeaking(false);
    setIsThinking(false);
    if (closeOverlay) onClose(liveTranscript);
  }

  const status = interrupted ? "Interrupted" : isThinking ? "Thinking..." : isSpeaking ? "Speaking..." : "Listening...";
  const confidenceClass = confidence == null ? "text-cyan" : confidence >= 0.8 ? "text-cyan" : confidence >= 0.5 ? "text-amber" : "text-danger";

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background/95 text-cyan backdrop-blur">
      <div className="absolute right-6 top-6 w-56">
        <LanguageSelector value={language} onChange={setVoiceLanguage} compact />
      </div>
      <div className={`voice-orb ${isListening ? "listening" : ""} ${isThinking ? "thinking" : ""} ${isSpeaking ? "speaking" : ""} ${interrupted ? "interrupted" : ""}`}>
        <canvas ref={canvasRef} width={260} height={260} className="absolute inset-[-30px]" />
        <span>W.A.Y.N.E</span>
        <span className="absolute -right-4 top-4 rounded border border-cyan/30 bg-background px-2 py-1 text-[10px] uppercase text-cyan">{language}</span>
      </div>
      <div className={`mt-10 font-heading text-lg ${isThinking ? "text-amber" : isSpeaking ? "text-success" : interrupted ? "text-danger" : "text-cyan"}`}>{status}</div>
      <div className="mt-2 text-xs text-cyan/50">Speak in any language. Transcription runs locally with Whisper.</div>
      <div className="mt-8 max-w-2xl px-8 text-center font-body">
        <p className={`min-h-8 transition-opacity ${confidenceClass}`}>{liveTranscript.split(/\s+/).slice(-28).join(" ")}</p>
        {confidence != null && <p className="mt-1 text-xs text-cyan/45">{languageName} | confidence {Math.round(confidence * 100)}%</p>}
        {lowConfidenceMessage && <p className="mt-2 text-xs text-danger">{lowConfidenceMessage}</p>}
        <p className="mt-3 min-h-8 text-amber transition-opacity">{aiResponse.split(/\s+/).slice(-28).join(" ")}</p>
      </div>
      <button onClick={() => stopDuplexSession(true)} className="absolute bottom-10 rounded-full border border-danger p-5 text-danger" title="Stop live voice">
        <X size={28} />
      </button>
    </div>
  );
}

function arrayBufferToBase64(buffer: ArrayBuffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let index = 0; index < bytes.byteLength; index += 1) binary += String.fromCharCode(bytes[index]);
  return btoa(binary);
}

function browserTtsLanguage(language: string) {
  const map: Record<string, string> = {
    ta: "ta-IN",
    hi: "hi-IN",
    te: "te-IN",
    kn: "kn-IN",
    ml: "ml-IN",
    fr: "fr-FR",
    es: "es-ES",
    de: "de-DE",
    ja: "ja-JP",
    zh: "zh-CN",
    ar: "ar-SA",
    pt: "pt-BR",
    ru: "ru-RU",
    ko: "ko-KR",
  };
  return map[language] || "en-US";
}
