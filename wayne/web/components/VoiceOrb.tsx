"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type SpeechRecognitionResultEvent = {
  results: {
    length: number;
    [index: number]: {
      isFinal?: boolean;
      [index: number]: { transcript: string };
    };
  };
};

type BrowserSpeechRecognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionResultEvent) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => BrowserSpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace("http://", "ws://").replace("https://", "wss://");

export function VoiceOrb({ onClose }: { onClose: (finalTranscript: string) => void }) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [interrupted, setInterrupted] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [aiResponse, setAiResponse] = useState("");
  const socketRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ttsBufferRef = useRef("");
  const interruptFramesRef = useRef(0);
  const detectorTimerRef = useRef<number | null>(null);
  const speechEndTimerRef = useRef<number | null>(null);
  const lastSentTranscriptRef = useRef("");

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

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const AudioCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioContext = new AudioCtor();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      startInterruptionDetector();

      startSpeechRecognition();
    } catch (error) {
      setIsListening(false);
      setIsThinking(false);
      setAiResponse(error instanceof Error ? error.message : "Voice session could not start.");
    }
  }

  function startSpeechRecognition() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return;
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      let text = "";
      let finalSeen = false;
      for (let index = 0; index < event.results.length; index += 1) {
        text += event.results[index]?.[0]?.transcript || "";
        finalSeen = finalSeen || Boolean(event.results[index]?.isFinal);
      }
      setLiveTranscript(text);
      queueSpeechEnd(text, finalSeen);
    };
    recognition.onend = () => {
      if (socketRef.current?.readyState === WebSocket.OPEN) recognition.start();
    };
    recognition.onerror = () => {
      setAiResponse("Browser speech recognition stopped. Check microphone permission, then reopen live voice.");
      setIsThinking(false);
      setIsListening(true);
    };
    recognition.start();
    recognitionRef.current = recognition;
  }

  function queueSpeechEnd(text: string, immediate = false) {
    const transcript = text.trim().replace(/\s+/g, " ");
    if (transcript.length < 2 || transcript === lastSentTranscriptRef.current) return;
    if (speechEndTimerRef.current) window.clearTimeout(speechEndTimerRef.current);

    const send = () => {
      if (transcript === lastSentTranscriptRef.current) return;
      lastSentTranscriptRef.current = transcript;
      setAiResponse("");
      setIsThinking(true);
      setIsListening(false);
      sendSocketMessage({ type: "speech_end", text: transcript });
    };

    if (immediate) {
      send();
      return;
    }
    speechEndTimerRef.current = window.setTimeout(send, 1100);
  }

  function handleWebSocketMessage(event: MessageEvent) {
    const message = JSON.parse(event.data);
    switch (message.type) {
      case "ready":
        setIsListening(true);
        speak("W.A.Y.N.E online. How can I assist you?");
        break;
      case "transcribing":
        setIsThinking(true);
        setIsListening(false);
        break;
      case "transcript":
        setLiveTranscript(message.text || "");
        setIsThinking(true);
        break;
      case "ai_token":
        setAiResponse((current) => current + message.token);
        speakIncremental(message.token);
        break;
      case "ai_done":
        flushSpeech();
        setIsThinking(false);
        setIsListening(true);
        break;
      case "interrupted":
        speechSynthesis.cancel();
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
    if (/[.?!]\s*$/.test(ttsBufferRef.current) || words.length >= 10) {
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
    utterance.voice = speechSynthesis.getVoices().find((voice) => voice.name.includes("Google UK English Male")) ?? speechSynthesis.getVoices()[0] ?? null;
    utterance.rate = 0.95;
    utterance.pitch = 0.85;
    utterance.onstart = () => {
      setIsSpeaking(true);
      setIsListening(false);
      setIsThinking(false);
    };
    utterance.onend = () => {
      setIsSpeaking(false);
      setIsListening(true);
    };
    speechSynthesis.speak(utterance);
  }

  function startInterruptionDetector() {
    const data = new Uint8Array(2048);
    const tick = () => {
      const analyser = analyserRef.current;
      if (analyser) {
        analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let index = 0; index < data.length; index += 1) {
          const value = data[index];
          const normalized = (value - 128) / 128;
          sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / data.length);
        if (rms > 0.02 && speechSynthesis.speaking) {
          interruptFramesRef.current += 1;
        } else {
          interruptFramesRef.current = 0;
        }
        if (interruptFramesRef.current >= 3) {
          speechSynthesis.cancel();
          sendSocketMessage({ type: "interrupt" });
          setIsSpeaking(false);
          setIsListening(true);
          setInterrupted(true);
          setTimeout(() => setInterrupted(false), 500);
          interruptFramesRef.current = 0;
        }
      }
      detectorTimerRef.current = window.setTimeout(tick, 100);
    };
    tick();
  }

  function sendSocketMessage(payload: Record<string, unknown>) {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  }

  function stopDuplexSession(closeOverlay = true) {
    sendSocketMessage({ type: "stop_voice" });
    if (socketRef.current && socketRef.current.readyState !== WebSocket.CLOSED) socketRef.current.close();
    try {
      recognitionRef.current?.stop();
    } catch {
      // SpeechRecognition can already be stopped during React dev reloads.
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close().catch(() => undefined);
    }
    if (detectorTimerRef.current) {
      window.clearTimeout(detectorTimerRef.current);
      detectorTimerRef.current = null;
    }
    if (speechEndTimerRef.current) {
      window.clearTimeout(speechEndTimerRef.current);
      speechEndTimerRef.current = null;
    }
    socketRef.current = null;
    recognitionRef.current = null;
    streamRef.current = null;
    audioContextRef.current = null;
    analyserRef.current = null;
    speechSynthesis.cancel();
    if (closeOverlay) speak("W.A.Y.N.E standing by.");
    setIsListening(false);
    setIsSpeaking(false);
    setIsThinking(false);
    if (closeOverlay) onClose(liveTranscript);
  }

  const status = interrupted ? "Interrupted" : isThinking ? "Thinking..." : isSpeaking ? "Speaking..." : "Listening...";

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background/95 text-cyan backdrop-blur">
      <div className={`voice-orb ${isListening ? "listening" : ""} ${isThinking ? "thinking" : ""} ${isSpeaking ? "speaking" : ""} ${interrupted ? "interrupted" : ""}`}>
        <canvas ref={canvasRef} width={260} height={260} className="absolute inset-[-30px]" />
        <span>W.A.Y.N.E</span>
      </div>
      <div className={`mt-10 font-heading text-lg ${isThinking ? "text-amber" : isSpeaking ? "text-success" : interrupted ? "text-danger" : "text-cyan"}`}>{status}</div>
      <div className="mt-8 max-w-2xl px-8 text-center font-body">
        <p className="min-h-8 text-cyan transition-opacity">{liveTranscript.split(/\s+/).slice(-28).join(" ")}</p>
        <p className="mt-3 min-h-8 text-amber transition-opacity">{aiResponse.split(/\s+/).slice(-28).join(" ")}</p>
      </div>
      <button onClick={() => stopDuplexSession(true)} className="absolute bottom-10 rounded-full border border-danger p-5 text-danger" title="Stop live voice">
        <X size={28} />
      </button>
    </div>
  );
}
