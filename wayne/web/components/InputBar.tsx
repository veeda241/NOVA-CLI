"use client";

import { Loader2, Mic, SendHorizontal } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { VoiceOrb } from "./VoiceOrb";

type OneShotSpeechRecognitionResultEvent = {
  results: {
    [index: number]: {
      [index: number]: {
        transcript: string;
      };
    };
  };
};

type BrowserSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  onresult: ((event: OneShotSpeechRecognitionResultEvent) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start: () => void;
};

type SpeechRecognitionConstructor = new () => BrowserSpeechRecognition;

export function InputBar({ onSend, loading }: { onSend: (text: string) => void; loading: boolean }) {
  const [value, setValue] = useState("");
  const [recording, setRecording] = useState(false);
  const [liveVoice, setLiveVoice] = useState(false);

  useEffect(() => {
    const openLiveVoice = () => setLiveVoice(true);
    window.addEventListener("wayne:start-voice", openLiveVoice);
    return () => window.removeEventListener("wayne:start-voice", openLiveVoice);
  }, []);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!value.trim()) return;
    onSend(value);
    setValue("");
  }

  function startVoice() {
    const speechWindow = window as unknown as { SpeechRecognition?: SpeechRecognitionConstructor; webkitSpeechRecognition?: SpeechRecognitionConstructor };
    const Recognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition;
    if (!Recognition) return;
    const recognition = new Recognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    setRecording(true);
    recognition.onresult = (event) => {
      const text = event.results[0]?.[0]?.transcript || "";
      setValue(text);
      if (text) onSend(text);
    };
    recognition.onend = () => setRecording(false);
    recognition.onerror = () => setRecording(false);
    recognition.start();
  }

  return (
    <>
      {liveVoice && (
        <VoiceOrb
          onClose={(finalTranscript) => {
            setLiveVoice(false);
            if (finalTranscript.trim()) setValue(finalTranscript);
          }}
        />
      )}
      <form onSubmit={submit} className="wayne-border flex items-center gap-2 bg-surface p-3">
        <button type="button" onClick={() => setLiveVoice(true)} className="wayne-border relative bg-panel p-3 text-cyan shadow-[0_0_18px_rgba(0,212,255,0.25)]" title="Live voice">
          <Mic size={18} />
          <span className="absolute -right-2 -top-2 rounded-full bg-cyan px-1.5 py-0.5 text-[9px] text-background">Live</span>
        </button>
        <button type="button" onClick={startVoice} className={`wayne-border p-3 ${recording ? "animate-pulseMic bg-danger/20 text-danger" : "bg-panel text-cyan/70"}`} title="One-shot voice input">
          <Mic size={18} />
        </button>
        <input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          disabled={loading}
          placeholder="Issue a command..."
          className="wayne-border min-w-0 flex-1 bg-background px-4 py-3 text-sm text-white outline-none focus:border-cyan"
        />
        <button disabled={loading} className="wayne-border bg-cyan/10 p-3 text-cyan disabled:opacity-50" title="Send">
          {loading ? <Loader2 size={18} className="animate-spin" /> : <SendHorizontal size={18} />}
        </button>
      </form>
    </>
  );
}
