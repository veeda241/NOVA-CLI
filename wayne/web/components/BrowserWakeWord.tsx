"use client";

import { useEffect, useRef } from "react";

type BrowserSpeechRecognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: { results: ArrayLike<{ [index: number]: { transcript: string } }> }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => BrowserSpeechRecognition;

export default function BrowserWakeWord({ onWake }: { onWake: () => void }) {
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const activeRef = useRef(true);

  useEffect(() => {
    const speechWindow = window as unknown as {
      SpeechRecognition?: SpeechRecognitionConstructor;
      webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
    const Recognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition;
    if (!Recognition) return;

    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript || "")
        .join(" ")
        .toLowerCase();
      if (transcript.includes("wayne") || transcript.includes("hey wayne")) onWake();
    };
    recognition.onend = () => {
      if (activeRef.current) {
        try {
          recognition.start();
        } catch {
          window.setTimeout(() => recognition.start(), 1000);
        }
      }
    };

    try {
      recognition.start();
      recognitionRef.current = recognition;
    } catch {
      return;
    }

    return () => {
      activeRef.current = false;
      recognition.stop();
    };
  }, [onWake]);

  return null;
}
