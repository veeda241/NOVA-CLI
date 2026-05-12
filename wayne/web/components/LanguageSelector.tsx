"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

const fallbackLanguages = [
  { code: "auto", name: "Auto Detect" },
  { code: "en", name: "English" },
  { code: "ta", name: "Tamil" },
  { code: "hi", name: "Hindi" },
  { code: "te", name: "Telugu" },
  { code: "kn", name: "Kannada" },
  { code: "ml", name: "Malayalam" },
  { code: "fr", name: "French" },
  { code: "es", name: "Spanish" },
  { code: "de", name: "German" },
  { code: "ja", name: "Japanese" },
  { code: "zh", name: "Chinese" },
  { code: "ar", name: "Arabic" },
  { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" },
  { code: "ko", name: "Korean" },
];

export function LanguageSelector({ value, onChange, compact = false }: { value?: string; onChange?: (language: string) => void; compact?: boolean }) {
  const [languages, setLanguages] = useState(fallbackLanguages);
  const [selected, setSelected] = useState(value || "auto");

  useEffect(() => {
    setSelected(value || "auto");
  }, [value]);

  useEffect(() => {
    fetch(`${API_URL}/voice/languages`)
      .then((response) => response.json())
      .then((data) => {
        if (Array.isArray(data.supported)) {
          setLanguages(data.supported);
          setSelected(value || data.current || "auto");
        }
      })
      .catch(() => undefined);
  }, [value]);

  async function change(next: string) {
    setSelected(next);
    onChange?.(next);
    await fetch(`${API_URL}/voice/language`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: next }),
    }).catch(() => undefined);
  }

  return (
    <label className={`block ${compact ? "" : "wayne-border bg-panel p-3"}`}>
      {!compact && <div className="mb-2 text-xs uppercase text-cyan/60">Voice Language</div>}
      <select value={selected} onChange={(event) => change(event.target.value)} className="wayne-border h-10 w-full bg-background px-2 text-xs text-cyan outline-none">
        {languages.map((language) => (
          <option key={language.code} value={language.code}>
            {flagFor(language.code)} {language.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function flagFor(code: string) {
  const flags: Record<string, string> = {
    auto: "🌐",
    en: "🇬🇧",
    ta: "🇮🇳",
    hi: "🇮🇳",
    te: "🇮🇳",
    kn: "🇮🇳",
    ml: "🇮🇳",
    fr: "🇫🇷",
    es: "🇪🇸",
    de: "🇩🇪",
    ja: "🇯🇵",
    zh: "🇨🇳",
    ar: "🇸🇦",
    pt: "🇧🇷",
    ru: "🇷🇺",
    ko: "🇰🇷",
  };
  return flags[code] || "🌐";
}
