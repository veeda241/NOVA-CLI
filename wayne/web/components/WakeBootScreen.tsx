"use client";

import { useEffect, useState } from "react";

const BOOT_STEPS = [
  "Initializing neural core...",
  "Loading Gemma local language model...",
  "Checking local backend link...",
  "Syncing task database...",
  "Mounting file system index...",
  "Activating device control layer...",
  "Establishing laptop agent link...",
  "Loading calendar integration...",
  "Warming up voice synthesis...",
  "All systems nominal."
];

const WAYNE_LOGO = ` ██╗    ██╗ █████╗ ██╗   ██╗███╗   ██╗███████╗
 ██║    ██║██╔══██╗╚██╗ ██╔╝████╗  ██║██╔════╝
 ██║ █╗ ██║███████║ ╚████╔╝ ██╔██╗ ██║█████╗
 ██║███╗██║██╔══██║  ╚██╔╝  ██║╚██╗██║██╔══╝
 ╚███╔███╔╝██║  ██║   ██║   ██║ ╚████║███████╗
  ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝╚══════╝`;

export default function WakeBootScreen({ mode = "wake" }: { mode?: "wake" | "sleep" }) {
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (mode === "sleep") {
      setDone(true);
      return;
    }
    let index = 0;
    const interval = window.setInterval(() => {
      if (index < BOOT_STEPS.length) {
        setCompletedSteps((current) => [...current, BOOT_STEPS[index]]);
        index += 1;
      } else {
        setDone(true);
        window.clearInterval(interval);
      }
    }, 350);
    return () => window.clearInterval(interval);
  }, [mode]);

  return (
    <div className="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-background text-cyan backdrop-blur">
      <pre className="mb-6 whitespace-pre text-[clamp(6px,1.5vw,13px)] leading-tight text-cyan drop-shadow-[0_0_18px_rgba(0,212,255,0.35)]">{WAYNE_LOGO}</pre>
      <p className="mb-8 text-center font-body text-[13px] tracking-[2px] text-amber">WIRELESS ARTIFICIAL YIELDING NETWORK ENGINE</p>
      {mode === "sleep" ? (
        <div className="wayne-border px-8 py-5 text-center">
          <p className="font-heading text-lg text-amber">W.A.Y.N.E STANDBY</p>
          <p className="mt-2 text-xs text-cyan/60">Say WAYNE to reinitialize.</p>
        </div>
      ) : (
        <>
          <div className="w-[min(480px,90vw)] text-left">
            {completedSteps.map((step, index) => (
              <div key={`${step}-${index}`} className={`animate-fade py-1 text-xs ${index === completedSteps.length - 1 ? "text-cyan" : "text-cyan/45"}`}>
                <span className="mr-2 text-success">✓</span>
                {step}
              </div>
            ))}
          </div>
          {done && (
            <div className="wayne-border mt-8 animate-fade px-8 py-4 text-center">
              <p className="font-heading text-base text-success">W.A.Y.N.E ONLINE</p>
              <p className="mt-1 text-[11px] text-cyan/55">Online. How can I assist you, Sir?</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
