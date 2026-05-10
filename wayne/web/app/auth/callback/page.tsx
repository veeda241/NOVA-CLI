"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_WAYNE_API_URL || "http://localhost:8000";

function AuthCallbackContent() {
  const params = useSearchParams();
  const [status, setStatus] = useState("Connecting Google Calendar...");

  useEffect(() => {
    const code = params.get("code");
    if (!code) {
      setStatus("Missing OAuth code.");
      return;
    }
    fetch(`${API_URL}/auth/callback?code=${encodeURIComponent(code)}`)
      .then((response) => {
        if (!response.ok) throw new Error("OAuth callback failed");
        setStatus("Google Calendar connected. You may return to W.A.Y.N.E.");
      })
      .catch(() => setStatus("Google Calendar connection failed. Check backend credentials."));
  }, [params]);

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-6 text-cyan">
      <section className="wayne-border bg-surface p-6 text-center">
        <h1 className="mb-3 font-heading text-2xl">W.A.Y.N.E Calendar Link</h1>
        <p className="text-sm text-cyan/80">{status}</p>
      </section>
    </main>
  );
}

export default function AuthCallback() {
  return (
    <Suspense
      fallback={
        <main className="flex min-h-screen items-center justify-center bg-background p-6 text-cyan">
          <section className="wayne-border bg-surface p-6 text-center">
            <h1 className="mb-3 font-heading text-2xl">W.A.Y.N.E Calendar Link</h1>
            <p className="text-sm text-cyan/80">Preparing callback...</p>
          </section>
        </main>
      }
    >
      <AuthCallbackContent />
    </Suspense>
  );
}
