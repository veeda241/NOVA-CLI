import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.WAYNE_API_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  const payload = await request.json();
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store"
    });
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("text/event-stream") && response.body) {
      return new Response(response.body, {
        status: response.status,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive"
        }
      });
    }
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { reply: "[OFFLINE MODE] [AI RESPONSE] W.A.Y.N.E backend unavailable. Start FastAPI on port 8000.", tool_used: null, messages: [] },
      { status: 200 }
    );
  }
}
