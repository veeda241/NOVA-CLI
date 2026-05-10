import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.WAYNE_API_URL || "http://localhost:8000";

export async function PATCH(request: NextRequest) {
  const payload = await request.json();
  try {
    const response = await fetch(`${API_URL}/learning/preferences`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store"
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ status: "failed" }, { status: 200 });
  }
}
