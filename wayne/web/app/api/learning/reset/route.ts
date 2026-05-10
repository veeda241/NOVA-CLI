import { NextResponse } from "next/server";

const API_URL = process.env.WAYNE_API_URL || "http://localhost:8000";

export async function DELETE() {
  try {
    const response = await fetch(`${API_URL}/learning/reset`, { method: "DELETE", cache: "no-store" });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ status: "failed" }, { status: 200 });
  }
}
