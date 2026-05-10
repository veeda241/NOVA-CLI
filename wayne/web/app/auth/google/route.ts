import { NextResponse } from "next/server";

const API_URL = process.env.WAYNE_API_URL || "http://localhost:8000";

export function GET() {
  return NextResponse.redirect(`${API_URL}/auth/google`);
}
