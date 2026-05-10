import { NextResponse } from "next/server";

const API_URL = process.env.WAYNE_API_URL || "http://localhost:8000";

export async function GET() {
  try {
    const response = await fetch(`${API_URL}/learning/stats`, { cache: "no-store" });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({
      total_interactions: 0,
      average_reward: 0,
      golden_responses_stored: 0,
      top_topics: [],
      learned_preferences: [],
      learning_score: 0
    });
  }
}
