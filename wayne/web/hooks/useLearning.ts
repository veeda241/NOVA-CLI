"use client";

import { useCallback, useEffect, useState } from "react";

export type LearningStats = {
  total_interactions: number;
  average_reward: number;
  learning_score: number;
  top_topics: { topic: string; frequency: number; avg_score: number }[];
  learned_preferences: { key: string; value: string; confidence: number; sample_count: number }[];
};

export function useLearning(refreshMs = 30000) {
  const [stats, setStats] = useState<LearningStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/learning/stats", { cache: "no-store" });
      if (!response.ok) throw new Error(`Learning stats failed: ${response.status}`);
      setStats(await response.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Learning stats unavailable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, refreshMs);
    return () => window.clearInterval(id);
  }, [refresh, refreshMs]);

  return { stats, loading, error, refresh };
}
