// apps/web/src/hooks/useRecommendations.ts
import { useState, useEffect } from "react";
import { recommendApi, RecommendedMovie, Movie } from "../lib/api";

export function useHomepageRecs() {
  const [recs, setRecs]       = useState<RecommendedMovie[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    recommendApi.homepage(20)
      .then((r) => setRecs(r.data.recommendations))
      .catch(() => setError("Failed to load recommendations"))
      .finally(() => setLoading(false));
  }, []);

  return { recs, loading, error };
}

export function useUserRecs(userId: number | null) {
  const [recs, setRecs]       = useState<RecommendedMovie[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!userId) return;
    setLoading(true);
    recommendApi.forUser(userId, 10)
      .then((r) => setRecs(r.data.recommendations))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [userId]);

  return { recs, loading };
}

export function useTrending() {
  const [movies, setMovies]   = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    recommendApi.trending()
      .then((r) => setMovies(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return { movies, loading };
}

export function usePopular() {
  const [movies, setMovies]   = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    recommendApi.popular()
      .then((r) => setMovies(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return { movies, loading };
}
