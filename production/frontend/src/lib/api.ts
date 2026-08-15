// apps/web/src/lib/api.ts
import axios from "axios";
import Cookies from "js-cookie";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE, timeout: 10000 });

// Attach JWT on every request
api.interceptors.request.use((config) => {
  const token = Cookies.get("ncf_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ── Types ──────────────────────────────────────────────────────────────────

export interface Movie {
  id: number;
  title: string;
  genres: string | null;
  poster_url: string | null;
  year: number | null;
}

export interface RecommendedMovie extends Movie {
  item_id: number;
  score: number;
  source: string;
  alpha: number;
}

export interface User {
  id: number;
  username: string;
  email: string;
  age_group?: string;
  gender?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface RecommendationResponse {
  user_id: number;
  n_interactions: number;
  alpha: number;
  strategy: string;
  recommendations: RecommendedMovie[];
}

export interface SearchResponse {
  query: string;
  results: Movie[];
  total: number;
}

// ── Auth ───────────────────────────────────────────────────────────────────

export const authApi = {
  register: (data: { username: string; email: string; password: string }) =>
    api.post<AuthResponse>("/api/v1/auth/register", data),

  login: (data: { username: string; password: string }) =>
    api.post<AuthResponse>("/api/v1/auth/login", data),

  profile: () => api.get<User>("/api/v1/users/profile"),
};

// ── Recommendations ────────────────────────────────────────────────────────

export const recommendApi = {
  homepage: (k = 20) =>
    api.get<{ strategy: string; recommendations: RecommendedMovie[] }>(
      `/api/v1/recommend/homepage?k=${k}`
    ),

  forUser: (userId: number, k = 10) =>
    api.get<RecommendationResponse>(`/api/v1/recommend/${userId}?k=${k}`),

  trending: () => api.get<Movie[]>("/api/v1/recommend/trending"),

  popular: () => api.get<Movie[]>("/api/v1/recommend/popular"),
};

// ── Search ─────────────────────────────────────────────────────────────────

export const searchApi = {
  search: (q: string, limit = 20) =>
    api.get<SearchResponse>(`/api/v1/search?q=${encodeURIComponent(q)}&limit=${limit}`),
};

// ── Interactions ───────────────────────────────────────────────────────────

export const interactApi = {
  record: (movie_id: number, event_type: "like" | "click" | "rate", rating?: number) =>
    api.post("/api/v1/interact", { movie_id, event_type, rating }),
};

export default api;
