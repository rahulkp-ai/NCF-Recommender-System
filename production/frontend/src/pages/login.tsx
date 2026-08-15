// apps/web/src/pages/login.tsx
import { useState } from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { useAuth } from "../hooks/useAuth";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);
  const { login }               = useAuth();
  const router                  = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(username, password);
      router.push("/");
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head><title>Sign In — Netfix</title></Head>

      {/* Full-page background */}
      <div className="min-h-screen bg-black flex items-center justify-center
                      bg-[url('/hero-bg.jpg')] bg-cover bg-center relative">
        <div className="absolute inset-0 bg-black/60" />

        <div className="relative z-10 w-full max-w-md mx-4">
          {/* Logo */}
          <p className="text-[#E50914] font-black text-4xl text-center mb-8 tracking-tight">
            NETFIX
          </p>

          {/* Card */}
          <div className="bg-black/80 rounded-md px-10 py-12">
            <h1 className="text-white text-3xl font-bold mb-8">Sign In</h1>

            {error && (
              <div className="bg-[#E50914]/20 border border-[#E50914]/50 text-[#ff6b6b]
                              text-sm px-4 py-3 rounded mb-6">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="input-dark"
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="input-dark"
              />

              <button
                type="submit"
                disabled={loading}
                className="btn-red w-full py-3 mt-2 text-base font-semibold
                           disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? "Signing in…" : "Sign In"}
              </button>
            </form>

            <p className="text-gray-400 text-sm mt-8">
              New to Netfix?{" "}
              <Link href="/signup" className="text-white hover:underline">
                Sign up now.
              </Link>
            </p>

            <p className="text-gray-500 text-xs mt-4">
              Demo accounts: <span className="text-gray-400">user_0001 — user_0010</span>
              <br />Password: <span className="text-gray-400">password</span>
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
