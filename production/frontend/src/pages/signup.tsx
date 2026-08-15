// apps/web/src/pages/signup.tsx
import { useState } from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { useAuth } from "../hooks/useAuth";

export default function SignupPage() {
  const [form, setForm]     = useState({ username: "", email: "", password: "" });
  const [error, setError]   = useState("");
  const [loading, setLoading] = useState(false);
  const { register }        = useAuth();
  const router              = useRouter();

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await register(form.username, form.email, form.password);
      router.push("/");
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head><title>Sign Up — Netfix</title></Head>

      <div className="min-h-screen bg-black flex items-center justify-center relative">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 to-black" />

        <div className="relative z-10 w-full max-w-md mx-4">
          <p className="text-[#E50914] font-black text-4xl text-center mb-8 tracking-tight">
            NETFIX
          </p>

          <div className="bg-black/80 rounded-md px-10 py-12 border border-gray-800">
            <h1 className="text-white text-3xl font-bold mb-2">Create Account</h1>
            <p className="text-gray-400 text-sm mb-8">
              Get personalised movie recommendations.
            </p>

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
                value={form.username}
                onChange={set("username")}
                required
                minLength={3}
                className="input-dark"
              />
              <input
                type="email"
                placeholder="Email address"
                value={form.email}
                onChange={set("email")}
                required
                className="input-dark"
              />
              <input
                type="password"
                placeholder="Password (min 6 chars)"
                value={form.password}
                onChange={set("password")}
                required
                minLength={6}
                className="input-dark"
              />

              <button
                type="submit"
                disabled={loading}
                className="btn-red w-full py-3 mt-2 text-base font-semibold
                           disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? "Creating account…" : "Get Started"}
              </button>
            </form>

            <p className="text-gray-400 text-sm mt-8">
              Already have an account?{" "}
              <Link href="/login" className="text-white hover:underline">
                Sign in.
              </Link>
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
