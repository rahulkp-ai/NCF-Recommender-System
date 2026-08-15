// apps/web/src/pages/movie/[id].tsx
// apps/web/src/pages/movie/[id].tsx
import { useState, useEffect } from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { FiPlay, FiThumbsUp, FiPlus, FiArrowLeft, FiStar } from "react-icons/fi";
import { Movie, interactApi } from "../../lib/api";
import api from "../../lib/api";

const PLACEHOLDER = "https://via.placeholder.com/300x450/1a1a1a/555?text=No+Poster";

export default function MovieDetailPage() {
  const router                            = useRouter();
  const { id }                            = router.query;
  const [movie, setMovie]                 = useState<Movie | null>(null);
  const [loading, setLoading]             = useState(true);
  const [error, setError]                 = useState(false);
  const [liked, setLiked]                 = useState(false);
  const [rating, setRating]               = useState(0);
  const [rated, setRated]                 = useState(false);
  const [similar, setSimilar]             = useState<Movie[]>([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  // Fetch movie details
  useEffect(() => {
    if (!id || id === "undefined") return;
    setLoading(true);
    setError(false);
    setSimilar([]);
    setLiked(false);
    api.get<Movie>(`/api/v1/recommend/movie/${id}`)
      .then((res) => setMovie(res.data))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [id]);

  // Called when user clicks Like
  const handleLike = async () => {
    if (!movie || liked) return;
    // Optimistically mark liked immediately
    setLiked(true);
    setLoadingSimilar(true);
    try {
      await interactApi.record(movie.id, "like");
    } catch {}
    // Fetch similar movies by genre
    try {
      const res = await api.get<Movie[]>(`/api/v1/recommend/similar/${movie.id}?limit=12`);
      setSimilar(res.data);
    } catch {}
    setLoadingSimilar(false);
  };

  const handleRate = async (stars: number) => {
    if (!movie || rated) return;
    setRating(stars);
    try { await interactApi.record(movie.id, "rate", stars); setRated(true); } catch {}
  };

  const genres = movie?.genres?.split("|") ?? [];

  if (loading) {
    return (
      <div className="min-h-screen bg-[#141414] flex items-center justify-center pt-20">
        <div className="w-12 h-12 border-4 border-[#E50914] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !movie) {
    return (
      <div className="min-h-screen bg-[#141414] flex flex-col items-center justify-center pt-20 gap-4">
        <p className="text-5xl">🎬</p>
        <p className="text-white text-xl font-semibold">Movie not found</p>
        <button onClick={() => router.back()} className="btn-secondary flex items-center gap-2">
          <FiArrowLeft /> Go Back
        </button>
      </div>
    );
  }

  return (
    <>
      <Head><title>{movie.title} — Netfix</title></Head>

      <div className="min-h-screen bg-[#141414] pt-16">
        {/* Hero backdrop */}
        <div className="relative h-[50vh] overflow-hidden">
          {movie.poster_url ? (
            <img
              src={movie.poster_url.replace("/w500/", "/original/")}
              alt={movie.title}
              className="w-full h-full object-cover object-top opacity-40"
            />
          ) : (
            <div className="w-full h-full bg-gradient-to-br from-gray-900 to-gray-800" />
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-[#141414] via-[#141414]/60 to-transparent" />
        </div>

        {/* Main content */}
        <div className="relative -mt-48 z-10 px-4 md:px-12 pb-8">
          <div className="flex flex-col md:flex-row gap-8">
            {/* Poster */}
            <div className="flex-shrink-0">
              <img
                src={movie.poster_url || PLACEHOLDER}
                alt={movie.title}
                className="w-48 md:w-64 rounded-lg shadow-2xl"
                onError={(e) => { (e.target as HTMLImageElement).src = PLACEHOLDER; }}
              />
            </div>

            {/* Info */}
            <div className="flex-1 pt-4 md:pt-12">
              <button
                onClick={() => router.back()}
                className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-4 text-sm"
              >
                <FiArrowLeft /> Back
              </button>

              <h1 className="text-white text-3xl md:text-4xl font-black mb-2 leading-tight">
                {movie.title.replace(/\s*\(\d{4}\)$/, "")}
              </h1>

              <div className="flex items-center gap-3 mb-4 flex-wrap">
                {movie.year && <span className="text-gray-400 text-sm">{movie.year}</span>}
                {genres.slice(0, 3).map((g) => (
                  <span key={g} className="text-xs border border-gray-600 text-gray-300 px-2 py-0.5 rounded">
                    {g}
                  </span>
                ))}
              </div>

              {/* Actions */}
              <div className="flex items-center gap-3 mb-6">
                <button className="btn-primary px-8 py-3 text-base">
                  <FiPlay className="fill-black" size={18} /> Play
                </button>

                {/* Like button — triggers similar movies */}
                <button
                  onClick={handleLike}
                  disabled={liked}
                  title={liked ? "Liked!" : "Like this movie"}
                  className={`w-11 h-11 border-2 rounded-full flex items-center justify-center
                    transition-all duration-300 disabled:cursor-default
                    ${liked
                      ? "border-[#E50914] text-[#E50914] bg-[#E50914]/10 scale-110"
                      : "border-white text-white hover:border-[#E50914] hover:text-[#E50914]"
                    }`}
                >
                  <FiThumbsUp size={18} className={liked ? "fill-[#E50914]" : ""} />
                </button>

                <button className="w-11 h-11 border-2 border-white rounded-full flex items-center justify-center text-white hover:border-gray-400 transition-colors">
                  <FiPlus size={18} />
                </button>
              </div>

              {/* Like prompt — shown before liking */}
              {!liked && (
                <p className="text-gray-500 text-sm mb-6 italic">
                  👍 Like this movie to see similar recommendations
                </p>
              )}

              {/* Star rating */}
              <div className="mb-6">
                <p className="text-gray-400 text-sm mb-2">Rate this movie</p>
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((s) => (
                    <button
                      key={s}
                      onClick={() => handleRate(s)}
                      disabled={rated}
                      className={`transition-colors disabled:cursor-not-allowed
                        ${s <= rating ? "text-yellow-400" : "text-gray-600 hover:text-yellow-300"}`}
                    >
                      <FiStar size={24} className={s <= rating ? "fill-yellow-400" : ""} />
                    </button>
                  ))}
                  {rated && <span className="text-gray-400 text-sm ml-2 self-center">Thanks!</span>}
                </div>
              </div>

              {/* Genre pills */}
              {genres.length > 0 && (
                <div>
                  <p className="text-gray-400 text-sm mb-2">Genres</p>
                  <div className="flex flex-wrap gap-2">
                    {genres.map((g) => (
                      <span key={g} className="bg-gray-800 text-gray-300 text-sm px-3 py-1 rounded-full">
                        {g}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Similar Movies (appears after Like) ─────────────────────────── */}
        {liked && (
          <div className="px-4 md:px-12 pb-16 mt-4">
            <div className="border-t border-gray-800 pt-8">
              <h2 className="text-white text-xl font-bold mb-1">
                Because you liked{" "}
                <span className="text-[#E50914]">
                  {movie.title.replace(/\s*\(\d{4}\)$/, "")}
                </span>
              </h2>
              <p className="text-gray-500 text-sm mb-6">
                Similar movies in {genres.slice(0, 2).join(", ")}
              </p>

              {/* Loading spinner */}
              {loadingSimilar && (
                <div className="flex gap-3">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="w-36 md:w-44 aspect-[2/3] bg-gray-800 rounded-md animate-pulse flex-shrink-0" />
                  ))}
                </div>
              )}

              {/* Similar movie grid */}
              {!loadingSimilar && similar.length > 0 && (
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
                  {similar.map((m) => (
                    <Link key={m.id} href={`/movie/${m.id}`}>
                      <div className="group cursor-pointer rounded-md overflow-hidden
                                      transition-transform duration-200 hover:scale-105 hover:shadow-xl">
                        <div className="relative aspect-[2/3]">
                          <img
                            src={m.poster_url || PLACEHOLDER}
                            alt={m.title}
                            className="w-full h-full object-cover"
                            onError={(e) => { (e.target as HTMLImageElement).src = PLACEHOLDER; }}
                          />
                          {/* Hover overlay */}
                          <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100
                                          transition-opacity duration-200 flex items-end p-2">
                            <p className="text-white text-xs font-semibold line-clamp-2">{m.title}</p>
                          </div>
                        </div>
                        {m.year && (
                          <p className="text-gray-500 text-xs text-center mt-1">{m.year}</p>
                        )}
                      </div>
                    </Link>
                  ))}
                </div>
              )}

              {/* No results */}
              {!loadingSimilar && similar.length === 0 && (
                <p className="text-gray-500 text-sm">No similar movies found.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}