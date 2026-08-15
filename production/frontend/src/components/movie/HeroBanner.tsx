// apps/web/src/components/movie/HeroBanner.tsx
import { useState } from "react";
import Link from "next/link";
import { FiPlay, FiInfo } from "react-icons/fi";
import { Movie, RecommendedMovie, interactApi } from "../../lib/api";

interface Props {
  movie: Movie | RecommendedMovie | null;
}

export default function HeroBanner({ movie }: Props) {
  const [muted, setMuted] = useState(true);

  if (!movie) {
    return (
      <div className="w-full h-[56vw] max-h-[700px] min-h-[400px] bg-gray-900 animate-pulse" />
    );
  }

  const poster = movie.poster_url
    ? movie.poster_url.replace("/w500/", "/original/")
    : null;

  const genres = movie.genres
    ? movie.genres.split("|").slice(0, 3).join(" • ")
    : "";

  const handlePlay = async () => {
    try { await interactApi.record(movie.id, "click"); } catch {}
  };

  return (
    <div className="relative w-full h-[56vw] max-h-[700px] min-h-[400px] overflow-hidden">
      {/* Background */}
      {poster ? (
        <img
          src={poster}
          alt={movie.title}
          className="absolute inset-0 w-full h-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 to-gray-800" />
      )}

      {/* Gradients for legibility */}
      <div className="absolute inset-0 bg-gradient-to-r from-black/70 via-black/30 to-transparent" />
      <div className="absolute bottom-0 left-0 right-0 h-48
                      bg-gradient-to-t from-[#141414] to-transparent" />

      {/* Content */}
      <div className="absolute bottom-[20%] left-4 md:left-12 max-w-lg">
        <h1 className="text-white text-3xl md:text-5xl font-black mb-3 text-shadow leading-tight">
          {movie.title.replace(/\s*\(\d{4}\)$/, "")}
        </h1>

        {genres && (
          <p className="text-gray-300 text-sm mb-4">{genres}</p>
        )}

        {movie.year && (
          <span className="inline-block border border-gray-400 text-gray-300
                           text-xs px-2 py-0.5 rounded mb-4">
            {movie.year}
          </span>
        )}

        <div className="flex items-center gap-3 mt-2">
          <button
            onClick={handlePlay}
            className="btn-primary text-base px-6 py-2.5"
          >
            <FiPlay className="fill-black" size={18} /> Play
          </button>
          <Link href={`/movie/${movie.id}`}>
            <button className="btn-secondary text-base px-6 py-2.5">
              <FiInfo size={18} /> More Info
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}
