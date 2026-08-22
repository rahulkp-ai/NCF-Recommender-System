// apps/web/src/components/movie/MovieCard.tsx
import { useState } from "react";
import Link from "next/link";
import { FiPlay, FiPlus, FiThumbsUp } from "react-icons/fi";
import { Movie, RecommendedMovie, interactApi } from "../../lib/api";

interface Props {
  movie: Movie | RecommendedMovie;
  showBadge?: boolean;
}

const PLACEHOLDER =
  "data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='450' viewBox='0 0 300 450'%3E%3Crect width='300' height='450' fill='%231a1a1a'/%3E%3Ctext x='150' y='225' font-family='sans-serif' font-size='18' fill='%23555' text-anchor='middle' dominant-baseline='middle'%3ENo Poster%3C/text%3E%3C/svg%3E";

export default function MovieCard({ movie, showBadge }: Props) {
  const [liked, setLiked]     = useState(false);
  const [hovered, setHovered] = useState(false);
  const poster = movie.poster_url || PLACEHOLDER;
  const source = (movie as RecommendedMovie).source;

  const handleLike = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await interactApi.record(movie.id, "like");
      setLiked(true);
    } catch {}
  };

  const handleClick = async () => {
    try { await interactApi.record(movie.id, "click"); } catch {}
  };

  return (
    <Link href={`/movie/${movie.id}`} onClick={handleClick}>
      <div
        className="relative flex-shrink-0 w-36 md:w-44 lg:w-52 cursor-pointer
                   rounded-md overflow-hidden transition-all duration-300
                   hover:scale-105 hover:z-10 hover:shadow-2xl group"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* Poster */}
        <img
          src={poster}
          alt={movie.title}
          className="w-full aspect-[2/3] object-cover"
          onError={(e) => { (e.target as HTMLImageElement).src = PLACEHOLDER; }}
        />

        {/* Source badge */}
        {showBadge && source && source !== "cold_start" && (
          <div className="absolute top-2 left-2 bg-[#E50914] text-white text-[10px]
                          font-bold px-1.5 py-0.5 rounded">
            {source === "ncf" ? "FOR YOU" : "TRENDING"}
          </div>
        )}

        {/* Hover overlay */}
        <div className={`absolute inset-0 bg-black/60 flex flex-col justify-end p-3
                         transition-opacity duration-200
                         ${hovered ? "opacity-100" : "opacity-0"}`}>
          <p className="text-white text-xs font-semibold line-clamp-2 mb-2">
            {movie.title}
          </p>
          <div className="flex items-center gap-2">
            <button className="w-7 h-7 bg-white rounded-full flex items-center justify-center
                               hover:bg-gray-200 transition-colors">
              <FiPlay size={12} className="text-black fill-black" />
            </button>
            <button
              onClick={handleLike}
              className={`w-7 h-7 border-2 rounded-full flex items-center justify-center
                          transition-colors
                          ${liked
                            ? "border-[#E50914] text-[#E50914]"
                            : "border-white text-white hover:border-gray-300"}`}
            >
              <FiThumbsUp size={12} />
            </button>
            <button className="w-7 h-7 border-2 border-white rounded-full flex items-center
                               justify-center text-white hover:border-gray-300 transition-colors">
              <FiPlus size={12} />
            </button>
          </div>
          {movie.year && (
            <p className="text-gray-400 text-[10px] mt-1">{movie.year}</p>
          )}
        </div>
      </div>
    </Link>
  );
}
