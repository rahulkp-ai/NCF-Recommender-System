// apps/web/src/components/movie/MovieRow.tsx
import { useRef } from "react";
import { FiChevronLeft, FiChevronRight } from "react-icons/fi";
import MovieCard from "./MovieCard";
import { Movie, RecommendedMovie } from "../../lib/api";

interface Props {
  title: string;
  movies: (Movie | RecommendedMovie)[];
  showBadge?: boolean;
  loading?: boolean;
}

export default function MovieRow({ title, movies, showBadge, loading }: Props) {
  const rowRef = useRef<HTMLDivElement>(null);

  const scroll = (dir: "left" | "right") => {
    if (!rowRef.current) return;
    const amount = rowRef.current.clientWidth * 0.75;
    rowRef.current.scrollBy({ left: dir === "right" ? amount : -amount, behavior: "smooth" });
  };

  return (
    <div className="mb-8 group/row">
      <h2 className="text-white text-lg md:text-xl font-semibold px-4 md:px-12 mb-3">
        {title}
      </h2>

      <div className="relative px-4 md:px-12">
        {/* Left arrow */}
        <button
          onClick={() => scroll("left")}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-10 h-full
                     bg-black/50 flex items-center justify-center
                     opacity-0 group-hover/row:opacity-100 transition-opacity
                     hover:bg-black/70"
        >
          <FiChevronLeft size={24} className="text-white" />
        </button>

        {/* Cards */}
        {loading ? (
          <div className="flex gap-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="flex-shrink-0 w-36 md:w-44 lg:w-52 aspect-[2/3]
                                      bg-gray-800 rounded-md animate-pulse" />
            ))}
          </div>
        ) : (
          <div ref={rowRef} className="row-scroll">
            {movies.map((m) => (
              <MovieCard key={m.id} movie={m} showBadge={showBadge} />
            ))}
          </div>
        )}

        {/* Right arrow */}
        <button
          onClick={() => scroll("right")}
          className="absolute right-0 top-1/2 -translate-y-1/2 z-10 w-10 h-full
                     bg-black/50 flex items-center justify-center
                     opacity-0 group-hover/row:opacity-100 transition-opacity
                     hover:bg-black/70"
        >
          <FiChevronRight size={24} className="text-white" />
        </button>
      </div>
    </div>
  );
}
