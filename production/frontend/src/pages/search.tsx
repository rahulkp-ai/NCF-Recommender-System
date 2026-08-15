// apps/web/src/pages/search.tsx
import { useState, useEffect } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import SearchBar from "../components/search/SearchBar";
import MovieCard from "../components/movie/MovieCard";
import { searchApi, Movie } from "../lib/api";

export default function SearchPage() {
  const router               = useRouter();
  const q                    = (router.query.q as string) || "";
  const [results, setResults] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    if (!q) return;
    setLoading(true);
    searchApi.search(q, 40)
      .then((res) => setResults(res.data.results))
      .catch(() => setResults([]))
      .finally(() => { setLoading(false); setSearched(true); });
  }, [q]);

  return (
    <>
      <Head>
        <title>{q ? `"${q}" — Netfix` : "Search — Netfix"}</title>
      </Head>

      <div className="min-h-screen bg-[#141414] pt-24 px-4 md:px-12 pb-16">
        {/* Search bar */}
        <div className="mb-10">
          <SearchBar initialQuery={q} />
        </div>

        {/* Results header */}
        {searched && (
          <p className="text-gray-400 text-sm mb-6">
            {results.length > 0
              ? `${results.length} result${results.length !== 1 ? "s" : ""} for "${q}"`
              : `No results for "${q}"`}
          </p>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {Array.from({ length: 18 }).map((_, i) => (
              <div key={i} className="aspect-[2/3] bg-gray-800 rounded-md animate-pulse" />
            ))}
          </div>
        )}

        {/* Results grid */}
        {!loading && results.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {results.map((m) => (
              <MovieCard key={m.id} movie={m} />
            ))}
          </div>
        )}

        {/* No results */}
        {!loading && searched && results.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 gap-4">
            <p className="text-5xl">🎬</p>
            <p className="text-white text-xl font-semibold">No titles found</p>
            <p className="text-gray-400 text-sm">
              Try different keywords or browse our catalogue.
            </p>
          </div>
        )}

        {/* Browse prompt when no query */}
        {!q && !loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-4">
            <p className="text-5xl">🔍</p>
            <p className="text-white text-xl font-semibold">Search for movies</p>
            <p className="text-gray-400 text-sm">
              Enter a title or genre above to find films.
            </p>
          </div>
        )}
      </div>
    </>
  );
}
