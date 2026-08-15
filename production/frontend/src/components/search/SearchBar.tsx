// apps/web/src/components/search/SearchBar.tsx
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/router";
import { FiSearch, FiX } from "react-icons/fi";
import { searchApi, Movie } from "../../lib/api";

export default function SearchBar({ initialQuery = "" }: { initialQuery?: string }) {
  const [query, setQuery]           = useState(initialQuery);
  const [suggestions, setSuggestions] = useState<Movie[]>([]);
  const [showSugg, setShowSugg]     = useState(false);
  const timerRef                    = useRef<NodeJS.Timeout>();
  const router                      = useRouter();

  useEffect(() => {
    if (query.length < 2) { setSuggestions([]); return; }
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      try {
        const res = await searchApi.search(query, 6);
        setSuggestions(res.data.results);
        setShowSugg(true);
      } catch {}
    }, 300);
    return () => clearTimeout(timerRef.current);
  }, [query]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/search?q=${encodeURIComponent(query.trim())}`);
      setShowSugg(false);
    }
  };

  return (
    <div className="relative w-full max-w-xl">
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <div className="relative flex-1">
          <FiSearch
            size={18}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => suggestions.length > 0 && setShowSugg(true)}
            placeholder="Search titles, genres…"
            className="input-dark pl-10 pr-10"
          />
          {query && (
            <button
              type="button"
              onClick={() => { setQuery(""); setSuggestions([]); }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400
                         hover:text-white transition-colors"
            >
              <FiX size={16} />
            </button>
          )}
        </div>
        <button type="submit" className="btn-red px-5 py-3">Search</button>
      </form>

      {/* Autocomplete dropdown */}
      {showSugg && suggestions.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-[#1a1a1a] border
                        border-gray-700 rounded shadow-2xl z-50 overflow-hidden">
          {suggestions.map((m) => (
            <button
              key={m.id}
              className="flex items-center gap-3 w-full px-4 py-2.5
                         hover:bg-gray-800 transition-colors text-left"
              onClick={() => {
                router.push(`/movie/${m.id}`);
                setShowSugg(false);
              }}
            >
              {m.poster_url && (
                <img src={m.poster_url} alt="" className="w-8 h-12 object-cover rounded" />
              )}
              <div>
                <p className="text-white text-sm">{m.title}</p>
                {m.year && <p className="text-gray-400 text-xs">{m.year}</p>}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
