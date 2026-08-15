// apps/web/src/components/layout/Navbar.tsx
import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import { FiSearch, FiBell, FiChevronDown } from "react-icons/fi";
import { useAuth } from "../../hooks/useAuth";

export default function Navbar() {
  const [scrolled, setScrolled]   = useState(false);
  const [searchOpen, setSearch]   = useState(false);
  const [searchVal, setSearchVal] = useState("");
  const { user, logout }          = useAuth();
  const router                    = useRouter();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchVal.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchVal.trim())}`);
      setSearch(false);
      setSearchVal("");
    }
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 flex items-center justify-between
        px-4 md:px-12 py-3 transition-all duration-300
        ${scrolled ? "bg-[#141414]" : "bg-gradient-to-b from-black/80 to-transparent"}`}
    >
      {/* Logo */}
      <div className="flex items-center gap-8">
        <Link href="/">
          <span className="text-[#E50914] font-black text-3xl tracking-tighter select-none">
            NETFIX
          </span>
        </Link>
        <div className="hidden md:flex items-center gap-5 text-sm text-gray-300">
          <Link href="/"        className="hover:text-white transition-colors">Home</Link>
          <Link href="/search"  className="hover:text-white transition-colors">Movies</Link>
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-4">
        {/* Search */}
        <form onSubmit={handleSearch} className="flex items-center">
          {searchOpen ? (
            <input
              autoFocus
              value={searchVal}
              onChange={(e) => setSearchVal(e.target.value)}
              placeholder="Titles, genres..."
              className="bg-black/80 border border-white text-white text-sm
                         px-3 py-1 rounded w-48 md:w-64 focus:outline-none"
              onBlur={() => { if (!searchVal) setSearch(false); }}
            />
          ) : (
            <button
              type="button"
              onClick={() => setSearch(true)}
              className="text-white hover:text-gray-300 transition-colors"
            >
              <FiSearch size={20} />
            </button>
          )}
        </form>

        <FiBell size={20} className="text-white cursor-pointer hover:text-gray-300" />

        {user ? (
          <div className="flex items-center gap-2 cursor-pointer group relative">
            <div className="w-8 h-8 bg-[#E50914] rounded flex items-center justify-center
                            text-white font-bold text-sm">
              {user.username[0].toUpperCase()}
            </div>
            <FiChevronDown size={14} className="text-white" />
            {/* Dropdown */}
            <div className="absolute top-10 right-0 bg-[#1a1a1a] border border-gray-700
                            rounded shadow-xl py-2 w-40 hidden group-hover:block">
              <Link href="/" className="block px-4 py-2 text-sm text-gray-300 hover:text-white">
                {user.username}
              </Link>
              <hr className="border-gray-700 my-1" />
              <button
                onClick={logout}
                className="block w-full text-left px-4 py-2 text-sm text-gray-300 hover:text-white"
              >
                Sign Out
              </button>
            </div>
          </div>
        ) : (
          <Link href="/login">
            <button className="btn-red text-sm px-4 py-1.5">Sign In</button>
          </Link>
        )}
      </div>
    </nav>
  );
}
