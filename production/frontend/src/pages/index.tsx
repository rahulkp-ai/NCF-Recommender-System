// apps/web/src/pages/index.tsx
import Head from "next/head";
import HeroBanner from "../components/movie/HeroBanner";
import MovieRow   from "../components/movie/MovieRow";
import {
  useHomepageRecs,
  useUserRecs,
  useTrending,
  usePopular,
} from "../hooks/useRecommendations";
import { useAuth } from "../hooks/useAuth";

export default function HomePage() {
  const { user }                    = useAuth();
  const { recs: homeRecs, loading: homeLoading } = useHomepageRecs();
  const { recs: userRecs, loading: userLoading } = useUserRecs(user?.id ?? null);
  const { movies: trending, loading: trendLoading } = useTrending();
  const { movies: popular,  loading: popLoading }   = usePopular();

  // Pick hero from trending or homepage recs
  const heroMovie =
    (trending.find((m) => m.poster_url)) ||
    (homeRecs.find((m) => m.poster_url)) ||
    null;

  return (
    <>
      <Head>
        <title>Netfix — Smart Recommendations</title>
        <meta name="description" content="Netflix-style movie recommendations powered by NCF" />
      </Head>

      <main className="bg-[#141414] min-h-screen">
        {/* Hero */}
        <HeroBanner movie={heroMovie} />

        {/* Rows */}
        <div className="relative z-10 -mt-16 pb-16">
          {user && (
            <MovieRow
              title="Recommended for You"
              movies={userRecs}
              showBadge
              loading={userLoading}
            />
          )}

          <MovieRow
            title="Trending Now"
            movies={trending}
            loading={trendLoading}
          />

          <MovieRow
            title="Popular on Netfix"
            movies={popular}
            loading={popLoading}
          />

          <MovieRow
            title={user ? "Because You Watched" : "Top Picks"}
            movies={homeRecs}
            showBadge
            loading={homeLoading}
          />
        </div>
      </main>
    </>
  );
}
