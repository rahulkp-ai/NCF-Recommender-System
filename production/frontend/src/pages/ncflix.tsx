import Head from "next/head";
import { useEffect, useState } from "react";
import { FiHeart } from "react-icons/fi";
import Link from "next/link";
import { useAuth } from "../hooks/useAuth";
import {
  interactApi,
  Movie,
  RecommendedMovie,
  recommendApi,
} from "../lib/api";

const WARMUP_INTERACTIONS = 20;
type InteractionHandler = (..._args: [number, "click" | "like"]) => Promise<boolean>;

function NCFlixCard({
  movie,
  onInteraction,
}: {
  movie: RecommendedMovie | Movie;
  onInteraction: InteractionHandler;
}) {
  const [liked, setLiked] = useState(false);
  const [saving, setSaving] = useState(false);
  const score = "score" in movie ? movie.score : null;
  const itemId = "item_id" in movie ? movie.item_id : movie.id;

  const recordClick = async () => {
    if (saving) return;
    setSaving(true);
    await onInteraction(itemId, "click");
    setSaving(false);
  };

  const saveLike = async (event: React.MouseEvent) => {
    event.stopPropagation();
    if (saving || liked) return;
    setSaving(true);
    if (await onInteraction(itemId, "like")) setLiked(true);
    setSaving(false);
  };

  return (
    <article
      onClick={recordClick}
      className="group relative w-[160px] shrink-0 cursor-pointer overflow-hidden rounded-lg border border-[#2a2a2a] bg-[#1a1a1a] transition-transform duration-200 hover:-translate-y-1"
    >
      <div className="relative aspect-[2/3] w-full overflow-hidden bg-[#1a1a2e]">
        {movie.poster_url ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="h-full w-full bg-[#1a1a2e]" />
        )}

        {score != null && (
          <div className="absolute left-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-[#7F77DD]">
            {Math.round(score * 100)}%
          </div>
        )}

        <button
          type="button"
          aria-label={liked ? "Unlike" : "Like"}
          onClick={saveLike}
          disabled={saving}
          className={`absolute right-2 top-2 flex h-[30px] w-[30px] items-center justify-center rounded-full border-0 transition-colors ${
            liked
              ? "bg-[#D85A30] text-white"
              : "bg-black/60 text-[#aaa]"
          }`}
        >
          <FiHeart size={14} fill={liked ? "currentColor" : "none"} />
        </button>
      </div>

      <div className="h-[48px] px-2.5 py-2">
        <p className="truncate text-xs font-medium text-[#ddd]">{movie.title}</p>
        <p className="mt-0.5 truncate text-[10px] text-[#666]">
          {movie.genres?.split("|")[0]}
          {movie.year ? ` · ${movie.year}` : ""}
        </p>
      </div>
    </article>
  );
}

function NCFlixRow({
  title,
  movies,
  loading,
  onInteraction,
}: {
  title: string;
  movies: (RecommendedMovie | Movie)[];
  loading: boolean;
  onInteraction: InteractionHandler;
}) {
  return (
    <section className="mb-8">
      <h2 className="mb-3 text-[15px] font-medium text-[#ccc]">{title}</h2>
      {loading ? (
        <div className="flex gap-3 overflow-hidden">
          {Array.from({ length: 6 }).map((_, index) => (
            <div
              key={index}
              className="h-[288px] w-[160px] shrink-0 animate-pulse rounded-lg bg-[#1a1a1a]"
            />
          ))}
        </div>
      ) : (
        <div className="ncflix-row flex gap-3 overflow-x-auto pb-2">
          {movies.map((movie) => (
            <NCFlixCard
              key={"item_id" in movie ? movie.item_id : movie.id}
              movie={movie}
              onInteraction={onInteraction}
            />
          ))}
        </div>
      )}
    </section>
  );
}

export default function NCFlixPage() {
  const { user } = useAuth();
  const [alpha, setAlpha] = useState<number>(0);
  const [beta, setBeta] = useState<number>(1);
  const [nInteractions, setNInteractions] = useState(0);
  const [strategy, setStrategy] = useState("cold_start");
  const [homeRecs, setHomeRecs] = useState<RecommendedMovie[]>([]);
  const [userRecs, setUserRecs] = useState<RecommendedMovie[]>([]);
  const [trending, setTrending] = useState<Movie[]>([]);
  const [loadingHome, setLoadingHome] = useState(true);
  const [loadingUser, setLoadingUser] = useState(false);
  const [loadingTrending, setLoadingTrending] = useState(true);
  const [interactionMessage, setInteractionMessage] = useState("");

  const handleInteraction = async (
    movieId: number,
    eventType: "click" | "like"
  ): Promise<boolean> => {
    if (!user) {
      setInteractionMessage("Sign in to save interactions and personalize recommendations.");
      return false;
    }

    try {
      await interactApi.record(movieId, eventType);
      const response = await recommendApi.forUser(user.id, 10);
      const { alpha: nextAlpha, n_interactions: nextCount, strategy: nextStrategy } =
        response.data;
      setUserRecs(response.data.recommendations);
      setAlpha(nextAlpha);
      setBeta(1 - nextAlpha);
      setNInteractions(nextCount);
      setStrategy(nextStrategy);
      setInteractionMessage("");
      window.dispatchEvent(
        new CustomEvent("ncf-interaction-updated", { detail: nextCount })
      );
      return true;
    } catch {
      setInteractionMessage("Could not save that interaction. Please try again.");
      return false;
    }
  };

  useEffect(() => {
    recommendApi
      .homepage(20)
      .then((response) => setHomeRecs(response.data.recommendations))
      .catch(() => setHomeRecs([]))
      .finally(() => setLoadingHome(false));

    recommendApi
      .trending()
      .then((response) => setTrending(response.data))
      .catch(() => setTrending([]))
      .finally(() => setLoadingTrending(false));

    if (user) {
      setLoadingUser(true);
      recommendApi
        .forUser(user.id, 10)
        .then((response) => {
          setUserRecs(response.data.recommendations);
          setAlpha(response.data.alpha);
          setBeta(1 - response.data.alpha);
          setNInteractions(response.data.n_interactions);
          setStrategy(response.data.strategy);
        })
        .catch(() => {
          setAlpha(0);
          setBeta(1);
          setNInteractions(0);
          setStrategy("cold_start");
        })
        .finally(() => setLoadingUser(false));
    } else {
      setUserRecs([]);
      setAlpha(0);
      setBeta(1);
      setNInteractions(0);
      setStrategy("cold_start");
      setLoadingUser(false);
    }
  }, [user]);

  const pct = Math.round(alpha * 100);
  const betaPct = Math.round(beta * 100);
  const indicator =
    strategy === "ncf"
      ? { label: "Personalised", color: "#7F77DD", desc: "NCF model driving recommendations" }
      : strategy === "blend"
        ? { label: "Learning…", color: "#1D9E75", desc: "Blending your taste with popular picks" }
        : { label: "Cold start", color: "#EF9F27", desc: "Showing popular & genre-based picks" };

  return (
    <>
      <Head>
        <title>NCFlix</title>
      </Head>
      <main className="min-h-screen bg-[#0a0a0a] px-8 pb-16 pt-6 text-[#eee]">
        <div className="mb-8 flex h-[68px] items-center gap-4 rounded-[10px] border border-[#333] bg-[#111] px-4 text-[13px] text-[#aaa]">
            <span
              className="whitespace-nowrap rounded-md border px-2.5 py-1 text-[11px] font-medium"
              style={{
                backgroundColor: `${indicator.color}22`,
                borderColor: `${indicator.color}55`,
                color: indicator.color,
              }}
            >
              {indicator.label}
            </span>

            <div className="min-w-0 flex-1">
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-[#666]">
                  NCF α = {pct}% · Cold-start β = {betaPct}%
                </span>
                <span className="text-[#555]">
                  {nInteractions} / {WARMUP_INTERACTIONS} interactions
                </span>
              </div>
              <div className="h-1 overflow-hidden rounded bg-[#222]">
                <div
                  className="h-full rounded transition-[width] duration-700"
                  style={{ width: `${pct}%`, backgroundColor: indicator.color }}
                />
              </div>
            </div>

            <span className="whitespace-nowrap text-[11px] text-[#555]">
              {indicator.desc}
            </span>
          </div>

        {interactionMessage && (
          <p role="status" className="-mt-5 mb-6 text-xs text-[#D85A30]">
            {interactionMessage}{" "}
            {!user && (
              <Link href="/login" className="underline">
                Sign in
              </Link>
            )}
          </p>
        )}

        {userRecs.length > 0 && (
          <NCFlixRow
            title="Recommended for you"
            movies={userRecs}
            loading={loadingUser}
            onInteraction={handleInteraction}
          />
        )}

        <NCFlixRow
          title="Trending now"
          movies={trending.length ? trending : homeRecs.slice(0, 10)}
          loading={loadingTrending || (trending.length === 0 && loadingHome)}
          onInteraction={handleInteraction}
        />

        {homeRecs.length > 10 && (
          <NCFlixRow
            title="Popular picks"
            movies={homeRecs.slice(10)}
            loading={loadingHome}
            onInteraction={handleInteraction}
          />
        )}
      </main>
    </>
  );
}