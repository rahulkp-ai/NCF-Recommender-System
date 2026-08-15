# `production/frontend`

Next.js 14 (App Router) + TypeScript + Tailwind frontend. Moved from
`apps/web/` in Phase 4.

## Run locally
```bash
npm install
npm run dev
```
Runs at http://localhost:3000, expects `backend` at the URL configured via
`NEXT_PUBLIC_API_URL` (see `.env.example`).

## Structure
- `app/` — routes (App Router)
- `components/` — shared UI components
- `hooks/`, `lib/`, `services/` — client-side data fetching / utilities
- `styles/` — Tailwind config and globals

## Testing
Scaffolded under `apps/web/__tests__/` (Jest + React Testing Library —
see root `docs/developer-guide/getting-started.md`).

## Deployment
Target: Vercel free tier — see `docs/deployment/README.md`.
