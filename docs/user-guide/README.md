# User Guide

This is a movie recommendation demo built on Neural Collaborative
Filtering (NCF) trained on MovieLens-1M, blended with content-based and
popularity signals for cold-start users.

## Using the app

1. Sign up / log in (`production/frontend`).
2. Rate a few movies, or search for titles you like.
3. Visit "Recommended for You" — this calls the hybrid engine
   (collaborative + content + popularity), with a cold-start fallback if
   you haven't rated enough movies yet.

## Where the model comes from

See the published research: `ieee/` (IEEE paper) and `thesis/` (full MSc
thesis with 14 phase notebooks) for the full methodology, benchmarks
(3.39× training speedup, 0.615 HR@10, 0.2257 BCE loss), and ablations.
