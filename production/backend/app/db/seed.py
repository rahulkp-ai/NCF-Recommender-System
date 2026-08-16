"""
production/backend/app/db/seed.py
Generates synthetic users, loads real MovieLens movies,
and simulates realistic interaction patterns.
Run: python -m production.backend.app.db.seed

Phase 5 correction: this file's root-detection was broken by the Phase 4
move and went unnoticed until now — production/backend/app/db/seed.py is
one directory deeper than the pre-move backend/app/db/seed.py, so
parents[3] (correct pre-move) silently pointed at production/ instead of
repo root after the move. Fixed to parents[4]. MOVIES_RAW also
repointed: data/raw/movielens/ml-1m/movies.dat was dropped from
production in Phase 4 (verified at the time that no *training* code
needed it) — but seed.py needs it to seed the demo DB, which Phase 4's
verification missed. Copied to production/artifacts/seed_data/movies.dat
(byte-identical, md5-verified) so it ships with the deployable artifact
set instead of depending on a data/ folder that no longer exists.
PROCESSED_DIR (unused dead reference to a path that also no longer
exists) is removed as part of this cleanup.
"""

import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import bcrypt
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from .connection import SessionLocal, create_tables
from .models import Interaction, Movie, User

ROOT_DIR = (
    Path(__file__).resolve().parents[4]
)  # .../production/backend/app/db/seed.py -> repo root (or /app in Docker)
MOVIES_RAW = ROOT_DIR / "production" / "artifacts" / "seed_data" / "movies.dat"

N_SYNTHETIC_USERS = 10
INTERACTIONS_PER_USER_MEAN = 60
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _parse_year(title: str):
    m = re.search(r"\((\d{4})\)$", title.strip())
    return int(m.group(1)) if m else None


def seed_movies(db: Session):
    if db.query(Movie).count() > 0:
        print("Movies already seeded — skipping")
        return
    df = pd.read_csv(
        MOVIES_RAW,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    movies = [
        Movie(id=int(row.item_id), title=row.title, genres=row.genres, year=_parse_year(row.title))
        for _, row in df.iterrows()
    ]
    db.bulk_save_objects(movies)
    db.commit()
    print(f"Seeded {len(movies):,} movies")


def seed_users(db: Session):
    if db.query(User).count() >= N_SYNTHETIC_USERS:
        print("Users already seeded — skipping")
        return
    age_groups = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    age_weights = [0.05, 0.20, 0.30, 0.20, 0.10, 0.10, 0.05]
    users = []
    for i in range(N_SYNTHETIC_USERS):
        username = f"user_{i + 1:04d}"
        users.append(
            User(
                id=i + 1,
                username=username,
                email=f"{username}@example.com",
                password_hash=hash_password("password"),
                age_group=random.choices(age_groups, weights=age_weights)[0],
                gender=random.choice(["M", "F"]),
                occupation=random.choice(range(21)),
            )
        )
    db.bulk_save_objects(users)
    db.commit()
    print(f"Seeded {len(users):,} synthetic users")


def seed_interactions(db: Session):
    if db.query(Interaction).count() > 0:
        print("Interactions already seeded — skipping")
        return
    all_movies = db.query(Movie).all()
    all_users = db.query(User).all()
    movie_ids = [m.id for m in all_movies]

    popularity = np.random.zipf(1.5, len(movie_ids))
    popularity = popularity / popularity.sum()

    now = datetime.utcnow()
    interactions = []
    for user in all_users:
        n = max(10, int(np.random.normal(INTERACTIONS_PER_USER_MEAN, 20)))
        chosen = np.random.choice(
            movie_ids, size=min(n, len(movie_ids)), replace=False, p=popularity
        )
        for movie_id in chosen:
            event_type = random.choices(["like", "click", "rate"], weights=[0.4, 0.4, 0.2])[0]
            rating = round(random.uniform(3.0, 5.0), 1) if event_type == "rate" else None
            ts = now - timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
            interactions.append(
                Interaction(
                    user_id=user.id,
                    movie_id=int(movie_id),
                    event_type=event_type,
                    rating=rating,
                    created_at=ts,
                )
            )
    db.bulk_save_objects(interactions)
    db.commit()
    print(f"Seeded {len(interactions):,} interactions")


def run_seed():
    create_tables()
    db: Session = SessionLocal()
    try:
        if db.query(User).count() == 0:
            print("Seeding database...")
            seed_movies(db)
            seed_users(db)
            seed_interactions(db)
            print("Seeding complete.")
        else:
            print("Database already seeded — skipping data insertion.")

        print("Synchronizing sequences...")
        db.execute(
            text("SELECT setval('users_id_seq', (SELECT COALESCE(MAX(id),0)+1 FROM users), false)")
        )
        db.execute(
            text(
                "SELECT setval('movies_id_seq', (SELECT COALESCE(MAX(id),0)+1 FROM movies), false)"
            )
        )
        db.execute(
            text(
                "SELECT setval('interactions_id_seq', (SELECT COALESCE(MAX(id),0)+1 FROM interactions), false)"
            )
        )
        db.commit()
        print("Server ready.")
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
