"""
collect_rawg_data.py
────────────────────
Collects video game data from the RAWG API and saves to CSV.

Filters to games from 2000 to 2025.

Usage:
    1. Create a .env file in the same folder with:
           RAWG_API_KEY=your_key_here
    2. Run:
           python collect_rawg_data.py

Output:
    rawg_games.csv  — one row per game, saved incrementally
"""

import csv
import os
import time
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────────────────────

API_KEY = "" + os.getenv("RAWG_API_KEY", "").strip()

if not API_KEY:
    raise ValueError(
        "No API key found. Create a .env file with RAWG_API_KEY=your_key_here"
    )

BASE_URL    = "https://api.rawg.io/api/games"
OUTPUT_FILE = Path("rawg_games.csv")

# How many games to collect (10,000 gives a rich dataset, well within free tier)
TARGET_GAMES = 10_000

# RAWG allows up to 40 results per page
PAGE_SIZE = 40

# Polite delay between requests (seconds) — avoids rate limiting
SLEEP_BETWEEN_REQUESTS = 0.25

# ── Fields to extract from each game record ────────────────────────────────────

CSV_COLUMNS = [
    "id",
    "name",
    "released",           # release date string e.g. "2023-11-10"
    "release_year",       # extracted from released
    "release_month",      # extracted from released — useful for seasonality feature
    "metacritic",         # critic score 1-100
    "rating",             # RAWG community rating 0-5
    "rating_count",       # number of RAWG ratings
    "playtime",           # average playtime in hours
    "esrb_rating",        # E / T / M / AO / RP
    "genres",             # pipe-separated e.g. "Action|RPG|Adventure"
    "platforms",          # pipe-separated e.g. "PC|PlayStation 5|Xbox Series S/X"
    "tags",               # top 5 tags pipe-separated e.g. "Singleplayer|Open World"
    "developer",          # primary developer
    "publisher",          # primary publisher
    "suggestions_count",  # how many similar games RAWG links — proxy for franchise size
]


# ── Helper functions ───────────────────────────────────────────────────────────

def extract_game_record(game: dict) -> dict:
    """
    Pull the fields we need from a single RAWG game object.
    Returns a flat dictionary ready to write as a CSV row.
    """
    # Release date — split into year and month for feature engineering
    released     = game.get("released") or ""
    release_year = ""
    release_month = ""
    if released and len(released) >= 7:
        parts = released.split("-")
        release_year  = parts[0] if len(parts) > 0 else ""
        release_month = parts[1] if len(parts) > 1 else ""

    # Genres — list of dicts with "name" key
    genres = "|".join(
        g["name"] for g in (game.get("genres") or [])
    )

    # Platforms — nested structure: list of {"platform": {"name": ...}}
    platforms = "|".join(
        p["platform"]["name"] for p in (game.get("platforms") or [])
    )

    # Tags — take top 5 by relevance (they come pre-sorted by RAWG)
    tags = "|".join(
        t["name"] for t in (game.get("tags") or [])[:5]
    )

    # ESRB rating — nested dict with "name" key
    esrb = ""
    if game.get("esrb_rating"):
        esrb = game["esrb_rating"].get("name", "")

    # Developers and publishers — take the first one listed
    developer = ""
    publisher = ""
    for entry in (game.get("developers") or []):
        developer = entry.get("name", "")
        break
    for entry in (game.get("publishers") or []):
        publisher = entry.get("name", "")
        break

    return {
        "id":               game.get("id", ""),
        "name":             game.get("name", ""),
        "released":         released,
        "release_year":     release_year,
        "release_month":    release_month,
        "metacritic":       game.get("metacritic", ""),
        "rating":           game.get("rating", ""),
        "rating_count":     game.get("ratings_count", ""),
        "playtime":         game.get("playtime", ""),
        "esrb_rating":      esrb,
        "genres":           genres,
        "platforms":        platforms,
        "tags":             tags,
        "developer":        developer,
        "publisher":        publisher,
        "suggestions_count": game.get("suggestions_count", ""),
    }


def fetch_page(page: int) -> dict | None:
    """
    Fetch one page of games from the RAWG API.
    Filters to only games with a Metacritic score.
    Returns the JSON response dict, or None on failure.
    """
    params = {
        "key":        API_KEY,
        "dates":      "2000-01-01,2025-12-31",  # filter from 2000s
        "page_size":  PAGE_SIZE,
        "page":       page,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"  HTTP error on page {page}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Request error on page {page}: {e}")
        return None


# ── Main collection loop ───────────────────────────────────────────────────────

def collect_games():
    """
    Main function — pages through RAWG API results and writes to CSV.
    Saves incrementally so progress is not lost if interrupted.
    """
    print(f"Starting RAWG data collection")
    print(f"Target: {TARGET_GAMES:,} games with Metacritic scores")
    print(f"Output: {OUTPUT_FILE}")
    print("-" * 50)

    games_collected = 0
    page            = 1
    total_available = None

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        while games_collected < TARGET_GAMES:
            print(f"Fetching page {page}...", end=" ")
            data = fetch_page(page)

            if data is None:
                print("Failed — stopping.")
                break

            # On first page, report how many games are available total
            if total_available is None:
                total_available = data.get("count", 0)
                print(f"(RAWG reports {total_available:,} games with Metacritic scores)")

            results = data.get("results", [])
            if not results:
                print("No results — reached end of data.")
                break

            # Write each game record to CSV
            for game in results:
                if games_collected >= TARGET_GAMES:
                    break
                record = extract_game_record(game)
                writer.writerow(record)
                games_collected += 1

            print(f"Collected {games_collected:,} games so far")

            # Stop if RAWG has no more pages
            if not data.get("next"):
                print("No more pages available.")
                break

            page += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("-" * 50)
    print(f"Done. {games_collected:,} games saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    collect_games()
