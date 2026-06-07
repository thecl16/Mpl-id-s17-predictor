"""
=============================================================
BAGIAN 1: DATA COLLECTION - MPL ID S17 Playoffs Scraper
=============================================================
Scrapes match data from Liquipedia MPL ID Season 17
Output: data/mpl_raw_data.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# MPL ID S17 Liquipedia URLs to scrape
URLS = [
    "https://liquipedia.net/mobilelegends/MPL/ID/Season_17/Playoffs",
    "https://liquipedia.net/mobilelegends/MPL/ID/Season_17/Regular_Season",
]

def scrape_liquipedia_matches(url: str) -> list[dict]:
    """Scrape match results from a Liquipedia page."""
    print(f"\n[INFO] Scraping: {url}")
    matches = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Liquipedia uses .match-row or .matchlist table structure
        match_rows = soup.select(".match-row, .wikitable tr, .brkts-popup-body")

        # --- Strategy 1: brkts (Bracket) match format ---
        brackets = soup.select(".brkts-match")
        for bracket in brackets:
            teams = bracket.select(".brkts-opponent-entry")
            scores = bracket.select(".brkts-opponent-score-inner, .brkts-score")
            if len(teams) >= 2 and len(scores) >= 2:
                team_a = teams[0].get_text(strip=True)
                team_b = teams[1].get_text(strip=True)
                try:
                    score_a = int(re.search(r"\d+", scores[0].get_text()).group())
                    score_b = int(re.search(r"\d+", scores[1].get_text()).group())
                except:
                    continue
                if team_a and team_b and team_a != team_b:
                    matches.append({
                        "team_a": team_a, "score_a": score_a,
                        "team_b": team_b, "score_b": score_b,
                        "result_a": "Win" if score_a > score_b else "Loss",
                        "result_b": "Win" if score_b > score_a else "Loss",
                        "date": datetime.today().strftime("%Y-%m-%d"),
                        "stage": "Playoffs" if "Playoffs" in url else "Regular",
                        "source": url
                    })

        # --- Strategy 2: wikitable rows ---
        tables = soup.select(".wikitable")
        for table in tables:
            rows = table.select("tr")[1:]  # skip header
            for row in rows:
                cells = row.select("td")
                if len(cells) >= 4:
                    try:
                        team_a = cells[0].get_text(strip=True)
                        score_text = cells[1].get_text(strip=True)
                        team_b = cells[2].get_text(strip=True)
                        scores = re.findall(r"\d+", score_text)
                        if len(scores) >= 2:
                            score_a, score_b = int(scores[0]), int(scores[1])
                            if team_a and team_b and team_a != team_b:
                                matches.append({
                                    "team_a": team_a, "score_a": score_a,
                                    "team_b": team_b, "score_b": score_b,
                                    "result_a": "Win" if score_a > score_b else "Loss",
                                    "result_b": "Win" if score_b > score_a else "Loss",
                                    "date": datetime.today().strftime("%Y-%m-%d"),
                                    "stage": "Playoffs" if "Playoffs" in url else "Regular",
                                    "source": url
                                })
                    except Exception:
                        continue

        print(f"[INFO] Found {len(matches)} matches from {url}")

    except requests.exceptions.ConnectionError:
        print(f"[WARNING] Cannot connect to {url} — generating synthetic fallback data.")
    except Exception as e:
        print(f"[ERROR] {e}")

    return matches


def generate_synthetic_data() -> pd.DataFrame:
    """
    Fallback: Generate realistic synthetic MPL ID S17 data
    based on real teams that competed in MPL ID.
    Use this when Liquipedia is unreachable (network restriction, etc.)
    """
    print("\n[INFO] Generating synthetic MPL ID S17 dataset as fallback...")

    TEAMS = [
        "EVOS Legends", "RRQ Hoshi", "Alter Ego", "Bigetron Alpha",
        "ONIC Esports", "Geek Fam", "Aura Fire", "MDH Esports",
        "Rebellion Zion", "Dewa United"
    ]

    import random
    random.seed(42)

    records = []
    match_id = 1

    # Regular Season: round-robin style
    for week in range(1, 8):
        pairs = [(TEAMS[i], TEAMS[j]) for i in range(len(TEAMS))
                 for j in range(i+1, len(TEAMS))]
        random.shuffle(pairs)
        pairs = pairs[:8]  # 8 matches per week

        for ta, tb in pairs:
            # Simulate team strength bias
            strength = {t: random.uniform(0.3, 0.8) for t in TEAMS}
            prob_a = strength[ta] / (strength[ta] + strength[tb])
            win_a = random.random() < prob_a
            score_a = random.choice([3, 2]) if win_a else random.choice([0, 1])
            score_b = random.choice([0, 1]) if win_a else random.choice([3, 2])

            records.append({
                "match_id": match_id,
                "date": f"2025-{week+1:02d}-{random.randint(1,28):02d}",
                "stage": "Regular Season",
                "team_a": ta, "score_a": score_a,
                "team_b": tb, "score_b": score_b,
                "result_a": "Win" if score_a > score_b else "Loss",
                "result_b": "Win" if score_b > score_a else "Loss",
            })
            match_id += 1

    # Playoffs: bracket
    playoff_matchups = [
        ("ONIC Esports", "Aura Fire", "Upper Bracket"),
        ("RRQ Hoshi", "Bigetron Alpha", "Upper Bracket"),
        ("EVOS Legends", "MDH Esports", "Upper Bracket"),
        ("Alter Ego", "Geek Fam", "Upper Bracket"),
        ("ONIC Esports", "EVOS Legends", "Semifinal"),
        ("RRQ Hoshi", "Alter Ego", "Semifinal"),
        ("EVOS Legends", "Alter Ego", "Lower Bracket"),
        ("ONIC Esports", "RRQ Hoshi", "Grand Final"),
    ]

    for ta, tb, stage in playoff_matchups:
        score_a = random.choice([3, 3, 2, 3])
        score_b = random.choice([0, 1, 3, 2])
        records.append({
            "match_id": match_id,
            "date": f"2025-08-{random.randint(1, 20):02d}",
            "stage": stage,
            "team_a": ta, "score_a": score_a,
            "team_b": tb, "score_b": score_b,
            "result_a": "Win" if score_a > score_b else "Loss",
            "result_b": "Win" if score_b > score_a else "Loss",
        })
        match_id += 1

    df = pd.DataFrame(records)
    print(f"[INFO] Synthetic dataset: {len(df)} matches generated.")
    return df


def main():
    all_matches = []

    for url in URLS:
        matches = scrape_liquipedia_matches(url)
        all_matches.extend(matches)
        time.sleep(2)  # polite delay

    if all_matches:
        df = pd.DataFrame(all_matches)
        df = df.drop_duplicates(subset=["team_a", "team_b", "score_a", "score_b"])
        df.insert(0, "match_id", range(1, len(df) + 1))
        print(f"\n[SUCCESS] Total unique matches scraped: {len(df)}")
    else:
        print("\n[INFO] Scraping returned no data — using synthetic fallback.")
        df = generate_synthetic_data()

    output_path = "data/mpl_raw_data.csv"
    df.to_csv(output_path, index=False)
    print(f"[SAVED] → {output_path}")
    print("\nSample data:")
    print(df.head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
