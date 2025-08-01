# Import required packages
from bertopic import BERTopic
import aiosqlite
import asyncio
from collections import defaultdict
import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import logging
import re
from spacy.lang.en.stop_words import STOP_WORDS
import os
import sys

"""
This script performs topic modeling on the posts in the database and
saves the results and models to disk.
"""

# === Logging and random seed setup ===
logging.basicConfig(level=logging.INFO)
random.seed(42)
np.random.seed(42)

# === Load sentence embedding model for BERTopic ===
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def clean_text(text: str) -> str:
    """
    Cleans a string by removing URLs, punctuation, numbers, extra whitespace, and stopwords.
    Preserves important negations like 'not' and 'no'.

    Args:
        text: Raw text input

    Returns:
        Cleaned and token-filtered string
    """
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)      # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

    # Tokenize and filter stopwords (preserving negations)
    tokens = [word for word in text.split() if word not in STOP_WORDS or word in {"not", "no"}]
    return " ".join(tokens)


def json_safe(obj: object) -> object:
    """Ensures compatibility of non-standard objects (e.g., NumPy arrays) with JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict(orient="records")
    return obj


async def fetch_posts_by_year() -> dict:
    """
    Loads all Reddit posts from the SQLite database and groups them by year.

    Returns:
        A dictionary with years as keys and cleaned post texts as values. Also includes key 'all' for global analysis.
    """
    posts_by_year = defaultdict(list)

    async with aiosqlite.connect("swiss_economy_reddit.db") as db:
        async with db.execute("SELECT title, selftext, year FROM reddit_posts") as cursor:
            async for title, selftext, year in cursor:
                raw_text = (title + " " + (selftext if selftext else "")).lower()
                text = clean_text(raw_text)

                # Skip short posts
                if len(text.split()) > 3:
                    posts_by_year[str(year)].append(text)
                    posts_by_year["all"].append(text)  # Combined set for overall analysis

    return posts_by_year


async def run_bertopic(posts_by_year: dict):
    """
    Performs topic modeling using BERTopic for each year (and 'all').

    Args:
        posts_by_year: Dictionary of posts grouped by year

    Saves:
        - JSON summary of topics
        - Individual BERTopic models per year to disk
    """
    all_bertopic_results = {}

    for year, posts in posts_by_year.items():
        if len(posts) < 10:
            print(f"Skipping BERTopic for {year} due to insufficient posts.")
            continue

        logging.info(f"Running BERTopic for {year}, {len(posts)} posts")

        # Initialize BERTopic model with embedding model and full topic discovery
        model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True, nr_topics=None)

        try:
            topics, probs = model.fit_transform(posts)
        except IndexError as e:
            logging.warning(f"Skipping {year} due to BERTopic reduction error: {e}")
            continue

        # Remove outlier topic assignments (-1)
        if probs is not None:
            filtered = [(t, p) for t, p in zip(topics, probs) if t != -1]
            if not filtered:
                logging.warning(f"All topics for {year} were outliers. Skipping.")
                continue

            topics, probs = zip(*filtered)
            topics = list(topics)
            probs = [p.tolist() if isinstance(p, np.ndarray) else p for p in probs]
        else:
            topics = [t for t in topics if t != -1]
            probs = []

        # Retrieve topic metadata (excluding outliers)
        topic_info = model.get_topic_info().to_dict(orient="records")
        topic_info = [t for t in topic_info if t["Topic"] != -1]

        # Store results
        all_bertopic_results[year] = {
            "topics": topics,
            "probs": probs,
            "topic_info": topic_info
        }

        # Display results
        print(f"Topics for {year}:")
        print(model.get_topic_info())

        # Save model to disk
        model.save(f"bertopic_model_{year}")  # Saving BERTopic model

    # Save all results to JSON
    with open('bertopic_results.json', 'w') as f:
        json.dump(all_bertopic_results, f, indent=2, default=json_safe)


async def main(db_path: str = "swiss_economy_reddit.db"):
    """Checks for database, loads data, runs BERTopic modeling, and saves output."""
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        print("Please perform data collection first!")
        sys.exit(1)

    posts_by_year = await fetch_posts_by_year()
    await run_bertopic(posts_by_year)

    print("Analysis on all posts completed.")

# Run script directly
if __name__ == "__main__":
    asyncio.run(main())
