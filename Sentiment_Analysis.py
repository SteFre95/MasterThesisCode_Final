# Import required packages
import sqlite3
from transformers import pipeline, AutoTokenizer
from transformers import logging as hf_logging
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import sys

"""
This script performs sentiment analysis on the posts stored in the database 
and writes the results to the database.
"""

# Suppress warnings about tokenizer truncation (e.g., long input)
hf_logging.set_verbosity_error()

# Download VADER lexicon (required for sentiment analysis)
nltk.download('vader_lexicon')

# Initialize sentiment analysis models
MODEL_NAME = "ProsusAI/finbert"
classifier = pipeline("text-classification", model=MODEL_NAME)  # Load FinBERT pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Load tokenizer for chunking
sia = SentimentIntensityAnalyzer()  # Initialize VADER analyzer


def chunk_text_by_tokens(text: str, max_tokens: int = 510, overlap: int = 50) -> list[str]:
    """
    Tokenizes long text into smaller overlapping chunks compatible with FinBERT (max 512 tokens).
    FinBERT expects sequences with [CLS] and [SEP] tokens, so we keep actual chunks at 510 tokens max.

     Args:
        text: Full string to tokenize and split
        max_tokens: Max number of tokens per chunk (excluding special tokens)
        overlap: Number of tokens to overlap between chunks to preserve context

    Returns:
        List of decoded text chunks
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)  # Encode text without special tokens
    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)  # Convert tokens back to text
        chunks.append(chunk_text)
        start += max_tokens - overlap  # Slide window with overlap
    return chunks


def analyze_sentiment_for_posts(db_path: str = "swiss_economy_reddit.db") -> None:
    """
    Connects to the Reddit SQLite DB and performs sentiment analysis (FinBERT + VADER)
    on posts that haven't been analyzed yet. Saves results back to the database.
    """
    # Ensure database exists before proceeding
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        print("Please perform data collection first!")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Select posts that still need sentiment score
    cursor.execute("""
        SELECT id, title, selftext 
        FROM reddit_posts 
        WHERE sentiment_VADER IS NULL AND sentiment_FinBERT IS NULL
    """)
    posts = cursor.fetchall()

    for idx, (post_id, title, selftext) in enumerate(posts, start=1):
        # Combine title and selftext into a single text block
        sep = "" if title.endswith(('.', '!', '?')) else "."
        text = f"{title}{sep} {selftext}" if selftext else title

        # === VADER Analysis ===
        vader_score = sia.polarity_scores(text)['compound']

        # === FinBERT Analysis ===
        finbert_text = text.lower()  # Lowercase input for FinBERT
        chunks = chunk_text_by_tokens(finbert_text)  # Split text into manageable chunks
        finbert_scores = []

        for chunk_idx, chunk in enumerate(chunks, start=1):
            print(f"Processing Post {idx}/{len(posts)} (ID: {post_id}) - Chunk {chunk_idx}/{len(chunks)}...")
            try:
                result = classifier(chunk, truncation=True)[0]  # Run sentiment prediction
                label = result["label"].lower()
                score = result["score"]
                # Convert label into a signed score for averaging
                if label == "positive":
                    finbert_scores.append(score)
                elif label == "negative":
                    finbert_scores.append(-score)
                else:  # Neutral
                    finbert_scores.append(0.0)
            except Exception as e:
                print(f"Error in chunk {chunk_idx}: {e}")
                continue  # Skip problematic chunks

        # Compute average FinBERT score across all chunks
        finbert_score = sum(finbert_scores) / len(finbert_scores) if finbert_scores else 0.0

        # === Update database with new sentiment scores ===
        cursor.execute("""
            UPDATE reddit_posts 
            SET sentiment_VADER = ?, sentiment_FINBERT = ? 
            WHERE id = ?
        """, (vader_score, finbert_score, post_id))
        conn.commit()

        print(f"Completed Post {idx}/{len(posts)} (ID: {post_id}) - Final Scores: FinBERT: {finbert_score:.4f}, "
              f"VADER: {vader_score:.4f}\n---")

    conn.close()  # Clean up connection


def main() -> None:
    """Main entry point: run sentiment analysis on database."""
    analyze_sentiment_for_posts()
    print("Analysis on all posts completed.")


# Run script directly
if __name__ == "__main__":
    main()
