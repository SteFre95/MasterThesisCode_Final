# Import required packages
import asyncpraw
import random
import asyncio
import aiosqlite
import json
from datetime import datetime
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

"""
This script collects posts using the Reddit API 
and stores them in an SQLite database.
"""

# Ensures consistent language detection results and reproducible sampling
DetectorFactory.seed = 0
random.seed(42)

# Load config file containing API credentials and custom parameters
with open("config.json", "r") as file:
    config = json.load(file)

# Extract credentials from config
client_id = config["client_id"]
client_secret = config["client_secret"]
user_agent = config["user_agent"]


def is_english(text: str) -> bool:
    """
    Detects if a given text is in English. Returns True if English, False otherwise or if the text is empty or
    an exception occurs.
    """
    if not text or text.strip() == "":
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:  # Handle detection failure
        return False


async def is_post_stored(post_id: str, db: aiosqlite.Connection) -> bool:
    """Checks if a post ID exists in the database. Returns True if the post ID exists, False otherwise."""
    async with db.execute("SELECT 1 FROM reddit_posts WHERE id = ?", (post_id,)) as cursor:
        result = await cursor.fetchone()
        return result is not None


async def setup_database() -> None:
    """Sets up the database schema, optionally resetting the table."""
    async with aiosqlite.connect("swiss_economy_reddit.db") as db:
        if config.get("reset_db", False):  # Reset database if specified in config
            await db.execute("DROP TABLE IF EXISTS reddit_posts")

        # Create the main table if it doesn't already exist
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id TEXT PRIMARY KEY NOT NULL,
                title TEXT NOT NULL,
                selftext TEXT NOT NULL,
                created_utc INTEGER NOT NULL,
                year INTEGER NOT NULL,
                sentiment_VADER REAL,
                sentiment_FinBERT REAL
            )
        """)
        await db.commit()


async def fetch_posts_for_year(year: int, db: aiosqlite.Connection, reddit: asyncpraw.Reddit,
                               limit: int = config["limit"]) -> None:
    """Collects posts for a given year and stores them in the database."""
    print(f"Collecting posts for {year}...")

    # Define UNIX timestamp boundaries for the year
    after = int(datetime(year, 1, 1).timestamp())
    before = int(datetime(year + 1, 1, 1).timestamp())

    subreddit = await reddit.subreddit(config["subreddit"])  # Subreddit from config
    query = " OR ".join(config["query_terms"])  # Join search terms using OR
    sorting_methods = config["sorting_methods"]  # e.g., relevance, top, new

    posts = []
    for sort in sorting_methods:  # Try each sorting method independently
        try:
            async for submission in subreddit.search(query, limit=limit, time_filter="all", sort=sort):
                # Filter posts to the correct year range
                if after <= submission.created_utc < before:
                    # Keep only English-language posts
                    if is_english(submission.title) and is_english(submission.selftext):
                        posts.append(submission)
        except Exception as e:
            print(f"Error fetching posts for year {year} with sort '{sort}': {e}")
            return  # Exit early on error

    # Remove duplicates and previously stored posts
    unique_posts = []
    for submission in posts:
        if not await is_post_stored(submission.id, db):
            unique_posts.append(submission)

    # Randomly sample from the filtered list (up to sample_size)
    sampled_posts = random.sample(unique_posts, min(config["sample_size"], len(unique_posts)))

    new_posts = []
    for submission in sampled_posts:
        if not await is_post_stored(submission.id, db):  # Double-check to avoid race condition
            new_posts.append((
                submission.id, submission.title, submission.selftext,
                int(submission.created_utc), year, None, None  # Placeholder for sentiment analysis
            ))

    # Insert posts into database
    if new_posts:
        await db.executemany("INSERT OR IGNORE INTO reddit_posts VALUES (?, ?, ?, ?, ?, ?, ?)", new_posts)
        await db.commit()


async def print_posts_per_year() -> None:
    """Retrieves and prints the count of posts stored per year and the overall number of posts."""
    async with aiosqlite.connect("swiss_economy_reddit.db") as db:
        # Query count of posts per year
        async with db.execute("SELECT year, COUNT(*) FROM reddit_posts GROUP BY year ORDER BY year") as cursor:
            print("\nPosts in the database per year:")
            async for year, count in cursor:
                print(f"{year}: {count} posts")
        # Query total post count
        async with db.execute("SELECT COUNT(*) FROM reddit_posts") as cursor:
            overall_count = await cursor.fetchone()
            print(f"\nTotal number of posts in the database: {overall_count[0]}")


async def main() -> None:
    """Main function: initializes DB, sets up Reddit client, fetches posts, and prints summary."""
    await setup_database()

    # Instantiate Reddit API client
    reddit = asyncpraw.Reddit(client_id=client_id,
                              client_secret=client_secret,
                              user_agent=user_agent)

    start_year = config["start_year"]
    end_year = datetime.now().year

    # Open one DB connection for all year-based tasks
    async with aiosqlite.connect("swiss_economy_reddit.db") as db:
        # Create one task per year and run them concurrently
        tasks = [fetch_posts_for_year(year, db, reddit) for year in range(start_year, end_year + 1)]
        await asyncio.gather(*tasks)

    await reddit.close()

    await print_posts_per_year()


if __name__ == "__main__":
    # Run the main function using a dedicated event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
