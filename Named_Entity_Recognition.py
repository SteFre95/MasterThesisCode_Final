# Import required packages
import sqlite3
import pandas as pd
import spacy
from typing import Tuple, List, Dict
from collections import Counter
import json
from tqdm import tqdm
import os
import sys

"""
This script performs named entity recognition on the posts in the database
and saves the results to disk.
"""

def load_data(db_path: str) -> pd.DataFrame:
    """
    Loads Reddit posts from the SQLite database into a DataFrame.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        DataFrame with 'id', 'title', 'selftext', 'year', and combined 'text' columns.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT id, title, selftext, year FROM reddit_posts", conn, index_col=None)
    # Replace missing selftext with empty string
    df['selftext'] = df['selftext'].fillna("")
    # Combine title and body for NER
    df['text'] = df['title'] + " " + df['selftext']
    return df


def extract_named_entities(df: pd.DataFrame, model: str = "en_core_web_trf") -> pd.DataFrame:
    """
    Uses spaCy to extract named entities from each post's text.

    Args:
        df: DataFrame containing the posts.
        model: spaCy model to use for NER (default: transformer-based English model).

    Returns:
        DataFrame with a new 'named_entities' column (list of (entity_text, entity_label) tuples).
    """
    nlp = spacy.load(model)  # Load spaCy model

    tqdm.pandas()  # Enable progress bar integration with pandas

    # Disable unused components for speed (only NER needed)
    with nlp.select_pipes(disable=["tagger", "parser", "lemmatizer"]):
        # Process texts in batches for efficiency
        docs = list(tqdm(nlp.pipe(df['text'], batch_size=8), total=len(df)))

    # Extract named entities from each processed doc
    df['named_entities'] = [[(ent.text, ent.label_) for ent in doc.ents] for doc in docs]
    return df


def analyze_entities(df: pd.DataFrame) -> Tuple[List[tuple[str, str]], Counter, Dict[str, List[tuple[str, str]]],
                                                Dict[str, Counter]]:
    """
    Analyzes and summarizes named entities overall and by year.

    Returns:
        - List of all entity tuples across all posts
        - Counter of overall entity label frequencies
        - Dictionary of yearly entity lists
        - Dictionary of yearly entity label counts
    """
    # Flatten all entity tuples from all posts
    all_entities = [ent for sublist in df['named_entities'] for ent in sublist]
    overall_entity_counts = Counter([ent[1] for ent in all_entities])

    yearly_entity_counts = {}
    yearly_entities = {}

    # Aggregate entities by year
    for year in df['year'].unique():
        year_df = df[df['year'] == year]
        year_entities = [ent for sublist in year_df['named_entities'] for ent in sublist]
        yearly_entities[str(year)] = year_entities
        yearly_entity_counts[str(year)] = Counter([ent[1] for ent in year_entities])

    return all_entities, overall_entity_counts, yearly_entities, yearly_entity_counts


def save_results(df: pd.DataFrame, all_entities: list[tuple[str, str]], overall_entity_counts: Counter,
                 yearly_entities: dict[str, list[tuple[str, str]]], yearly_entity_counts: dict[str, Counter],
                 output_file: str) -> None:
    """
    Serializes and saves named entity data and metadata to a JSON file.

    Args:
        df: The full DataFrame with entities.
        all_entities: Flattened list of all named entities.
        overall_entity_counts: Frequency count of entity types across all posts.
        yearly_entities: Year-specific lists of named entities.
        yearly_entity_counts: Year-specific frequency counts.
        output_file: Path to the output JSON file.
    """
    # Reduce DataFrame to relevant fields and convert to list of dicts
    df_serialized = df[['id', 'title', 'selftext', 'year', 'named_entities']].to_dict(orient='records')

    # Package everything for JSON serialization
    data = {
        "df": df_serialized,
        "all_entities": all_entities,
        "overall_entity_counts": dict(sorted(overall_entity_counts.items(), key=lambda x: x[1], reverse=True)),
        "yearly_entities": {year: ents for year, ents in yearly_entities.items()},
        "yearly_entity_counts": {year: dict(cnts.most_common()) for year, cnts in yearly_entity_counts.items()}
    }

    # Save to disk
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Entry point: load Reddit posts, run NER, aggregate results, and save to file."""
    db_path = "swiss_economy_reddit.db"
    model = "en_core_web_trf"  # spaCy transformer model (accurate but slow)
    output_file = "ner_results.json"

    # Make sure the DB exists
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        print("Please perform data collection first!")
        sys.exit(1)

    # Load data and process
    df = load_data(db_path)
    df = extract_named_entities(df, model)
    all_entities, overall_entity_counts, yearly_entities, yearly_entity_counts = analyze_entities(df)

    # Save all outputs
    save_results(df, all_entities, overall_entity_counts, yearly_entities, yearly_entity_counts, output_file)

    print("Analysis on all posts completed.")


# Run script directly
if __name__ == "__main__":
    main()
