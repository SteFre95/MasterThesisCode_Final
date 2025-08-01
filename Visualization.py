# Import required packages
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from typing import Optional, Dict, List, Tuple, Any
import plotly.express as px
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS
import os
import json
from matplotlib_venn import venn2
from bertopic import BERTopic

"""
This script visualizes Reddit post analysis results:
- Sentiment Analysis
- Named Entity Recognition (NER)
- Topic Modeling
"""


def ensure_dir(path: str) -> None:
    """Ensure the output directory exists, create it if necessary."""
    os.makedirs(path, exist_ok=True)

# === 1. Sentiment Analysis ===


def load_sentiment_data(db_path: str) -> pd.DataFrame:
    """
       Load yearly sentiment data from SQLite DB.
       Returns average sentiment scores and post counts per year.
    """
    query = """
    SELECT year, AVG(sentiment_VADER) AS avg_sentiment_vader,
           AVG(sentiment_FinBERT) AS avg_sentiment_finbert,
           COUNT(*) AS num_posts
    FROM reddit_posts
    WHERE sentiment_VADER IS NOT NULL AND sentiment_FinBERT IS NOT NULL
    GROUP BY year
    ORDER BY year
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    df["year"] = df["year"].astype(int)
    return df


def compute_averages(df: pd.DataFrame) -> tuple[float, float]:
    """Compute overall average sentiment scores across all years."""
    avg_vader = df["avg_sentiment_vader"].mean()
    avg_finbert = df["avg_sentiment_finbert"].mean()
    return avg_vader, avg_finbert


def plot_sentiment_analysis(df: pd.DataFrame, avg_vader: float, avg_finbert: float) -> None:
    """Plot post counts (bar) and sentiment scores (line) over years."""
    print("Visualizing sentiment analysis...")
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar: number of posts
    ax1.bar(df["year"], df["num_posts"], alpha=0.6, color="#ff7f0e", width=0.5, label="Post Count")
    ax1.set_ylabel("Number of Posts", fontsize=12, color="#ff7f0e")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.tick_params(axis='y', colors="#ff7f0e")
    ax1.set_ylim(0, max(df["num_posts"]) * 1.2)

    # Line: sentiment scores (on secondary y-axis)
    ax2 = ax1.twinx()
    ax2.plot(df["year"], df["avg_sentiment_vader"], marker="o", linestyle="-", color="#1f77b4", linewidth=2,
             label="Avg VADER Sentiment per year")
    ax2.plot(df["year"], df["avg_sentiment_finbert"], marker="s", linestyle="--", color="#2ca02c", linewidth=2,
             label="Avg FinBERT Sentiment per year")
    ax2.set_ylabel("Average Sentiment Score", fontsize=12)
    ax2.tick_params(axis='y')
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.7)

    # Adjust y-axis range to fit lines
    sentiment_min = min(df[["avg_sentiment_vader", "avg_sentiment_finbert"]].min()) - 0.1
    sentiment_max = max(df[["avg_sentiment_vader", "avg_sentiment_finbert"]].max()) + 0.1
    ax2.set_ylim(sentiment_min, sentiment_max)

    # Horizontal lines for overall averages
    ax2.axhline(avg_vader, color="#1f77b4", linestyle=":", linewidth=1.5, alpha=0.7,
                label="Avg VADER Sentiment (All Years)")
    ax2.axhline(avg_finbert, color="#2ca02c", linestyle=":", linewidth=1.5, alpha=0.7,
                label="Avg FinBERT Sentiment (All Years)")

    # Titles & legends
    plt.title("Sentiment Analysis Results of r/Switzerland Posts on the Economy",
              fontsize=14, fontweight="bold")
    plt.xticks(df["year"], rotation=45)

    # Combine both legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.15, -0.15),
               fontsize=10, frameon=False)

    # Adjust layout
    fig.subplots_adjust(bottom=0.3)

    # Save to disk
    out_dir = "visualizations/sentiment_analysis"
    ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, "sentiment_overview.png"))
    plt.close()


def sa_vis() -> None:
    """Main sentiment analysis visualization runner."""
    db_path = "swiss_economy_reddit.db"
    if not os.path.exists(db_path):
        print(f"[Sentiment Visualization] Skipping: Database file '{db_path}' not found.")
        return

    df = load_sentiment_data(db_path)

    if df.empty:
        print(f"[Sentiment Visualization] Skipping: No valid sentiment data found in '{db_path}'. Make sure sentiment "
              f"analysis was performed previously.")
        return  # Skip the visualization

    avg_vader, avg_finbert = compute_averages(df)
    plot_sentiment_analysis(df, avg_vader, avg_finbert)


# === 2. Named Entity Recognition (NER) ===

def load_data(json_path: str) -> tuple[dict, pd.DataFrame]:
    """Load NER results from JSON file into dict and DataFrame."""
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data["df"])
    return data, df


def is_valid_entity(text: str, label: Optional[str] = None) -> bool:
    """Filter out invalid or noisy entities."""
    text = text.strip()
    if label in {"DATE", "CARDINAL", "MONEY", "PERCENT", "TIME"}:
        return re.match(r"^\w[\w\-'., ]+$", text, re.UNICODE)
    return (
        text.lower() not in STOP_WORDS
        and 1 < len(text) < 30
        and re.match(r"^\w[\w\-' ]+$", text, re.UNICODE)
        and not text.isdigit()
    )


def normalize_entities(data: Dict) -> Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]]]:
    """Lowercase and filter all and yearly entity lists."""
    all_entities = [(ent[0].lower(), ent[1]) for ent in data["all_entities"] if is_valid_entity(ent[0], ent[1])]
    yearly_entities = {
        year: [(ent[0].lower(), ent[1]) for ent in ents if is_valid_entity(ent[0], ent[1])]
        for year, ents in data["yearly_entities"].items()
    }
    return all_entities, yearly_entities


def compute_entity_counts(all_entities: list[tuple[str, str]], yearly_entities: dict[str, list[tuple[str, str]]]) \
        -> tuple[Counter[str], dict[str, Counter[str]]]:
    """Count entity types overall and by year."""
    overall_entity_counts = Counter([ent[1] for ent in all_entities])
    yearly_entity_counts = {
        year: Counter([ent[1] for ent in ents]) for year, ents in yearly_entities.items()
    }
    return overall_entity_counts, yearly_entity_counts

# === Visualization Utilities for NER ===


def plot_bar_chart(df: pd.DataFrame, title: str, filename: str) -> None:
    """Create horizontal bar chart of entity types."""
    fig = px.bar(df, x='Count', y='Entity Type', orientation='h',
                 title=title, text='Count', color='Entity Type')
    out_dir = "visualizations/ner"
    ensure_dir(out_dir)
    fig.write_html(os.path.join(out_dir, filename))


def plot_wordcloud(freq_dict: Dict[str, int], title: str, filename: str) -> None:
    """Generate word cloud from entity frequencies."""
    wordcloud = WordCloud(width=1000, height=500, background_color='white',
                          colormap='viridis').generate_from_frequencies(freq_dict)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)

    out_dir = "visualizations/ner"
    ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_treemap(df: pd.DataFrame, title: str, filename: str) -> None:
    """Treemap showing entity types and individual entities."""
    fig = px.treemap(df, path=["Entity Type", "Entity"], values="Count",
                     title=title, color="Entity Type",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    out_dir = "visualizations/ner"
    ensure_dir(out_dir)
    fig.write_html(os.path.join(out_dir, filename))


def plot_trend_chart(entity_trends: pd.DataFrame, top_entity_types: list[str]) -> None:
    """Line chart showing entity type usage trends over time."""
    filtered = entity_trends[top_entity_types]
    ax = filtered.plot(kind='line', figsize=(12, 6), marker='o', markersize=6, linewidth=2)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.title("Top 5 Entity Type Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend(title="Entity Type")

    out_dir = "visualizations/ner"
    ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entity_type_trends.png"))
    plt.close()

# === NER Visualization Orchestration ===


def visualize_overall(all_entities: list[tuple[str, str]], overall_entity_counts: Counter) -> None:
    """Generate overall NER visualizations."""
    # Top 10 Entity Types (Bar Chart)
    df = pd.DataFrame(overall_entity_counts.items(), columns=['Entity Type', 'Count'])
    df = df.sort_values(by='Count', ascending=False).head(10)
    plot_bar_chart(df, "Top 10 Named Entity Types (Overall)", "ner_bar_overall.html")

    # Word Cloud
    freq = Counter([ent[0] for ent in all_entities])
    plot_wordcloud(dict(freq.most_common(100)), "Top Named Entities in Collected Posts (Overall)",
                   "ner_wordcloud_overall.png")

    # Treemap
    entity_counter = Counter(all_entities)
    treemap_df = pd.DataFrame(
        [{"Entity Type": k[1], "Entity": k[0], "Count": v} for k, v in entity_counter.items()]
    )
    plot_treemap(treemap_df, "Overall Named Entity Treemap", "ner_treemap_overall.html")


def visualize_yearly(yearly_entities: dict[str, list[tuple[str, str]]],
                     yearly_entity_counts: dict[str, Counter]) -> None:
    """Create NER plots for each year."""
    for year, entities in yearly_entities.items():
        print(f"\nVisualizing NER for {year}...")

        # Bar Chart
        df = pd.DataFrame(yearly_entity_counts[year].items(), columns=['Entity Type', 'Count'])
        df = df.sort_values(by='Count', ascending=False).head(10)
        plot_bar_chart(df, f"Top 10 Named Entity Types in {year}",
                       f"ner_bar_{year}.html")

        # Word Cloud
        freq = Counter([ent[0] for ent in entities])
        plot_wordcloud(dict(freq.most_common(100)), f"Top Named Entities in {year}",
                       f"ner_wordcloud_{year}.png")

        # Treemap
        entity_counter = Counter(entities)
        treemap_df = pd.DataFrame(
            [{"Entity Type": k[1], "Entity": k[0], "Count": v} for k, v in entity_counter.items()]
        )
        plot_treemap(treemap_df, f"Named Entity Treemap for {year}",
                     f"ner_treemap_{year}.html")


def visualize_trends(yearly_entity_counts: dict[str, Counter], overall_entity_counts: Counter) -> None:
    """Plot time-series trends of most frequent entity types."""
    top_entity_types = [etype for etype, _ in overall_entity_counts.most_common(5)]
    trend_matrix = pd.DataFrame({int(year): dict(cnt) for year, cnt in yearly_entity_counts.items()}).fillna(0).T
    trend_matrix.index = trend_matrix.index.astype(int)
    trend_matrix = trend_matrix.sort_index()
    plot_trend_chart(trend_matrix, top_entity_types)


def ner_vis() -> None:
    """Main NER visualization runner."""
    json_path = "ner_results.json"

    if not os.path.exists(json_path):
        print(f"[NER Visualization] Skipping: '{json_path}' not found. Make sure named entity recognition was "
              f"performed previously.")
        return  # Skip the visualization entirely

    data, df = load_data(json_path)
    all_entities, yearly_entities = normalize_entities(data)
    overall_counts, yearly_counts = compute_entity_counts(all_entities, yearly_entities)

    visualize_overall(all_entities, overall_counts)
    visualize_trends(yearly_counts, overall_counts)
    visualize_yearly(yearly_entities, yearly_counts)


# === 3. Topic Modeling ===

def process_year(year: str, data: Dict[str, Any]) -> None:
    """Generate visualizations for BERTopic results of a given year."""
    print(f"\nVisualizing topic modeling for {year}...")
    out_dir = "visualizations/topic_modeling"
    ensure_dir(out_dir)

    model_path = f"bertopic_model_{year}"
    model = BERTopic.load(model_path)
    topic_info = model.get_topic_info()
    n_topics = len(topic_info[topic_info.Topic != -1])  # Exclude outliers

    if not data.get("topic_info") or n_topics < 1:
        print(f"Skipping {year}: no topic info")
        return

    # Bar chart: topic sizes
    topics = [t["Name"] for t in data["topic_info"]]
    sizes = [t["Count"] for t in data["topic_info"]]

    plt.figure(figsize=(10, 6))
    plt.barh(topics, sizes, color="skyblue")
    plt.title(f"Topic Sizes for {year}")
    plt.xlabel("Number of Documents")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"topic_sizes_{year}.png"))
    plt.close()

    # Interactive Barchart
    print(f"Computing interactive barchart for {year}")
    fig = model.visualize_barchart(top_n_topics=None, n_words=10)
    fig.update_layout(title_text=f"Top Words per Topic — {year}")
    fig.write_html(os.path.join(out_dir, f"topic_barchart_{year}.html"))

    # Venn: only when 2 topics
    if n_topics == 2:
        try:
            topics_ids = topic_info[topic_info.Topic != -1]["Topic"].tolist()
            topic_0 = set(word for word, _ in model.get_topic(topics_ids[0]))
            topic_1 = set(word for word, _ in model.get_topic(topics_ids[1]))
            venn2([topic_0, topic_1], set_labels=("Topic 0", "Topic 1"))
            plt.title(f"Keyword Overlap Between Topics - {year}")
            plt.savefig(os.path.join(out_dir, f"topic_venn_{year}.png"))
            plt.close()
        except Exception as e:
            print(f"Venn diagram failed for {year}: {e}")

    # UMAP: only when ≥3 topics
    if n_topics > 2:
        try:
            print(f"Computing topic embedding plot for {year}")
            embeddings = model.topic_embeddings_
            if embeddings is None or len(embeddings) < 5:
                print(f"Skipping UMAP: Not enough topic embeddings")
            else:
                fig = model.visualize_topics()
                fig.write_html(os.path.join(out_dir, f"topic_umap_{year}.html"))
        except Exception as e:
            print(f"UMAP visualization failed for {year}: {e}")
    else:
        print(f"Skipping UMAP visualization for {year}: only {n_topics} topics")


def tm_vis() -> None:
    """Main topic modeling visualization runner."""
    json_path = "bertopic_results.json"
    if not os.path.exists(json_path):
        print(f"[Topic Modeling Visualization] Skipping: '{json_path}' not found. Make sure topic modeling was "
              f"performed previously.")
        return

    with open(json_path) as f:
        results = json.load(f)

    available_years = [
        year for year in results if os.path.exists(f"bertopic_model_{year}")
    ]

    if not available_years:
        print("[Topic Modeling Visualization] Skipping: No BERTopic models found for any year. Make sure topic "
              f"modeling was performed previously.")
        return

    for year in available_years:
        process_year(year, results.get(year, {}))


# === Entry Point ===

if __name__ == "__main__":
    sa_vis()
    ner_vis()
    tm_vis()
