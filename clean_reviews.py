"""
Stage 3: Data Cleaning & Exploratory Analysis.

Takes data/reviews_raw.csv (88k+ reviews from Stage 2) and produces:
  - data/reviews_clean.csv (cleaned, filtered to analysis window)
  - figures/data_overview_*.png (exploratory figures for the report)
  - Summary statistics printed to terminal

Cleaning steps:
  1. Drop reviews with no text
  2. Parse submission_time to proper datetime, drop unparseable
  3. Filter to analysis window: 2023-01-01 -> 2026-04-22
  4. Drop duplicate review_ids (keep first)
  5. Normalise whitespace in text and title
  6. Flag very short reviews (<20 chars) as low-info
  7. Create combined `full_text` = title + text (for NLP later)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# --- Config ----------------------------------------------------------------

DATA_DIR = Path("data")
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "reviews_raw.csv"
OUTPUT_FILE = DATA_DIR / "reviews_clean.csv"

ANALYSIS_START = pd.Timestamp("2023-01-01", tz="UTC")
ANALYSIS_END = pd.Timestamp("2026-04-23", tz="UTC")

# Apply a consistent, professional figure style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


# --- Cleaning helpers ------------------------------------------------------

def normalise_whitespace(s):
    """Collapse all whitespace to single spaces, strip ends."""
    if pd.isna(s):
        return s
    return re.sub(r"\s+", " ", str(s)).strip()


def load_and_clean(path):
    """Load raw CSV, apply cleaning steps, return cleaned DataFrame."""
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df):,}")
    
    # Drop rows with no text (ratings-only reviews)
    before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"  After dropping empty text: {len(df):,} (removed {before - len(df):,})")
    
    # Parse datetime
    df["submission_time"] = pd.to_datetime(
        df["submission_time"], errors="coerce", utc=True
    )
    before = len(df)
    df = df.dropna(subset=["submission_time"])
    print(f"  After dropping bad dates: {len(df):,} (removed {before - len(df):,})")
    
    # Filter to analysis window
    before = len(df)
    df = df[(df["submission_time"] >= ANALYSIS_START) &
            (df["submission_time"] <= ANALYSIS_END)]
    print(f"  After filtering to analysis window "
          f"[{ANALYSIS_START.date()} -> {ANALYSIS_END.date()}]: "
          f"{len(df):,} (removed {before - len(df):,})")
    
    # Dedupe on review_id
    before = len(df)
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    print(f"  After deduplication: {len(df):,} (removed {before - len(df):,})")
    
    # Normalise whitespace
    df["text"] = df["text"].apply(normalise_whitespace)
    df["title"] = df["title"].apply(normalise_whitespace)
    
    # Combined text field for NLP
    df["full_text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    df["full_text"] = df["full_text"].apply(normalise_whitespace)
    
    # Length features
    df["text_length_chars"] = df["text"].astype(str).str.len()
    df["text_length_words"] = df["text"].astype(str).str.split().str.len()
    
    # Flag very short reviews (signal, not filter - we keep them but can exclude for LDA)
    df["is_short"] = df["text_length_chars"] < 20
    
    # Date features for temporal analysis
    df["date"] = df["submission_time"].dt.date
    df["year_month"] = df["submission_time"].dt.to_period("M").astype(str)
    df["year"] = df["submission_time"].dt.year
    
    # Sentiment ground truth: collapse 1-5 rating to positive/neutral/negative
    def rating_to_sentiment(r):
        if pd.isna(r): return None
        r = int(r)
        if r <= 2: return "negative"
        if r == 3: return "neutral"
        return "positive"
    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)
    
    return df


# --- Exploratory figures ---------------------------------------------------

def plot_rating_distribution(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["rating"].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#F06C00")
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Figure 1. Distribution of star ratings\n"
                 f"(n = {len(df):,} reviews)")
    # Add percentage labels
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v/total:.1%}", ha="center", va="bottom", fontsize=11)
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_reviews_over_time(df, out_path):
    monthly = df.groupby("year_month").size().reset_index(name="count")
    monthly["year_month_dt"] = pd.to_datetime(monthly["year_month"])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="year_month_dt", y="count", ax=ax,
                 color="#F06C00", linewidth=2, marker="o", markersize=6)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Figure 2. Review volume over time")
    ax.tick_params(axis="x", rotation=45)
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_category_counts(df, out_path):
    cat_counts = df["category_keyword"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(y=cat_counts.index, x=cat_counts.values, ax=ax,
                color="#F06C00")
    ax.set_xlabel("Number of reviews")
    ax.set_ylabel("Category")
    ax.set_title("Figure 3. Reviews by product category")
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_text_length_distribution(df, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Clip at 99th percentile for readability
    p99 = df["text_length_words"].quantile(0.99)
    plot_df = df[df["text_length_words"] <= p99]
    sns.histplot(data=plot_df, x="text_length_words", bins=50, ax=ax,
                 color="#F06C00")
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Number of reviews")
    ax.set_title(f"Figure 4. Distribution of review length (clipped at 99th percentile = {int(p99)} words)")
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_avg_rating_by_category(df, out_path):
    cat_ratings = (df.groupby("category_keyword")["rating"]
                     .agg(["mean", "count"])
                     .sort_values("mean"))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#D32F2F" if m < 4 else "#F06C00" if m < 4.5 else "#2E7D32"
              for m in cat_ratings["mean"]]
    ax.barh(cat_ratings.index, cat_ratings["mean"], color=colors)
    ax.set_xlim(1, 5)
    ax.axvline(df["rating"].mean(), color="black", linestyle="--",
               label=f"Overall mean = {df['rating'].mean():.2f}")
    ax.set_xlabel("Mean star rating")
    ax.set_ylabel("Category")
    ax.set_title("Figure 5. Average rating by category")
    ax.legend()
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


# --- Main -------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run scrape_reviews.py first.")
        return
    
    df = load_and_clean(INPUT_FILE)
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Cleaned reviews: {len(df):,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print(f"Categories: {df['category_keyword'].nunique()}")
    print(f"Date range: {df['submission_time'].min().date()} to {df['submission_time'].max().date()}")
    print(f"Mean rating: {df['rating'].mean():.2f}")
    print(f"Median review length: {int(df['text_length_words'].median())} words")
    
    print("\nRating distribution:")
    print(df["rating"].value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.1%}"))
    
    print("\nRating sentiment distribution (our ground truth):")
    print(df["rating_sentiment"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))
    
    print("\nReviews per year:")
    print(df["year"].value_counts().sort_index())
    
    # Save cleaned data
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")
    
    # Figures
    print("\nGenerating exploratory figures...")
    plot_rating_distribution(df, FIGS_DIR / "fig01_rating_distribution.png")
    plot_reviews_over_time(df, FIGS_DIR / "fig02_reviews_over_time.png")
    plot_category_counts(df, FIGS_DIR / "fig03_category_counts.png")
    plot_text_length_distribution(df, FIGS_DIR / "fig04_text_length.png")
    plot_avg_rating_by_category(df, FIGS_DIR / "fig05_avg_rating_by_category.png")
    
    print(f"\n[OK] All figures saved to {FIGS_DIR}/")


if __name__ == "__main__":
    main()