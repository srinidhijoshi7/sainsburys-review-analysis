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

# Import pandas for loading, cleaning, and saving tabular data
import pandas as pd

# Import numpy for numerical operations (used in some calculations)
import numpy as np

# Import matplotlib for creating and saving figures
import matplotlib.pyplot as plt

# Import seaborn for styled statistical plots
import seaborn as sns

# Import Path for cross-platform file path handling
from pathlib import Path

# Import re for regular expressions (used to normalise whitespace)
import re

# --- Config ----------------------------------------------------------------

# Folder containing input/output data files
DATA_DIR = Path("data")

# Folder where we save output figures (created if it doesn't exist)
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)

# Path to the raw reviews file produced by scrape_reviews.py
INPUT_FILE = DATA_DIR / "reviews_raw.csv"

# Path where the cleaned reviews CSV will be saved
OUTPUT_FILE = DATA_DIR / "reviews_clean.csv"

# Start and end of the analysis window (UTC-aware timestamps)
ANALYSIS_START = pd.Timestamp("2023-01-01", tz="UTC")
ANALYSIS_END = pd.Timestamp("2026-04-23", tz="UTC")

# Apply a consistent, professional visual style to all figures
sns.set_theme(style="whitegrid", context="talk")

# Set screen resolution for interactive viewing
plt.rcParams["figure.dpi"] = 120

# Set output resolution for saved figures (higher = sharper)
plt.rcParams["savefig.dpi"] = 200

# Automatically crop whitespace around saved figures
plt.rcParams["savefig.bbox"] = "tight"

# --- Cleaning helpers ------------------------------------------------------

def normalise_whitespace(s):
    """Collapse all whitespace to single spaces, strip ends."""

    # Return NaN as-is (don't try to process missing values)
    if pd.isna(s):
        return s

    # Replace any sequence of whitespace characters (spaces, tabs, newlines) with a single space
    # Then strip leading/trailing whitespace
    return re.sub(r"\s+", " ", str(s)).strip()


def load_and_clean(path):
    """Load raw CSV, apply cleaning steps, return cleaned DataFrame."""

    # Print progress message
    print(f"Loading {path}...")

    # Read the raw CSV into a DataFrame
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df):,}")

    # Step 1: Drop reviews that have no text body (rating-only submissions are not useful for NLP)
    before = len(df)
    df = df.dropna(subset=["text"])                              # Remove rows where text is NaN
    df = df[df["text"].astype(str).str.strip() != ""]           # Remove rows where text is empty/whitespace
    print(f"  After dropping empty text: {len(df):,} (removed {before - len(df):,})")

    # Step 2: Parse submission_time as a timezone-aware datetime
    df["submission_time"] = pd.to_datetime(
        df["submission_time"], errors="coerce", utc=True  # 'coerce' turns unparseable dates into NaT
    )

    # Drop rows where the date couldn't be parsed (NaT values)
    before = len(df)
    df = df.dropna(subset=["submission_time"])
    print(f"  After dropping bad dates: {len(df):,} (removed {before - len(df):,})")

    # Step 3: Filter to the analysis window (only keep reviews within the specified date range)
    before = len(df)
    df = df[(df["submission_time"] >= ANALYSIS_START) &
            (df["submission_time"] <= ANALYSIS_END)]
    print(f"  After filtering to analysis window "
          f"[{ANALYSIS_START.date()} -> {ANALYSIS_END.date()}]: "
          f"{len(df):,} (removed {before - len(df):,})")

    # Step 4: Remove duplicate reviews (same review_id appearing more than once — keep the first)
    before = len(df)
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    print(f"  After deduplication: {len(df):,} (removed {before - len(df):,})")

    # Step 5: Normalise whitespace in the review body and title fields
    df["text"] = df["text"].apply(normalise_whitespace)
    df["title"] = df["title"].apply(normalise_whitespace)

    # Step 6: Create a combined text field by concatenating title and text (used by NLP models later)
    df["full_text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    df["full_text"] = df["full_text"].apply(normalise_whitespace)

    # Step 7: Calculate character and word length features for each review
    df["text_length_chars"] = df["text"].astype(str).str.len()         # Number of characters
    df["text_length_words"] = df["text"].astype(str).str.split().str.len()  # Number of words

    # Flag very short reviews as potentially low-information (< 20 chars)
    # We keep them in the dataset but can exclude them for LDA topic modelling
    df["is_short"] = df["text_length_chars"] < 20

    # Step 8: Create useful date/time derived columns for temporal analysis
    df["date"] = df["submission_time"].dt.date                           # Calendar date (no time)
    df["year_month"] = df["submission_time"].dt.to_period("M").astype(str)  # e.g. "2024-03"
    df["year"] = df["submission_time"].dt.year                           # Just the year

    # Step 9: Convert the numeric star rating into a 3-class sentiment label (ground truth)
    def rating_to_sentiment(r):
        # Return None if the rating is missing
        if pd.isna(r): return None
        r = int(r)
        if r <= 2: return "negative"   # 1 or 2 stars = negative
        if r == 3: return "neutral"    # 3 stars = neutral
        return "positive"              # 4 or 5 stars = positive

    # Apply the rating-to-sentiment function to every row
    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    # Return the cleaned DataFrame
    return df


# --- Exploratory figures ---------------------------------------------------

def plot_rating_distribution(df, out_path):
    """Create a bar chart showing how many reviews were left at each star rating."""

    # Create a figure with a specific size
    fig, ax = plt.subplots(figsize=(8, 5))

    # Count how many reviews have each star rating (1–5) and sort by rating
    counts = df["rating"].value_counts().sort_index()

    # Draw a bar chart with Sainsbury's orange colour
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#F06C00")

    # Axis labels and title
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Figure 1. Distribution of star ratings\n"
                 f"(n = {len(df):,} reviews)")

    # Add percentage labels above each bar
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v/total:.1%}", ha="center", va="bottom", fontsize=11)

    # Save the figure and close it to free memory
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_reviews_over_time(df, out_path):
    """Create a line chart showing review volume per month over the analysis window."""

    # Count reviews per year-month period and create a proper datetime column for the x-axis
    monthly = df.groupby("year_month").size().reset_index(name="count")
    monthly["year_month_dt"] = pd.to_datetime(monthly["year_month"])

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw the line with markers at each month
    sns.lineplot(data=monthly, x="year_month_dt", y="count", ax=ax,
                 color="#F06C00", linewidth=2, marker="o", markersize=6)

    # Axis labels, title, and tick rotation
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Figure 2. Review volume over time")
    ax.tick_params(axis="x", rotation=45)

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_category_counts(df, out_path):
    """Create a horizontal bar chart showing how many reviews exist per category keyword."""

    # Count reviews per category, sorted by frequency (most common at top)
    cat_counts = df["category_keyword"].value_counts()

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw horizontal bars — categories on y-axis, counts on x-axis
    sns.barplot(y=cat_counts.index, x=cat_counts.values, ax=ax, color="#F06C00")

    # Axis labels and title
    ax.set_xlabel("Number of reviews")
    ax.set_ylabel("Category")
    ax.set_title("Figure 3. Reviews by product category")

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_text_length_distribution(df, out_path):
    """Create a histogram of review word counts, clipped at the 99th percentile."""

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Clip at the 99th percentile so extreme outliers don't compress the histogram
    p99 = df["text_length_words"].quantile(0.99)
    plot_df = df[df["text_length_words"] <= p99]

    # Draw the histogram with 50 bins
    sns.histplot(data=plot_df, x="text_length_words", bins=50, ax=ax, color="#F06C00")

    # Axis labels and title
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Number of reviews")
    ax.set_title(f"Figure 4. Distribution of review length (clipped at 99th percentile = {int(p99)} words)")

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_avg_rating_by_category(df, out_path):
    """Create a horizontal bar chart of the mean star rating per category."""

    # Calculate mean rating and review count per category, sorted by mean rating ascending
    cat_ratings = (df.groupby("category_keyword")["rating"]
                   .agg(["mean", "count"])
                   .sort_values("mean"))

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colour-code bars: red for low ratings, orange for medium, green for high
    colors = ["#D32F2F" if m < 4 else "#F06C00" if m < 4.5 else "#2E7D32"
              for m in cat_ratings["mean"]]

    # Draw horizontal bars
    ax.barh(cat_ratings.index, cat_ratings["mean"], color=colors)

    # Set x-axis range to 1–5 (star rating scale)
    ax.set_xlim(1, 5)

    # Add a vertical dashed line for the overall mean rating
    ax.axvline(df["rating"].mean(), color="black", linestyle="--",
               label=f"Overall mean = {df['rating'].mean():.2f}")

    # Axis labels, title, and legend
    ax.set_xlabel("Mean star rating")
    ax.set_ylabel("Category")
    ax.set_title("Figure 5. Average rating by category")
    ax.legend()

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


# --- Main -------------------------------------------------------------------

def main():
    # Check that the raw reviews file exists before proceeding
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run scrape_reviews.py first.")
        return

    # Run the cleaning pipeline and get the cleaned DataFrame
    df = load_and_clean(INPUT_FILE)

    # Print a section header
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Print key statistics about the cleaned dataset
    print(f"Cleaned reviews: {len(df):,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print(f"Categories: {df['category_keyword'].nunique()}")
    print(f"Date range: {df['submission_time'].min().date()} to {df['submission_time'].max().date()}")
    print(f"Mean rating: {df['rating'].mean():.2f}")
    print(f"Median review length: {int(df['text_length_words'].median())} words")

    # Print the percentage of reviews at each star rating
    print("\nRating distribution:")
    print(df["rating"].value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.1%}"))

    # Print the percentage breakdown into negative/neutral/positive (our ground truth labels)
    print("\nRating sentiment distribution (our ground truth):")
    print(df["rating_sentiment"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

    # Print how many reviews fell in each year
    print("\nReviews per year:")
    print(df["year"].value_counts().sort_index())

    # Save the cleaned data to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")

    # Generate all exploratory figures and save them to the figures folder
    print("\nGenerating exploratory figures...")
    plot_rating_distribution(df, FIGS_DIR / "fig01_rating_distribution.png")
    plot_reviews_over_time(df, FIGS_DIR / "fig02_reviews_over_time.png")
    plot_category_counts(df, FIGS_DIR / "fig03_category_counts.png")
    plot_text_length_distribution(df, FIGS_DIR / "fig04_text_length.png")
    plot_avg_rating_by_category(df, FIGS_DIR / "fig05_avg_rating_by_category.png")

    # Confirm all figures have been saved
    print(f"\n[OK] All figures saved to {FIGS_DIR}/")


# Only run main() when this script is executed directly
if __name__ == "__main__":
    main()
