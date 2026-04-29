"""
Stage 6: Event-annotated temporal sentiment analysis.

Takes BERT's per-review sentiment scores and aggregates them by month,
then overlays verified UK events to test whether sentiment spikes align
with real-world business moments.

Events compiled from: ONS CPI bulletins (March 2023 peak), House of
Commons Library inflation briefing, Sainsbury's corporate press releases
(Nectar Prices launch 11 Apr 2023), and UK news reports of the 16 Mar
2024 IT outage.

Outputs:
- figures/fig13_sentiment_with_events.png
- figures/fig14_topic_volume_with_events.png
- figures/fig15_category_sentiment_heatmap.png
- data/monthly_sentiment.csv
- data/event_impact_stats.csv (+/- 30 day comparison around each event)
"""

# Import pandas for loading and aggregating the data
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import matplotlib for creating and saving figures
import matplotlib.pyplot as plt

# Import matplotlib's date formatting utilities for the x-axis of time-series plots
import matplotlib.dates as mdates

# Import seaborn for heatmap styling
import seaborn as sns

# Import Path for file path operations
from pathlib import Path

# --- Config ----------------------------------------------------------------

# Folder containing input/output data files
DATA_DIR = Path("data")

# Folder where figures will be saved
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)  # Create the folder if it doesn't already exist

# Input: full-dataset BERT sentiment predictions from apply_bert_full.py
SENTIMENT_FILE = DATA_DIR / "reviews_full_sentiment.csv"

# Input: negative reviews with LDA topic assignments from topic_modelling.py
TOPICS_FILE = DATA_DIR / "negative_reviews_with_topics.csv"

# Output: monthly aggregated sentiment for reporting
MONTHLY_OUT = DATA_DIR / "monthly_sentiment.csv"

# Output: event impact table (sentiment 30 days before vs. after each event)
EVENT_IMPACT_OUT = DATA_DIR / "event_impact_stats.csv"

# Apply consistent professional styling to all figures
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# --- Events ----------------------------------------------------------------

# A list of real-world events to annotate on the sentiment timeline.
# All dates and descriptions were verified against official sources
# (see module docstring for the source list).
EVENTS = [
    # UK food inflation peaked at 19.1% — likely associated with more price complaints
    {"date": "2023-03-31", "label": "UK food inflation peaks (19.1%)",
     "type": "macro", "color": "#D32F2F"},

    # Sainsbury's launched its Nectar Prices loyalty discount scheme
    {"date": "2023-04-11", "label": "Sainsbury's launches Nectar Prices",
     "type": "sainsburys", "color": "#F06C00"},

    # Aldi Price Match scheme expanded to 400+ products
    {"date": "2023-08-24", "label": "Aldi Price Match expansion (400+)",
     "type": "sainsburys", "color": "#F06C00"},

    # Sainsbury's suffered an IT outage affecting online ordering
    {"date": "2024-03-16", "label": "Sainsbury's IT outage",
     "type": "sainsburys", "color": "#D32F2F"},

    # UK food inflation fell back to normal levels (1.7%)
    {"date": "2024-05-31", "label": "Food inflation normalises (1.7%)",
     "type": "macro", "color": "#2E7D32"},

    # Nectar Prices scheme reached 5,000 discounted products
    {"date": "2024-09-21", "label": "Nectar Prices hits 5,000 products",
     "type": "sainsburys", "color": "#F06C00"},

    # Aldi Price Match extended to Sainsbury's Local convenience stores
    {"date": "2024-11-05", "label": "Aldi Price Match in Local stores",
     "type": "sainsburys", "color": "#F06C00"},
]

# --- Helpers ---------------------------------------------------------------

def load_sentiment():
    """Load the full-dataset BERT sentiment CSV and add a year_month column."""

    # Read the CSV produced by apply_bert_full.py
    df = pd.read_csv(SENTIMENT_FILE)

    # Parse submission_time as a UTC-aware datetime
    df["submission_time"] = pd.to_datetime(df["submission_time"], utc=True)

    # Create a year_month column truncated to the start of each calendar month (for groupby)
    df["year_month"] = df["submission_time"].dt.to_period("M").dt.to_timestamp()

    return df


def aggregate_monthly(df):
    """Monthly mean sentiment + share of negative reviews."""

    # Map BERT's string labels to a numeric sentiment score: -1, 0, +1
    score_map = {"negative": -1, "neutral": 0, "positive": 1}

    # Work on a copy so we don't modify the original DataFrame
    df = df.copy()

    # Create a numeric sentiment score column
    df["sentiment_numeric"] = df["bert_pred"].map(score_map)

    # Group by year_month and compute summary statistics for each month
    monthly = df.groupby("year_month").agg(
        mean_sentiment=("sentiment_numeric", "mean"),   # Average sentiment score (-1 to +1)
        pct_negative=("bert_pred",
                       lambda x: (x == "negative").sum() / len(x) * 100),  # % negative reviews
        review_count=("bert_pred", "count"),            # Total number of reviews this month
        mean_rating=("rating", "mean"),                 # Average star rating this month
    ).reset_index()

    # Drop months with fewer than 100 reviews (too few to give a stable monthly average)
    monthly = monthly[monthly["review_count"] >= 100].copy()

    return monthly


def plot_sentiment_with_events(monthly, events, out_path):
    """Two-panel figure: mean sentiment (top) and % negative (bottom), with event annotations."""

    # Create a figure with two stacked panels sharing the same x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})

    # --- Top panel: mean monthly sentiment score ---
    ax1.plot(monthly["year_month"], monthly["mean_sentiment"],
             marker="o", linewidth=2.5, color="#F06C00", markersize=6,
             label="Mean monthly BERT sentiment")

    # Draw a horizontal reference line at 0 (neutral)
    ax1.axhline(0, color="gray", linewidth=0.7, alpha=0.6)

    # Axis label and title for the top panel
    ax1.set_ylabel("Mean sentiment\n(-1 negative, +1 positive)")
    ax1.set_title("Figure 13. Sainsbury's customer review sentiment over time,\n"
                  "annotated with UK retail and macroeconomic events",
                  fontsize=15)

    # --- Bottom panel: percentage of reviews predicted negative ---
    ax2.fill_between(monthly["year_month"], 0, monthly["pct_negative"],
                     color="#D32F2F", alpha=0.35)  # Shaded area under the line

    ax2.plot(monthly["year_month"], monthly["pct_negative"],
             marker="s", linewidth=2, color="#D32F2F", markersize=5,
             label="% negative reviews (BERT)")

    # Axis labels for the bottom panel
    ax2.set_ylabel("% of reviews\npredicted negative")
    ax2.set_xlabel("Month")

    # --- Add vertical event lines to BOTH panels ---
    for event in events:
        # Parse the event date as a pandas Timestamp
        ev_date = pd.Timestamp(event["date"])

        # Skip events that fall outside our data's date range
        if ev_date < monthly["year_month"].min() or ev_date > monthly["year_month"].max():
            continue

        # Draw a vertical dashed line on both panels at the event date
        for ax in (ax1, ax2):
            ax.axvline(ev_date, color=event["color"], linestyle="--",
                       alpha=0.7, linewidth=1.5)

        # Add a text label on the top panel only, staggered vertically to avoid overlap
        y_positions = [0.95, 0.85, 0.75, 0.65]
        idx = events.index(event)           # Position index of this event
        y_pos = y_positions[idx % len(y_positions)]

        ax1.annotate(event["label"], xy=(ev_date, 1),
                     xycoords=("data", "axes fraction"),  # x in data coords, y in fraction of axes
                     xytext=(5, -15 - (idx % 4) * 18),   # Offset the label slightly right and down
                     textcoords="offset points",
                     fontsize=8.5, color=event["color"],
                     rotation=0, ha="left", va="top",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="white", edgecolor=event["color"],
                               alpha=0.9))   # White box background for readability

    # Format x-axis to show every 3 months
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Rotate x-axis labels 45 degrees to prevent overlapping
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add legends to both panels
    ax1.legend(loc="lower right", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)

    # Adjust layout to prevent panel labels from overlapping
    plt.tight_layout()

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_topic_volume_with_events(events, out_path):
    """Stacked bar chart: monthly negative-review volume by topic, with events."""

    # Check the topics file exists (it's created by topic_modelling.py)
    if not TOPICS_FILE.exists():
        print("  [skip] topics file not found")
        return

    # Load the negative reviews with their LDA topic assignments
    topics = pd.read_csv(TOPICS_FILE)

    # Parse submission_time to UTC-aware datetime
    topics["submission_time"] = pd.to_datetime(topics["submission_time"], utc=True)

    # Create year_month column for monthly grouping
    topics["year_month"] = topics["submission_time"].dt.to_period("M").dt.to_timestamp()

    # Human-readable names for each LDA topic (matches topic_modelling.py)
    topic_labels = {
        0: "Meat/Protein Quality",
        1: "Reformulation/Recipe",
        2: "Taste & Texture",
        3: "Price-Value",
        4: "Prepared Food/Premium",
    }

    # Map topic IDs to their human-readable names
    topics["topic_name"] = topics["dominant_topic"].map(topic_labels)

    # Count the number of reviews per month per topic, filling missing combinations with 0
    counts = (topics.groupby(["year_month", "topic_name"])
              .size().unstack(fill_value=0))

    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 7))

    # Define distinct colours for each topic bar segment
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Draw a stacked bar chart (each bar = one month, segments = topics)
    counts.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.9)

    # Only label every 3rd x-axis tick (monthly bars get crowded)
    ax.set_xticks(range(0, len(counts), 3))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in counts.index[::3]],
                       rotation=45, ha="right")

    # Axis labels and title
    ax.set_xlabel("Month")
    ax.set_ylabel("Negative review volume")
    ax.set_title("Figure 14. Volume of negative reviews by complaint topic over time")

    # Place the legend outside the plot to the right
    ax.legend(title="Topic", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_category_sentiment_heatmap(df, out_path):
    """Heatmap of mean BERT sentiment by product category × year-quarter."""

    # Work on a copy
    df = df.copy()

    # Map sentiment labels to numeric scores
    df["sentiment_numeric"] = df["bert_pred"].map({"negative": -1, "neutral": 0, "positive": 1})

    # Create a year-quarter column (e.g. "2024Q1") for column headers
    df["quarter"] = df["submission_time"].dt.to_period("Q").astype(str)

    # Build a pivot table: rows = category, cols = quarter, values = mean sentiment
    pivot = df.pivot_table(index="category_keyword",
                           columns="quarter",
                           values="sentiment_numeric",
                           aggfunc="mean")

    # Drop any category that is missing data for more than 40% of quarters
    pivot = pivot.dropna(thresh=pivot.shape[1] * 0.6)

    # Sort categories by their overall mean sentiment (worst at top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw a red-yellow-green heatmap centred at 0 (neutral)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-0.5, vmax=1, ax=ax,
                cbar_kws={"label": "Mean sentiment (-1 neg to +1 pos)"},
                annot_kws={"size": 8})

    # Title and axis labels
    ax.set_title("Figure 15. Mean BERT sentiment by category and quarter\n"
                 "(lower values = worse customer sentiment)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Product category")

    # Rotate x-axis labels so they don't overlap
    plt.xticks(rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def compute_event_impact(df, events):
    """
    For each event, compare mean sentiment in the 30 days BEFORE vs AFTER.
    This gives a quantitative estimate of whether customer sentiment shifted
    around the time of each event. This is associational, NOT causal.
    """

    # Work on a copy and add a numeric sentiment score column
    df = df.copy()
    df["sentiment_numeric"] = df["bert_pred"].map({"negative": -1, "neutral": 0, "positive": 1})

    # List to hold one results row per event
    rows = []

    # Loop over each event in the list
    for event in events:
        # Parse the event date as a timezone-aware timestamp (UTC)
        ev_date = pd.Timestamp(event["date"], tz="UTC")

        # Select reviews in the 30 days immediately BEFORE the event
        before = df[(df["submission_time"] >= ev_date - pd.Timedelta(days=30)) &
                    (df["submission_time"] < ev_date)]

        # Select reviews in the 30 days immediately AFTER the event
        after = df[(df["submission_time"] >= ev_date) &
                   (df["submission_time"] < ev_date + pd.Timedelta(days=30))]

        # Skip events with too few reviews in either window (not enough to be meaningful)
        if len(before) < 30 or len(after) < 30:
            continue

        # Build a summary row for this event
        rows.append({
            "event_date": event["date"],
            "event_label": event["label"],
            "n_before": len(before),                                    # Number of reviews before event
            "n_after": len(after),                                      # Number of reviews after event
            "mean_sent_before": before["sentiment_numeric"].mean(),     # Mean sentiment before
            "mean_sent_after": after["sentiment_numeric"].mean(),       # Mean sentiment after
            "delta": after["sentiment_numeric"].mean() - before["sentiment_numeric"].mean(),  # Change
            "pct_negative_before": (before["bert_pred"] == "negative").mean() * 100,  # % negative before
            "pct_negative_after": (after["bert_pred"] == "negative").mean() * 100,    # % negative after
        })

    # Convert the list of dictionaries to a DataFrame and return it
    return pd.DataFrame(rows)


# --- Main -------------------------------------------------------------------

def main():
    # Check the input file exists before proceeding
    if not SENTIMENT_FILE.exists():
        print(f"Missing {SENTIMENT_FILE}. Run apply_bert_full.py first.")
        return

    # Load the full-dataset BERT sentiment data
    print(f"Loading {SENTIMENT_FILE}...")
    df = load_sentiment()
    print(f"Total reviews: {len(df):,}")

    # Aggregate by month
    monthly = aggregate_monthly(df)

    # Save the monthly summary to CSV for use in the report
    monthly.to_csv(MONTHLY_OUT, index=False)
    print(f"[OK] Monthly sentiment saved: {MONTHLY_OUT} ({len(monthly)} months)")

    # --- Generate Figures ---
    print("\nGenerating figures...")

    # Two-panel time-series figure with event annotations
    plot_sentiment_with_events(monthly, EVENTS,
                               FIGS_DIR / "fig13_sentiment_with_events.png")

    # Stacked bar of negative review volume by topic over time
    plot_topic_volume_with_events(EVENTS,
                                  FIGS_DIR / "fig14_topic_volume_with_events.png")

    # Heatmap of sentiment by category and quarter
    plot_category_sentiment_heatmap(df,
                                    FIGS_DIR / "fig15_category_sentiment_heatmap.png")

    # --- Compute Event Impact Table ---
    impact = compute_event_impact(df, EVENTS)

    # Save the event impact statistics to CSV
    impact.to_csv(EVENT_IMPACT_OUT, index=False)

    # Print the event impact table to the terminal
    print("\n" + "=" * 70)
    print("EVENT IMPACT: 30-day window before vs. after each event")
    print("=" * 70)
    print("(Positive delta = sentiment IMPROVED; negative = WORSENED)")
    print("(Associational only, not causal)\n")

    # Print one row per event with before/after comparison
    for _, r in impact.iterrows():
        print(f"{r['event_date']} {r['event_label']}")
        print(f"  n_before = {r['n_before']:4d}  n_after = {r['n_after']:4d}")
        print(f"  Mean sentiment: {r['mean_sent_before']:+.3f} -> {r['mean_sent_after']:+.3f}"
              f"  (Δ = {r['delta']:+.3f})")
        print(f"  % negative: {r['pct_negative_before']:5.1f}% -> "
              f"{r['pct_negative_after']:5.1f}%")
        print()

    # Confirm outputs were saved
    print(f"[OK] Event impact saved: {EVENT_IMPACT_OUT}")
    print(f"[OK] All figures saved to {FIGS_DIR}/")


# Only run main() when this script is executed directly
if __name__ == "__main__":
    main()
