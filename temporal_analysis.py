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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# --- Config ----------------------------------------------------------------

DATA_DIR = Path("data")
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)

SENTIMENT_FILE = DATA_DIR / "reviews_full_sentiment.csv"
TOPICS_FILE = DATA_DIR / "negative_reviews_with_topics.csv"
MONTHLY_OUT = DATA_DIR / "monthly_sentiment.csv"
EVENT_IMPACT_OUT = DATA_DIR / "event_impact_stats.csv"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# --- Events ----------------------------------------------------------------
# All dates and events verified via ONS, House of Commons Library,
# Sainsbury's corporate press releases, and UK news outlets (see temporal
# analysis script header comment for source list).

EVENTS = [
    {"date": "2023-03-31", "label": "UK food inflation peaks (19.1%)",
     "type": "macro", "color": "#D32F2F"},
    {"date": "2023-04-11", "label": "Sainsbury's launches Nectar Prices",
     "type": "sainsburys", "color": "#F06C00"},
    {"date": "2023-08-24", "label": "Aldi Price Match expansion (400+)",
     "type": "sainsburys", "color": "#F06C00"},
    {"date": "2024-03-16", "label": "Sainsbury's IT outage",
     "type": "sainsburys", "color": "#D32F2F"},
    {"date": "2024-05-31", "label": "Food inflation normalises (1.7%)",
     "type": "macro", "color": "#2E7D32"},
    {"date": "2024-09-21", "label": "Nectar Prices hits 5,000 products",
     "type": "sainsburys", "color": "#F06C00"},
    {"date": "2024-11-05", "label": "Aldi Price Match in Local stores",
     "type": "sainsburys", "color": "#F06C00"},
]


# --- Helpers ---------------------------------------------------------------

def load_sentiment():
    df = pd.read_csv(SENTIMENT_FILE)
    df["submission_time"] = pd.to_datetime(df["submission_time"], utc=True)
    df["year_month"] = df["submission_time"].dt.to_period("M").dt.to_timestamp()
    return df


def aggregate_monthly(df):
    """Monthly mean sentiment + share of negative reviews."""
    score_map = {"negative": -1, "neutral": 0, "positive": 1}
    df = df.copy()
    df["sentiment_numeric"] = df["bert_pred"].map(score_map)
    
    monthly = df.groupby("year_month").agg(
        mean_sentiment=("sentiment_numeric", "mean"),
        pct_negative=("bert_pred",
                      lambda x: (x == "negative").sum() / len(x) * 100),
        review_count=("bert_pred", "count"),
        mean_rating=("rating", "mean"),
    ).reset_index()
    
    # Drop months with very few reviews (noisy)
    monthly = monthly[monthly["review_count"] >= 100].copy()
    return monthly


def plot_sentiment_with_events(monthly, events, out_path):
    """Two-panel plot: top = mean sentiment, bottom = % negative."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})
    
    # Panel 1: mean sentiment
    ax1.plot(monthly["year_month"], monthly["mean_sentiment"],
             marker="o", linewidth=2.5, color="#F06C00", markersize=6,
             label="Mean monthly BERT sentiment")
    ax1.axhline(0, color="gray", linewidth=0.7, alpha=0.6)
    ax1.set_ylabel("Mean sentiment\n(-1 negative, +1 positive)")
    ax1.set_title("Figure 13. Sainsbury's customer review sentiment over time,\n"
                  "annotated with UK retail and macroeconomic events",
                  fontsize=15)
    
    # Panel 2: % negative
    ax2.fill_between(monthly["year_month"], 0, monthly["pct_negative"],
                      color="#D32F2F", alpha=0.35)
    ax2.plot(monthly["year_month"], monthly["pct_negative"],
             marker="s", linewidth=2, color="#D32F2F", markersize=5,
             label="% negative reviews (BERT)")
    ax2.set_ylabel("% of reviews\npredicted negative")
    ax2.set_xlabel("Month")
    
    # Add events as vertical lines on BOTH panels
    for event in events:
        ev_date = pd.Timestamp(event["date"])
        # Only plot if event is within our data range
        if ev_date < monthly["year_month"].min() or ev_date > monthly["year_month"].max():
            continue
        for ax in (ax1, ax2):
            ax.axvline(ev_date, color=event["color"], linestyle="--",
                       alpha=0.7, linewidth=1.5)
        # Label on top axis, rotated, staggered vertically to avoid overlap
        y_positions = [0.95, 0.85, 0.75, 0.65]
        idx = events.index(event)
        y_pos = y_positions[idx % len(y_positions)]
        ax1.annotate(event["label"], xy=(ev_date, 1),
                     xycoords=("data", "axes fraction"),
                     xytext=(5, -15 - (idx % 4) * 18),
                     textcoords="offset points",
                     fontsize=8.5, color=event["color"],
                     rotation=0, ha="left", va="top",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="white", edgecolor=event["color"],
                               alpha=0.9))
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    ax1.legend(loc="lower right", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_topic_volume_with_events(events, out_path):
    """Stacked bar chart: monthly negative-review volume by topic, with events."""
    if not TOPICS_FILE.exists():
        print("  [skip] topics file not found")
        return
    
    topics = pd.read_csv(TOPICS_FILE)
    topics["submission_time"] = pd.to_datetime(topics["submission_time"], utc=True)
    topics["year_month"] = topics["submission_time"].dt.to_period("M").dt.to_timestamp()
    
    topic_labels = {
        0: "Meat/Protein Quality",
        1: "Reformulation/Recipe",
        2: "Taste & Texture",
        3: "Price-Value",
        4: "Prepared Food/Premium",
    }
    topics["topic_name"] = topics["dominant_topic"].map(topic_labels)
    
    # Monthly counts by topic
    counts = (topics.groupby(["year_month", "topic_name"])
                    .size().unstack(fill_value=0))
    
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    counts.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.9)
    
    # Replace numerical x-ticks with month labels, every 3rd
    ax.set_xticks(range(0, len(counts), 3))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in counts.index[::3]],
                       rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel("Negative review volume")
    ax.set_title("Figure 14. Volume of negative reviews by complaint topic over time")
    ax.legend(title="Topic", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_category_sentiment_heatmap(df, out_path):
    """Heatmap of mean sentiment by category × year-quarter."""
    df = df.copy()
    df["sentiment_numeric"] = df["bert_pred"].map({"negative": -1, "neutral": 0, "positive": 1})
    df["quarter"] = df["submission_time"].dt.to_period("Q").astype(str)
    
    pivot = df.pivot_table(index="category_keyword",
                           columns="quarter",
                           values="sentiment_numeric",
                           aggfunc="mean")
    # Keep categories with full coverage
    pivot = pivot.dropna(thresh=pivot.shape[1] * 0.6)
    # Sort by overall mean
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-0.5, vmax=1, ax=ax,
                cbar_kws={"label": "Mean sentiment (-1 neg to +1 pos)"},
                annot_kws={"size": 8})
    ax.set_title("Figure 15. Mean BERT sentiment by category and quarter\n"
                 "(lower values = worse customer sentiment)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Product category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def compute_event_impact(df, events):
    """
    For each event, compare mean sentiment in the 30 days BEFORE vs AFTER.
    Gives a rough quantitative view of 'did the event coincide with a shift?'.
    This is associational — we are NOT claiming causation in the report.
    """
    df = df.copy()
    df["sentiment_numeric"] = df["bert_pred"].map({"negative": -1, "neutral": 0, "positive": 1})
    
    rows = []
    for event in events:
        ev_date = pd.Timestamp(event["date"], tz="UTC")
        before = df[(df["submission_time"] >= ev_date - pd.Timedelta(days=30)) &
                    (df["submission_time"] < ev_date)]
        after = df[(df["submission_time"] >= ev_date) &
                   (df["submission_time"] < ev_date + pd.Timedelta(days=30))]
        if len(before) < 30 or len(after) < 30:
            continue
        rows.append({
            "event_date": event["date"],
            "event_label": event["label"],
            "n_before": len(before),
            "n_after": len(after),
            "mean_sent_before": before["sentiment_numeric"].mean(),
            "mean_sent_after": after["sentiment_numeric"].mean(),
            "delta": after["sentiment_numeric"].mean() - before["sentiment_numeric"].mean(),
            "pct_negative_before": (before["bert_pred"] == "negative").mean() * 100,
            "pct_negative_after": (after["bert_pred"] == "negative").mean() * 100,
        })
    return pd.DataFrame(rows)


# --- Main -------------------------------------------------------------------

def main():
    if not SENTIMENT_FILE.exists():
        print(f"Missing {SENTIMENT_FILE}. Run apply_bert_full.py first.")
        return
    
    print(f"Loading {SENTIMENT_FILE}...")
    df = load_sentiment()
    print(f"Total reviews: {len(df):,}")
    
    # Monthly aggregation
    monthly = aggregate_monthly(df)
    monthly.to_csv(MONTHLY_OUT, index=False)
    print(f"[OK] Monthly sentiment saved: {MONTHLY_OUT} ({len(monthly)} months)")
    
    # Figures
    print("\nGenerating figures...")
    plot_sentiment_with_events(monthly, EVENTS,
                                FIGS_DIR / "fig13_sentiment_with_events.png")
    plot_topic_volume_with_events(EVENTS,
                                   FIGS_DIR / "fig14_topic_volume_with_events.png")
    plot_category_sentiment_heatmap(df,
                                     FIGS_DIR / "fig15_category_sentiment_heatmap.png")
    
    # Event impact table
    impact = compute_event_impact(df, EVENTS)
    impact.to_csv(EVENT_IMPACT_OUT, index=False)
    
    print("\n" + "=" * 70)
    print("EVENT IMPACT: 30-day window before vs. after each event")
    print("=" * 70)
    print("(Positive delta = sentiment IMPROVED; negative = WORSENED)")
    print("(Associational only, not causal)\n")
    for _, r in impact.iterrows():
        print(f"{r['event_date']}  {r['event_label']}")
        print(f"  n_before = {r['n_before']:4d}   n_after = {r['n_after']:4d}")
        print(f"  Mean sentiment:  {r['mean_sent_before']:+.3f}  ->  {r['mean_sent_after']:+.3f}"
              f"   (Δ = {r['delta']:+.3f})")
        print(f"  % negative:      {r['pct_negative_before']:5.1f}%  -> "
              f"{r['pct_negative_after']:5.1f}%")
        print()
    
    print(f"[OK] Event impact saved: {EVENT_IMPACT_OUT}")
    print(f"[OK] All figures saved to {FIGS_DIR}/")


if __name__ == "__main__":
    main()