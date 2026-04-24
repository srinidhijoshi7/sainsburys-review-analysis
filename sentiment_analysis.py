"""
Stage 4: Sentiment Analysis with BERT + Ground-Truth Validation.

Applies a transformer-based sentiment classifier (RoBERTa fine-tuned for
sentiment on Twitter) to a stratified sample of Sainsbury's product reviews,
then validates the model's predictions against the star-rating ground truth.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
  - RoBERTa-base architecture (125M parameters)
  - Fine-tuned on ~124M tweets for 3-class sentiment (negative/neutral/positive)
  - Handles short, informal user-generated text better than generic BERT

Ground truth construction:
  - Stars 1-2 => negative
  - Stars 3   => neutral
  - Stars 4-5 => positive

Outputs:
  - data/reviews_with_sentiment.csv (sample with BERT predictions)
  - figures/fig06_confusion_matrix.png
  - figures/fig07_bert_vs_rating_heatmap.png
  - figures/fig08_sentiment_over_time.png
  - Printed classification report (precision/recall/F1 per class)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# --- Config ----------------------------------------------------------------

DATA_DIR = Path("data")
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "reviews_clean.csv"
OUTPUT_FILE = DATA_DIR / "reviews_with_sentiment.csv"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SAMPLES_PER_STAR = 1000       # stratified: 1000 per star rating, total 5000
MAX_TOKEN_LENGTH = 256        # truncate longer reviews (few are longer)
BATCH_SIZE = 16               # batched inference for speed
RANDOM_SEED = 42

# Mapping from model's label indices to human-readable labels
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


# --- Sampling --------------------------------------------------------------

def stratified_sample(df, n_per_group, group_col="rating", seed=RANDOM_SEED):
    """Take n_per_group rows per rating class (or all if fewer available)."""
    parts = []
    for star in sorted(df[group_col].dropna().unique()):
        subset = df[df[group_col] == star]
        take = min(len(subset), n_per_group)
        parts.append(subset.sample(n=take, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# --- Model loading ---------------------------------------------------------

def load_model(model_name):
    """Load tokenizer + model. First call downloads ~500MB."""
    print(f"Loading model: {model_name}")
    print("(first run downloads ~500MB - subsequent runs use local cache)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # inference mode - disables dropout
    # Use MPS (Apple Silicon GPU) if available, else CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    return tokenizer, model, device


# --- Inference -------------------------------------------------------------

def predict_batch(texts, tokenizer, model, device):
    """Run sentiment prediction on a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LENGTH,
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    scores = outputs.logits.cpu().numpy()
    probs = softmax(scores, axis=1)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict_all(df, tokenizer, model, device, text_col="full_text"):
    """Run model over all reviews in df, batched."""
    all_preds = []
    all_probs = []
    
    texts = df[text_col].astype(str).tolist()
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT inference"):
        batch = texts[i:i + BATCH_SIZE]
        preds, probs = predict_batch(batch, tokenizer, model, device)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
    
    df = df.copy()
    df["bert_pred_idx"] = all_preds
    df["bert_pred"] = [LABEL_MAP[p] for p in all_preds]
    probs_arr = np.array(all_probs)
    df["bert_prob_negative"] = probs_arr[:, 0]
    df["bert_prob_neutral"] = probs_arr[:, 1]
    df["bert_prob_positive"] = probs_arr[:, 2]
    df["bert_confidence"] = probs_arr.max(axis=1)
    return df


# --- Evaluation figures ----------------------------------------------------

def plot_confusion_matrix(df, out_path):
    """BERT prediction vs. star-rating ground truth."""
    labels = ["negative", "neutral", "positive"]
    y_true = df["rating_sentiment"]
    y_pred = df["bert_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "% of ground-truth row"},
                ax=ax, annot_kws={"size": 14})
    ax.set_xlabel("BERT prediction")
    ax.set_ylabel("Ground truth (from star rating)")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    ax.set_title(f"Figure 6. Confusion matrix: BERT vs. star-rating ground truth\n"
                 f"Accuracy = {acc:.2%}, Macro-F1 = {f1:.3f}")
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_bert_vs_rating_heatmap(df, out_path):
    """Full 5x3 heatmap: star rating vs BERT prediction (more granular than CM)."""
    ct = pd.crosstab(df["rating"], df["bert_pred"],
                     normalize="index") * 100
    # Ensure column order
    col_order = [c for c in ["negative", "neutral", "positive"] if c in ct.columns]
    ct = ct[col_order]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues",
                cbar_kws={"label": "% of reviews at this star rating"},
                ax=ax, annot_kws={"size": 14})
    ax.set_xlabel("BERT prediction")
    ax.set_ylabel("Star rating")
    ax.set_title("Figure 7. Distribution of BERT predictions by star rating")
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_sentiment_over_time(df_all, out_path):
    """
    Apply BERT predictions to ALL reviews (via lookup from our sampled
    df), OR if we only have sample predictions, plot the sample's
    sentiment-vs-rating over time.
    
    For now: plot rolling average of BERT-predicted sentiment score
    for the sampled reviews over time.
    """
    df = df_all.copy()
    df["submission_time"] = pd.to_datetime(df["submission_time"], utc=True)
    # Map BERT prediction to numeric score: -1, 0, +1
    score_map = {"negative": -1, "neutral": 0, "positive": 1}
    df["bert_score"] = df["bert_pred"].map(score_map)
    # Similarly for ground truth
    df["truth_score"] = df["rating_sentiment"].map(score_map)
    # Monthly aggregation
    df["year_month"] = df["submission_time"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("year_month").agg(
        bert_mean=("bert_score", "mean"),
        truth_mean=("truth_score", "mean"),
        n=("bert_score", "count")
    ).reset_index()
    # Drop months with <20 reviews (noisy)
    monthly = monthly[monthly["n"] >= 20]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["year_month"], monthly["bert_mean"], marker="o",
            label="BERT predicted sentiment", color="#F06C00", linewidth=2)
    ax.plot(monthly["year_month"], monthly["truth_mean"], marker="s",
            label="Star-rating ground truth", color="#2E7D32", linewidth=2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Mean sentiment score (-1 = negative, +1 = positive)")
    ax.set_xlabel("Month")
    ax.set_title("Figure 8. Mean sentiment over time: BERT vs. ground truth\n"
                 "(sampled reviews only, months with n ≥ 20)")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=45)
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


# --- Main -------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run clean_reviews.py first.")
        return
    
    print(f"Loading cleaned reviews from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["rating", "full_text", "rating_sentiment"])
    df["rating"] = df["rating"].astype(int)
    print(f"  Total cleaned reviews: {len(df):,}")
    
    # Stratified sample
    print(f"\nTaking stratified sample: {SAMPLES_PER_STAR} per star rating")
    sample = stratified_sample(df, SAMPLES_PER_STAR)
    print(f"Sample size: {len(sample):,}")
    print("Sample rating distribution:")
    print(sample["rating"].value_counts().sort_index())
    
    # Load model + predict
    print("")
    tokenizer, model, device = load_model(MODEL_NAME)
    print("")
    sample = predict_all(sample, tokenizer, model, device)
    
    # Save
    sample.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved predictions to {OUTPUT_FILE}")
    
    # --- Evaluation ---
    print("\n" + "=" * 60)
    print("MODEL VALIDATION: BERT vs. star-rating ground truth")
    print("=" * 60)
    
    y_true = sample["rating_sentiment"]
    y_pred = sample["bert_pred"]
    
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro F1:  {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))
    
    # Figures
    print("\nGenerating evaluation figures...")
    plot_confusion_matrix(sample, FIGS_DIR / "fig06_confusion_matrix.png")
    plot_bert_vs_rating_heatmap(sample, FIGS_DIR / "fig07_bert_vs_rating_heatmap.png")
    plot_sentiment_over_time(sample, FIGS_DIR / "fig08_sentiment_over_time.png")
    
    print(f"\n[OK] All figures saved to {FIGS_DIR}/")


if __name__ == "__main__":
    main()