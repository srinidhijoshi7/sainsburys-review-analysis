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
- Stars 3 => neutral
- Stars 4-5 => positive

Outputs:
- data/reviews_with_sentiment.csv (sample with BERT predictions)
- figures/fig06_confusion_matrix.png
- figures/fig07_bert_vs_rating_heatmap.png
- figures/fig08_sentiment_over_time.png
- Printed classification report (precision/recall/F1 per class)
"""

# Import pandas for loading and manipulating the reviews DataFrame
import pandas as pd

# Import numpy for array operations (e.g. argmax over probability arrays)
import numpy as np

# Import matplotlib for creating and saving figures
import matplotlib.pyplot as plt

# Import seaborn for styled heatmaps and line plots
import seaborn as sns

# Import Path for file path operations
from pathlib import Path

# Import tqdm to show progress bars during batch inference
from tqdm import tqdm

# Import the Hugging Face tokenizer and model classes for loading RoBERTa
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import softmax to convert raw logits into probability distributions
from scipy.special import softmax

# Import PyTorch (needed to run the neural network model)
import torch

# Import sklearn metrics for evaluating model accuracy against ground truth
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# --- Config ----------------------------------------------------------------

# Folder containing data files
DATA_DIR = Path("data")

# Folder where figures will be saved
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)  # Create the folder if it doesn't already exist

# Input: the cleaned reviews file produced by clean_reviews.py
INPUT_FILE = DATA_DIR / "reviews_clean.csv"

# Output: the sampled reviews with BERT predictions attached
OUTPUT_FILE = DATA_DIR / "reviews_with_sentiment.csv"

# Hugging Face model identifier for the Cardiff NLP RoBERTa sentiment model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Number of reviews to sample per star rating (1–5), giving 5,000 total
SAMPLES_PER_STAR = 1000

# Maximum number of tokens to pass to the model (truncate longer reviews)
MAX_TOKEN_LENGTH = 256

# How many reviews to process in a single forward pass through the model
BATCH_SIZE = 16

# Random seed for reproducibility of the stratified sample
RANDOM_SEED = 42

# Maps the model's output class indices to human-readable sentiment labels
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Apply consistent professional styling to all figures
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# --- Sampling --------------------------------------------------------------

def stratified_sample(df, n_per_group, group_col="rating", seed=RANDOM_SEED):
    """Take n_per_group rows per rating class (or all if fewer available)."""

    # List to hold the sampled subset for each star rating
    parts = []

    # Iterate over each unique star rating (1, 2, 3, 4, 5)
    for star in sorted(df[group_col].dropna().unique()):
        # Get all reviews with this star rating
        subset = df[df[group_col] == star]

        # Take n_per_group reviews, or all of them if fewer are available
        take = min(len(subset), n_per_group)

        # Randomly sample the specified number of rows from this group
        parts.append(subset.sample(n=take, random_state=seed))

    # Concatenate all groups, then shuffle the combined DataFrame randomly
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# --- Model loading ---------------------------------------------------------

def load_model(model_name):
    """Load tokenizer + model. First call downloads ~500MB."""

    # Tell the user what model is being loaded
    print(f"Loading model: {model_name}")
    print("(first run downloads ~500MB - subsequent runs use local cache)")

    # Load the tokenizer — this converts raw text into token IDs the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the pre-trained classification model (RoBERTa fine-tuned for sentiment)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Set the model to evaluation mode (disables dropout — we're not training)
    model.eval()

    # Use Apple Silicon GPU (MPS) if available, otherwise use CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Move the model's weights to the selected device
    model = model.to(device)

    print(f"Model loaded on device: {device}")

    # Return the tokenizer, model, and the device they're running on
    return tokenizer, model, device


# --- Inference -------------------------------------------------------------

def predict_batch(texts, tokenizer, model, device):
    """Run sentiment prediction on a batch of texts."""

    # Tokenize a list of texts:
    # - return PyTorch tensors ("pt")
    # - truncate texts longer than MAX_TOKEN_LENGTH tokens
    # - pad shorter texts so all in the batch are the same length
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LENGTH,
    ).to(device)  # Move tokenised input tensors to the same device as the model

    # Disable gradient computation — we're just doing inference, not training
    with torch.no_grad():
        # Pass the tokenized inputs through the model to get raw output logits
        outputs = model(**inputs)

    # Move the logits back to CPU and convert to a numpy array
    scores = outputs.logits.cpu().numpy()

    # Apply softmax to convert logits to probabilities (sum to 1 across classes)
    probs = softmax(scores, axis=1)

    # Take the class index with the highest probability as the predicted label
    preds = np.argmax(probs, axis=1)

    # Return the predicted class indices and the full probability arrays
    return preds, probs


def predict_all(df, tokenizer, model, device, text_col="full_text"):
    """Run model over all reviews in df, batched."""

    # Lists to collect predictions and probabilities for every review
    all_preds = []
    all_probs = []

    # Convert the text column to a plain Python list of strings
    texts = df[text_col].astype(str).tolist()

    # Process the texts in batches, showing a progress bar
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT inference"):
        # Slice out one batch of texts
        batch = texts[i:i + BATCH_SIZE]

        # Run inference on this batch
        preds, probs = predict_batch(batch, tokenizer, model, device)

        # Accumulate results
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    # Work on a copy of the DataFrame so we don't modify the original
    df = df.copy()

    # Store the raw predicted class index (0, 1, or 2)
    df["bert_pred_idx"] = all_preds

    # Map predicted index to a human-readable label ("negative", "neutral", "positive")
    df["bert_pred"] = [LABEL_MAP[p] for p in all_preds]

    # Convert the list of probability arrays to a 2D numpy array for easier column extraction
    probs_arr = np.array(all_probs)

    # Store the probability for each sentiment class as separate columns
    df["bert_prob_negative"] = probs_arr[:, 0]   # Probability of being negative
    df["bert_prob_neutral"]  = probs_arr[:, 1]   # Probability of being neutral
    df["bert_prob_positive"] = probs_arr[:, 2]   # Probability of being positive

    # Store the maximum probability (confidence in the predicted class)
    df["bert_confidence"] = probs_arr.max(axis=1)

    # Return the DataFrame with all new BERT prediction columns added
    return df


# --- Evaluation figures ----------------------------------------------------

def plot_confusion_matrix(df, out_path):
    """BERT prediction vs. star-rating ground truth."""

    # Define the class order for both axes of the confusion matrix
    labels = ["negative", "neutral", "positive"]

    # Ground truth labels (from star ratings)
    y_true = df["rating_sentiment"]

    # Model's predicted labels
    y_pred = df["bert_pred"]

    # Compute the confusion matrix — rows = true class, cols = predicted class
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert raw counts to percentages of each ground-truth row (how often BERT got each class right)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw a heatmap: cell colour = percentage, cell annotation = raw count
    sns.heatmap(cm_pct, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "% of ground-truth row"},
                ax=ax, annot_kws={"size": 14})

    # Axis labels
    ax.set_xlabel("BERT prediction")
    ax.set_ylabel("Ground truth (from star rating)")

    # Calculate accuracy and macro F1 to display in the title
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    ax.set_title(f"Figure 6. Confusion matrix: BERT vs. star-rating ground truth\n"
                 f"Accuracy = {acc:.2%}, Macro-F1 = {f1:.3f}")

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_bert_vs_rating_heatmap(df, out_path):
    """Full 5x3 heatmap: star rating vs BERT prediction (more granular than CM)."""

    # Create a cross-tabulation: rows = star ratings (1–5), cols = BERT predictions
    # normalize="index" converts each row to a percentage (each row sums to 100%)
    ct = pd.crosstab(df["rating"], df["bert_pred"],
                     normalize="index") * 100

    # Ensure columns appear in a consistent order
    col_order = [c for c in ["negative", "neutral", "positive"] if c in ct.columns]
    ct = ct[col_order]

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the heatmap with one decimal place in annotations
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues",
                cbar_kws={"label": "% of reviews at this star rating"},
                ax=ax, annot_kws={"size": 14})

    # Axis labels and title
    ax.set_xlabel("BERT prediction")
    ax.set_ylabel("Star rating")
    ax.set_title("Figure 7. Distribution of BERT predictions by star rating")

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_sentiment_over_time(df_all, out_path):
    """
    Plot rolling average of BERT-predicted sentiment score for the sampled
    reviews over time, compared to the star-rating ground truth.
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = df_all.copy()

    # Parse submission_time to UTC-aware datetime
    df["submission_time"] = pd.to_datetime(df["submission_time"], utc=True)

    # Map BERT label to a numeric score: negative=-1, neutral=0, positive=+1
    score_map = {"negative": -1, "neutral": 0, "positive": 1}
    df["bert_score"] = df["bert_pred"].map(score_map)

    # Map ground-truth sentiment to the same numeric scale
    df["truth_score"] = df["rating_sentiment"].map(score_map)

    # Create a year-month column for monthly aggregation (as a datetime for the x-axis)
    df["year_month"] = df["submission_time"].dt.to_period("M").dt.to_timestamp()

    # Aggregate per month: mean sentiment scores and review count
    monthly = df.groupby("year_month").agg(
        bert_mean=("bert_score", "mean"),    # Average BERT sentiment score this month
        truth_mean=("truth_score", "mean"),  # Average ground-truth sentiment score this month
        n=("bert_score", "count")            # Number of reviews this month
    ).reset_index()

    # Drop months with fewer than 20 reviews (too few to be statistically meaningful)
    monthly = monthly[monthly["n"] >= 20]

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot BERT's monthly mean sentiment as an orange line with circular markers
    ax.plot(monthly["year_month"], monthly["bert_mean"], marker="o",
            label="BERT predicted sentiment", color="#F06C00", linewidth=2)

    # Plot ground-truth (star rating) monthly mean sentiment as a green dashed line
    ax.plot(monthly["year_month"], monthly["truth_mean"], marker="s",
            label="Star-rating ground truth", color="#2E7D32", linewidth=2, linestyle="--")

    # Draw a horizontal line at 0 (neutral boundary)
    ax.axhline(0, color="gray", linewidth=0.5)

    # Set the y-axis to run from -1 (most negative) to +1 (most positive)
    ax.set_ylim(-1, 1)

    # Axis labels and title
    ax.set_ylabel("Mean sentiment score (-1 = negative, +1 = positive)")
    ax.set_xlabel("Month")
    ax.set_title("Figure 8. Mean sentiment over time: BERT vs. ground truth\n"
                 "(sampled reviews only, months with n ≥ 20)")

    # Add a legend in the bottom-right corner
    ax.legend(loc="lower right")

    # Rotate x-axis labels 45 degrees so they don't overlap
    ax.tick_params(axis="x", rotation=45)

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


# --- Main -------------------------------------------------------------------

def main():
    # Check that the cleaned reviews file exists
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run clean_reviews.py first.")
        return

    # Load the cleaned reviews
    print(f"Loading cleaned reviews from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Drop rows that are missing a rating, full_text, or sentiment label (we need all three)
    df = df.dropna(subset=["rating", "full_text", "rating_sentiment"])

    # Ensure the rating column is stored as integer
    df["rating"] = df["rating"].astype(int)
    print(f"  Total cleaned reviews: {len(df):,}")

    # Draw a stratified sample: 1,000 reviews per star rating = ~5,000 total
    print(f"\nTaking stratified sample: {SAMPLES_PER_STAR} per star rating")
    sample = stratified_sample(df, SAMPLES_PER_STAR)
    print(f"Sample size: {len(sample):,}")
    print("Sample rating distribution:")
    print(sample["rating"].value_counts().sort_index())

    # Load the BERT model and run inference on the sample
    print("")
    tokenizer, model, device = load_model(MODEL_NAME)
    print("")
    sample = predict_all(sample, tokenizer, model, device)

    # Save the sample with BERT predictions to CSV
    sample.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved predictions to {OUTPUT_FILE}")

    # --- Evaluation: compare BERT predictions to star-rating ground truth ---
    print("\n" + "=" * 60)
    print("MODEL VALIDATION: BERT vs. star-rating ground truth")
    print("=" * 60)

    # Ground truth and predicted labels for the sample
    y_true = sample["rating_sentiment"]
    y_pred = sample["bert_pred"]

    # Print overall accuracy (percentage of exactly correct predictions)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

    # Print macro F1 (average F1 across all classes, treating each class equally)
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.3f}")

    # Print weighted F1 (accounts for class imbalance)
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.3f}")

    # Print a detailed classification report with precision, recall, and F1 per class
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

    # Generate and save all three evaluation figures
    print("\nGenerating evaluation figures...")
    plot_confusion_matrix(sample, FIGS_DIR / "fig06_confusion_matrix.png")
    plot_bert_vs_rating_heatmap(sample, FIGS_DIR / "fig07_bert_vs_rating_heatmap.png")
    plot_sentiment_over_time(sample, FIGS_DIR / "fig08_sentiment_over_time.png")

    print(f"\n[OK] All figures saved to {FIGS_DIR}/")


# Only run main() when executed directly
if __name__ == "__main__":
    main()
