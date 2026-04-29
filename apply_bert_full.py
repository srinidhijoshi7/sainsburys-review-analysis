"""
Stage 4b: Apply validated BERT sentiment model to the FULL cleaned dataset.

After establishing in Stage 4 that the RoBERTa model performs reliably
(76.8% accuracy, strong agreement with ground truth at the monthly
aggregate level), we now apply it to all 64,916 reviews to enable
rich downstream analysis (temporal trends, category-level sentiment,
topic modelling on negative reviews, etc.).

Output: data/reviews_full_sentiment.csv
"""

# Import pandas for loading the input CSV and saving the output CSV
import pandas as pd

# Import numpy for array operations (argmax, softmax output handling)
import numpy as np

# Import Path for cross-platform file path handling
from pathlib import Path

# Import tqdm to display a progress bar during batch inference
from tqdm import tqdm

# Import the Hugging Face tokenizer and model classes for RoBERTa
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import softmax to convert model logits into probability distributions
from scipy.special import softmax

# Import PyTorch to run the neural network inference
import torch

# --- Config ----------------------------------------------------------------

# Folder containing all data files
DATA_DIR = Path("data")

# Input: cleaned reviews produced by clean_reviews.py
INPUT_FILE = DATA_DIR / "reviews_clean.csv"

# Output: same reviews with BERT sentiment predictions added
OUTPUT_FILE = DATA_DIR / "reviews_full_sentiment.csv"

# The Hugging Face model ID for Cardiff NLP's RoBERTa sentiment classifier
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Maximum token length passed to the model (truncate longer reviews)
MAX_TOKEN_LENGTH = 256

# Number of reviews processed per forward pass (larger = faster on GPU)
BATCH_SIZE = 32  # bigger batches are fine on the full dataset

# Maps the model's output class indices to human-readable sentiment labels
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_model():
    """Load the tokenizer and model from the local Hugging Face cache."""

    # Print which model we're loading (it will use the locally cached version after first run)
    print(f"Loading model from local cache: {MODEL_NAME}")

    # Load the tokenizer — converts text strings into integer token IDs
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the pre-trained classification model weights
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Switch model to evaluation mode (disables dropout for deterministic output)
    model.eval()

    # Use Apple Silicon GPU (MPS) if available, otherwise fall back to CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Move model parameters to the selected device
    model = model.to(device)

    print(f"Device: {device}")

    # Return the tokenizer, model, and device so they can be used in predict_all()
    return tokenizer, model, device


def predict_all(texts, tokenizer, model, device):
    """Run BERT sentiment inference over all texts in batches."""

    # Lists to accumulate predictions and probabilities across all batches
    all_preds = []
    all_probs = []

    # Iterate over the texts in chunks of BATCH_SIZE, with a progress bar
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT inference"):

        # Slice out the current batch of texts
        batch = texts[i:i + BATCH_SIZE]

        # Tokenize the batch:
        # - truncate to MAX_TOKEN_LENGTH tokens (cuts off very long reviews)
        # - pad shorter texts to the same length as the longest in this batch
        # - return PyTorch tensors
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=MAX_TOKEN_LENGTH,
        ).to(device)  # Move tokenised tensors to the same device as the model

        # Disable gradient computation for inference (saves memory, speeds up)
        with torch.no_grad():
            # Forward pass: get raw logit scores for each sentiment class
            outputs = model(**inputs)

        # Move logits to CPU and convert to numpy for further processing
        scores = outputs.logits.cpu().numpy()

        # Apply softmax row-wise to get probabilities for each class
        probs = softmax(scores, axis=1)

        # Take the class with the highest probability as the prediction
        preds = np.argmax(probs, axis=1)

        # Accumulate results from this batch
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    # Return the list of predicted class indices and the full probability arrays
    return all_preds, all_probs


def main():
    # Check that the cleaned reviews file exists before proceeding
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run clean_reviews.py first.")
        return

    # Load the cleaned reviews CSV into a DataFrame
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Drop any rows where full_text is missing (can't run inference without text)
    df = df.dropna(subset=["full_text"]).reset_index(drop=True)
    print(f"Total reviews to classify: {len(df):,}")

    # Estimate roughly how long inference will take (assuming ~5 batches/second on MPS)
    est_mins = len(df) / BATCH_SIZE / 5 / 60
    print(f"Estimated runtime: ~{est_mins:.0f} minutes")

    # Load the model, tokenizer, and device
    tokenizer, model, device = load_model()

    # Extract the full_text column as a list of plain strings (required by the tokenizer)
    texts = df["full_text"].astype(str).tolist()

    # Run inference on all reviews
    preds, probs = predict_all(texts, tokenizer, model, device)

    # Convert the list of probability lists to a 2D numpy array for easy column slicing
    probs_arr = np.array(probs)

    # Store the predicted class index (0=negative, 1=neutral, 2=positive)
    df["bert_pred_idx"] = preds

    # Map predicted index to a human-readable label using LABEL_MAP
    df["bert_pred"] = [LABEL_MAP[p] for p in preds]

    # Store the probability of each sentiment class as separate columns
    df["bert_prob_negative"] = probs_arr[:, 0]   # Probability of being negative
    df["bert_prob_neutral"]  = probs_arr[:, 1]   # Probability of being neutral
    df["bert_prob_positive"] = probs_arr[:, 2]   # Probability of being positive

    # Store the highest probability across the three classes (model confidence)
    df["bert_confidence"] = probs_arr.max(axis=1)

    # Create a signed sentiment score: positive_prob minus negative_prob
    # Ranges from -1 (purely negative) to +1 (purely positive), used for temporal plots
    df["bert_score"] = probs_arr[:, 2] - probs_arr[:, 0]

    # Save the enriched DataFrame to the output CSV (no row index column)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")

    # Print the distribution of predicted sentiment labels across the full dataset
    print("\nBERT prediction distribution on full dataset:")
    print(df["bert_pred"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

    # Print the mean confidence score (how certain the model was overall)
    print(f"\nMean BERT confidence: {df['bert_confidence'].mean():.3f}")

    # Count high-confidence negative predictions (useful for LDA topic modelling in Stage 5)
    print(f"High-confidence negative reviews (>0.80): "
          f"{(df[df['bert_pred']=='negative']['bert_confidence'] > 0.80).sum():,}")


# Only run main() when this script is executed directly
if __name__ == "__main__":
    main()
