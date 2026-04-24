"""
Stage 4b: Apply validated BERT sentiment model to the FULL cleaned dataset.

After establishing in Stage 4 that the RoBERTa model performs reliably
(76.8% accuracy, strong agreement with ground truth at the monthly
aggregate level), we now apply it to all 64,916 reviews to enable
rich downstream analysis (temporal trends, category-level sentiment,
topic modelling on negative reviews, etc.).

Output: data/reviews_full_sentiment.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "reviews_clean.csv"
OUTPUT_FILE = DATA_DIR / "reviews_full_sentiment.csv"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKEN_LENGTH = 256
BATCH_SIZE = 32  # bigger batches are fine on the full dataset

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_model():
    print(f"Loading model from local cache: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    print(f"Device: {device}")
    return tokenizer, model, device


def predict_all(texts, tokenizer, model, device):
    all_preds = []
    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT inference"):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=MAX_TOKEN_LENGTH,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits.cpu().numpy()
        probs = softmax(scores, axis=1)
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
    return all_preds, all_probs


def main():
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run clean_reviews.py first.")
        return
    
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["full_text"]).reset_index(drop=True)
    print(f"Total reviews to classify: {len(df):,}")
    
    # Estimated runtime
    est_mins = len(df) / BATCH_SIZE / 5 / 60  # ~5 batches/sec on MPS
    print(f"Estimated runtime: ~{est_mins:.0f} minutes")
    
    tokenizer, model, device = load_model()
    
    texts = df["full_text"].astype(str).tolist()
    preds, probs = predict_all(texts, tokenizer, model, device)
    
    # Add to DataFrame
    probs_arr = np.array(probs)
    df["bert_pred_idx"] = preds
    df["bert_pred"] = [LABEL_MAP[p] for p in preds]
    df["bert_prob_negative"] = probs_arr[:, 0]
    df["bert_prob_neutral"] = probs_arr[:, 1]
    df["bert_prob_positive"] = probs_arr[:, 2]
    df["bert_confidence"] = probs_arr.max(axis=1)
    # Signed score for temporal plots
    df["bert_score"] = probs_arr[:, 2] - probs_arr[:, 0]  # pos_prob - neg_prob
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")
    
    print("\nBERT prediction distribution on full dataset:")
    print(df["bert_pred"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))
    print(f"\nMean BERT confidence: {df['bert_confidence'].mean():.3f}")
    print(f"High-confidence negative reviews (>0.80): {(df[df['bert_pred']=='negative']['bert_confidence'] > 0.80).sum():,}")


if __name__ == "__main__":
    main()