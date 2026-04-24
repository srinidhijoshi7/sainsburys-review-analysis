"""
Stage 5: LDA Topic Modelling on BERT-identified negative reviews.

Applies Latent Dirichlet Allocation (Blei, Ng & Jordan 2003) to the
5,465 high-confidence negative reviews identified by RoBERTa in Stage 4.
Two-stage pipeline (BERT filter -> LDA) isolates the negative signal,
giving cleaner topics than running LDA on the full corpus.

Model selection: coherence score (c_v) across K = 5, 7, 10, 12 topics
to choose the best number of topics rather than picking arbitrarily.

Outputs:
  - data/negative_reviews_with_topics.csv
  - figures/fig09_lda_coherence.png
  - figures/fig10_topic_prevalence_over_time.png
  - figures/fig11_topic_by_category.png
  - figures/fig12_top_words_per_topic.png
  - figures/lda_interactive.html  (pyLDAvis - open in browser)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models

# --- Config ----------------------------------------------------------------

DATA_DIR = Path("data")
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "reviews_full_sentiment.csv"
OUTPUT_FILE = DATA_DIR / "negative_reviews_with_topics.csv"

NEGATIVE_CONFIDENCE_THRESHOLD = 0.80
K_VALUES_TO_TRY = [5, 7, 10, 12]
RANDOM_SEED = 42
NUM_PASSES = 10  # LDA training passes
TOP_N_WORDS = 15

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


# --- NLTK setup ------------------------------------------------------------

def ensure_nltk_resources():
    """Download NLTK data if not already present."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


# --- Preprocessing ---------------------------------------------------------

# Extended stopword list: English + common review-noise words that appear
# in nearly every review regardless of topic (would hurt topic coherence)
REVIEW_STOPWORDS = {
    # Company / platform noise
    "sainsbury", "sainsburys", "sainsbury's",
    # Review structure noise
    "product", "item", "bought", "buy", "buying", "purchase", "purchased",
    "would", "could", "should", "get", "got", "use", "used", "using",
    "one", "two", "say", "said", "really", "quite", "also",
    "like", "good", "bad", "nice", "great", "lovely", "amazing",
    "taste", "flavour",  # too generic for food reviews - we want more specific
    # Generic negation that won't differentiate topics
    "not", "no", "nothing", "none",
    # Review meta-language
    "review", "reviewing", "star", "rating",
    # Time
    "day", "week", "month", "year", "today", "yesterday",
}


def preprocess_text(text, stopword_set, lemmatizer):
    """
    Lowercase, tokenise, remove punctuation/digits/stopwords, lemmatise,
    return list of tokens.
    """
    if pd.isna(text):
        return []
    # Lowercase
    text = str(text).lower()
    # Remove URLs (precaution)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove punctuation and digits - keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Tokenise
    tokens = word_tokenize(text)
    # Filter: length > 2, not stopword, lemmatise
    cleaned = []
    for tok in tokens:
        if len(tok) <= 2:
            continue
        if tok in stopword_set:
            continue
        lem = lemmatizer.lemmatize(tok)
        if lem in stopword_set:
            continue
        if len(lem) <= 2:
            continue
        cleaned.append(lem)
    return cleaned


# --- LDA model selection ---------------------------------------------------

def compute_coherence(corpus, dictionary, texts, k, seed=RANDOM_SEED):
    """Train an LDA model with k topics, return (model, coherence_score)."""
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=seed,
        passes=NUM_PASSES,
        alpha="auto",
        eta="auto",
        iterations=100,
    )
    coherence = CoherenceModel(
        model=lda, texts=texts, dictionary=dictionary, coherence="c_v"
    ).get_coherence()
    return lda, coherence


def plot_coherence(k_values, coherence_scores, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, coherence_scores, marker="o", linewidth=2,
            markersize=10, color="#F06C00")
    best_k = k_values[int(np.argmax(coherence_scores))]
    ax.axvline(best_k, color="#2E7D32", linestyle="--",
               label=f"Best K = {best_k}")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Coherence score (c_v)")
    ax.set_title("Figure 9. LDA model selection via topic coherence")
    ax.legend()
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


# --- Topic analysis figures ------------------------------------------------

def extract_topic_top_words(lda, n_words=TOP_N_WORDS):
    """Return dict: topic_id -> list of (word, weight) tuples."""
    topics = {}
    for i in range(lda.num_topics):
        topics[i] = lda.show_topic(i, topn=n_words)
    return topics


def assign_topic_to_doc(lda, bow):
    """Return (dominant_topic_id, probability) for a single document."""
    topic_probs = lda.get_document_topics(bow, minimum_probability=0)
    topic_probs_sorted = sorted(topic_probs, key=lambda x: x[1], reverse=True)
    return topic_probs_sorted[0]


def plot_top_words_per_topic(topics, topic_labels, out_path):
    """Grid of horizontal bar charts - top words per topic."""
    n = len(topics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i in range(n):
        words_weights = topics[i]
        words = [w for w, _ in words_weights][:10]
        weights = [w for _, w in words_weights][:10]
        ax = axes[i]
        ax.barh(range(len(words))[::-1], weights, color="#F06C00")
        ax.set_yticks(range(len(words))[::-1])
        ax.set_yticklabels(words, fontsize=10)
        label = topic_labels.get(i, f"Topic {i}")
        ax.set_title(f"Topic {i}: {label}", fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
    
    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Figure 12. Top 10 words per LDA topic "
                 "(negative reviews only)", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_topic_over_time(df, topic_labels, out_path):
    df = df.copy()
    df["submission_time"] = pd.to_datetime(df["submission_time"], utc=True)
    df["quarter"] = df["submission_time"].dt.to_period("Q").dt.to_timestamp()
    
    # Count topic occurrences per quarter
    topic_counts = (df.groupby(["quarter", "dominant_topic"])
                      .size().unstack(fill_value=0))
    topic_pct = topic_counts.div(topic_counts.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = sns.color_palette("tab10", n_colors=len(topic_pct.columns))
    for i, col in enumerate(topic_pct.columns):
        label = topic_labels.get(col, f"Topic {col}")
        ax.plot(topic_pct.index, topic_pct[col], marker="o",
                label=f"{col}: {label}", color=colors[i], linewidth=2)
    
    ax.set_xlabel("Quarter")
    ax.set_ylabel("% of negative reviews with this dominant topic")
    ax.set_title("Figure 10. Topic prevalence among negative reviews over time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.tick_params(axis="x", rotation=45)
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_topic_by_category(df, topic_labels, out_path):
    ct = pd.crosstab(df["category_keyword"], df["dominant_topic"],
                     normalize="index") * 100
    # Rename columns to include labels
    ct.columns = [f"T{c}: {topic_labels.get(c, '')}" for c in ct.columns]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="Oranges",
                cbar_kws={"label": "% of negative reviews in category"},
                ax=ax, annot_kws={"size": 9})
    ax.set_xlabel("Topic")
    ax.set_ylabel("Product category")
    ax.set_title("Figure 11. Distribution of complaint topics by category\n"
                 "(row-normalised: each row sums to 100%)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def generate_interactive_lda(lda, corpus, dictionary, out_path):
    """Generate pyLDAvis interactive visualisation (saved as HTML)."""
    try:
        vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(vis, str(out_path))
        print(f"  [interactive] {out_path}")
    except Exception as e:
        print(f"  [WARN] pyLDAvis failed: {e}")


# --- Main -------------------------------------------------------------------

def main():
    ensure_nltk_resources()
    
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run apply_bert_full.py first.")
        return
    
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter to high-confidence negative reviews only
    neg = df[(df["bert_pred"] == "negative") &
             (df["bert_confidence"] >= NEGATIVE_CONFIDENCE_THRESHOLD)].copy()
    print(f"High-confidence negative reviews: {len(neg):,}")
    
    # Preprocess
    print("\nPreprocessing text (lemmatisation, stopword removal)...")
    stopword_set = set(stopwords.words("english")) | REVIEW_STOPWORDS
    lemmatizer = WordNetLemmatizer()
    
    from tqdm import tqdm
    tqdm.pandas(desc="Tokenising")
    neg["tokens"] = neg["full_text"].progress_apply(
        lambda t: preprocess_text(t, stopword_set, lemmatizer)
    )
    # Drop reviews with too few tokens (not informative for LDA)
    neg = neg[neg["tokens"].str.len() >= 5].reset_index(drop=True)
    print(f"After filtering short docs: {len(neg):,}")
    
    texts = neg["tokens"].tolist()
    
    # Build dictionary and corpus
    print("\nBuilding dictionary and corpus...")
    dictionary = Dictionary(texts)
    # Remove tokens that appear in fewer than 10 docs or more than 50% of docs
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(f"Dictionary size: {len(dictionary):,} tokens")
    print(f"Corpus size: {len(corpus):,} documents")
    
    # --- Model selection via coherence ---
    print("\nFitting LDA models for K in", K_VALUES_TO_TRY)
    coherences = []
    models = {}
    for k in K_VALUES_TO_TRY:
        print(f"  Fitting K={k}...")
        lda, coh = compute_coherence(corpus, dictionary, texts, k)
        print(f"    Coherence (c_v): {coh:.4f}")
        coherences.append(coh)
        models[k] = lda
    
    best_k = K_VALUES_TO_TRY[int(np.argmax(coherences))]
    best_lda = models[best_k]
    print(f"\nBest K by coherence: {best_k} (c_v = {max(coherences):.4f})")
    
    plot_coherence(K_VALUES_TO_TRY, coherences, FIGS_DIR / "fig09_lda_coherence.png")
    
    # --- Print topics ---
    print("\n" + "=" * 60)
    print(f"TOP {TOP_N_WORDS} WORDS PER TOPIC (K = {best_k})")
    print("=" * 60)
    topics = extract_topic_top_words(best_lda, n_words=TOP_N_WORDS)
    for topic_id, words_weights in topics.items():
        words = ", ".join([f"{w}({wt:.3f})" for w, wt in words_weights[:10]])
        print(f"\nTopic {topic_id}:")
        print(f"  {words}")
    
    # --- Assign topics to documents ---
    print("\nAssigning dominant topic to each review...")
    dominant_topics = []
    topic_probs_list = []
    for bow in tqdm(corpus, desc="Assigning"):
        top_id, top_prob = assign_topic_to_doc(best_lda, bow)
        dominant_topics.append(top_id)
        topic_probs_list.append(top_prob)
    neg["dominant_topic"] = dominant_topics
    neg["topic_probability"] = topic_probs_list
    
    # --- Topic labels derived from top words + exemplar review inspection ---
    # Labels were assigned after reviewing the top 15 words and the 3
    # highest-probability exemplar reviews per topic. Each label captures
    # the dominant business-meaningful complaint theme in the topic.
    topic_labels = {
        0: "Meat/Protein Quality Failures",
        1: "Reformulation / Recipe Changes",
        2: "Taste & Texture Disappointment",
        3: "Price-Value & Packaging",
        4: "Prepared Food & Premium Disappointment",
    }
    
    # --- Figures ---
    print("\nGenerating topic figures...")
    plot_top_words_per_topic(topics, topic_labels, FIGS_DIR / "fig12_top_words_per_topic.png")
    plot_topic_over_time(neg, topic_labels, FIGS_DIR / "fig10_topic_prevalence_over_time.png")
    plot_topic_by_category(neg, topic_labels, FIGS_DIR / "fig11_topic_by_category.png")
    generate_interactive_lda(best_lda, corpus, dictionary, FIGS_DIR / "lda_interactive.html")
    
    # --- Save ---
    out_cols = ["review_id", "product_id", "product_name", "category_keyword",
                "rating", "title", "text", "full_text",
                "submission_time", "bert_pred", "bert_confidence",
                "dominant_topic", "topic_probability"]
    out_cols = [c for c in out_cols if c in neg.columns]
    neg[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")
    
    # --- Sample exemplar reviews per topic ---
    print("\n" + "=" * 60)
    print("EXEMPLAR REVIEWS PER TOPIC (highest topic probability)")
    print("=" * 60)
    for t in range(best_k):
        subset = neg[neg["dominant_topic"] == t].nlargest(3, "topic_probability")
        print(f"\nTopic {t} ({len(neg[neg['dominant_topic'] == t])} reviews total):")
        for _, row in subset.iterrows():
            text_snip = str(row["full_text"])[:180]
            print(f"  [{row['category_keyword']}, {row['rating']}*] {text_snip}...")


if __name__ == "__main__":
    main()