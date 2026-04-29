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
- figures/lda_interactive.html (pyLDAvis - open in browser)
"""

# Import pandas for loading data and saving results
import pandas as pd

# Import numpy for numerical operations (argmax, array indexing)
import numpy as np

# Import matplotlib for creating and saving figures
import matplotlib.pyplot as plt

# Import seaborn for styled heatmaps
import seaborn as sns

# Import Path for file path operations
from pathlib import Path

# Import re for regular expressions (used in text preprocessing)
import re

# Suppress any runtime warnings that clutter the output
import warnings
warnings.filterwarnings("ignore")

# Import NLTK for natural language processing utilities
import nltk

# Import the English stopwords list (common words like "the", "is", "a")
from nltk.corpus import stopwords

# Import lemmatizer — reduces words to their base form (e.g. "running" -> "run")
from nltk.stem import WordNetLemmatizer

# Import word tokenizer — splits text into individual words
from nltk.tokenize import word_tokenize

# Import Gensim's Dictionary — maps words to integer IDs for LDA
from gensim.corpora import Dictionary

# Import LDA model and coherence evaluator from Gensim
from gensim.models import LdaModel, CoherenceModel

# Import pyLDAvis for generating an interactive browser-based topic visualisation
import pyLDAvis

# Adapter to use pyLDAvis with Gensim LDA models
import pyLDAvis.gensim_models

# --- Config ----------------------------------------------------------------

# Folder containing input data files
DATA_DIR = Path("data")

# Folder where output figures will be saved
FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)  # Create if it doesn't exist

# Input: full-dataset BERT predictions from apply_bert_full.py
INPUT_FILE = DATA_DIR / "reviews_full_sentiment.csv"

# Output: negative reviews with their assigned LDA topics
OUTPUT_FILE = DATA_DIR / "negative_reviews_with_topics.csv"

# Only include reviews where BERT is at least this confident the review is negative
NEGATIVE_CONFIDENCE_THRESHOLD = 0.80

# Topic counts to try during model selection
K_VALUES_TO_TRY = [5, 7, 10, 12]

# Random seed for reproducible LDA results
RANDOM_SEED = 42

# Number of training passes through the corpus for LDA
NUM_PASSES = 10

# How many top words to extract and display per topic
TOP_N_WORDS = 15

# Apply consistent styling to all figures
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# --- NLTK setup ------------------------------------------------------------

def ensure_nltk_resources():
    """Download NLTK data if not already present."""

    # List of NLTK data packages we need, as (lookup_path, download_name) pairs
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),   # Newer punctuation-aware tokenizer
        ("tokenizers/punkt", "punkt"),            # Standard tokenizer (fallback)
        ("corpora/stopwords", "stopwords"),       # English stopword list
        ("corpora/wordnet", "wordnet"),           # WordNet lexical database for lemmatization
    ]

    # For each required resource, check if it's already downloaded
    for path, name in resources:
        try:
            nltk.data.find(path)   # Raises LookupError if not found
        except LookupError:
            print(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)  # Download silently in the background


# --- Preprocessing ---------------------------------------------------------

# Domain-specific stopwords to remove on top of the standard English list.
# These are words so common in product reviews that they'd appear in every topic
# and reduce topic coherence (they don't help distinguish topics from each other).
REVIEW_STOPWORDS = {
    # Brand-specific noise (appears in almost every review)
    "sainsbury", "sainsburys", "sainsbury's",
    # Generic purchase/usage language that doesn't signal a specific complaint
    "product", "item", "bought", "buy", "buying", "purchase", "purchased",
    "would", "could", "should", "get", "got", "use", "used", "using",
    "one", "two", "say", "said", "really", "quite", "also",
    # Generic quality adjectives — too vague to separate complaint themes
    "like", "good", "bad", "nice", "great", "lovely", "amazing",
    # Highly generic food words that appear across all food categories
    "taste", "flavour",
    # Generic negation words — too ubiquitous in negative reviews to be informative
    "not", "no", "nothing", "none",
    # Review meta-language that's not about the product itself
    "review", "reviewing", "star", "rating",
    # Time words — don't signal a product complaint
    "day", "week", "month", "year", "today", "yesterday",
}


def preprocess_text(text, stopword_set, lemmatizer):
    """
    Lowercase, tokenise, remove punctuation/digits/stopwords, lemmatise,
    return list of tokens suitable for LDA.
    """

    # Return an empty list if the input is NaN/None
    if pd.isna(text):
        return []

    # Convert to lowercase so "BAD" and "bad" are treated as the same word
    text = str(text).lower()

    # Remove URLs (unlikely in reviews, but a precaution)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Keep only alphabetic characters and spaces (remove punctuation, digits, symbols)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize: split the cleaned text string into a list of individual words
    tokens = word_tokenize(text)

    # Filter and lemmatize each token
    cleaned = []
    for tok in tokens:
        # Skip tokens that are too short (single letters or two-letter words like "it")
        if len(tok) <= 2:
            continue

        # Skip if the token is in our combined stopword set
        if tok in stopword_set:
            continue

        # Lemmatize: convert the word to its base/root form
        lem = lemmatizer.lemmatize(tok)

        # Check again if the lemmatized form is a stopword (e.g. "running" -> "run" might be filtered)
        if lem in stopword_set:
            continue

        # Skip if the lemmatized form is also very short
        if len(lem) <= 2:
            continue

        # Add the cleaned, lemmatized token to the output list
        cleaned.append(lem)

    # Return the processed list of tokens for this review
    return cleaned


# --- LDA model selection ---------------------------------------------------

def compute_coherence(corpus, dictionary, texts, k, seed=RANDOM_SEED):
    """Train an LDA model with k topics, return (model, coherence_score)."""

    # Train the LDA model with the specified number of topics
    lda = LdaModel(
        corpus=corpus,          # Bag-of-words representation of the corpus
        id2word=dictionary,     # Mapping from word IDs back to words
        num_topics=k,           # Number of topics to discover
        random_state=seed,      # Fixed seed for reproducibility
        passes=NUM_PASSES,      # Number of passes through the corpus during training
        alpha="auto",           # Automatically learn document-topic distribution prior
        eta="auto",             # Automatically learn topic-word distribution prior
        iterations=100,         # Maximum E-step iterations per document per pass
    )

    # Compute the c_v coherence score — measures how semantically similar
    # the top words of each topic are (higher = more coherent, interpretable topics)
    coherence = CoherenceModel(
        model=lda, texts=texts, dictionary=dictionary, coherence="c_v"
    ).get_coherence()

    # Return both the trained model and its coherence score
    return lda, coherence


def plot_coherence(k_values, coherence_scores, out_path):
    """Plot coherence score vs number of topics to visualise model selection."""

    # Create the figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot coherence score for each K value tried
    ax.plot(k_values, coherence_scores, marker="o", linewidth=2,
            markersize=10, color="#F06C00")

    # Find the K with the highest coherence score
    best_k = k_values[int(np.argmax(coherence_scores))]

    # Mark the best K with a vertical dashed green line
    ax.axvline(best_k, color="#2E7D32", linestyle="--",
               label=f"Best K = {best_k}")

    # Axis labels, title, and legend
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Coherence score (c_v)")
    ax.set_title("Figure 9. LDA model selection via topic coherence")
    ax.legend()

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


# --- Topic analysis figures ------------------------------------------------

def extract_topic_top_words(lda, n_words=TOP_N_WORDS):
    """Return dict: topic_id -> list of (word, weight) tuples."""

    # Initialise empty dict to hold top words for each topic
    topics = {}

    # Iterate over every topic the model discovered
    for i in range(lda.num_topics):
        # Get the top n_words words and their weights for topic i
        topics[i] = lda.show_topic(i, topn=n_words)

    # Return the dictionary of topics
    return topics


def assign_topic_to_doc(lda, bow):
    """Return (dominant_topic_id, probability) for a single document."""

    # Get the probability distribution over all topics for this document
    # minimum_probability=0 ensures every topic is included even if weight is near 0
    topic_probs = lda.get_document_topics(bow, minimum_probability=0)

    # Sort by probability descending so the most dominant topic is first
    topic_probs_sorted = sorted(topic_probs, key=lambda x: x[1], reverse=True)

    # Return the top topic's ID and its probability
    return topic_probs_sorted[0]


def plot_top_words_per_topic(topics, topic_labels, out_path):
    """Create a grid of horizontal bar charts — one per topic, showing top 10 words."""

    # Number of topics
    n = len(topics)

    # Use 3 columns in the grid
    ncols = 3

    # Calculate how many rows are needed (ceil division)
    nrows = int(np.ceil(n / ncols))

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))

    # Flatten to a 1D array for easy indexing (handle edge case of only 1 axis)
    axes = axes.flatten() if n > 1 else [axes]

    # Fill each subplot with a topic's top words
    for i in range(n):
        # Get the (word, weight) list for topic i
        words_weights = topics[i]

        # Extract the top 10 words and their weights
        words = [w for w, _ in words_weights][:10]
        weights = [w for _, w in words_weights][:10]

        # Select the subplot for this topic
        ax = axes[i]

        # Draw horizontal bars — reversed so the most important word is at the top
        ax.barh(range(len(words))[::-1], weights, color="#F06C00")

        # Label each bar with the word name
        ax.set_yticks(range(len(words))[::-1])
        ax.set_yticklabels(words, fontsize=10)

        # Set the subplot title to the human-readable topic label
        label = topic_labels.get(i, f"Topic {i}")
        ax.set_title(f"Topic {i}: {label}", fontsize=11)

        # Reduce x-tick label size
        ax.tick_params(axis="x", labelsize=9)

    # Hide any unused subplot panels (when n < nrows * ncols)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Add a main title spanning all subplots
    fig.suptitle("Figure 12. Top 10 words per LDA topic "
                 "(negative reviews only)", fontsize=14, y=1.00)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_topic_over_time(df, topic_labels, out_path):
    """Line chart showing how each topic's share of negative reviews changes by quarter."""

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Parse submission_time to a UTC-aware datetime
    df["submission_time"] = pd.to_datetime(df["submission_time"], utc=True)

    # Create a quarterly period column and convert to a datetime for the x-axis
    df["quarter"] = df["submission_time"].dt.to_period("Q").dt.to_timestamp()

    # Count how many reviews per quarter belong to each dominant topic
    topic_counts = (df.groupby(["quarter", "dominant_topic"])
                    .size().unstack(fill_value=0))

    # Convert counts to percentages of total negative reviews in that quarter
    topic_pct = topic_counts.div(topic_counts.sum(axis=1), axis=0) * 100

    # Create the figure
    fig, ax = plt.subplots(figsize=(13, 6))

    # Create a colour palette with one colour per topic
    colors = sns.color_palette("tab10", n_colors=len(topic_pct.columns))

    # Plot a line for each topic
    for i, col in enumerate(topic_pct.columns):
        # Get the human-readable label for this topic
        label = topic_labels.get(col, f"Topic {col}")

        # Draw the line with circular markers
        ax.plot(topic_pct.index, topic_pct[col], marker="o",
                label=f"{col}: {label}", color=colors[i], linewidth=2)

    # Axis labels and title
    ax.set_xlabel("Quarter")
    ax.set_ylabel("% of negative reviews with this dominant topic")
    ax.set_title("Figure 10. Topic prevalence among negative reviews over time")

    # Place the legend outside the plot to avoid overlapping the lines
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Rotate x-axis labels 45 degrees
    ax.tick_params(axis="x", rotation=45)

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def plot_topic_by_category(df, topic_labels, out_path):
    """Heatmap showing what % of each product category's negative reviews belong to each topic."""

    # Build a cross-tabulation: rows = product category, cols = dominant topic
    # normalize="index" makes each row sum to 100% (row-normalised)
    ct = pd.crosstab(df["category_keyword"], df["dominant_topic"],
                     normalize="index") * 100

    # Rename the column headers to include the human-readable topic labels
    ct.columns = [f"T{c}: {topic_labels.get(c, '')}" for c in ct.columns]

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap: warm colours = higher share of this topic in this category
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="Oranges",
                cbar_kws={"label": "% of negative reviews in category"},
                ax=ax, annot_kws={"size": 9})

    # Axis labels and title
    ax.set_xlabel("Topic")
    ax.set_ylabel("Product category")
    ax.set_title("Figure 11. Distribution of complaint topics by category\n"
                 "(row-normalised: each row sums to 100%)")

    # Rotate x-axis labels so they don't overlap
    plt.xticks(rotation=45, ha="right", fontsize=9)

    # Save and close
    plt.savefig(out_path)
    plt.close()
    print(f"  [fig] {out_path}")


def generate_interactive_lda(lda, corpus, dictionary, out_path):
    """Generate a pyLDAvis interactive HTML visualisation of the topic model."""

    try:
        # Prepare the visualisation data from the Gensim LDA model
        vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)

        # Save the visualisation as a standalone HTML file (open in browser to view)
        pyLDAvis.save_html(vis, str(out_path))

        print(f"  [interactive] {out_path}")

    except Exception as e:
        # If pyLDAvis fails for any reason, log a warning and continue
        print(f"  [WARN] pyLDAvis failed: {e}")


# --- Main -------------------------------------------------------------------

def main():
    # Make sure all required NLTK data packages are downloaded
    ensure_nltk_resources()

    # Check that the input file exists before proceeding
    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}. Run apply_bert_full.py first.")
        return

    # Load all reviews with BERT predictions
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Filter to only reviews where BERT predicted "negative" with high confidence
    neg = df[(df["bert_pred"] == "negative") &
             (df["bert_confidence"] >= NEGATIVE_CONFIDENCE_THRESHOLD)].copy()
    print(f"High-confidence negative reviews: {len(neg):,}")

    # --- Text Preprocessing ---
    print("\nPreprocessing text (lemmatisation, stopword removal)...")

    # Combine standard English stopwords with our domain-specific review stopwords
    stopword_set = set(stopwords.words("english")) | REVIEW_STOPWORDS

    # Initialise the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Import tqdm here (used for progress bar on the tokenisation step)
    from tqdm import tqdm

    # Enable tqdm.pandas() so we can use df.progress_apply() below
    tqdm.pandas(desc="Tokenising")

    # Apply the preprocessing function to every review's full_text column
    neg["tokens"] = neg["full_text"].progress_apply(
        lambda t: preprocess_text(t, stopword_set, lemmatizer)
    )

    # Drop reviews that ended up with fewer than 5 tokens after cleaning
    # (too short to be informative for LDA)
    neg = neg[neg["tokens"].str.len() >= 5].reset_index(drop=True)
    print(f"After filtering short docs: {len(neg):,}")

    # Extract the token lists as a plain Python list of lists
    texts = neg["tokens"].tolist()

    # --- Build Gensim Dictionary and Corpus ---
    print("\nBuilding dictionary and corpus...")

    # Create a Gensim Dictionary: maps each unique word to an integer ID
    dictionary = Dictionary(texts)

    # Filter the dictionary:
    # - Remove words appearing in fewer than 10 documents (too rare to form a topic)
    # - Remove words appearing in more than 50% of documents (too common to distinguish topics)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    # Convert each tokenized review to a bag-of-words (BoW) representation
    # Each BoW is a list of (word_id, word_count) tuples
    corpus = [dictionary.doc2bow(text) for text in texts]

    print(f"Dictionary size: {len(dictionary):,} tokens")
    print(f"Corpus size: {len(corpus):,} documents")

    # --- Model Selection via Coherence Score ---
    print("\nFitting LDA models for K in", K_VALUES_TO_TRY)

    # Lists to store results for each K value
    coherences = []
    models = {}

    # Try each candidate K value
    for k in K_VALUES_TO_TRY:
        print(f"  Fitting K={k}...")

        # Train LDA and compute coherence for this K
        lda, coh = compute_coherence(corpus, dictionary, texts, k)
        print(f"  Coherence (c_v): {coh:.4f}")

        # Store the coherence score and the trained model
        coherences.append(coh)
        models[k] = lda

    # Select the K value that gave the highest coherence score
    best_k = K_VALUES_TO_TRY[int(np.argmax(coherences))]
    best_lda = models[best_k]
    print(f"\nBest K by coherence: {best_k} (c_v = {max(coherences):.4f})")

    # Save the coherence score plot
    plot_coherence(K_VALUES_TO_TRY, coherences, FIGS_DIR / "fig09_lda_coherence.png")

    # --- Print Top Words per Topic ---
    print("\n" + "=" * 60)
    print(f"TOP {TOP_N_WORDS} WORDS PER TOPIC (K = {best_k})")
    print("=" * 60)

    # Extract top words for each discovered topic
    topics = extract_topic_top_words(best_lda, n_words=TOP_N_WORDS)

    # Print to terminal with word weights
    for topic_id, words_weights in topics.items():
        words = ", ".join([f"{w}({wt:.3f})" for w, wt in words_weights[:10]])
        print(f"\nTopic {topic_id}:")
        print(f"  {words}")

    # --- Assign Dominant Topic to Each Review ---
    print("\nAssigning dominant topic to each review...")

    dominant_topics = []   # List of topic IDs (one per review)
    topic_probs_list = []  # List of dominant topic probabilities

    # Iterate over every review's BoW representation
    for bow in tqdm(corpus, desc="Assigning"):
        top_id, top_prob = assign_topic_to_doc(best_lda, bow)
        dominant_topics.append(top_id)      # Most likely topic
        topic_probs_list.append(top_prob)   # Probability of that topic

    # Add the topic assignments to the DataFrame
    neg["dominant_topic"] = dominant_topics
    neg["topic_probability"] = topic_probs_list

    # --- Human-readable topic labels (assigned after manual inspection of top words) ---
    topic_labels = {
        0: "Meat/Protein Quality Failures",
        1: "Reformulation / Recipe Changes",
        2: "Taste & Texture Disappointment",
        3: "Price-Value & Packaging",
        4: "Prepared Food & Premium Disappointment",
    }

    # --- Generate Figures ---
    print("\nGenerating topic figures...")

    # Bar chart of top words per topic
    plot_top_words_per_topic(topics, topic_labels, FIGS_DIR / "fig12_top_words_per_topic.png")

    # Line chart of topic prevalence over time
    plot_topic_over_time(neg, topic_labels, FIGS_DIR / "fig10_topic_prevalence_over_time.png")

    # Heatmap of topic distribution by product category
    plot_topic_by_category(neg, topic_labels, FIGS_DIR / "fig11_topic_by_category.png")

    # Interactive pyLDAvis visualisation saved as HTML
    generate_interactive_lda(best_lda, corpus, dictionary, FIGS_DIR / "lda_interactive.html")

    # --- Save Output CSV ---
    # Select only the columns we need in the output file
    out_cols = ["review_id", "product_id", "product_name", "category_keyword",
                "rating", "title", "text", "full_text",
                "submission_time", "bert_pred", "bert_confidence",
                "dominant_topic", "topic_probability"]

    # Filter to only columns that actually exist in the DataFrame (defensive check)
    out_cols = [c for c in out_cols if c in neg.columns]

    # Save to CSV
    neg[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")

    # --- Print Exemplar Reviews per Topic ---
    print("\n" + "=" * 60)
    print("EXEMPLAR REVIEWS PER TOPIC (highest topic probability)")
    print("=" * 60)

    # For each topic, show the 3 reviews most strongly associated with it
    for t in range(best_k):
        subset = neg[neg["dominant_topic"] == t].nlargest(3, "topic_probability")
        print(f"\nTopic {t} ({len(neg[neg['dominant_topic'] == t])} reviews total):")

        for _, row in subset.iterrows():
            # Truncate the review text to the first 180 characters for readability
            text_snip = str(row["full_text"])[:180]
            print(f"  [{row['category_keyword']}, {row['rating']}*] {text_snip}...")


# Only run main() when this script is executed directly
if __name__ == "__main__":
    main()
