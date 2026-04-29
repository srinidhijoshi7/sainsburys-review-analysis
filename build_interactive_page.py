"""
Builds an enhanced, branded interactive HTML page that wraps the
pyLDAvis visualisation with explanatory panels, topic labels, and
a how-to-read guide. This improves on the raw pyLDAvis output by:

- Adding Sainsbury's-branded CSS styling
- Providing a plain-English explanation of the lambda (relevance) slider
- Mapping pyLDAvis topic numbers (1-5) to our report topic names
- Adding a How-to-Read guide for non-technical readers
- Including attribution for citations
"""

# Import pandas for loading data and manipulating the reviews DataFrame
import pandas as pd

# Import numpy for numerical operations (not directly used here, but part of the wider toolkit)
import numpy as np

# Import Path for cross-platform file path operations
from pathlib import Path

# Import re for regular expressions (used in text preprocessing)
import re

# Suppress runtime warnings that would clutter the terminal output
import warnings
warnings.filterwarnings("ignore")

# Import NLTK for natural language processing
import nltk

# Import the English stopwords list from NLTK
from nltk.corpus import stopwords

# Import the lemmatizer to reduce words to their base form
from nltk.stem import WordNetLemmatizer

# Import the word tokenizer to split text into individual words
from nltk.tokenize import word_tokenize

# Import Gensim's Dictionary for mapping words to integer IDs
from gensim.corpora import Dictionary

# Import Gensim's LDA model for topic modelling
from gensim.models import LdaModel

# Import pyLDAvis for generating the interactive topic visualisation
import pyLDAvis

# Adapter to prepare Gensim LDA models for pyLDAvis
import pyLDAvis.gensim_models

# --- Paths -----------------------------------------------------------------

# Folder containing input data files
DATA_DIR = Path("data")

# Folder where the output HTML file will be saved
FIGS_DIR = Path("figures")

# Input: full-dataset BERT sentiment predictions from apply_bert_full.py
INPUT_FILE = DATA_DIR / "reviews_full_sentiment.csv"

# Output: the enhanced branded HTML interactive page
OUTPUT_FILE = FIGS_DIR / "lda_interactive_enhanced.html"

# --- Review-specific stopwords (same as topic_modelling.py) ---------------

# Domain-specific words that are too generic to distinguish complaint topics.
# These are excluded from the LDA vocabulary to improve topic coherence.
REVIEW_STOPWORDS = {
    "sainsbury", "sainsburys", "sainsbury's",           # Brand name noise
    "product", "item", "bought", "buy", "buying", "purchase", "purchased",  # Purchase language
    "would", "could", "should", "get", "got", "use", "used", "using",
    "one", "two", "say", "said", "really", "quite", "also",
    "like", "good", "bad", "nice", "great", "lovely", "amazing",
    "taste", "flavour",        # Too generic across food categories
    "not", "no", "nothing", "none",       # Generic negation words
    "review", "reviewing", "star", "rating",   # Review meta-language
    "day", "week", "month", "year", "today", "yesterday",  # Time words
}


def preprocess_text(text, stopword_set, lemmatizer):
    """Tokenise, clean, and lemmatise one review text string."""

    # Return empty list for missing values
    if pd.isna(text):
        return []

    # Convert to lowercase so "BAD" and "bad" are treated identically
    text = str(text).lower()

    # Remove any URLs from the text
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove everything except lowercase letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split the string into individual words
    tokens = word_tokenize(text)

    # Filter and lemmatize
    cleaned = []
    for tok in tokens:
        # Skip tokens that are 2 characters or fewer
        if len(tok) <= 2 or tok in stopword_set:
            continue

        # Lemmatize: reduce to root form (e.g. "prices" -> "price")
        lem = lemmatizer.lemmatize(tok)

        # Skip if lemmatized form is a stopword or too short
        if lem in stopword_set or len(lem) <= 2:
            continue

        # Add the cleaned token to the list
        cleaned.append(lem)

    return cleaned


# --- Topic labels and descriptions -----------------------------------------

# A dictionary mapping 0-indexed Gensim topic IDs to (title, description) tuples.
# These labels were assigned after inspecting the top 15 words and exemplar reviews.
TOPIC_LABELS = {
    0: ("Meat/Protein Quality Failures",
        "Complaints about tough, tasteless, or inedible meat, chicken, and fish. "
        "Reviewers frequently describe throwing the product in the bin or calling it a "
        "'waste of money'. Dominant in beef (73%), chicken (52%) and salmon (40%) categories."),

    1: ("Reformulation / Recipe Changes",
        "Long-standing customers expressing disappointment with product reformulations — "
        "'the new recipe is awful'. Concentrated heavily in shampoo (78%), washing powder (66%), "
        "and coffee (49%)."),

    2: ("Taste & Texture Disappointment",
        "Subjective quality complaints — bland, dry, too sweet, hard, disappointing. "
        "Bread (40%), sandwich (39%), cereal (35%) and biscuits (34%) dominate this theme."),

    3: ("Price-Value & Shrinkflation",
        "Cost-of-living frustrations: price hikes, reduced product size, Nectar pricing disputes, "
        "unreliable packaging. Toilet paper (69%), beer (66%), tea (59%) and juice (53%) lead this theme."),

    4: ("Prepared Food & Premium Disappointment",
        "Ready-to-eat and premium own-brand (Taste the Difference) not meeting expectations. "
        "Pizza (49%), ice cream (48%), ready meal (29%) and Taste the Difference (28%) concentrate here."),
}


def rebuild_lda_model():
    """
    Re-run the LDA preprocessing and training to get the Gensim model objects
    needed to generate the pyLDAvis visualisation with the best K=5.
    """

    # Download NLTK resources if not already cached
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    # Load the full-dataset BERT sentiment predictions
    print("Loading sentiment data...")
    df = pd.read_csv(INPUT_FILE)

    # Filter to only high-confidence negative reviews (same threshold as topic_modelling.py)
    neg = df[(df["bert_pred"] == "negative") & (df["bert_confidence"] >= 0.80)].copy()
    print(f"Negative reviews: {len(neg):,}")

    # Preprocess text: tokenise, remove stopwords, lemmatise
    print("Preprocessing text...")
    stopword_set = set(stopwords.words("english")) | REVIEW_STOPWORDS
    lemmatizer = WordNetLemmatizer()

    # Apply preprocessing to every review's full_text
    neg["tokens"] = neg["full_text"].apply(lambda t: preprocess_text(t, stopword_set, lemmatizer))

    # Drop reviews with fewer than 5 tokens (too short to be informative)
    neg = neg[neg["tokens"].str.len() >= 5].reset_index(drop=True)

    # Build the Gensim Dictionary (word-to-ID mapping)
    print("Building dictionary & corpus...")
    dictionary = Dictionary(neg["tokens"].tolist())

    # Filter out very rare and very common words (same thresholds as topic_modelling.py)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    # Convert each tokenized review to a bag-of-words list of (word_id, count) tuples
    corpus = [dictionary.doc2bow(text) for text in neg["tokens"].tolist()]

    # Train the LDA model with K=5 (the best K found by coherence analysis in topic_modelling.py)
    print("Training LDA (K=5)...")
    lda = LdaModel(
        corpus=corpus,          # Bag-of-words corpus
        id2word=dictionary,     # Word-to-ID mapping
        num_topics=5,           # Number of topics (best K)
        random_state=42,        # Fixed seed for reproducibility
        passes=10,              # Training passes through the corpus
        alpha="auto",           # Automatically learn document-topic prior
        eta="auto",             # Automatically learn topic-word prior
        iterations=100,         # Maximum E-step iterations per document
    )

    # Return the trained model, corpus, dictionary, and the filtered negative reviews DataFrame
    return lda, corpus, dictionary, neg


def build_enhanced_page(lda, corpus, dictionary, output_path):
    """
    Generate the pyLDAvis visualisation HTML and embed it inside a
    full branded HTML page with explanatory panels and topic legend cards.
    """

    # Generate the pyLDAvis visualisation data structure
    print("Generating pyLDAvis visualisation...")
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)

    # Extract the raw HTML of the pyLDAvis visualisation (JavaScript + SVG)
    raw_html = pyLDAvis.prepared_data_to_html(vis)

    # Determine the mapping from pyLDAvis topic numbers (1-based, sorted by size)
    # to Gensim topic IDs (0-based, in training order)
    try:
        topic_order = vis.topic_order   # pyLDAvis provides this ordering
    except AttributeError:
        # Fallback: assume topics are in the same order as Gensim trained them
        topic_order = list(range(lda.num_topics))

    # Build a mapping from pyLDAvis number (1–5) to our human-readable topic label
    pyldavis_to_label = {}
    for pyldavis_num, gensim_id in enumerate(topic_order, start=1):
        # pyLDAvis may use 1-based or 0-based gensim IDs — normalise to 0-based
        gensim_id_zero_indexed = gensim_id - 1 if min(topic_order) == 1 else gensim_id
        # Look up the corresponding title and description from TOPIC_LABELS
        pyldavis_to_label[pyldavis_num] = TOPIC_LABELS[gensim_id_zero_indexed]

    # Build the HTML for the topic legend card grid (one card per topic)
    legend_cards = ""
    for pyldavis_num in range(1, 6):
        # Unpack the title and description for this topic
        label, description = pyldavis_to_label[pyldavis_num]

        # Append a styled card div to the legend_cards string
        legend_cards += f"""
        <div class="topic-card" data-topic="{pyldavis_num}">
          <div class="topic-num">Topic {pyldavis_num}</div>
          <div class="topic-title">{label}</div>
          <div class="topic-desc">{description}</div>
        </div>
        """

    # Build the complete branded HTML page as a multi-line f-string.
    # Double braces {{ }} are used where we want literal CSS braces in the output.
    page = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Interactive LDA Topic Model — Sainsbury's Customer Complaints</title>
  <!-- Load pyLDAvis default CSS for the visualisation panel -->
  <link rel="stylesheet" type="text/css"
        href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">
  <style>
    /* CSS custom properties for Sainsbury's brand colours */
    :root {{
      --sb-orange: #F06C00;
      --sb-dark: #1a1a1a;
      --sb-grey: #4a4a4a;
      --sb-light-grey: #f5f5f5;
      --sb-border: #e0e0e0;
      --sb-accent: #ffebd6;
    }}

    /* Reset box model for all elements */
    * {{ box-sizing: border-box; }}

    /* Page body: clean sans-serif font, light background */
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
      margin: 0; padding: 0;
      color: var(--sb-dark);
      background: #fafafa;
      line-height: 1.55;
    }}

    /* Orange gradient header bar */
    header {{
      background: linear-gradient(135deg, var(--sb-orange) 0%, #d95500 100%);
      color: white;
      padding: 32px 48px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }}
    header h1 {{ margin: 0 0 6px 0; font-size: 26px; font-weight: 600; }}
    header .subtitle {{ font-size: 15px; opacity: 0.95; }}

    /* Main content container with max width */
    .container {{ max-width: 1500px; margin: 0 auto; padding: 24px 48px 48px 48px; }}

    /* Reusable white card panel */
    .panel {{
      background: white;
      border: 1px solid var(--sb-border);
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .panel h2 {{ margin: 0 0 12px 0; font-size: 17px; color: var(--sb-orange); font-weight: 600; }}
    .panel p {{ margin: 6px 0; font-size: 14px; color: var(--sb-grey); }}

    /* How-to-read panel has a left orange border accent */
    .how-to-read {{
      background: var(--sb-accent);
      border-left: 4px solid var(--sb-orange);
    }}
    .how-to-read ol {{ margin: 8px 0 0 0; padding-left: 22px; }}
    .how-to-read li {{ margin-bottom: 6px; font-size: 14px; color: var(--sb-grey); }}

    /* Lambda explanation: two-column side-by-side boxes */
    .lambda-box {{ display: flex; gap: 16px; margin-top: 12px; }}
    .lambda-end {{
      flex: 1;
      padding: 14px;
      border-radius: 6px;
      border: 1px solid var(--sb-border);
    }}
    .lambda-end.left {{ background: #fff8f0; }}   /* Warm tint for lambda=0 */
    .lambda-end.right {{ background: #f0f6ff; }}  /* Cool tint for lambda=1 */
    .lambda-end strong {{ display: block; font-size: 13px; text-transform: uppercase;
                          letter-spacing: 0.5px; margin-bottom: 4px; }}
    .lambda-end.left strong {{ color: var(--sb-orange); }}
    .lambda-end.right strong {{ color: #2962ff; }}
    .lambda-end p {{ margin: 0; font-size: 13px; }}

    /* Topic card grid */
    .topic-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}

    /* Individual topic card with an orange left border */
    .topic-card {{
      background: white;
      border: 1px solid var(--sb-border);
      border-left: 4px solid var(--sb-orange);
      border-radius: 6px;
      padding: 12px 14px;
    }}
    .topic-num {{ font-size: 11px; color: var(--sb-orange); font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.8px; }}
    .topic-title {{ font-size: 15px; font-weight: 600; margin: 3px 0 6px 0; color: var(--sb-dark); }}
    .topic-desc {{ font-size: 12.5px; color: var(--sb-grey); line-height: 1.5; }}

    /* Container for the pyLDAvis visualisation */
    .viz-container {{
      background: white;
      border: 1px solid var(--sb-border);
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}

    /* Page footer */
    footer {{
      text-align: center;
      padding: 24px;
      font-size: 12px;
      color: #888;
      border-top: 1px solid var(--sb-border);
      margin-top: 40px;
    }}
    footer code {{
      background: #eee;
      padding: 1px 6px;
      border-radius: 3px;
      font-family: 'SF Mono', Menlo, Consolas, monospace;
      font-size: 11px;
    }}
  </style>
</head>
<body>

<!-- Branded orange header -->
<header>
  <h1>Interactive LDA Topic Model — Sainsbury's Customer Complaints</h1>
  <div class="subtitle">Exploring 5,257 high-confidence negative reviews (Jan 2023 – Apr 2026) · Appendix visualisation</div>
</header>

<div class="container">

  <!-- How-to-Read guide panel -->
  <div class="panel how-to-read">
    <h2>How to read this visualisation</h2>
    <ol>
      <li><strong>The left panel</strong> shows each topic as a circle — size = prevalence, distance = distinctiveness.</li>
      <li><strong>Click or hover on any circle</strong> to see that topic's top terms on the right.</li>
      <li><strong>The right panel</strong> shows the top 30 words — red bars = frequency in topic, blue = overall frequency.</li>
      <li><strong>Adjust the λ slider</strong> at the top right to control what counts as a "top" word.</li>
    </ol>
  </div>

  <!-- Lambda slider explanation panel -->
  <div class="panel lambda-explanation">
    <h2>What does the λ (lambda) slider do?</h2>
    <p>The slider controls a trade-off between <strong>frequency</strong> and <strong>distinctiveness</strong> (Sievert and Shirley, 2014):</p>
    <div class="lambda-box">
      <div class="lambda-end left">
        <strong>λ near 0 — distinctive</strong>
        <p>Shows words <em>unique to this topic</em>. Best for understanding what makes each topic different.</p>
      </div>
      <div class="lambda-end right">
        <strong>λ near 1 — frequent</strong>
        <p>Shows the most <em>common</em> words in the topic, even if they appear in others too.</p>
      </div>
    </div>
    <!-- Recommended setting based on Sievert & Shirley (2014) -->
    <p style="margin-top:12px;"><strong>Recommended: try λ ≈ 0.6</strong> for the most interpretable view.</p>
  </div>

  <!-- Topic legend cards panel -->
  <div class="panel">
    <h2>What are Topics 1–5?</h2>
    <p style="margin-bottom:0;">Each topic was labelled after inspecting the top 15 words and three exemplar reviews per topic.</p>
    <!-- The 5 topic cards are injected here by the Python code above -->
    <div class="topic-grid">
      {legend_cards}
    </div>
  </div>

  <!-- The pyLDAvis visualisation itself -->
  <div class="panel">
    <h2>The visualisation</h2>
    <p>Click a topic circle on the left, or use Previous/Next Topic to step through them.</p>
    <div class="viz-container">
      {raw_html}
    </div>
  </div>

</div>

<!-- Attribution footer -->
<footer>
  Generated using <code>pyLDAvis</code> (Mabey 2018) and <code>gensim</code> (Řehůřek and Sojka 2010).
  LDA: Blei, Ng and Jordan (2003). Relevance metric: Sievert and Shirley (2014).<br>
  Sainsbury's Customer Review Analysis · University of Bristol · Social Media and Web Analytics (EFIMM0139) · 2026
</footer>

</body>
</html>
"""

    # Write the complete HTML page to the output file (UTF-8 encoding for special characters)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[OK] Enhanced HTML saved to {output_path}")


def main():
    # Rebuild the LDA model so we have the Gensim objects needed for pyLDAvis
    lda, corpus, dictionary, _ = rebuild_lda_model()

    # Generate and save the enhanced branded HTML page
    build_enhanced_page(lda, corpus, dictionary, OUTPUT_FILE)

    # Print a reminder of how to open the file in a browser
    print(f"\nOpen with: open {OUTPUT_FILE}")


# Only run main() when this script is executed directly
if __name__ == "__main__":
    main()
