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

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

DATA_DIR = Path("data")
FIGS_DIR = Path("figures")
INPUT_FILE = DATA_DIR / "reviews_full_sentiment.csv"
OUTPUT_FILE = FIGS_DIR / "lda_interactive_enhanced.html"

REVIEW_STOPWORDS = {
    "sainsbury", "sainsburys", "sainsbury's",
    "product", "item", "bought", "buy", "buying", "purchase", "purchased",
    "would", "could", "should", "get", "got", "use", "used", "using",
    "one", "two", "say", "said", "really", "quite", "also",
    "like", "good", "bad", "nice", "great", "lovely", "amazing",
    "taste", "flavour",
    "not", "no", "nothing", "none",
    "review", "reviewing", "star", "rating",
    "day", "week", "month", "year", "today", "yesterday",
}


def preprocess_text(text, stopword_set, lemmatizer):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        if len(tok) <= 2 or tok in stopword_set:
            continue
        lem = lemmatizer.lemmatize(tok)
        if lem in stopword_set or len(lem) <= 2:
            continue
        cleaned.append(lem)
    return cleaned


# Topic labels - these match our final report labels
# pyLDAvis re-orders topics by size, so we also map from gensim index to pyLDAvis number
# (pyLDAvis uses 1-based indexing and sorts by topic marginal probability)
TOPIC_LABELS = {
    0: ("Meat/Protein Quality Failures",
        "Complaints about tough, tasteless, or inedible meat, chicken, and fish. Reviewers frequently describe throwing the product in the bin or calling it a 'waste of money'. Dominant in beef (73%), chicken (52%) and salmon (40%) categories."),
    1: ("Reformulation / Recipe Changes",
        "Long-standing customers expressing disappointment with product reformulations — 'the new recipe is awful'. Concentrated heavily in shampoo (78%), washing powder (66%), and coffee (49%)."),
    2: ("Taste & Texture Disappointment",
        "Subjective quality complaints — bland, dry, too sweet, hard, disappointing. Bread (40%), sandwich (39%), cereal (35%) and biscuits (34%) dominate this theme."),
    3: ("Price-Value & Shrinkflation",
        "Cost-of-living frustrations: price hikes, reduced product size, Nectar pricing disputes, unreliable packaging. Toilet paper (69%), beer (66%), tea (59%) and juice (53%) lead this theme."),
    4: ("Prepared Food & Premium Disappointment",
        "Ready-to-eat and premium own-brand (Taste the Difference) not meeting expectations. Pizza (49%), ice cream (48%), ready meal (29%) and Taste the Difference (28%) concentrate here."),
}


def rebuild_lda_model():
    """Rebuild the LDA model so we can regenerate the pyLDAvis viz with fresh data."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
    
    print("Loading sentiment data...")
    df = pd.read_csv(INPUT_FILE)
    neg = df[(df["bert_pred"] == "negative") & (df["bert_confidence"] >= 0.80)].copy()
    print(f"Negative reviews: {len(neg):,}")
    
    print("Preprocessing text...")
    stopword_set = set(stopwords.words("english")) | REVIEW_STOPWORDS
    lemmatizer = WordNetLemmatizer()
    neg["tokens"] = neg["full_text"].apply(lambda t: preprocess_text(t, stopword_set, lemmatizer))
    neg = neg[neg["tokens"].str.len() >= 5].reset_index(drop=True)
    
    print("Building dictionary & corpus...")
    dictionary = Dictionary(neg["tokens"].tolist())
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in neg["tokens"].tolist()]
    
    print("Training LDA (K=5)...")
    lda = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=5,
        random_state=42, passes=10, alpha="auto", eta="auto", iterations=100,
    )
    return lda, corpus, dictionary, neg


def build_enhanced_page(lda, corpus, dictionary, output_path):
    """Generate pyLDAvis HTML and wrap it in a branded page."""
    print("Generating pyLDAvis visualisation...")
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    raw_html = pyLDAvis.prepared_data_to_html(vis)
    
    # Determine the pyLDAvis -> gensim topic mapping
    # pyLDAvis sorts topics by marginal probability (largest = Topic 1)
    # We extract the ordering from the vis object's topic_order
    try:
        topic_order = vis.topic_order  # list of gensim topic IDs in pyLDAvis order
    except AttributeError:
        topic_order = list(range(lda.num_topics))
    
    # Build: pyLDAvis number (1-5) -> our label
    # pyLDAvis's topic_order returns 1-indexed gensim IDs, but TOPIC_LABELS
    # uses 0-indexed keys, so we convert.
    pyldavis_to_label = {}
    for pyldavis_num, gensim_id in enumerate(topic_order, start=1):
        # Normalise: if topic_order is 1-indexed, subtract 1 to match TOPIC_LABELS
        gensim_id_zero_indexed = gensim_id - 1 if min(topic_order) == 1 else gensim_id
        pyldavis_to_label[pyldavis_num] = TOPIC_LABELS[gensim_id_zero_indexed]
    
    # Build the topic legend cards
    legend_cards = ""
    for pyldavis_num in range(1, 6):
        label, description = pyldavis_to_label[pyldavis_num]
        legend_cards += f"""
        <div class="topic-card" data-topic="{pyldavis_num}">
            <div class="topic-num">Topic {pyldavis_num}</div>
            <div class="topic-title">{label}</div>
            <div class="topic-desc">{description}</div>
        </div>
        """
    
    # Full HTML page
    page = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interactive LDA Topic Model — Sainsbury's Customer Complaints</title>
<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">
<style>
    :root {{
        --sb-orange: #F06C00;
        --sb-dark: #1a1a1a;
        --sb-grey: #4a4a4a;
        --sb-light-grey: #f5f5f5;
        --sb-border: #e0e0e0;
        --sb-accent: #ffebd6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
        margin: 0;
        padding: 0;
        color: var(--sb-dark);
        background: #fafafa;
        line-height: 1.55;
    }}
    header {{
        background: linear-gradient(135deg, var(--sb-orange) 0%, #d95500 100%);
        color: white;
        padding: 32px 48px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }}
    header h1 {{
        margin: 0 0 6px 0;
        font-size: 26px;
        font-weight: 600;
        letter-spacing: -0.3px;
    }}
    header .subtitle {{
        font-size: 15px;
        opacity: 0.95;
    }}
    .container {{
        max-width: 1500px;
        margin: 0 auto;
        padding: 24px 48px 48px 48px;
    }}
    .panel {{
        background: white;
        border: 1px solid var(--sb-border);
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .panel h2 {{
        margin: 0 0 12px 0;
        font-size: 17px;
        color: var(--sb-orange);
        font-weight: 600;
    }}
    .panel p {{
        margin: 6px 0;
        font-size: 14px;
        color: var(--sb-grey);
    }}
    .how-to-read {{
        background: var(--sb-accent);
        border-left: 4px solid var(--sb-orange);
    }}
    .how-to-read ol {{
        margin: 8px 0 0 0;
        padding-left: 22px;
    }}
    .how-to-read li {{
        margin-bottom: 6px;
        font-size: 14px;
        color: var(--sb-grey);
    }}
    .lambda-explanation {{
        background: white;
    }}
    .lambda-box {{
        display: flex;
        gap: 16px;
        margin-top: 12px;
    }}
    .lambda-end {{
        flex: 1;
        padding: 14px;
        border-radius: 6px;
        border: 1px solid var(--sb-border);
    }}
    .lambda-end.left {{ background: #fff8f0; }}
    .lambda-end.right {{ background: #f0f6ff; }}
    .lambda-end strong {{
        display: block;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .lambda-end.left strong {{ color: var(--sb-orange); }}
    .lambda-end.right strong {{ color: #2962ff; }}
    .lambda-end p {{ margin: 0; font-size: 13px; }}
    .topic-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
        margin-top: 12px;
    }}
    .topic-card {{
        background: white;
        border: 1px solid var(--sb-border);
        border-left: 4px solid var(--sb-orange);
        border-radius: 6px;
        padding: 12px 14px;
    }}
    .topic-num {{
        font-size: 11px;
        color: var(--sb-orange);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    .topic-title {{
        font-size: 15px;
        font-weight: 600;
        margin: 3px 0 6px 0;
        color: var(--sb-dark);
    }}
    .topic-desc {{
        font-size: 12.5px;
        color: var(--sb-grey);
        line-height: 1.5;
    }}
    .viz-container {{
        background: white;
        border: 1px solid var(--sb-border);
        border-radius: 8px;
        padding: 16px;
        overflow: auto;
    }}
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

<header>
    <h1>Interactive LDA Topic Model — Sainsbury's Customer Complaints</h1>
    <div class="subtitle">Exploring 5,257 high-confidence negative reviews (Jan 2023 – Apr 2026) · Appendix visualisation</div>
</header>

<div class="container">

    <div class="panel how-to-read">
        <h2>How to read this visualisation</h2>
        <ol>
            <li><strong>The left panel</strong> shows each of the five discovered complaint topics as a circle. The size of the circle shows how common that topic is across all negative reviews. The distance between circles shows how different the topics are from each other — circles far apart are highly distinct.</li>
            <li><strong>Click or hover on any circle</strong> to see that topic's top terms on the right.</li>
            <li><strong>The right panel</strong> shows the top 30 words for whichever topic you select. Red = how often the word appears in this topic; blue = how often it appears overall.</li>
            <li><strong>Adjust the slider</strong> at the top right (λ, 'lambda') to change what counts as a 'top' word — see the next panel for what this means.</li>
        </ol>
    </div>

    <div class="panel lambda-explanation">
        <h2>What does the λ (lambda) slider do?</h2>
        <p>The slider controls a trade-off between <strong>how frequent</strong> a word is versus <strong>how distinctive</strong> it is to the selected topic (Sievert and Shirley, 2014):</p>
        <div class="lambda-box">
            <div class="lambda-end left">
                <strong>λ near 0 — distinctive</strong>
                <p>Shows words that are <em>specific to this topic</em>. For example, with λ=0.2, Topic 2 (Price &amp; Shrinkflation) surfaces 'shrinkflation', 'nectar', 'hike', 'screw cap' — words that only appear in this topic. Best for understanding what makes a topic unique.</p>
            </div>
            <div class="lambda-end right">
                <strong>λ near 1 — frequent</strong>
                <p>Shows the most <em>common</em> words in this topic, regardless of whether they also appear elsewhere. At λ=1, you will often see generic words like 'awful', 'disappointing', 'price' even for different topics. Useful for seeing overall volume.</p>
            </div>
        </div>
        <p style="margin-top:12px;"><strong>Recommended: try λ ≈ 0.6</strong>. Sievert and Shirley found this gives the most interpretable picture — a blend of distinctiveness and frequency.</p>
    </div>

    <div class="panel">
        <h2>What are Topics 1–5?</h2>
        <p style="margin-bottom:0;">Each topic was labelled by the research team after inspecting the top 15 words and three exemplar reviews per topic. The numbering on this visualisation (1–5) matches the 'Selected Topic' control at the top of the chart below. Categories in brackets show which product groups dominate each theme.</p>
        <div class="topic-grid">
            {legend_cards}
        </div>
    </div>

    <div class="panel">
        <h2>The visualisation</h2>
        <p>Click a topic circle on the left, or use Previous/Next Topic to step through them. Adjust λ at the top right to control the word list on the right.</p>
        <div class="viz-container">
            {raw_html}
        </div>
    </div>

</div>

<footer>
    Generated using <code>pyLDAvis</code> (Mabey 2018) and <code>gensim</code> (Řehůřek and Sojka 2010). LDA algorithm: Blei, Ng and Jordan (2003). Relevance metric: Sievert and Shirley (2014).<br>
    Sainsbury's Customer Review Analysis · University of Bristol · Social Media and Web Analytics (EFIMM0XXX) · 2026
</footer>

</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[OK] Enhanced HTML saved to {output_path}")


def main():
    lda, corpus, dictionary, _ = rebuild_lda_model()
    build_enhanced_page(lda, corpus, dictionary, OUTPUT_FILE)
    print(f"\nOpen with:  open {OUTPUT_FILE}")


if __name__ == "__main__":
    main()