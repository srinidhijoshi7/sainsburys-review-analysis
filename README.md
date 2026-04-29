# Sainsbury's Customer Review Analysis

## Overview

This repository contains the full analysis pipeline for a web-scraping and NLP study of 64,916 Sainsbury's customer product reviews published between January 2023 and April 2026. The study period captures the peak of UK food inflation (19.1%, March 2023) and Sainsbury's key strategic responses including the Nectar Prices launch (April 2023).

A two-stage NLP pipeline is applied:
1. **RoBERTa sentiment classification** (cardiffnlp/twitter-roberta-base-sentiment-latest) validated against star-rating ground truth on a stratified 5,000-review sample (accuracy = 76.8%, macro-F1 = 0.630)
2. **Latent Dirichlet Allocation** topic modelling on 5,257 high-confidence negative reviews, producing five coherent complaint themes

## Repository Structure

| File | Stage | Description |
|------|-------|-------------|
| `discover_products.py` | 1 | Product discovery via Sainsbury's public search API |
| `scrape_reviews.py` | 2 | Review harvesting via Bazaarvoice endpoint |
| `clean_reviews.py` | 3 | Data cleaning, deduplication, feature engineering |
| `sentiment_analysis.py` | 4a | BERT sentiment validation on stratified sample |
| `apply_bert_full.py` | 4b | BERT applied to full 64,916-review corpus |
| `topic_modelling.py` | 5 | LDA topic modelling with coherence-based K selection |
| `temporal_analysis.py` | 6 | Event-annotated temporal sentiment analysis |
| `build_interactive_page.py` | 7 | Interactive pyLDAvis HTML dashboard |

## Key Findings

- The Nectar Prices launch (April 2023) coincided with the largest positive sentiment shift in the dataset (Δ = +0.257)
- Five complaint themes identified: Meat/Protein Quality (28%), Price-Value & Shrinkflation (21%), Prepared Food & Premium Disappointment (19%), Taste & Texture (18%), Reformulation/Recipe Changes (13%)
- The March 2024 IT outage produced no measurable change in product-review sentiment (Δ = −0.002), confirming that product reviews cannot detect service incidents

## Interactive Dashboard

The LDA topic model is available as an interactive visualisation:  
👉 https://srinidhijoshi7.github.io/sainsburys-review-analysis/figures/lda_interactive_enhanced.html

## How to Reproduce

```bash
git clone https://github.com/srinidhijoshi7/sainsburys-review-analysis
cd sainsburys-review-analysis
# Run scripts in order: Stage 1 → 7
python discover_products.py
python scrape_reviews.py
python clean_reviews.py
python sentiment_analysis.py
python apply_bert_full.py
python topic_modelling.py
python temporal_analysis.py
python build_interactive_page.py
```

## Ethics

All data collected is publicly available customer content posted under self-chosen nicknames. No authentication was bypassed, no personally identifiable information was retained, and scraping adhered to a polite rate limit of one request per 0.8 seconds. This study follows British Psychological Society (2021) guidelines on internet-mediated research.
