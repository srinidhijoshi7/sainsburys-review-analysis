"""
Stage 2: Review Harvester.

For each product in data/products.csv, fetch reviews from the
Bazaarvoice-powered Sainsbury's reviews API and save incrementally
to data/reviews_raw.csv.

Endpoint: https://reviews.sainsburys-groceries.co.uk/data/reviews.json
Discovered via Chrome DevTools > Network > Fetch/XHR on a product page.

Sampling design:
  - Up to MAX_REVIEWS_PER_PRODUCT most recent reviews per product
  - Skip products with fewer than MIN_REVIEWS_PER_PRODUCT reviews
  - Incremental writes: survives interruptions
  - Resume: re-running skips already-scraped products
"""

import requests
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm

# --- Config ----------------------------------------------------------------

DATA_DIR = Path("data")
PRODUCTS_FILE = DATA_DIR / "products.csv"
REVIEWS_FILE = DATA_DIR / "reviews_raw.csv"

REVIEWS_URL = "https://reviews.sainsburys-groceries.co.uk/data/reviews.json"

MAX_REVIEWS_PER_PRODUCT = 50   # cap per product to balance diversity
MIN_REVIEWS_PER_PRODUCT = 5    # skip very-low-review products
BATCH_SIZE = 100               # Bazaarvoice max per request
SLEEP_BETWEEN_REQUESTS = 0.8   # polite rate-limiting
CHECKPOINT_EVERY_N_PRODUCTS = 50  # flush to disk every N products

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Referer": "https://www.sainsburys.co.uk/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


# --- Core fetchers ---------------------------------------------------------

def fetch_reviews_page(product_id, offset=0, limit=BATCH_SIZE):
    """Fetch one batch of reviews for a given product ID."""
    params = {
        "ApiVersion": "5.4",
        "Filter": f"ProductId:{product_id}",
        "Sort": "SubmissionTime:desc",   # newest first
        "Offset": offset,
        "Limit": limit,
        "Include": "products",
    }
    response = requests.get(REVIEWS_URL, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_review_fields(raw_review, category_keyword):
    """Pull the fields we care about from a raw Bazaarvoice review record."""
    secondary = raw_review.get("SecondaryRatings") or {}
    quality = (secondary.get("Quality") or {}).get("Value")
    value = (secondary.get("Value") or {}).get("Value")
    
    return {
        "review_id": raw_review.get("Id"),
        "product_id": raw_review.get("ProductId"),
        "product_name": raw_review.get("OriginalProductName"),
        "category_keyword": category_keyword,
        "rating": raw_review.get("Rating"),
        "rating_quality": quality,
        "rating_value": value,
        "title": raw_review.get("Title"),
        "text": raw_review.get("ReviewText"),
        "submission_time": raw_review.get("SubmissionTime"),
        "user_nickname": raw_review.get("UserNickname"),
        "user_location": raw_review.get("UserLocation"),
        "is_recommended": raw_review.get("IsRecommended"),
        "helpfulness": raw_review.get("Helpfulness"),
        "campaign_id": raw_review.get("CampaignId"),
        "is_syndicated": raw_review.get("IsSyndicated"),
    }


def fetch_reviews_for_product(product_uid, category_keyword, max_reviews):
    """Fetch up to max_reviews reviews for one product, paginating as needed."""
    product_id = f"{product_uid}-P"  # Bazaarvoice convention
    reviews = []
    offset = 0
    
    while len(reviews) < max_reviews:
        limit = min(BATCH_SIZE, max_reviews - len(reviews))
        try:
            data = fetch_reviews_page(product_id, offset=offset, limit=limit)
        except Exception as e:
            # Log and move on - don't let one product break the whole run
            print(f"  [WARN] product {product_uid} offset {offset} failed: {e}")
            break
        
        results = data.get("Results", [])
        if not results:
            break
        
        for raw in results:
            reviews.append(extract_review_fields(raw, category_keyword))
        
        total_available = data.get("TotalResults", 0)
        offset += len(results)
        
        # Stop if we've pulled everything available
        if offset >= total_available:
            break
        
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    return reviews


# --- Orchestrator ----------------------------------------------------------

def load_already_scraped_product_ids():
    """If reviews_raw.csv exists, return set of product_uids already scraped."""
    if not REVIEWS_FILE.exists():
        return set()
    try:
        existing = pd.read_csv(REVIEWS_FILE, usecols=["product_id"])
        # product_id is "XXXXX-P", we want the "XXXXX" part
        return set(existing["product_id"].str.replace("-P", "", regex=False).unique())
    except Exception as e:
        print(f"[WARN] Could not read existing reviews file: {e}")
        return set()


def append_to_csv(new_rows, file_path):
    """Append rows to CSV, writing header only if file doesn't exist yet."""
    if not new_rows:
        return
    df = pd.DataFrame(new_rows)
    header = not file_path.exists()
    df.to_csv(file_path, mode="a", header=header, index=False)


def main():
    if not PRODUCTS_FILE.exists():
        print(f"Missing {PRODUCTS_FILE}. Run discover_products.py first.")
        return
    
    products_df = pd.read_csv(PRODUCTS_FILE)
    products_df["review_count"] = pd.to_numeric(products_df["review_count"], errors="coerce").fillna(0)
    
    # Apply sampling rules
    eligible = products_df[products_df["review_count"] >= MIN_REVIEWS_PER_PRODUCT].copy()
    
    # Resume: skip products we've already done
    already_done = load_already_scraped_product_ids()
    if already_done:
        print(f"Resuming: {len(already_done)} products already scraped, skipping them")
        eligible = eligible[~eligible["product_uid"].astype(str).isin(already_done)]
    
    print(f"Products to scrape: {len(eligible)}")
    print(f"Cap per product: {MAX_REVIEWS_PER_PRODUCT}")
    print(f"Expected max reviews: ~{len(eligible) * MAX_REVIEWS_PER_PRODUCT:,}")
    print(f"Estimated runtime: ~{len(eligible) * 1.5 / 60:.0f} minutes")
    print("=" * 60)
    
    if len(eligible) == 0:
        print("Nothing to do.")
        return
    
    buffer = []
    products_done_this_run = 0
    total_reviews_this_run = 0
    
    for _, row in tqdm(eligible.iterrows(), total=len(eligible), desc="Products"):
        product_uid = str(row["product_uid"])
        keyword = row["category_keyword"]
        # Per-product cap: min of overall cap and reviews actually available
        cap = min(MAX_REVIEWS_PER_PRODUCT, int(row["review_count"]))
        
        reviews = fetch_reviews_for_product(product_uid, keyword, cap)
        buffer.extend(reviews)
        total_reviews_this_run += len(reviews)
        products_done_this_run += 1
        
        # Checkpoint flush
        if products_done_this_run % CHECKPOINT_EVERY_N_PRODUCTS == 0:
            append_to_csv(buffer, REVIEWS_FILE)
            buffer = []
            tqdm.write(f"  [checkpoint] {products_done_this_run} products done, "
                       f"{total_reviews_this_run:,} reviews collected so far")
    
    # Final flush
    append_to_csv(buffer, REVIEWS_FILE)
    
    print("=" * 60)
    print(f"[OK] Done. Products scraped this run: {products_done_this_run}")
    print(f"Reviews collected this run: {total_reviews_this_run:,}")
    
    # Final stats
    if REVIEWS_FILE.exists():
        final = pd.read_csv(REVIEWS_FILE)
        print(f"Total reviews in {REVIEWS_FILE}: {len(final):,}")
        print(f"Unique products: {final['product_id'].nunique()}")


if __name__ == "__main__":
    main()