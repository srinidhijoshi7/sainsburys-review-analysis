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

# Import requests to make HTTP calls to the Bazaarvoice review API
import requests

# Import pandas for reading the products list and writing the reviews CSV
import pandas as pd

# Import time so we can pause between requests (rate-limiting)
import time

# Import Path to work with file paths in a portable way
from pathlib import Path

# Import tqdm to show a progress bar in the terminal while scraping
from tqdm import tqdm

# --- Config ----------------------------------------------------------------

# Folder where all data files are stored
DATA_DIR = Path("data")

# Input: the product list created by discover_products.py
PRODUCTS_FILE = DATA_DIR / "products.csv"

# Output: the raw reviews CSV we'll build in this script
REVIEWS_FILE = DATA_DIR / "reviews_raw.csv"

# Bazaarvoice reviews API endpoint (same for all Sainsbury's products)
REVIEWS_URL = "https://reviews.sainsburys-groceries.co.uk/data/reviews.json"

# Maximum number of reviews to fetch per product (to balance breadth vs. depth)
MAX_REVIEWS_PER_PRODUCT = 50

# Minimum reviews a product must have to be worth scraping
MIN_REVIEWS_PER_PRODUCT = 5

# Bazaarvoice allows up to 100 reviews per API request
BATCH_SIZE = 100

# How long to wait between each API request (seconds) — being polite
SLEEP_BETWEEN_REQUESTS = 0.8

# How often to flush the in-memory buffer to disk (every N products)
CHECKPOINT_EVERY_N_PRODUCTS = 50

# Browser-like headers to avoid being blocked by the API
HEADERS = {
    # We expect a JSON response
    "Accept": "application/json",
    # Standard content-type for JSON APIs
    "Content-Type": "application/json",
    # Referer makes the request appear to come from the Sainsbury's website
    "Referer": "https://www.sainsburys.co.uk/",
    # Full Chrome 124 user-agent string to mimic a real browser
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# --- Core fetchers ---------------------------------------------------------

def fetch_reviews_page(product_id, offset=0, limit=BATCH_SIZE):
    """Fetch one batch of reviews for a given product ID."""

    # Build the query parameters for this API request
    params = {
        "ApiVersion": "5.4",                    # Bazaarvoice API version
        "Filter": f"ProductId:{product_id}",    # Filter to this specific product
        "Sort": "SubmissionTime:desc",           # Newest reviews first
        "Offset": offset,                        # Skip this many reviews (for pagination)
        "Limit": limit,                          # How many reviews to return
        "Include": "products",                   # Include product metadata in the response
    }

    # Make the HTTP GET request with a 30-second timeout
    response = requests.get(REVIEWS_URL, headers=HEADERS, params=params, timeout=30)

    # Raise an exception if the server returned an error status code
    response.raise_for_status()

    # Parse and return the JSON response body
    return response.json()


def extract_review_fields(raw_review, category_keyword):
    """Pull the fields we care about from a raw Bazaarvoice review record."""

    # Get the 'SecondaryRatings' sub-object (or empty dict if missing)
    secondary = raw_review.get("SecondaryRatings") or {}

    # Extract the quality sub-rating value (None if not provided by reviewer)
    quality = (secondary.get("Quality") or {}).get("Value")

    # Extract the value-for-money sub-rating (None if not provided)
    value = (secondary.get("Value") or {}).get("Value")

    # Return a flat dictionary with all the fields we want to store
    return {
        "review_id": raw_review.get("Id"),                     # Unique review identifier
        "product_id": raw_review.get("ProductId"),             # Bazaarvoice product ID
        "product_name": raw_review.get("OriginalProductName"), # Product name at time of review
        "category_keyword": category_keyword,                  # Which search keyword found this product
        "rating": raw_review.get("Rating"),                    # Overall star rating (1–5)
        "rating_quality": quality,                             # Sub-rating: quality
        "rating_value": value,                                 # Sub-rating: value for money
        "title": raw_review.get("Title"),                      # Review headline
        "text": raw_review.get("ReviewText"),                  # Full review body
        "submission_time": raw_review.get("SubmissionTime"),   # ISO datetime string
        "user_nickname": raw_review.get("UserNickname"),       # Reviewer's display name
        "user_location": raw_review.get("UserLocation"),       # Reviewer's location (if provided)
        "is_recommended": raw_review.get("IsRecommended"),     # Whether reviewer recommends the product
        "helpfulness": raw_review.get("Helpfulness"),          # Helpfulness score from other users
        "campaign_id": raw_review.get("CampaignId"),           # If the review was part of a campaign
        "is_syndicated": raw_review.get("IsSyndicated"),       # Whether review came from another platform
    }


def fetch_reviews_for_product(product_uid, category_keyword, max_reviews):
    """Fetch up to max_reviews reviews for one product, paginating as needed."""

    # Bazaarvoice uses a "-P" suffix on the product UID in its filter parameter
    product_id = f"{product_uid}-P"

    # List to accumulate all reviews for this product
    reviews = []

    # Start at the first review
    offset = 0

    # Keep fetching pages until we have enough reviews
    while len(reviews) < max_reviews:
        # How many reviews to request this page (don't exceed the remaining cap)
        limit = min(BATCH_SIZE, max_reviews - len(reviews))

        try:
            # Fetch one page of reviews starting at the current offset
            data = fetch_reviews_page(product_id, offset=offset, limit=limit)

        except Exception as e:
            # Log the error but don't crash — move on to the next product
            print(f"  [WARN] product {product_uid} offset {offset} failed: {e}")
            break

        # Get the list of review objects from this page
        results = data.get("Results", [])

        # If the API returned no reviews, we've reached the end — stop paginating
        if not results:
            break

        # Extract the fields we want from each raw review and add to our list
        for raw in results:
            reviews.append(extract_review_fields(raw, category_keyword))

        # How many reviews this product has in total on the platform
        total_available = data.get("TotalResults", 0)

        # Advance the offset by the number of reviews we just retrieved
        offset += len(results)

        # If we've fetched all available reviews for this product, stop
        if offset >= total_available:
            break

        # Pause before the next request to be a good citizen
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Return all reviews collected for this product
    return reviews


# --- Orchestrator ----------------------------------------------------------

def load_already_scraped_product_ids():
    """If reviews_raw.csv exists, return set of product_uids already scraped."""

    # If the reviews file doesn't exist yet, nothing has been scraped
    if not REVIEWS_FILE.exists():
        return set()

    try:
        # Read only the product_id column to keep memory usage low
        existing = pd.read_csv(REVIEWS_FILE, usecols=["product_id"])

        # product_id is stored as "XXXXX-P" in the CSV; strip the "-P" to match products.csv
        return set(existing["product_id"].str.replace("-P", "", regex=False).unique())

    except Exception as e:
        # If we can't read the file for any reason, start from scratch
        print(f"[WARN] Could not read existing reviews file: {e}")
        return set()


def append_to_csv(new_rows, file_path):
    """Append rows to CSV, writing header only if file doesn't exist yet."""

    # Nothing to write — return early
    if not new_rows:
        return

    # Convert the list of dicts to a DataFrame
    df = pd.DataFrame(new_rows)

    # Write the header row only if the CSV doesn't already exist
    header = not file_path.exists()

    # Append to the existing file (or create it if it doesn't exist)
    df.to_csv(file_path, mode="a", header=header, index=False)


def main():
    # Check that the products list exists — it must be created by discover_products.py first
    if not PRODUCTS_FILE.exists():
        print(f"Missing {PRODUCTS_FILE}. Run discover_products.py first.")
        return

    # Load the product list into a DataFrame
    products_df = pd.read_csv(PRODUCTS_FILE)

    # Ensure review_count is numeric; fill NaN with 0
    products_df["review_count"] = pd.to_numeric(products_df["review_count"], errors="coerce").fillna(0)

    # Keep only products with at least the minimum number of reviews
    eligible = products_df[products_df["review_count"] >= MIN_REVIEWS_PER_PRODUCT].copy()

    # Resume logic: identify products we've already scraped and skip them
    already_done = load_already_scraped_product_ids()
    if already_done:
        print(f"Resuming: {len(already_done)} products already scraped, skipping them")
        # Remove products whose UIDs are already in the existing reviews file
        eligible = eligible[~eligible["product_uid"].astype(str).isin(already_done)]

    # Print how many products remain to be scraped
    print(f"Products to scrape: {len(eligible)}")
    print(f"Cap per product: {MAX_REVIEWS_PER_PRODUCT}")
    print(f"Expected max reviews: ~{len(eligible) * MAX_REVIEWS_PER_PRODUCT:,}")
    print(f"Estimated runtime: ~{len(eligible) * 1.5 / 60:.0f} minutes")
    print("=" * 60)

    # If there's nothing to do, exit early
    if len(eligible) == 0:
        print("Nothing to do.")
        return

    # In-memory buffer: accumulate reviews here before flushing to disk
    buffer = []

    # Counters to track progress
    products_done_this_run = 0
    total_reviews_this_run = 0

    # Iterate over each eligible product with a progress bar
    for _, row in tqdm(eligible.iterrows(), total=len(eligible), desc="Products"):
        # Get the product's unique ID as a string
        product_uid = str(row["product_uid"])

        # Get the category keyword that found this product
        keyword = row["category_keyword"]

        # Respect both the global cap and the actual number of reviews available
        cap = min(MAX_REVIEWS_PER_PRODUCT, int(row["review_count"]))

        # Fetch reviews for this product
        reviews = fetch_reviews_for_product(product_uid, keyword, cap)

        # Add to the in-memory buffer
        buffer.extend(reviews)

        # Accumulate total review count
        total_reviews_this_run += len(reviews)

        # Increment the product counter
        products_done_this_run += 1

        # Every CHECKPOINT_EVERY_N_PRODUCTS products, flush the buffer to disk
        if products_done_this_run % CHECKPOINT_EVERY_N_PRODUCTS == 0:
            append_to_csv(buffer, REVIEWS_FILE)
            # Clear the buffer after writing to free memory
            buffer = []
            # Print a progress update without interrupting the progress bar
            tqdm.write(f"  [checkpoint] {products_done_this_run} products done, "
                       f"{total_reviews_this_run:,} reviews collected so far")

    # Write any remaining reviews in the buffer that weren't flushed at a checkpoint
    append_to_csv(buffer, REVIEWS_FILE)

    # Print final summary
    print("=" * 60)
    print(f"[OK] Done. Products scraped this run: {products_done_this_run}")
    print(f"Reviews collected this run: {total_reviews_this_run:,}")

    # Print totals from the complete output file (including previous runs)
    if REVIEWS_FILE.exists():
        final = pd.read_csv(REVIEWS_FILE)
        print(f"Total reviews in {REVIEWS_FILE}: {len(final):,}")
        print(f"Unique products: {final['product_id'].nunique()}")


# Only run main() if this script is executed directly
if __name__ == "__main__":
    main()
