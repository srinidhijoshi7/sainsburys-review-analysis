"""
Stage 1: Product Discovery Scraper.

Queries Sainsbury's search API across diverse category keywords to build
a seed list of product IDs for Stage 2 (review harvesting).

Endpoint: https://www.sainsburys.co.uk/groceries-api/gol-services/product/v1/product

Discovered via Chrome DevTools > Network > Fetch/XHR on a search results page.

Output: data/products.csv
"""

# Import the requests library to send HTTP GET requests to the Sainsbury's API
import requests

# Import pandas for storing and manipulating tabular data (products list)
import pandas as pd

# Import time so we can pause between requests (polite rate-limiting)
import time

# Import Path so we can work with file/folder paths in a cross-platform way
from pathlib import Path

# Define the folder where we will save our output CSV file
OUTPUT_DIR = Path("data")

# Create the 'data' folder if it doesn't already exist
OUTPUT_DIR.mkdir(exist_ok=True)

# The Sainsbury's product search API endpoint (discovered via browser DevTools)
SEARCH_URL = "https://www.sainsburys.co.uk/groceries-api/gol-services/product/v1/product"

# HTTP headers that mimic a real Chrome browser request.
# These are required to pass Akamai bot detection — without them the API may block us.
HEADERS = {
    # Tell the server we accept JSON or plain text responses
    "Accept": "application/json, text/plain, */*",
    # Preferred language is British English
    "Accept-Language": "en-GB,en;q=0.9",
    # Tell the server we support compressed responses
    "Accept-Encoding": "gzip, deflate, br",
    # Make the request look like it came from the Sainsbury's search results page
    "Referer": "https://www.sainsburys.co.uk/gol-ui/SearchResults/milk",
    # Same-origin header to match browser behaviour
    "Origin": "https://www.sainsburys.co.uk",
    # These Sec-Fetch-* headers signal the request is from a legitimate browser context
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    # Chrome browser brand identification string
    "Sec-Ch-Ua": '"Chromium";v="124", "Not=A?Brand";v="24", "Google Chrome";v="124"',
    # Tells server this is not a mobile device
    "Sec-Ch-Ua-Mobile": "?0",
    # Tells server the operating system is macOS
    "Sec-Ch-Ua-Platform": '"macOS"',
    # Full user-agent string mimicking Chrome 124 on macOS
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# A list of search keywords spanning many grocery categories.
# Using diverse categories ensures the topic model later surfaces different complaint types.
CATEGORY_KEYWORDS = [
    # Fresh staple foods
    "milk", "bread", "eggs", "chicken", "beef", "salmon", "bananas", "apples",
    # Packaged / ambient grocery items
    "pasta", "cereal", "biscuits", "coffee", "tea",
    # Ready meals and convenience food
    "ready meal", "pizza", "sandwich",
    # Frozen products
    "ice cream", "frozen vegetables",
    # Drinks
    "wine", "beer", "juice",
    # Household / non-food items
    "washing powder", "toilet paper", "shampoo",
    # Sainsbury's premium own-brand line
    "taste the difference",
]


def fetch_page(keyword, page_number=1, page_size=60):
    """Fetch one page of search results for a given keyword."""

    # Build the query parameters for the API request
    params = {
        "filter[keyword]": keyword,   # The search term to look up
        "page_number": page_number,   # Which page of results to fetch (1-indexed)
        "page_size": page_size,       # How many results to return per page
        "sort_order": "FAVOURITES_FIRST",  # Sort by popularity so we get well-known products first
    }

    # Send the GET request with our headers and query params, wait up to 30 seconds
    response = requests.get(
        SEARCH_URL, headers=HEADERS, params=params, timeout=30
    )

    # Raise an exception if the server returned an HTTP error (e.g. 403, 500)
    response.raise_for_status()

    # Parse the response body as JSON and return it
    return response.json()


def extract_products(response_data, keyword):
    """Extract product records from the search API response."""

    # Initialise an empty list to hold all product dictionaries for this page
    products = []

    # Iterate over each product in the 'products' list in the API response
    for p in response_data.get("products", []):

        # Safely get the nested 'reviews' dict (use empty dict if missing)
        reviews_info = p.get("reviews") or {}

        # Safely get the nested 'retail_price' dict (use empty dict if missing)
        retail_price = p.get("retail_price") or {}

        # Build a flat dictionary of the fields we care about for this product
        products.append({
            "product_uid": p.get("product_uid"),          # Unique product identifier
            "name": p.get("name"),                         # Product display name
            "category_keyword": keyword,                   # The keyword we searched for
            "price": retail_price.get("price"),            # Current retail price
            "review_count": reviews_info.get("total", 0) or 0,    # Number of reviews (default 0)
            "average_rating": reviews_info.get("average_rating"),  # Star rating (1–5)
            "brand": p.get("brand"),                       # Brand name
            "product_type": p.get("product_type"),         # Type/classification of product
        })

    # Return all extracted product records for this page
    return products


def discover_products_for_keyword(keyword, max_pages=2):
    """For a given keyword, fetch up to max_pages of search results."""

    # Print progress to terminal so we can monitor the scrape
    print(f"\n[Keyword: {keyword}]")

    # Accumulate all products found for this keyword across all pages
    all_products = []

    # Iterate through each page number starting at 1
    for page_num in range(1, max_pages + 1):
        try:
            # Fetch this page of results from the API
            data = fetch_page(keyword, page_number=page_num)

            # Extract individual product records from the response
            products = extract_products(data, keyword)

            # Get the total number of products matching this keyword (for logging)
            total = data.get("controls", {}).get("total_record_count", "?")

            # Print how many products we retrieved this page and the total available
            print(f"  Page {page_num}: got {len(products)} products "
                  f"(keyword has {total} total matches)")

            # Add this page's products to the running list
            all_products.extend(products)

            # If we got fewer results than the page size, there are no more pages to fetch
            if len(products) < 60:
                break

            # Wait 1.5 seconds before the next request (polite rate limiting)
            time.sleep(1.5)

        except Exception as e:
            # Log the error but don't crash — skip to the next keyword
            print(f"  Error on page {page_num}: {e}")
            break

    # Return all products found for this keyword
    return all_products


def main():
    # Print a summary header so the user knows what's happening
    print(f"Discovering products across {len(CATEGORY_KEYWORDS)} keywords")
    print("=" * 60)

    # Master list to collect products from all keywords
    all_products = []

    # Loop through every search keyword
    for keyword in CATEGORY_KEYWORDS:
        # Fetch up to 2 pages of results for this keyword
        products = discover_products_for_keyword(keyword, max_pages=2)

        # Add this keyword's products to the master list
        all_products.extend(products)

        # Pause 1 second between keywords to avoid hammering the API
        time.sleep(1)

    # Convert the list of product dicts into a pandas DataFrame
    df = pd.DataFrame(all_products)

    # Print a separator and summary stats
    print(f"\n{'=' * 60}")
    print(f"Total products discovered (before dedup): {len(df)}")

    # If nothing was retrieved, warn the user and exit
    if len(df) == 0:
        print("No products retrieved. Check if requests are being blocked.")
        return

    # Remove duplicate products (same product can appear in multiple keyword searches)
    df = df.drop_duplicates(subset="product_uid").reset_index(drop=True)
    print(f"Unique products after dedup: {len(df)}")

    # Convert review_count to a number; fill missing values with 0
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)

    # Keep only products that have at least one review (no point scraping review-less products)
    df_with_reviews = df[df["review_count"] > 0].copy()
    print(f"Products with at least 1 review: {len(df_with_reviews)}")

    # Print a summary table: how many products and total reviews per category keyword
    print("\nProducts with reviews by category:")
    summary = (df_with_reviews
               .groupby("category_keyword")
               .agg(product_count=("product_uid", "count"),
                    total_reviews=("review_count", "sum"))
               .sort_values("total_reviews", ascending=False))
    print(summary)

    # Define the output file path
    output_file = OUTPUT_DIR / "products.csv"

    # Save the filtered product list to CSV (no row index column needed)
    df_with_reviews.to_csv(output_file, index=False)
    print(f"\n[OK] Saved {len(df_with_reviews)} products to {output_file}")

    # Print the total number of reviews available to scrape in Stage 2
    print(f"Total reviews available across these products: "
          f"{int(df_with_reviews['review_count'].sum()):,}")


# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
