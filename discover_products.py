"""
Stage 1: Product Discovery Scraper.

Queries Sainsbury's search API across diverse category keywords to build
a seed list of product IDs for Stage 2 (review harvesting).

Endpoint: https://www.sainsburys.co.uk/groceries-api/gol-services/product/v1/product
Discovered via Chrome DevTools > Network > Fetch/XHR on a search results page.

Output: data/products.csv
"""

import requests
import pandas as pd
import time
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

SEARCH_URL = "https://www.sainsburys.co.uk/groceries-api/gol-services/product/v1/product"

# Headers that satisfy Akamai's bot detection. The Sec-Fetch-* and Origin
# headers signal that the request is coming from a legitimate browser context.
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.sainsburys.co.uk/gol-ui/SearchResults/milk",
    "Origin": "https://www.sainsburys.co.uk",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Ch-Ua": '"Chromium";v="124", "Not=A?Brand";v="24", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Category keywords for BREADTH across Sainsbury's grocery range.
# Chosen to span: fresh food, packaged goods, ready meals, frozen,
# drinks, household, and the Taste the Difference premium own-brand.
# This diversity matters because our topic modelling (Stage 3) will
# surface different customer concerns across product types.
CATEGORY_KEYWORDS = [
    # Fresh / staples
    "milk", "bread", "eggs", "chicken", "beef", "salmon", "bananas", "apples",
    # Packaged / ambient
    "pasta", "cereal", "biscuits", "coffee", "tea",
    # Ready meals / convenience
    "ready meal", "pizza", "sandwich",
    # Frozen
    "ice cream", "frozen vegetables",
    # Drinks
    "wine", "beer", "juice",
    # Household
    "washing powder", "toilet paper", "shampoo",
    # Premium own-brand
    "taste the difference",
]


def fetch_page(keyword, page_number=1, page_size=60):
    """Fetch one page of search results for a given keyword."""
    params = {
        "filter[keyword]": keyword,
        "page_number": page_number,
        "page_size": page_size,
        "sort_order": "FAVOURITES_FIRST",
    }
    response = requests.get(
        SEARCH_URL, headers=HEADERS, params=params, timeout=30
    )
    response.raise_for_status()
    return response.json()


def extract_products(response_data, keyword):
    """Extract product records from the search API response."""
    products = []
    for p in response_data.get("products", []):
        reviews_info = p.get("reviews") or {}
        retail_price = p.get("retail_price") or {}
        products.append({
            "product_uid": p.get("product_uid"),
            "name": p.get("name"),
            "category_keyword": keyword,
            "price": retail_price.get("price"),
            "review_count": reviews_info.get("total", 0) or 0,
            "average_rating": reviews_info.get("average_rating"),
            "brand": p.get("brand"),
            "product_type": p.get("product_type"),
        })
    return products


def discover_products_for_keyword(keyword, max_pages=2):
    """For a given keyword, fetch up to max_pages of search results."""
    print(f"\n[Keyword: {keyword}]")
    all_products = []
    
    for page_num in range(1, max_pages + 1):
        try:
            data = fetch_page(keyword, page_number=page_num)
            products = extract_products(data, keyword)
            total = data.get("controls", {}).get("total_record_count", "?")
            print(f"  Page {page_num}: got {len(products)} products "
                  f"(keyword has {total} total matches)")
            all_products.extend(products)
            
            # Stop early if we got fewer results than requested (no more pages)
            if len(products) < 60:
                break
            
            # Be polite: sleep between requests
            time.sleep(1.5)
        except Exception as e:
            print(f"  Error on page {page_num}: {e}")
            break
    
    return all_products


def main():
    print(f"Discovering products across {len(CATEGORY_KEYWORDS)} keywords")
    print("=" * 60)
    
    all_products = []
    for keyword in CATEGORY_KEYWORDS:
        products = discover_products_for_keyword(keyword, max_pages=2)
        all_products.extend(products)
        time.sleep(1)  # extra sleep between keywords
    
    df = pd.DataFrame(all_products)
    
    print(f"\n{'=' * 60}")
    print(f"Total products discovered (before dedup): {len(df)}")
    
    if len(df) == 0:
        print("No products retrieved. Check if requests are being blocked.")
        return
    
    # Remove duplicates (a product can match multiple keywords)
    df = df.drop_duplicates(subset="product_uid").reset_index(drop=True)
    print(f"Unique products after dedup: {len(df)}")
    
    # Keep only products that actually have reviews
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df_with_reviews = df[df["review_count"] > 0].copy()
    print(f"Products with at least 1 review: {len(df_with_reviews)}")
    
    # Summary by category
    print("\nProducts with reviews by category:")
    summary = (df_with_reviews
               .groupby("category_keyword")
               .agg(product_count=("product_uid", "count"),
                    total_reviews=("review_count", "sum"))
               .sort_values("total_reviews", ascending=False))
    print(summary)
    
    output_file = OUTPUT_DIR / "products.csv"
    df_with_reviews.to_csv(output_file, index=False)
    print(f"\n[OK] Saved {len(df_with_reviews)} products to {output_file}")
    print(f"Total reviews available across these products: "
          f"{int(df_with_reviews['review_count'].sum()):,}")


if __name__ == "__main__":
    main()