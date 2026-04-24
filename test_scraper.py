"""
Test scraper v3 - Sainsbury's Bazaarvoice review API.

This endpoint was discovered via Chrome DevTools > Network > Fetch/XHR
on a Sainsbury's product page. It returns reviews as clean JSON.

Goal of this test: fetch ONE page of reviews for one product (milk),
print the first 3 reviews, and confirm the field names so we know
what the scale-up scraper needs to extract.
"""

import requests
import json

# Review API endpoint - same structure for every product, just swap ProductId
BASE_URL = "https://reviews.sainsburys-groceries.co.uk/data/reviews.json"

# Product ID for Sainsbury's Semi-Skimmed Milk 2.27L (from user's URL)
# Note: the "-P" suffix at the end is part of the product ID in Bazaarvoice
PRODUCT_ID = "357937-P"

# Browser-like headers - these match what Chrome actually sent
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

# Query parameters - matches exactly what the browser sent
params = {
    "ApiVersion": "5.4",
    "Filter": f"ProductId:{PRODUCT_ID}",
    "Offset": 0,
    "Limit": 10,
    "Include": "products",
}


def test_fetch():
    """Fetch one page and show us what's in it."""
    print(f"Fetching reviews for product: {PRODUCT_ID}")
    print(f"Endpoint: {BASE_URL}")
    
    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    print(f"Status: {response.status_code}")
    response.raise_for_status()
    
    data = response.json()
    
    # Top-level structure of Bazaarvoice response
    print("\n--- Top-level keys in response ---")
    print(list(data.keys()))
    
    # How many reviews total for this product?
    total = data.get("TotalResults", "unknown")
    print(f"\nTotal reviews available: {total}")
    
    # Get the list of reviews from this page
    results = data.get("Results", [])
    print(f"Reviews returned this page: {len(results)}")
    
    # Show the first review's fields so we know what to extract later
    if results:
        print("\n--- Fields in first review ---")
        first_review = results[0]
        for key in first_review.keys():
            value = first_review[key]
            # Abbreviate long values
            display = str(value)
            if len(display) > 80:
                display = display[:80] + "..."
            print(f"  {key}: {display}")
    
    # Show the first 3 reviews as a sanity check
    print("\n--- First 3 reviews ---")
    for i, review in enumerate(results[:3], 1):
        print(f"\nReview {i}:")
        print(f"  Rating: {review.get('Rating')}")
        print(f"  Date: {review.get('SubmissionTime')}")
        print(f"  Title: {review.get('Title', '')}")
        text = review.get('ReviewText', '') or ''
        print(f"  Text (first 120 chars): {text[:120]}")
    
    return data


if __name__ == "__main__":
    data = test_fetch()
    print("\n[OK] Test complete.")