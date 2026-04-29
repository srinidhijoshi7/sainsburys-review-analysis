"""
Test scraper v3 - Sainsbury's Bazaarvoice review API.

This endpoint was discovered via Chrome DevTools > Network > Fetch/XHR
on a Sainsbury's product page. It returns reviews as clean JSON.

Goal of this test: fetch ONE page of reviews for one product (milk),
print the first 3 reviews, and confirm the field names so we know
what the scale-up scraper needs to extract.
"""

# Import requests to send HTTP requests to the review API
import requests

# Import json (not strictly needed here since we use response.json(), but useful for debugging)
import json

# The Bazaarvoice-powered review API endpoint used by Sainsbury's
# This is the same endpoint for every product — we just change the ProductId parameter
BASE_URL = "https://reviews.sainsburys-groceries.co.uk/data/reviews.json"

# The product ID for Sainsbury's Semi-Skimmed Milk 2.27L
# The "-P" suffix is part of how Bazaarvoice formats product IDs on this platform
PRODUCT_ID = "357937-P"

# HTTP headers that mimic what Chrome sends — needed to avoid being blocked
HEADERS = {
    # Tell the server we want a JSON response
    "Accept": "application/json",
    # Content-Type signals we could also send JSON (standard browser behaviour)
    "Content-Type": "application/json",
    # Referer makes the request look like it came from the Sainsbury's site
    "Referer": "https://www.sainsburys.co.uk/",
    # Full Chrome user-agent string to appear as a real browser
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Query parameters for the API request — these match what the browser actually sent
params = {
    "ApiVersion": "5.4",               # Bazaarvoice API version to use
    "Filter": f"ProductId:{PRODUCT_ID}",  # Filter reviews to this specific product
    "Offset": 0,                        # Start from the first review (page 1)
    "Limit": 10,                        # Request 10 reviews per page
    "Include": "products",              # Also include product metadata in the response
}


def test_fetch():
    """Fetch one page and show us what's in it."""

    # Print what we're about to do for visibility in the terminal
    print(f"Fetching reviews for product: {PRODUCT_ID}")
    print(f"Endpoint: {BASE_URL}")

    # Send the GET request, wait up to 30 seconds for a response
    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)

    # Print the HTTP status code (200 = OK, 403 = blocked, 500 = server error, etc.)
    print(f"Status: {response.status_code}")

    # Raise an exception if the server returned any error status code
    response.raise_for_status()

    # Parse the response body from JSON into a Python dictionary
    data = response.json()

    # Print the top-level keys in the Bazaarvoice response so we understand its structure
    print("\n--- Top-level keys in response ---")
    print(list(data.keys()))

    # Get the total number of reviews this product has on the platform
    total = data.get("TotalResults", "unknown")
    print(f"\nTotal reviews available: {total}")

    # Get the list of reviews returned in this page
    results = data.get("Results", [])
    print(f"Reviews returned this page: {len(results)}")

    # If there are any results, inspect the first one to see what fields are available
    if results:
        print("\n--- Fields in first review ---")
        first_review = results[0]

        # Iterate over every key-value pair in the first review dictionary
        for key in first_review.keys():
            value = first_review[key]

            # Convert value to string for display
            display = str(value)

            # If the string is very long, truncate it to keep output readable
            if len(display) > 80:
                display = display[:80] + "..."

            # Print the field name and its (possibly truncated) value
            print(f"  {key}: {display}")

    # Print the first 3 reviews as a sanity check of the data quality
    print("\n--- First 3 reviews ---")
    for i, review in enumerate(results[:3], 1):
        # Print the review number (1-indexed)
        print(f"\nReview {i}:")

        # Print the star rating (1–5)
        print(f"  Rating: {review.get('Rating')}")

        # Print the date the review was submitted
        print(f"  Date: {review.get('SubmissionTime')}")

        # Print the review title (headline)
        print(f"  Title: {review.get('Title', '')}")

        # Get the full review body text (default to empty string if missing)
        text = review.get('ReviewText', '') or ''

        # Print only the first 120 characters of the review text to keep output concise
        print(f"  Text (first 120 chars): {text[:120]}")

    # Return the raw data in case the caller wants to inspect it further
    return data


# Only run if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Run the test fetch and store the returned data
    data = test_fetch()

    # Confirm the test finished without errors
    print("\n[OK] Test complete.")
