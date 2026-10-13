
import os
from pathlib import Path

from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load API key from .env file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("SERPAPI_API_KEY")

if not api_key:
    print("Error: SERPAPI_API_KEY not found in .env file.")
    raise SystemExit(1)

# Get company name from the user
company_name = input("Enter a company name: ").strip()

if not company_name:
    print("Please enter a valid company name.")
    raise SystemExit(1)

# Search for publicly available employee profiles
params = {
    "engine": "google",
    "q": f'site:linkedin.com/in/ "{company_name}" engineer',
    "api_key": api_key,
    "num": 10,
}

try:
    print(f"\nSearching for employee information at {company_name}...\n")

    search = GoogleSearch(params)
    results = search.get_dict()

    # Handle API errors
    if results.get("error"):
        print("Search API error:", results["error"])
        raise SystemExit(1)

    organic_results = results.get("organic_results", [])

    if not organic_results:
        print("No relevant search results found.")
        raise SystemExit(0)

    # Display search results
    for index, item in enumerate(organic_results, start=1):
        print(f"Result {index}")
        print("Title:", item.get("title", "Not found"))
        print("Link:", item.get("link", "Not found"))
        print("Snippet:", item.get("snippet", "Not found"))
        print("-" * 60)

except Exception as error:
    print(f"An error occurred: {error}")
