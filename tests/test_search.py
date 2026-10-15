import os
from pathlib import Path
from dotenv import load_dotenv
from serpapi import GoogleSearch

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("SERPAPI_API_KEY")

if not api_key:
    print("Error: SERPAPI_API_KEY not found in .env file.")
    raise SystemExit(1)

company_name = input("Enter a company name: ").strip() or "OpenAI"

params = {
    "engine": "google",
    "q": f'site:linkedin.com/in/ "{company_name}" engineer',
    "api_key": api_key,
    "num": 5,
}

try:
    print(f"\nSearching for employee profiles at {company_name}...\n")
    search = GoogleSearch(params)
    results = search.get_dict()

    if results.get("error"):
        print("Search API error:", results["error"])
        raise SystemExit(1)

    organic_results = results.get("organic_results", [])
    if not organic_results:
        print("No search results found.")
    else:
        for index, item in enumerate(organic_results, start=1):
            print(f"Result {index}")
            print("Title:", item.get("title", "N/A"))
            print("Link:", item.get("link", "N/A"))
            print("Snippet:", item.get("snippet", "N/A"))
            print("-" * 50)

except Exception as error:
    print(f"An error occurred: {error}")

