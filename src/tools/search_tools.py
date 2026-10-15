import json
from typing import Optional
from serpapi import GoogleSearch
from langchain_core.tools import tool

from src.config import SERPAPI_API_KEY, EXCLUDED_DOMAINS, EXCLUDED_PATHS


def _execute_serpapi_search(query: str, num_results: int = 8) -> list[dict]:
    """Helper to query SerpApi and filter irrelevant domains and paths."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    }
    try:
        response = GoogleSearch(params).get_dict()
    except Exception as error:
        return [{"error": f"Search API failed: {error}"}]

    if response.get("error"):
        return [{"error": response["error"]}]

    results = []
    seen_urls = set()

    for item in response.get("organic_results", []):
        url = item.get("link", "").strip()
        if not url or url in seen_urls:
            continue

        lower_url = url.lower()
        if any(domain in lower_url for domain in EXCLUDED_DOMAINS):
            continue
        if any(path in lower_url for path in EXCLUDED_PATHS):
            continue

        seen_urls.add(url)
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": url,
        })

    return results


@tool
def search_company_employees(company: str, role_or_department: Optional[str] = None) -> str:
    """Search public directories and LinkedIn profiles for employees at a given company.
    Optionally filter by a specific role or department (e.g., 'engineers', 'product manager').
    Returns a JSON string of candidate profile results with title, snippet, and url.
    """
    if role_or_department:
        query = f'site:linkedin.com/in/ "{company}" {role_or_department}'
    else:
        query = f'site:linkedin.com/in/ "{company}"'

    results = _execute_serpapi_search(query, num_results=10)
    if not results:
        query_fallback = f'"{company}" team members engineers professional profiles'
        results = _execute_serpapi_search(query_fallback, num_results=8)

    return json.dumps(results[:8], ensure_ascii=False)


@tool
def search_company_leadership(company: str) -> str:
    """Search for executive leadership, founders, C-suite, and management team members at a given company.
    Returns a JSON string of candidate profile results with title, snippet, and url.
    """
    query = f'site:linkedin.com/in/ "{company}" (founder OR CEO OR CTO OR VP OR Director OR "Head of")'
    results = _execute_serpapi_search(query, num_results=10)
    if not results:
        query_fallback = f'site:theorg.com/org/ "{company}" team'
        results = _execute_serpapi_search(query_fallback, num_results=8)

    return json.dumps(results[:8], ensure_ascii=False)

