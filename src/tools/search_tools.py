import json
import re
from typing import Optional
from urllib.parse import urlparse
from serpapi import GoogleSearch
from langchain_core.tools import tool

from src.config import SERPAPI_API_KEY, EXCLUDED_DOMAINS, EXCLUDED_PATHS


def parse_company_and_domain(input_text: str) -> tuple[str, str]:
    """Extracts a clean company name and domain from either a URL or raw text.
    
    Examples:
      'https://www.twelve.co/about' -> ('twelve', 'twelve.co')
      'twelve.co'                   -> ('twelve', 'twelve.co')
      'Stripe'                      -> ('Stripe', '')
    """
    clean_input = input_text.strip()
    
    # Check if input looks like a URL or domain
    if re.search(r'^(https?://|www\.)', clean_input) or re.search(r'\.[a-z]{2,}(/|$)', clean_input, re.I):
        if not clean_input.startswith(('http://', 'https://')):
            clean_input = 'https://' + clean_input
        parsed = urlparse(clean_input)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Derive readable name from domain (e.g. 'twelve.co' -> 'Twelve')
        name_part = domain.split('.')[0]
        company_name = name_part.capitalize()
        return company_name, domain

    # Plain company name provided
    return clean_input, ""


def classify_seniority(role: str) -> str:
    """Classifies a job title into a standard leadership seniority tier."""
    role_lower = role.lower()
    
    if any(k in role_lower for k in ("ceo", "cto", "cfo", "coo", "cmo", "cpo", "cro", "cio", "chief")):
        return "C-Suite"
    if any(k in role_lower for k in ("founder", "co-founder", "president", "partner", "owner")):
        return "Founder / Executive"
    if any(k in role_lower for k in ("vice president", "vp", "director", "head of")):
        return "VP / Director"
    if any(k in role_lower for k in ("lead", "manager", "principal", "architect", "staff")):
        return "Lead / Manager"
    
    return "Senior Leadership"


def _execute_serpapi_search(query: str, num_results: int = 8) -> list[dict]:
    """Helper to query SerpApi Google search and filter irrelevant sites."""
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
def search_company_website_leadership(company_or_url: str) -> str:
    """Searches the official company website and public corporate pages for executive team members.
    Works with either a company name or website URL (e.g. 'twelve.co' or 'Stripe').
    Queries team rosters, leadership, and 'about us' pages on the target domain.
    """
    company_name, domain = parse_company_and_domain(company_or_url)
    results = []
    
    if domain:
        query_domain = f'site:{domain} (team OR leadership OR executives OR founders OR "about us" OR management)'
        results.extend(_execute_serpapi_search(query_domain, num_results=6))
    
    query_web = f'"{company_name}" ("executive team" OR "leadership team" OR "board of directors" OR "management team")'
    results.extend(_execute_serpapi_search(query_web, num_results=6))
    
    # Deduplicate by URL
    seen = set()
    clean_results = []
    for r in results:
        if r.get("url") and r["url"] not in seen:
            seen.add(r["url"])
            clean_results.append(r)

    return json.dumps(clean_results[:8], ensure_ascii=False)


@tool
def search_directories_and_linkedin(company_or_url: str, role_filter: Optional[str] = None) -> str:
    """Searches executive directories (The Org, Crunchbase, RocketReach) and professional profiles (LinkedIn)
    specifically for higher-level leadership roles (CEO, CTO, CFO, Founders, VPs, Directors, Leads, Managers).
    Accepts company name or URL.
    """
    company_name, _ = parse_company_and_domain(company_or_url)
    
    role_query = role_filter if role_filter else '(CEO OR CTO OR CFO OR COO OR "Vice President" OR VP OR Director OR "Head of" OR Founder OR "Co-Founder" OR Lead OR Manager)'
    
    queries = [
        # LinkedIn high-level roles
        f'site:linkedin.com/in/ "{company_name}" {role_query}',
        # The Org leadership chart
        f'site:theorg.com/org/ "{company_name}"',
        # Crunchbase people
        f'site:crunchbase.com/person "{company_name}"',
        # RocketReach management
        f'site:rocketreach.co "{company_name}" (leadership OR executive OR management)',
    ]
    
    all_results = []
    seen = set()
    
    for q in queries:
        batch = _execute_serpapi_search(q, num_results=5)
        for r in batch:
            if r.get("url") and r["url"] not in seen:
                seen.add(r["url"])
                all_results.append(r)
        if len(all_results) >= 12:
            break

    return json.dumps(all_results[:10], ensure_ascii=False)


# Legacy backward compatibility aliases
search_company_employees = search_directories_and_linkedin
search_company_leadership = search_company_website_leadership
