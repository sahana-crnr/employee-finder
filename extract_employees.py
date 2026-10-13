
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# =====================================================
# 1. Configuration
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY is missing from .env")

MODEL_NAME = "llama3.2"


# =====================================================
# 2. Initialize LangChain and Ollama
# =====================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    format="json",
)

prompt = ChatPromptTemplate.from_template("""
You extract publicly available professional information.

Company being searched: {company}

Search results:
{search_results}

Return a JSON object in this format:
{{
  "employees": [
    {{
      "source_index": 0,
      "name": "Full name",
      "job_title": "Professional role",
      "company": "Company association"
    }}
  ]
}}

Rules:
1. Extract identifiable individual people only.
2. Use only facts supported by the supplied title and snippet.
3. Include people whose association with the requested company
   is supported by the supplied information.
4. Exclude job listings, reviews, salary pages, generic company
   pages, and unrelated articles.
5. Do not invent names, job titles, or employment details.
6. Use the exact source_index of the supplied search result.
7. Use an empty string if a job title is unavailable.
8. If no relevant people are found, return:
   {{"employees": []}}
9. Return valid JSON only. Do not include Markdown.
""")

chain = prompt | llm


# =====================================================
# 3. Search public information
# =====================================================

def search_profiles(company):
    queries = [
        f'site:linkedin.com/in/ "{company}"',
        f'"{company}" management team people names roles',
        f'"{company}" engineers team members professional profiles',
        f'site:theorg.com/org/ "{company}" team',
        f'site:rocketreach.co "{company}" management team',
    ]

    results_by_url = {}

    excluded_domains = (
        "indeed.com",
        "glassdoor.",
        "youtube.com",
        "facebook.com",
        "instagram.com",
        "reddit.com",
        "scribd.com",
        "ziprecruiter.com",
    )

    excluded_paths = (
        "/jobs",
        "/reviews",
        "/salary",
        "/benefits",
        "/community/",
        "/careers",
    )

    for query in queries:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 10,
        }

        try:
            response = GoogleSearch(params).get_dict()
        except Exception:
            continue

        if response.get("error"):
            continue

        for item in response.get("organic_results", []):
            url = item.get("link", "").strip()

            if not url or url in results_by_url:
                continue

            lower_url = url.lower()

            if any(domain in lower_url for domain in excluded_domains):
                continue

            if any(path in lower_url for path in excluded_paths):
                continue

            results_by_url[url] = {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": url,
            }

    return list(results_by_url.values())


# =====================================================
# 4. Extract professional information
# =====================================================

def extract_employees(company, search_results):
    if not search_results:
        return []

    # Remove obvious irrelevant results before calling the model.
    excluded_phrases = (
        "employee reviews",
        "salary",
        "benefits",
        "job openings",
        "jobs in india",
        "company profile",
        "acquires ",
    )

    candidates = []

    for result in search_results:
        text = (
            result.get("title", "") + " " +
            result.get("snippet", "")
        ).lower()

        if any(phrase in text for phrase in excluded_phrases):
            continue

        candidates.append(result)

    # Limit input to reduce local model processing time.
    candidates = candidates[:10]

    if not candidates:
        return []

    indexed_results = [
        {
            "source_index": index,
            "title": result["title"],
            "snippet": result["snippet"],
        }
        for index, result in enumerate(candidates)
    ]

    response = chain.invoke({
        "company": company,
        "search_results": json.dumps(
            indexed_results,
            ensure_ascii=False,
        ),
    })

    try:
        data = json.loads(response.content)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Ollama returned invalid JSON.") from error

    if not isinstance(data, dict):
        raise ValueError("Unexpected response format from Ollama.")

    employees = data.get("employees", [])

    if not isinstance(employees, list):
        raise ValueError("The employees field must be a list.")

    validated = []
    seen_names = set()

    for person in employees:
        if not isinstance(person, dict):
            continue

        index = person.get("source_index")

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(candidates)
        ):
            continue

        name = person.get("name")
        role = person.get("job_title", "")
        company_association = person.get("company")

        if not isinstance(name, str) or not name.strip():
            continue

        if (
            not isinstance(company_association, str)
            or not company_association.strip()
        ):
            continue

        name = name.strip()
        unique_key = name.casefold()

        if unique_key in seen_names:
            continue

        source = candidates[index]

        validated.append({
            "name": name,
            "job_title": (
                role.strip()
                if isinstance(role, str)
                else ""
            ),
            "company": company_association.strip(),
            "source_title": source["title"],
            "source_url": source["url"],
            "source_snippet": source["snippet"],
        })

        seen_names.add(unique_key)

    return validated


# =====================================================
# 5. Display clean results
# =====================================================

def display_results(company, employees):
    print(f"\nPublic professional information: {company}")
    print("=" * 55)

    if not employees:
        print("No sufficiently supported individual profiles found.")
        print("Try another company name or check public search sources.")
        return

    for number, person in enumerate(employees, start=1):
        print(f"\nPerson {number}")
        print("-" * 35)
        print("Name:", person["name"])
        print("Role:", person["job_title"] or "Not available")
        print("Company association:", person["company"])
        print("Source:", person["source_url"])

    print("\n" + "=" * 55)
    print("Candidate profiles found:", len(employees))
    print("Verify source links to confirm current employment.")

# 6. Main


def main():
    company = input("Enter a company name: ").strip()

    if not company:
        print("Please enter a valid company name.")
        return

    try:
        search_results = search_profiles(company)

        if not search_results:
            print("No public search results found.")
            return

        employees = extract_employees(company, search_results)
        display_results(company, employees)

    except KeyboardInterrupt:
        print("\nSearch cancelled.")

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
