import json
import re
from typing import Any, Optional
from urllib.parse import quote, urlparse

from langchain_core.tools import tool
from serpapi import GoogleSearch

from src.config import EXCLUDED_DOMAINS, EXCLUDED_PATHS, SERPAPI_API_KEY


LEADERSHIP_TERMS = (
    "ceo", "chief", "founder", "co-founder", "president", "partner",
    "vice president", "vp", "director", "head of", "general manager",
    "senior manager", "architect", "lead", "manager",
)

REJECT_HEADLINE_PHRASES = (
    "immediate joiner", "immediate joining", "join immediately", "open to work",
    "opentowork", "serving notice", "serving notice period", "on notice",
    "notice period", "actively looking", "seeking opportunities",
    "seeking new opportunities", "looking for new", "looking for next",
    "looking for opportunities", "available immediately", "immediately available",
    "ready to join", "job seeker", "jobseeker", "former ", "ex-", "past:",
    "previously at", "previous ", "retired", "resigned", "alumni",
)

CURRENT_MARKER_RE = re.compile(
    r"\b(present|current(?:ly)?|to date|incumbent|our new|welcomes?|delighted to welcome|anniversary|celebrating|stepping in as)\b|"
    r"\b\d{4}\s*[-\u2013\u2014]\s*present\b|"
    r"\b(?:is|as)\s+(?:the\s+)?(?:ceo|cto|cfo|coo|founder|co-founder|president|vp|director|architect|head|lead|manager)\b|"
    r"\b(?:co-founder,\s*ceo|founder\s*&\s*ceo)\b|"
    r"\b(?:ceo|cto|cfo|founder|co-founder|president|vp|director|architect|lead|manager)\s+(?:at|of|with|for)\b|"
    r"\b\w+(?:'s)?\s+(?:ceo|cto|cfo|founder|co-founder|president|vp|director|architect|lead|manager)\b",
    re.IGNORECASE,
)
EMPLOYMENT_TIMELINE_RE = re.compile(
    r"((?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+)?(?:19|20)\d{2}\s*(?:[-\u2013\u2014]|\bto\b)\s*"
    r"(?:present|current|now))",
    re.IGNORECASE,
)

NEGATIVE_AFFILIATION_PHRASES = (
    "not directly affiliated with",
    "not affiliated with",
    "not associated with",
    "no affiliation with",
    "unrelated to",
    "does not work at",
    "doesn't work at",
    "not employed by",
    "not working at",
)


def parse_company_and_domain(input_text: str) -> tuple[str, str]:
    """Return the supplied company name or a readable name and domain from a URL."""
    clean_input = input_text.strip()
    if re.search(r"^(https?://|www\.)", clean_input, re.IGNORECASE) or re.search(
        r"\.[a-z]{2,}(/|$)", clean_input, re.IGNORECASE
    ):
        parsed = urlparse(
            clean_input if clean_input.startswith(("http://", "https://"))
            else f"https://{clean_input}"
        )
        domain = (parsed.hostname or "").lower().removeprefix("www.")
        company_name = domain.split(".")[0].replace("-", " ").title()
        return company_name, domain
    return clean_input, ""


def classify_seniority(role: str) -> str:
    role_lower = role.lower()
    if any(k in role_lower for k in ("ceo", "cto", "cfo", "coo", "cmo", "cpo", "cro", "cio", "chief")):
        return "C-Suite"
    if any(k in role_lower for k in ("founder", "co-founder", "president", "partner", "owner")):
        return "Founder / Executive"
    if any(k in role_lower for k in ("vice president", "vp", "director", "head of")):
        return "VP / Director"
    if any(k in role_lower for k in ("manager",)):
        return "Manager"
    if any(k in role_lower for k in ("lead", "architect", "principal", "staff")):
        return "Lead / Architect"
    return "Other"


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def clean_brand_name(company_name: str) -> str:
    """Strip generic corporate suffixes to obtain the core brand name (e.g. 'Carnera Technologies' -> 'Carnera')."""
    clean = re.sub(
        r"(?i)\b(technologies|technology|tech|inc\.?|llc\.?|corp\.?|corporation|ltd\.?|limited|pvt\.?|private|solutions|group|services|company|co\.?)\b",
        "",
        company_name,
    ).strip()
    return clean or company_name


def _contains_full_company_name(text: str, company_name: str) -> bool:
    normalized_full = _normalized_text(company_name)
    normalized_brand = _normalized_text(clean_brand_name(company_name))
    haystack = f" {_normalized_text(text)} "
    if normalized_full and f" {normalized_full} " in haystack:
        return True
    if normalized_brand and f" {normalized_brand} " in haystack:
        return True
    return False


def _has_direct_company_affiliation(text: str, company_name: str) -> bool:
    """Match common title and snippet layouts while requiring company affiliation."""
    if any(phrase in text.lower() for phrase in NEGATIVE_AFFILIATION_PHRASES):
        return False
    candidates = [re.escape(company_name.strip())]
    brand = clean_brand_name(company_name).strip()
    if brand and brand.lower() != company_name.strip().lower():
        candidates.append(re.escape(brand))

    boundary = r"(?=\b|[,|/\u00b7']|\s|$)"
    role = (
        r"(?:ceo|chief(?: executive| technology| financial| operating)? officer|"
        r"founder|co-founder|president|partner|vice president|vp|director|head of|"
        r"(?:general|senior|technical|project|product|engineering|delivery|operations|account|program)?\s*manager|"
        r"manager|architect|lead)"
    )

    for co in candidates:
        patterns = (
            rf"(?:\bat\s+|\bwith\s+|\bfor\s+|\bof\s+|@\s*){co}{boundary}",
            rf"\b{role}\b\s*(?:[-|/:\u2013\u2014\u00b7.,]\s*)+{co}{boundary}",
            rf"{co}(?:'s)?{boundary}\s*(?:[-|/:\u2013\u2014\u00b7.,]\s*)+\b{role}\b",
            rf"{co}(?:'s)?\s+\b{role}\b",
            rf"\bcurrent(?:ly)?\s+(?:company|employer)\s*:?\s*{co}{boundary}",
            rf"{co}\b.{{0,80}}?\b(?:delighted to welcome|welcome|appoints?|hires?|celebrating|announced|stepping in as)\b.{{0,60}}?\b{role}\b",
            rf"{co}\b.{{0,60}}?\b(?:work anniversary|celebrating)\b",
            rf"\b{role}\b.{{0,30}}?\b(?:at|with|for|in|@)\s+{co}{boundary}",
        )
        if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
            return True
    return False


def _has_rejected_company_affiliation(text: str, company_name: str) -> bool:
    """Reject former/resigned status only when it is tied to the searched company."""
    company = re.escape(company_name.strip())
    if not company:
        return False
    boundary = r"(?=\b|[,|/\u00b7])"
    patterns = (
        rf"\b(?:former|ex|previously|past)\b[^.;|,\u00b7]{{0,50}}?\b(?:at|with|for|of)\s+{company}{boundary}",
        rf"\b(?:resigned|retired|left)\b.{{0,50}}?\b(?:from|at)\s+{company}{boundary}",
        rf"\b{company}{boundary}.{{0,70}}?\b(?:former employee|no longer|resigned|retired|left the company)\b",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _employment_timeline(text: str) -> str:
    match = EMPLOYMENT_TIMELINE_RE.search(text)
    if not match:
        return "Present"
    timeline = re.sub(r"\s+", " ", match.group(1)).strip()
    return re.sub(r"\s*(?:-|\u2013|\u2014|to)\s*", " - ", timeline, flags=re.IGNORECASE)


def _company_verification_search(company_name: str, query: str, source: str) -> dict:
    """Return one exact-name Google or Instagram company identity match."""
    try:
        response = GoogleSearch({"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": 8}).get_dict()
    except Exception as error:
        return {"verified": False, "company": company_name, "source": source, "error": f"{source} verification failed: {error}"}
    if response.get("error"):
        return {"verified": False, "company": company_name, "source": source, "error": str(response["error"])}
    for item in response.get("organic_results", []):
        url = (item.get("link") or "").strip()
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        title, snippet = item.get("title", ""), item.get("snippet", "")
        if not _contains_full_company_name(f"{title} {snippet}", company_name):
            continue
        if source == "Instagram":
            if not (host == "instagram.com" or host.endswith(".instagram.com")) or len([p for p in parsed.path.split("/") if p]) != 1:
                continue
            if any(word in f"{title} {snippet}".lower() for word in ("post", "reel", "stories")):
                continue
        else:
            if any(domain in host for domain in ("linkedin.", "instagram.", "facebook.", "x.com", "twitter.", "youtube.", "tiktok.", "glassdoor.", "ambitionbox.", "zaubacorp.", "builtin.", "naukri.", "indeed.", "quora.")):
                continue
            if any(part in host for part in ("crunchbase", "theorg", "bloomberg", "zoominfo", "pitchbook", "myworkdayjobs")):
                continue
        return {"verified": True, "company": company_name, "company_url": url, "title": title, "snippet": snippet, "source": source, "query": query}
    return {"verified": False, "company": company_name, "source": source, "query": query, "error": f"No exact-name {source} company result was found."}


@tool
def verify_company_google(company_or_url: str) -> str:
    """Verify company identity using Google results only; does not search for employees."""
    company_name, _ = parse_company_and_domain(company_or_url)
    if not company_name:
        return json.dumps({"verified": False, "company": "", "error": "A company name is required."})
    query = f'"{company_name.replace(chr(34), " ").strip()}" company official website'
    return json.dumps(_company_verification_search(company_name, query, "Google"), ensure_ascii=False)


@tool
def verify_company_instagram(company_or_url: str) -> str:
    """Verify company identity from an exact-name Instagram profile only."""
    company_name, _ = parse_company_and_domain(company_or_url)
    if not company_name:
        return json.dumps({"verified": False, "company": "", "error": "A company name is required."})
    query = f'site:instagram.com "{company_name.replace(chr(34), " ").strip()}" company'
    return json.dumps(_company_verification_search(company_name, query, "Instagram"), ensure_ascii=False)


verify_company_linkedin = verify_company_google


def _is_leadership_result(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in LEADERSHIP_TERMS):
        return True
    return False


def _execute_serpapi_search(
    query: str,
    company_name: str,
    num_results: int = 10,
    allowed_domain: Optional[str] = None,
    allow_general_web: bool = False,
    instagram_only: bool = False,
    company_match: Optional[str] = None,
) -> list[dict]:
    """Make one SerpApi request and retain only exact-company, current leadership evidence."""
    try:
        response = GoogleSearch(
            {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": num_results,
            }
        ).get_dict()
    except Exception as error:
        return [{"error": f"Search API failed: {error}"}]

    if response.get("error"):
        return [{"error": response["error"]}]

    results = []
    seen_urls = set()
    clean_domain = (allowed_domain or "").lower().removeprefix("www.")
    for item in response.get("organic_results", []):
        url = (item.get("link") or "").strip()
        if not url or url in seen_urls:
            continue
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        is_instagram = host == "instagram.com" or host.endswith(".instagram.com")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        evidence_text = f"{title} {snippet}"
        lower_evidence = evidence_text.lower()

        if any(phrase in lower_evidence for phrase in REJECT_HEADLINE_PHRASES):
            continue
        if any(
            domain in host and not (is_instagram and domain == "instagram.com")
            for domain in EXCLUDED_DOMAINS
        ):
            continue
        if any(path in parsed.path.lower() for path in EXCLUDED_PATHS):
            continue

        is_linkedin = host == "linkedin.com" or host.endswith(".linkedin.com")
        is_company_site = bool(clean_domain) and (
            host == clean_domain or host.endswith(f".{clean_domain}")
        )
        is_theorg = host == "theorg.com" or host.endswith(".theorg.com")
        if instagram_only and not is_instagram:
            continue
        if not (is_linkedin or is_company_site or is_theorg or allow_general_web):
            continue
        if is_linkedin and not any(part in parsed.path.lower() for part in ("/in/", "/posts/", "/company/")):
            continue
        if is_instagram and any(
            segment in parsed.path.lower().split("/")
            for segment in ("p", "reel", "reels", "stories", "explore")
        ):
            continue
        if host == "crunchbase.com" or host.endswith(".crunchbase.com"):
            continue

        # Match the complete supplied company phrase, not a partial company-name token.
        matched_company = company_match or company_name
        if not _contains_full_company_name(evidence_text, matched_company):
            continue
        if _has_rejected_company_affiliation(evidence_text, matched_company):
            continue
        is_roster_page = is_company_site or (is_theorg and "/org/" in parsed.path.lower())
        if not is_roster_page and not _has_direct_company_affiliation(evidence_text, matched_company):
            continue
        if not _is_leadership_result(evidence_text):
            continue

        # A current team page is current evidence. LinkedIn results must show a current marker.
        if not is_roster_page and not CURRENT_MARKER_RE.search(evidence_text):
            continue

        seen_urls.add(url)
        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
                "source_domain": host,
                "current_employment_evidence": "official current team listing" if is_roster_page else "current marker in search result",
                "employment_timeline": _employment_timeline(evidence_text),
                "current_employment_status": "Presently working",
                "company_match": "first-word fallback" if company_match else "full company name",
            }
        )
    return results


def _normalize_role_filter(role_filter: Optional[Any] = None) -> str:
    """Keep the agent tool signature compatible while limiting results to senior roles."""
    if not role_filter:
        return '(CEO OR CTO OR Founder OR President OR VP OR Director OR Lead OR Manager)'
    if isinstance(role_filter, str):
        return role_filter
    if isinstance(role_filter, (list, tuple)):
        terms = [str(item).strip() for item in role_filter if item]
        if terms:
            return "(" + " OR ".join(f'"{term}"' for term in terms) + ")"
    return '(CEO OR CTO OR Founder OR President OR VP OR Director OR Lead OR Manager)'


@tool
def search_linkedin_leadership(
    company_or_url: Optional[str] = None,
    company: Optional[str] = None,
    role_filter: Optional[Any] = None,
    domain: Optional[str] = None,
) -> str:
    """Search LinkedIn profiles and verified company sources for current senior leaders."""
    target = company or company_or_url or ""
    company_name, parsed_domain = parse_company_and_domain(target)
    active_domain = domain or parsed_domain
    if active_domain and any(ex in active_domain.lower() for ex in ("linkedin", "instagram", "facebook", "twitter", "glassdoor", "naukri", "theorg", "crunchbase")):
        active_domain = None
    if not company_name:
        return json.dumps({"error": "A company name or domain is required.", "results": []})

    safe_name = company_name.replace('"', " ").strip().title()
    brand = clean_brand_name(safe_name).title()
    role_query = _normalize_role_filter(role_filter)
    query = f'site:linkedin.com "{safe_name}" {role_query}'
    results = _execute_serpapi_search(
        query,
        company_name=company_name,
        num_results=20,
    )
    exact_matches = [item for item in results if isinstance(item, dict) and item.get("url")]

    used_fallback = False
    fallback_query = ""
    if len(exact_matches) < 3:
        if active_domain:
            fallback_query = f'"{brand}" (CEO OR CTO OR Founder OR President OR Director OR VP) "{active_domain}"'
            exec_results = _execute_serpapi_search(
                fallback_query,
                company_name=company_name,
                num_results=15,
                company_match=brand,
            )
            seen = {item["url"] for item in exact_matches}
            for item in exec_results:
                if isinstance(item, dict) and item.get("url") and item["url"] not in seen:
                    exact_matches.append(item)
                    seen.add(item["url"])

            mgr_query = f'"{brand}" (Manager OR Lead OR Architect OR Head) "{active_domain}"'
            mgr_results = _execute_serpapi_search(
                mgr_query,
                company_name=company_name,
                num_results=15,
                company_match=brand,
            )
            for item in mgr_results:
                if isinstance(item, dict) and item.get("url") and item["url"] not in seen:
                    exact_matches.append(item)
                    seen.add(item["url"])

            if len(exact_matches) < 3:
                li_query = f'site:linkedin.com "{brand}" (CEO OR CTO OR Founder OR President OR Director OR Manager OR Lead)'
                li_results = _execute_serpapi_search(
                    li_query,
                    company_name=company_name,
                    num_results=20,
                    company_match=brand,
                )
                for item in li_results:
                    if isinstance(item, dict) and item.get("url") and item["url"] not in seen:
                        exact_matches.append(item)
                        seen.add(item["url"])
        else:
            fallback_query = f'site:linkedin.com "{brand}" (CEO OR CTO OR Founder OR President OR Director OR Manager OR Lead)'
            fallback_results = _execute_serpapi_search(
                fallback_query,
                company_name=company_name,
                num_results=20,
                company_match=brand,
            )
            seen = {item["url"] for item in exact_matches}
            for item in fallback_results:
                if isinstance(item, dict) and item.get("url") and item["url"] not in seen:
                    exact_matches.append(item)
                    seen.add(item["url"])
        used_fallback = True

    return json.dumps(
        {
            "query": query,
            "fallback_query": fallback_query,
            "used_first_word_fallback": used_fallback,
            "results": exact_matches[:20],
        },
        ensure_ascii=False,
    )


@tool
def search_company_website_leadership(
    company_or_url: Optional[str] = None,
    company: Optional[str] = None,
) -> str:
    """Search the supplied official company domain for its current leadership, using one SerpApi request."""
    target = company or company_or_url or ""
    company_name, domain = parse_company_and_domain(target)
    if not domain:
        return json.dumps({"error": "Provide the official company website URL to search this source without a domain lookup.", "results": []})
    query = (
        f'site:{domain} "{company_name}" '
        '(leadership OR "executive team" OR "management team" OR "our team")'
    )
    results = _execute_serpapi_search(
        query,
        company_name=company_name,
        num_results=10,
        allowed_domain=domain,
    )
    return json.dumps({"query": query, "results": results[:8]}, ensure_ascii=False)


@tool
def find_member_linkedin(
    name: str,
    company: Optional[str] = None,
    company_or_url: Optional[str] = None,
) -> str:
    """Create a LinkedIn people-search link without consuming SerpApi searches."""
    target = company or company_or_url or ""
    comp_name, _ = parse_company_and_domain(target) if target else ("", "")
    encoded = quote(f"{name} {comp_name}".strip())
    return json.dumps({"name": name, "linkedin_url": f"https://www.linkedin.com/search/results/people/?keywords={encoded}"})


# Compatibility aliases
search_google_leadership = search_linkedin_leadership
search_instagram_leadership = verify_company_instagram
search_leadership_sources = search_linkedin_leadership
search_directories_and_linkedin = search_linkedin_leadership
search_company_employees = search_linkedin_leadership
search_company_leadership = search_company_website_leadership
