import json
import re
from typing import Any, Generator, Optional
from urllib.parse import urlparse, quote

from langchain.agents import create_agent

from src.config import llm as default_llm
from src.prompts import AGENT_SYSTEM_PROMPT
from src.tools import (
    classify_seniority,
    parse_company_and_domain,
    search_linkedin_leadership,
    verify_company_google,
    verify_company_instagram,
)
from src.tools.search_tools import (
    NEGATIVE_AFFILIATION_PHRASES,
    REJECT_HEADLINE_PHRASES,
    _has_direct_company_affiliation,
    _has_rejected_company_affiliation,
)


def create_employee_agent(llm_instance=None):
    """Create an evidence-only extractor. SerpApi is called separately once per search."""
    return create_agent(
        model=llm_instance or default_llm,
        tools=[],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


agent = create_employee_agent()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _extract_json(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    candidates = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    candidates.append(cleaned)
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{[\s\S]*\}", candidate)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
    return None


def _text_contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    normalized_phrase = " ".join(re.findall(r"[a-z0-9]+", phrase.lower()))
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_text} "


def _role_is_supported(role: str, evidence_text: str) -> bool:
    normalized_role = " ".join(re.findall(r"[a-z0-9]+", role.lower()))
    normalized_evidence = " ".join(re.findall(r"[a-z0-9]+", evidence_text.lower()))
    if normalized_role and normalized_role in normalized_evidence:
        return True
    if any(k in normalized_role for k in ("manager", "architect", "lead", "director", "head", "vp", "chief", "founder", "ceo", "cto", "cfo", "coo", "president")):
        if any(k in normalized_evidence for k in ("manager", "architect", "lead", "director", "head", "vp", "chief", "founder", "ceo", "cto", "cfo", "coo", "officer", "president")):
            return True
    aliases = {
        "ceo": ("chief executive officer", "chief executive"),
        "cto": ("chief technology officer",),
        "cfo": ("chief financial officer",),
        "coo": ("chief operating officer",),
        "cmo": ("chief marketing officer",),
        "cpo": ("chief product officer",),
        "cio": ("chief information officer",),
        "vp": ("vice president",),
    }
    for short, longs in aliases.items():
        for long in longs:
            if (short in normalized_role and long in normalized_evidence) or (long in normalized_role and short in normalized_evidence):
                return True
    return False


def _find_exact_linkedin_profile(
    name: str,
    company: str,
    search_results: Optional[list[dict]] = None,
) -> Optional[str]:
    """Resolve the person's exact LinkedIn profile URL from already-fetched search results without consuming extra searches."""
    clean_name = name.lower().replace(".", "").strip()
    parts = [p for p in clean_name.split() if len(p) > 1]

    # Check already-fetched search results for an exact /in/ personal profile
    for item in (search_results or []):
        url = item.get("url", "")
        if "/in/" in url.lower():
            text = f"{item.get('title', '')} {item.get('snippet', '')}".lower().replace(".", "")
            if parts and all(p in text for p in parts):
                return url

    return None


def parse_agent_json(
    raw_text: str,
    fallback_company: str,
    search_results: Optional[list[dict]] = None,
) -> dict:
    """Parse model output and retain only profiles grounded in the actual search results."""
    data = _extract_json(raw_text)
    candidates = data.get("employees", []) if isinstance(data, dict) else []
    if not isinstance(candidates, list):
        candidates = []

    evidence_by_url = {
        item.get("url", ""): item
        for item in (search_results or [])
        if isinstance(item, dict) and item.get("url")
    }
    standardized = []
    seen_names = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("full_name") or "").strip()
        role = str(item.get("role") or item.get("job_title") or item.get("title") or "").strip()
        source_url = str(item.get("source_url") or item.get("url") or "").strip()
        if not name or not role or name.lower() in seen_names:
            continue

        evidence = evidence_by_url.get(source_url)
        if not evidence:
            clean_s = source_url.rstrip("/").split("?")[0].lower()
            for ev_url, ev_item in evidence_by_url.items():
                if ev_url.rstrip("/").split("?")[0].lower() == clean_s:
                    evidence = ev_item
                    break
        if not evidence:
            for ev_item in (search_results or []):
                if isinstance(ev_item, dict) and _text_contains_phrase(
                    f"{ev_item.get('title', '')} {ev_item.get('snippet', '')}", name
                ):
                    evidence = ev_item
                    break
        if not evidence:
            continue
        evidence_text = f"{evidence.get('title', '')} {evidence.get('snippet', '')}".lower()
        if any(term in evidence_text for term in REJECT_HEADLINE_PHRASES):
            continue
        if any(term in evidence_text for term in NEGATIVE_AFFILIATION_PHRASES):
            continue
        company_evidence_name = fallback_company
        if evidence.get("company_match") == "first-word fallback":
            company_evidence_name = fallback_company.split()[0]
        if _has_rejected_company_affiliation(evidence_text, company_evidence_name):
            continue
        is_roster = (
            evidence.get("current_employment_evidence") == "official current team listing"
            or "theorg.com" in evidence.get("source_domain", "")
            or "theorg.com" in source_url
        )
        if not is_roster and not _has_direct_company_affiliation(
            evidence_text, company_evidence_name
        ):
            continue
        if not _text_contains_phrase(evidence_text, name):
            continue
        if not _role_is_supported(role, evidence_text):
            continue
        if classify_seniority(role) == "Other":
            continue

        source_domain = evidence.get("source_domain") or (urlparse(source_url).hostname or "")
        seniority = classify_seniority(role)
        linkedin_url = source_url if "/in/" in source_url.lower() else ""
        if not linkedin_url:
            exact_profile = _find_exact_linkedin_profile(name, fallback_company, search_results)
            if exact_profile:
                linkedin_url = exact_profile
        if not linkedin_url:
            encoded_kw = quote(f"{name} {fallback_company}".strip())
            linkedin_url = f"https://www.linkedin.com/search/results/people/?keywords={encoded_kw}"

        display_profile = linkedin_url or source_url
        standardized.append(
            {
                "name": name,
                "role": role,
                "seniority": seniority,
                "linkedin_url": linkedin_url,
                "profile_url": display_profile,
                "source_domain": "linkedin.com" if "linkedin.com" in display_profile else source_domain,
                "source_url": display_profile,
                "source_snippet": evidence.get("snippet", "") or evidence.get("current_employment_evidence", ""),
                "current_employment_status": evidence.get("current_employment_status", "Presently working"),
                "employment_timeline": evidence.get("employment_timeline", "Present"),
            }
        )
        seen_names.add(name.lower())

    return {"company": fallback_company, "employees": standardized, "raw_output": raw_text}


def run_agent_stream(
    company_or_url: str,
    role_preference: Optional[str] = None,
    agent_instance=None,
) -> Generator[dict[str, Any], None, None]:
    """Verify the company with Google and Instagram, then search LinkedIn for current leaders."""
    company_name, domain = parse_company_and_domain(company_or_url)
    active_agent = agent_instance or agent
    verifications = []
    for tool_name, verify_tool in (
        ("verify_company_google", verify_company_google),
        ("verify_company_instagram", verify_company_instagram),
    ):
        yield {"type": "action", "tool": tool_name, "args": {"company": company_name}}
        try:
            verification = json.loads(verify_tool.invoke({"company_or_url": company_or_url}))
        except Exception as error:
            verification = {"verified": False, "source": tool_name, "error": f"Verification failed: {error}"}
        verifications.append(verification)
        yield {"type": "observation", "tool": tool_name}
        if verification.get("verified"):
            break

    verified = next((item for item in verifications if item.get("verified")), None)
    if not verified:
        verification_errors = "; ".join(item.get("error", "") for item in verifications if item.get("error"))
        yield {
            "type": "result",
            "data": {
                "company": company_name,
                "company_verified": False,
                "company_page_url": "",
                "employees": [],
                "raw_output": verification_errors or "Google and Instagram did not confirm an exact-name company presence.",
            },
            "raw": "",
        }
        return

    # Extract official company domain from verification or input
    discovered_domain = domain
    social_and_generic_domains = (
        "google", "instagram", "squarespace", "facebook", "twitter",
        "linkedin", "youtube", "tiktok", "gmail", "yahoo", "glassdoor",
        "ambitionbox", "zaubacorp", "naukri", "indeed", "builtin",
        "theorg", "crunchbase"
    )
    if discovered_domain and any(ex in discovered_domain.lower() for ex in social_and_generic_domains):
        discovered_domain = ""

    if verified:
        snip = verified.get("snippet", "")
        email_m = re.search(r"@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", snip)
        if email_m:
            cand = email_m.group(1).lower().removeprefix("www.")
            if not any(ex in cand for ex in social_and_generic_domains):
                discovered_domain = cand
        if not discovered_domain:
            comp_url = verified.get("company_url", "")
            if comp_url:
                parsed_c = urlparse(comp_url)
                cand = (parsed_c.hostname or "").lower().removeprefix("www.")
                if not any(ex in cand for ex in social_and_generic_domains):
                    discovered_domain = cand

    search_results = []
    search_errors = []
    seen_urls = set()
    for tool_name, search_tool in (("search_linkedin_leadership", search_linkedin_leadership),):
        yield {
            "type": "action",
            "tool": tool_name,
            "args": {"company": company_name},
        }
        try:
            search_data = json.loads(
                search_tool.invoke({
                    "company_or_url": company_or_url,
                    "domain": discovered_domain or "",
                    "role_filter": role_preference or "",
                })
            )
        except Exception as error:
            search_data = {"error": f"Search failed: {error}", "results": []}

        raw_results = search_data.get("results", []) if isinstance(search_data, dict) else []
        for item in raw_results:
            if isinstance(item, dict) and item.get("url") and item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                search_results.append(item)
        if search_data.get("error"):
            search_errors.append(str(search_data["error"]))
        for item in raw_results:
            if isinstance(item, dict) and item.get("error"):
                search_errors.append(str(item["error"]))
        yield {"type": "observation", "tool": tool_name}

    if not search_results:
        yield {
            "type": "result",
            "data": {
                "company": company_name,
                "company_verified": True,
                "company_page_url": verified.get("company_url", ""),
                "company_verification_source": verified.get("source", ""),
                "employees": [],
                "raw_output": " ".join(search_errors) or "The company was verified. LinkedIn search did not return indexed profiles with enough evidence of a current senior role at this company; this does not mean the company has no members.",
            },
            "raw": "",
        }
        return

    extraction_request = (
        f"Extract ALL qualifying current leaders, executives, founders, architects, and managers for {company_name!r}. "
        "Be thorough: extract every single qualifying individual mentioned in the search evidence. "
        "Do not stop after one or two people; include every qualifying leader and manager found across all snippets. "
        "Return the complete list in 'employees' in valid JSON. Search evidence:\n"
        + json.dumps(search_results, ensure_ascii=False)
    )
    final_message = ""
    try:
        response = active_agent.invoke(
            {"messages": [("user", extraction_request)]}
        )
        messages = response.get("messages", []) if isinstance(response, dict) else []
        if messages:
            final_message = _message_text(getattr(messages[-1], "content", ""))
    except Exception as error:
        err_msg = str(error)
        if "10061" in err_msg or "actively refused" in err_msg.lower():
            err_msg = (
                "Local Ollama server is not running on localhost:11434. "
                "Please start Ollama or run 'ollama serve' in a terminal. "
                "Alternatively, set LLM_PROVIDER=groq or openai in your .env file."
            )
        yield {
            "type": "result",
            "data": {
                "company": company_name,
                "employees": [],
                "raw_output": f"Profile extraction failed: {err_msg}",
            },
            "raw": "",
        }
        return

    parsed_result = parse_agent_json(final_message, company_name, search_results)
    parsed_result["company_verified"] = True
    parsed_result["company_page_url"] = verified.get("company_url", "")
    parsed_result["company_verification_source"] = verified.get("source", "")
    yield {"type": "result", "data": parsed_result, "raw": final_message}


def run_agent(company: str, role_preference: Optional[str] = None, agent_instance=None):
    """Run the leadership search from the command line."""
    print(f"Employee Finder search for: {company}")
    print("-" * 60)
    try:
        final_data = None
        for event in run_agent_stream(company, role_preference, agent_instance):
            if event["type"] == "action":
                print(f"Searching with {event['tool']}…")
            elif event["type"] == "result":
                final_data = event["data"]
        print(json.dumps(final_data or {"company": company, "employees": []}, indent=2))
    except KeyboardInterrupt:
        print("\nSearch cancelled.")
    except Exception as error:
        print(f"\nSearch failed: {error}")
