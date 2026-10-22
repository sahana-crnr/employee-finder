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
    clean_brand_name,
    is_disqualified_as_former_or_left,
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
    role_tokens = [t for t in normalized_role.split() if len(t) > 2]
    if role_tokens and any(t in normalized_evidence for t in role_tokens):
        return True
    if any(k in normalized_role for k in ("manager", "architect", "lead", "director", "head", "vp", "chief", "founder", "ceo", "cto", "cfo", "coo", "president", "engineer", "developer", "specialist", "analyst", "consultant", "sales", "talent", "recruiter", "builder")):
        if any(k in normalized_evidence for k in ("manager", "architect", "lead", "director", "head", "vp", "chief", "founder", "ceo", "cto", "cfo", "coo", "officer", "president", "engineer", "developer", "specialist", "analyst", "consultant", "sales", "talent", "recruiter", "builder")):
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


def _normalize_person_key(name: str) -> str:
    """Normalize names for robust deduplication (e.g. 'Nick AJ', 'Nick A.', 'Nick A' -> 'nick a')."""
    clean = re.sub(r"[^a-z0-9\s]", "", name.lower())
    tokens = [t for t in clean.split() if t]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    first = tokens[0]
    last_initial = tokens[1][0]
    return f"{first} {last_initial}"


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


def extract_person_from_result(item: dict, company_name: str) -> tuple[str, str]:
    """Extract person name and clean role from a verified search result title and snippet."""
    title = item.get("title", "")
    snippet = item.get("snippet", "")
    clean = re.sub(r"\s*[-|]\s*linkedin.*$", "", title, flags=re.IGNORECASE).strip()
    parts = re.split(r"\s*[-–—:]\s*", clean, maxsplit=1)
    if not parts or not parts[0].strip():
        return "", ""
    name = parts[0].strip()
    brand_clean = clean_brand_name(company_name)
    role = ""
    if len(parts) >= 2:
        raw_role = parts[1].strip()
        role = re.sub(
            rf"\s*(?:at|@|with|for)\s+(?:{re.escape(company_name)}|{re.escape(brand_clean)}).*$",
            "",
            raw_role,
            flags=re.IGNORECASE,
        ).strip()
        role = re.sub(
            rf"^(?:{re.escape(company_name)}|{re.escape(brand_clean)})\b.*$",
            "",
            role,
            flags=re.IGNORECASE,
        ).strip()

        # Handle multiple titles separated by | (e.g. Co-Founder | CEO or Founder | Builder)
        if " | " in role:
            sub_parts = [p.strip() for p in role.split(" | ") if p.strip()]
            title_tokens = ("ceo", "cto", "cfo", "coo", "founder", "co-founder", "builder", "director", "manager", "lead", "architect", "engineer", "developer", "president")
            kept = [p for p in sub_parts if any(t in p.lower() for t in title_tokens)]
            if len(kept) > 1:
                role = " & ".join(kept)
            elif kept:
                role = kept[0]
            elif sub_parts:
                role = sub_parts[0]

    if not role or role.lower() in (company_name.lower(), brand_clean.lower()):
        role_m = re.search(
            r"\b(?:lead\s+data\s+engineer|data\s+engineer|software\s+engineer\s+ii|software\s+engineer|associate\s+engineering\s+manager|engineering\s+manager|technical\s+project\s+manager|project\s+manager|product\s+manager|technical\s+lead|solution\s+architect|cto|ceo|founder|co-founder|inside\s+sales|talent\s+acquisition|account\s+manager|sales\s+director|consultant|specialist)\b",
            snippet,
            re.IGNORECASE,
        )
        if role_m:
            role = role_m.group(0).title()
        else:
            spear_m = re.search(r"\bspearhead\s+initiatives\s+in\s+([A-Za-z0-9\s,\&]+?)(?:,?\s+enabling|\.|$)", snippet, re.IGNORECASE)
            if spear_m:
                role = "Lead - " + spear_m.group(1).strip()
            else:
                role = "Team Member"
    return name, role


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
    seen_keys = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("full_name") or "").strip()
        role = str(item.get("role") or item.get("job_title") or item.get("title") or "").strip()
        source_url = str(item.get("source_url") or item.get("url") or "").strip()
        if not name or not role:
            continue

        person_key = _normalize_person_key(name)
        if not person_key:
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
        disqualified, _ = is_disqualified_as_former_or_left(
            evidence.get("title", ""),
            evidence.get("snippet", ""),
            company_evidence_name,
        )
        if disqualified:
            continue
        if _has_rejected_company_affiliation(evidence_text, company_evidence_name):
            continue
        is_roster = (
            evidence.get("current_employment_evidence") == "official current team listing"
            or any(d in evidence.get("source_domain", "") or d in source_url for d in ("theorg.com", "crunchbase.com", "builtin.com", "pitchbook.com"))
        )
        if not is_roster and not _has_direct_company_affiliation(
            evidence_text, company_evidence_name
        ):
            continue
        if not _text_contains_phrase(evidence_text, name):
            continue
        if not _role_is_supported(role, evidence_text):
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
        record = {
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

        if person_key in seen_keys:
            existing_idx = seen_keys[person_key]
            existing_item = standardized[existing_idx]
            # Upgrade existing if current record has a direct /in/ profile URL and existing does not
            if "/in/" in display_profile.lower() and "/in/" not in existing_item.get("profile_url", "").lower():
                standardized[existing_idx] = record
            elif len(role) > len(existing_item.get("role", "")):
                existing_item["role"] = role
                existing_item["seniority"] = seniority
            continue

        seen_keys[person_key] = len(standardized)
        standardized.append(record)

    # Reconciliation pass: add any verified current employees/managers from search_results omitted by LLM
    for ev_item in (search_results or []):
        if not isinstance(ev_item, dict):
            continue
        ev_title = ev_item.get("title", "")
        ev_snippet = ev_item.get("snippet", "")
        ev_url = ev_item.get("url", "")
        if not ev_title or not ev_url:
            continue

        company_evidence_name = fallback_company
        if ev_item.get("company_match") == "first-word fallback":
            company_evidence_name = fallback_company.split()[0]

        disqualified, _ = is_disqualified_as_former_or_left(
            ev_title, ev_snippet, company_evidence_name
        )
        if disqualified:
            continue

        is_roster = (
            ev_item.get("current_employment_evidence") == "official current team listing"
            or any(d in ev_item.get("source_domain", "") or d in ev_url for d in ("theorg.com", "crunchbase.com", "builtin.com", "pitchbook.com"))
        )
        if not is_roster and not _has_direct_company_affiliation(
            f"{ev_title} {ev_snippet}", company_evidence_name
        ):
            continue

        cand_name, cand_role = extract_person_from_result(ev_item, fallback_company)
        if not cand_name:
            continue
        cand_key = _normalize_person_key(cand_name)
        if not cand_key or cand_key in seen_keys:
            continue

        cand_seniority = classify_seniority(cand_role)
        display_profile = ev_url
        rec = {
            "name": cand_name,
            "role": cand_role,
            "seniority": cand_seniority,
            "linkedin_url": display_profile if "/in/" in display_profile.lower() else "",
            "profile_url": display_profile,
            "source_domain": "linkedin.com" if "linkedin.com" in display_profile else ev_item.get("source_domain", ""),
            "source_url": display_profile,
            "source_snippet": ev_snippet or ev_item.get("current_employment_evidence", ""),
            "current_employment_status": ev_item.get("current_employment_status", "Presently working"),
            "employment_timeline": ev_item.get("employment_timeline", "Present"),
        }
        seen_keys[cand_key] = len(standardized)
        standardized.append(rec)

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
                "raw_output": " ".join(search_errors) or "The company was verified. LinkedIn search did not return indexed profiles with enough evidence of current employment at this company; this does not mean the company has no members.",
            },
            "raw": "",
        }
        return

    extraction_request = (
        f"Extract EVERY qualifying current employee, leader, executive, founder, architect, engineer, and team member for {company_name!r}. "
        "IMPORTANT: A single search snippet or team directory often contains multiple people (e.g. both a CEO and a Founder). "
        "Carefully examine every snippet and extract each and every distinct qualifying individual mentioned across all snippets without omitting anyone. "
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
