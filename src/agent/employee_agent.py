import json
from typing import Optional, Generator, Any
from langchain.agents import create_agent

from src.config import llm as default_llm
from src.prompts import AGENT_SYSTEM_PROMPT
from src.tools import (
    search_company_website_leadership,
    search_directories_and_linkedin,
    classify_seniority,
    parse_company_and_domain,
)


def create_employee_agent(llm_instance=None):
    """Initializes and returns a compiled LangChain AI Agent.
    Accepts an optional custom LLM (e.g. from Streamlit model selector).
    """
    model = llm_instance or default_llm
    tools = [search_company_website_leadership, search_directories_and_linkedin]
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


# Default agent instance
agent = create_employee_agent()


import re
from urllib.parse import urlparse


def _extract_profiles_from_text_fallback(raw_text: str, fallback_company: str) -> list[dict]:
    """Fallback heuristic extractor that parses bullet points and text lines when LLM outputs markdown instead of JSON."""
    profiles = []
    lines = raw_text.splitlines()
    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("#") or "{" in line_s or "}" in line_s:
            continue

        # Match markdown bullets e.g. - **Name**: Role (URL) or - Name - Role - URL
        m1 = re.search(r"[-*•]\s*\**([A-Z][a-zA-Z\.\s]+?)\**\s*(?::|[-–—]|\|)\s*([^(\n]+?)(?:\s*\((https?://[^\s)]+)\)|\s*[-–—|]\s*(https?://\S+)|$)", line_s)
        if m1:
            name = m1.group(1).strip()
            role = m1.group(2).strip()
            url = m1.group(3) or m1.group(4) or ""
            if len(name.split()) <= 4 and not any(kw in name.lower() for kw in ("company", "leadership", "executive", "verified", "source", "http", "note")):
                profiles.append({
                    "name": name,
                    "role": role,
                    "seniority": classify_seniority(role),
                    "source_url": url,
                })
                continue

        # Match numbered list e.g. 1. Name - Role (URL)
        m2 = re.search(r"^\d+\.\s*\**([A-Z][a-zA-Z\.\s]+?)\**\s*(?::|[-–—]|\|)\s*([^(\n]+?)(?:\s*\((https?://[^\s)]+)\)|$)", line_s)
        if m2:
            name = m2.group(1).strip()
            role = m2.group(2).strip()
            url = m2.group(3) or ""
            if len(name.split()) <= 4 and not any(kw in name.lower() for kw in ("company", "leadership", "executive", "verified", "source")):
                profiles.append({
                    "name": name,
                    "role": role,
                    "seniority": classify_seniority(role),
                    "source_url": url,
                })

    return profiles


def parse_agent_json(raw_text: str, fallback_company: str) -> dict:
    """Extracts and validates JSON from LLM response text with multiple resilient fallbacks."""
    if not raw_text:
        return {"company": fallback_company, "employees": []}

    cleaned = raw_text.strip()
    data = None

    # Step 1: Look for markdown code fence blocks (```json ... ``` or ``` ... ```)
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    candidates = fences + [cleaned]

    for cand in candidates:
        c_str = cand.strip()
        if not c_str:
            continue
        try:
            data = json.loads(c_str)
            break
        except Exception:
            pass
        # Try regex extract of outermost {...} or [...]
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", c_str)
        if m:
            try:
                data = json.loads(m.group(1))
                break
            except Exception:
                pass

    employees = []
    if isinstance(data, dict):
        for key in ("employees", "leadership", "executives", "profiles", "team", "people"):
            if key in data and isinstance(data[key], list):
                employees = data[key]
                break
        if not employees:
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    employees = v
                    break
    elif isinstance(data, list):
        employees = data

    # Step 2: Fallback to heuristic text extraction if JSON yielded no employees
    if not employees:
        fallback_emps = _extract_profiles_from_text_fallback(cleaned, fallback_company)
        if fallback_emps:
            employees = fallback_emps

    # Standardize and classify each employee profile
    standardized = []
    seen_names = set()
    for emp in employees:
        if not isinstance(emp, dict):
            continue
        name = emp.get("name") or emp.get("full_name") or ""
        name = str(name).strip()
        if not name or name.lower() in seen_names:
            continue

        role = emp.get("role") or emp.get("job_title") or emp.get("title") or ""
        role = str(role).strip()
        seniority = emp.get("seniority") or classify_seniority(role)
        source_url = emp.get("source_url") or emp.get("url") or emp.get("link") or ""

        source_domain = emp.get("source_domain") or ""
        if not source_domain and source_url:
            netloc = urlparse(source_url).netloc.lower()
            source_domain = netloc.replace("www.", "")

        seen_names.add(name.lower())
        standardized.append({
            "name": name,
            "role": role,
            "seniority": seniority,
            "source_domain": source_domain,
            "source_url": source_url,
        })

    return {
        "company": fallback_company,
        "employees": standardized,
        "raw_output": raw_text,
    }


def run_agent_stream(
    company_or_url: str,
    role_preference: Optional[str] = None,
    agent_instance=None,
) -> Generator[dict[str, Any], None, None]:
    """Generator yielding real-time agent execution events for Streamlit UI.
    
    Yields dicts with:
      {"type": "action", "tool": str, "args": dict}
      {"type": "observation", "tool": str}
      {"type": "result", "data": dict, "raw": str}
    """
    active_agent = agent_instance or agent
    company_name, domain = parse_company_and_domain(company_or_url)
    
    user_query = (
        f"Find key leadership, C-suite, founders, directors, and managers for the company '{company_name}'"
    )
    if domain:
        user_query += f" with official website domain '{domain}'"
    if role_preference:
        user_query += f". Target role focus: '{role_preference}'"
    user_query += "."

    inputs = {"messages": [("user", user_query)]}
    final_message = ""

    for chunk in active_agent.stream(inputs, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {
                            "type": "action",
                            "tool": tc.get("name", "Tool"),
                            "args": tc.get("args", {}),
                        }
                elif hasattr(msg, "name") and msg.name:
                    yield {
                        "type": "observation",
                        "tool": msg.name,
                    }
                elif hasattr(msg, "content") and msg.content:
                    final_message = msg.content

    parsed_result = parse_agent_json(final_message, company_name)
    yield {
        "type": "result",
        "data": parsed_result,
        "raw": final_message,
    }


def run_agent(company: str, role_preference: Optional[str] = None, agent_instance=None):
    """Executes the agent in CLI mode and prints streaming updates and JSON report."""
    print(f"LangChain AI Agent activated for: {company}")
    print("-" * 60)

    try:
        final_data = None
        for event in run_agent_stream(company, role_preference, agent_instance):
            if event["type"] == "action":
                print(f"\n[Agent Action] Calling Tool: {event['tool']}")
                print(f"  Arguments: {event['args']}")
            elif event["type"] == "observation":
                print(f"[Tool Observation] Completed call for: {event['tool']}")
            elif event["type"] == "result":
                final_data = event["data"]

        print("\nFinal Extracted Employee Information (JSON)")
        print("-" * 60)
        if final_data:
            print(json.dumps(final_data, indent=2))
        else:
            print(json.dumps({"company": company, "employees": []}, indent=2))

    except KeyboardInterrupt:
        print("\nAgent execution cancelled by user.")
    except Exception as error:
        print(f"\nError during agent execution: {error}")
