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


def parse_agent_json(raw_text: str, fallback_company: str) -> dict:
    """Extracts and validates JSON from LLM response text."""
    if not raw_text:
        return {"company": fallback_company, "employees": []}

    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "employees" in data:
            # Ensure each employee has standard fields and seniority classification
            for emp in data.get("employees", []):
                role = emp.get("role") or emp.get("job_title", "")
                emp["role"] = role
                if not emp.get("seniority"):
                    emp["seniority"] = classify_seniority(role)
                if not emp.get("source_domain") and emp.get("source_url"):
                    from urllib.parse import urlparse
                    netloc = urlparse(emp["source_url"]).netloc
                    emp["source_domain"] = netloc.replace("www.", "")
            return data
    except Exception:
        pass

    return {"company": fallback_company, "employees": [], "raw_output": raw_text}


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
