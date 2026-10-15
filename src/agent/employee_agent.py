import json
from typing import Optional
from langchain.agents import create_agent

from src.config import llm
from src.prompts import AGENT_SYSTEM_PROMPT
from src.tools import search_company_employees, search_company_leadership

def create_employee_agent():
    """Initializes and returns the compiled LangChain AI Agent."""
    tools = [search_company_employees, search_company_leadership]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

# Singleton agent instance
agent = create_employee_agent()


def run_agent(company: str, role_preference: Optional[str] = None):
    """Executes the LangChain AI Agent and displays reasoning steps and results."""
    user_query = f"Find verified public employee profiles for the company '{company}'."
    if role_preference:
        user_query += f" Focus especially on '{role_preference}' roles."

    print(f"LangChain AI Agent activated for company: {company}")
    print("-" * 60)

    inputs = {"messages": [("user", user_query)]}

    try:
        final_message = ""
        for chunk in agent.stream(inputs, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    # Print agent tool decisions
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n[Agent Action] Calling Tool: {tc['name']}")
                            print(f"  Arguments: {tc.get('args')}")
                    # Print tool observations
                    elif hasattr(msg, "name") and msg.name:
                        print(f"[Tool Observation] Completed call for: {msg.name}")
                    # Capture final assistant response
                    elif hasattr(msg, "content") and msg.content:
                        final_message = msg.content

        print("Final Extracted Employee Information (JSON)")
        print("-" * 60)
        if final_message:
            cleaned = final_message.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            try:
                parsed = json.loads(cleaned)
                print(json.dumps(parsed, indent=2))
            except Exception:
                print(final_message)
        else:
            print(json.dumps({"company": company, "employees": []}, indent=2))

    except KeyboardInterrupt:
        print("\nAgent execution cancelled by user.")
    except Exception as error:
        print(f"\nError during agent execution: {error}")

