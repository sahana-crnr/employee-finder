# Employee Finder AI Agent

An autonomous research AI agent built with **LangChain** and **Ollama (llama3.2)**. Equipped with dynamic search tools via SerpApi, the agent autonomously decides how to search, investigates public employee profiles and leadership rosters, extracts verified personnel details, and reports its findings.

## AI Agent Capabilities

- **Autonomous Tool Calling:** Uses LangChain's agent architecture (`create_agent`) with `@tool` integrations for company employee and leadership searches.
- **Dynamic Search Formulation:** Autonomously formulates targeted search queries for specific companies, departments, and roles.
- **Noise Filtering:** Pre-filters irrelevant results such as job listings, reviews, salary portals, and social media.
- **Grounded Verification:** Extracts verified individual names, roles, and company affiliations backed by public URLs, preventing hallucinations.
- **Real-Time Agent Visibility:** Streams agent actions, tool calls, arguments, and observations directly in the terminal.

---

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** running locally:
   ```bash
   ollama pull llama3.2
   ```
3. **[SerpApi API Key](https://serpapi.com/)** for performing Google searches.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sahana-crnr/employee-finder.git
   cd employee-finder
   ```

2. **Create and activate a virtual environment:**
   - **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env`:
   - **Windows (PowerShell):**
     ```powershell
     Copy-Item .env.example .env
     ```
   - **Linux / macOS:**
     ```bash
     cp .env.example .env
     ```
   Open `.env` and fill in your SerpApi API key:
   ```env
   SERPAPI_API_KEY=your_serpapi_api_key_here
   ```

---

## Usage

### Run the AI Agent
```bash
python main.py
```
*(You can also run `python extract_employees.py` which calls the same agent).*

Enter the target company name when prompted (e.g., `Anthropic`, `Stripe`, `OpenAI`), and optionally specify a role (e.g., `engineer`, `leadership`).

### Test Scripts
- **Test SerpApi search:**
  ```bash
  python tests/test_search.py
  ```
- **Test LangChain & Ollama connection:**
  ```bash
  python tests/test_langchain.py
  ```

---

## Project Structure

```text
employee-finder/
├── main.py                     # Primary CLI entrypoint
├── extract_employees.py        # Entrypoint bridge
├── requirements.txt            # Project Python dependencies
├── .env.example                # Sample environment variable template
├── .gitignore                  # Git ignore rules for virtual environments, secrets, and cache
├── README.md                   # Project documentation
│
├── src/                        # Modular application package
│   ├── __init__.py
│   ├── config.py               # Environment variables, LLM initialization, constants
│   ├── schemas.py              # Pydantic data schemas (EmployeeProfile)
│   ├── prompts.py              # Centralized agent system prompt & guidelines
│   ├── tools/                  # LangChain tools (@tool)
│   │   ├── __init__.py
│   │   └── search_tools.py     # SerpApi employee and leadership search tools
│   └── agent/                  # LangChain Agent logic
│       ├── __init__.py
│       └── employee_agent.py   # Agent initialization (create_agent) and streaming runner
│
└── tests/                      # Test suite
    ├── __init__.py
    ├── test_search.py          # Direct SerpApi integration test
    └── test_langchain.py       # Direct Ollama LLM test
```

---

## License

This project is licensed under the MIT License.
