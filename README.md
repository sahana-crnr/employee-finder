# Employee Finder AI Agent

An employee research app built with **LangChain** and **SerpApi**. It searches for senior leaders and returns only profiles supported by current-employment evidence and a complete company-name match.

## AI Agent Capabilities

- **Leadership focus:** Targets executives, founders, vice presidents, directors, department heads, and senior managers.
- **Current employment checks:** Excludes former employees and job seekers; profiles need current-status evidence.
- **Company matching:** Tries the complete name first. If LinkedIn has no qualifying profile results, retries with the first company-name word and still requires a direct current role match.
- **Company verification and member search:** Uses Google results and Instagram company profiles only to verify the company. After verification, searches LinkedIn profiles for current leaders; Google and Instagram are not used to find members. The language model only extracts details from returned LinkedIn evidence.
- **Source links:** Results include the supporting source and current-employment snippet.

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

### 1. Run the Streamlit Web Application
```bash
streamlit run app.py
```
This opens the web app with model settings, company search, leadership results, and CSV/JSON downloads.

### 2. Run via Terminal CLI
```bash
python main.py
```
*(Or `python extract_employees.py`)*. Enter the target company name or website URL when prompted.

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
