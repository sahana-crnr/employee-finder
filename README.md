# Employee Finder

An intelligent CLI tool that gathers and extracts publicly available professional profile information for a specified company. It combines search aggregation via SerpApi with structured LLM parsing using LangChain and a local Ollama model (`llama3.2`).

## Features

- **Targeted Public Search:** Queries SerpApi across multiple search patterns (e.g., LinkedIn profiles, company directories, team pages).
- **Noise Filtering:** Pre-filters irrelevant results such as job listings, reviews, salary portals, and social media.
- **LLM-Powered Extraction:** Employs LangChain and local Ollama (`llama3.2`) to extract identifiable names, roles, and company associations in structured JSON format.
- **Deduplication & Validation:** Cross-checks source indices, eliminates duplicate names, and formats the final verified profiles cleanly in the terminal.

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

### Run the Main Extractor
```bash
python extract_employees.py
```
Enter the target company name when prompted (e.g., `Anthropic`, `Stripe`, `OpenAI`).

### Test Scripts
- **Test SerpApi search:**
  ```bash
  python test_search.py
  ```
- **Test LangChain & Ollama connection:**
  ```bash
  python test_langchain.py
  ```

---

## Project Structure

```text
employee-finder/
├── extract_employees.py  # Main script: searches, extracts, and displays profiles
├── test_search.py        # Utility script to test SerpApi integration
├── test_langchain.py     # Utility script to test Ollama + LangChain setup
├── requirements.txt      # Project Python dependencies
├── .env.example          # Sample environment variable template
├── .gitignore            # Git ignore rules for virtual environments, secrets, and cache
└── README.md             # Project documentation
```

---

## License

This project is licensed under the MIT License.
