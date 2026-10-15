import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# API Keys & Secrets
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY is missing from .env file.")

# Model Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))


def get_llm():
    """Initializes and returns the configured ChatModel based on LLM_PROVIDER."""
    if LLM_PROVIDER == "ollama":
        model = MODEL_NAME or "llama3.2"
        return ChatOllama(model=model, temperature=MODEL_TEMPERATURE)

    elif LLM_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Groq selected. Please install: pip install langchain-groq")
        model = MODEL_NAME or "llama-3.3-70b-versatile"
        return ChatGroq(model=model, temperature=MODEL_TEMPERATURE)

    elif LLM_PROVIDER == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("OpenAI selected. Please install: pip install langchain-openai")
        model = MODEL_NAME or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=MODEL_TEMPERATURE)

    elif LLM_PROVIDER == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Gemini selected. Please install: pip install langchain-google-genai")
        model = MODEL_NAME or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=model, temperature=MODEL_TEMPERATURE)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Supported options: 'ollama', 'groq', 'openai', 'gemini'."
        )


# Initialize configured LLM
llm = get_llm()


# Search Filtering Rules
EXCLUDED_DOMAINS = (
    "indeed.com",
    "glassdoor.",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "scribd.com",
    "ziprecruiter.com",
)

EXCLUDED_PATHS = (
    "/jobs",
    "/reviews",
    "/salary",
    "/benefits",
    "/community/",
    "/careers",
)

