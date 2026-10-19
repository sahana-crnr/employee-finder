import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# API Keys & Secrets
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY is missing from .env file.")

# Model Configuration Defaults
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME")
DEFAULT_MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
):
    """Initializes and returns a LangChain ChatModel.
    Can be called with custom parameters (from Streamlit UI) or defaults from .env.
    """
    prov = (provider or DEFAULT_LLM_PROVIDER).lower().strip()
    temp = DEFAULT_MODEL_TEMPERATURE if temperature is None else float(temperature)

    if prov == "ollama":
        model = model_name or DEFAULT_MODEL_NAME or "llama3.2"
        return ChatOllama(model=model, temperature=temp)

    elif prov == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Groq selected. Please install: pip install langchain-groq")
        key = api_key or os.getenv("GROQ_API_KEY")
        model = model_name or DEFAULT_MODEL_NAME or "llama-3.3-70b-versatile"
        return ChatGroq(model=model, temperature=temp, api_key=key)

    elif prov == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("OpenAI selected. Please install: pip install langchain-openai")
        key = api_key or os.getenv("OPENAI_API_KEY")
        model = model_name or DEFAULT_MODEL_NAME or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temp, api_key=key)

    elif prov == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Gemini selected. Please install: pip install langchain-google-genai")
        key = api_key or os.getenv("GOOGLE_API_KEY")
        model = model_name or DEFAULT_MODEL_NAME or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=model, temperature=temp, api_key=key)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{prov}'. "
            f"Supported options: 'ollama', 'groq', 'openai', 'gemini'."
        )


# Initialize default configured LLM
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
