import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve configuration from environment variable or Streamlit Cloud secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# API Keys & Secrets
SERPAPI_API_KEY = _get_secret_or_env("SERPAPI_API_KEY", "")

# Model Configuration Defaults
DEFAULT_LLM_PROVIDER = (_get_secret_or_env("LLM_PROVIDER", "ollama") or "ollama").lower().strip()
DEFAULT_MODEL_NAME = _get_secret_or_env("MODEL_NAME")
DEFAULT_MODEL_TEMPERATURE = float(_get_secret_or_env("MODEL_TEMPERATURE", "0") or "0")


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
):
    """Initializes and returns a LangChain ChatModel.
    Can be called with custom parameters (from Streamlit UI) or defaults from .env/secrets.
    """
    prov = (provider or DEFAULT_LLM_PROVIDER).lower().strip()
    temp = DEFAULT_MODEL_TEMPERATURE if temperature is None else float(temperature)

    if prov == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Ollama selected. Please install: pip install langchain-ollama")
        model = model_name or DEFAULT_MODEL_NAME or "llama3.2"
        return ChatOllama(model=model, temperature=temp)

    elif prov == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Groq selected. Please install: pip install langchain-groq")
        key = api_key or _get_secret_or_env("GROQ_API_KEY")
        model = model_name or DEFAULT_MODEL_NAME or "llama-3.3-70b-versatile"
        return ChatGroq(model=model, temperature=temp, api_key=key)

    elif prov == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("OpenAI selected. Please install: pip install langchain-openai")
        key = api_key or _get_secret_or_env("OPENAI_API_KEY")
        model = model_name or DEFAULT_MODEL_NAME or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temp, api_key=key)

    elif prov == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Gemini selected. Please install: pip install langchain-google-genai")
        key = api_key or _get_secret_or_env("GOOGLE_API_KEY")
        model = model_name or DEFAULT_MODEL_NAME or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=model, temperature=temp, api_key=key)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{prov}'. "
            f"Supported options: 'ollama', 'groq', 'openai', 'gemini'."
        )


# Initialize default configured LLM safely
try:
    llm = get_llm()
except Exception:
    llm = None


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
