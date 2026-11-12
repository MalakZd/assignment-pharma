import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
CACHE_DIR = ROOT / ".cache"


def _env(name: str, default: str | None = None) -> str | None:
    """Read an env var, tolerating stray whitespace and surrounding quotes.

    Docker's --env-file keeps quotes literally, so a value like 'https://x'
    would otherwise arrive with the quotes included.
    """
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip("'\"").strip()
    return value or default


API_KEY = _env("OPENAI_API_KEY")

BASE_URL = _env("OPENAI_BASE_UR") or _env("OPENAI_BASE_URL")
CHAT_MODEL = _env("OPENAI_MODEL", "gpt-5.6-terra")
EMBED_MODEL = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")



@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Provide it via .env or --env-file.")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=5, timeout=90.0)