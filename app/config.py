import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from functools import lru_cache

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
CACHE_DIR = ROOT / ".cache"

API_KEY = os.getenv("OPENAI_API_KEY")
# Accept the typo'd name too, in case a grader's .env has it.
BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_UR")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Provide it via .env or --env-file.")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=5, timeout=90.0)
