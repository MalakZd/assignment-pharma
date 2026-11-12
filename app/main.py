import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import APIConnectionError, APIStatusError, RateLimitError
from pydantic import BaseModel, field_validator

from . import config
from .pipeline import answer_question
from .retriever import Retriever
from .schemas import AskResponse

log = logging.getLogger("agent")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class _State:
    retriever: Retriever | None = None
    lock = threading.Lock()


state = _State()


def get_retriever() -> Retriever:
    """Build the index once; if building fails it is retried on the next call."""
    with state.lock:
        if state.retriever is None:
            state.retriever = Retriever()

        return state.retriever


def _warm_up() -> None:
    try:
        get_retriever()
        log.info("Index ready")
    except Exception as e:  # /ask will retry and report the real error
        log.warning("Index warm-up failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the index in the background so /health answers immediately.
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Pharma Research Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > 2000:
            raise ValueError("question must be at most 2000 characters")

        return v


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": state.retriever is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not config.API_KEY:
        raise HTTPException(503, "Server is not configured: OPENAI_API_KEY is missing.")
    try:
        return answer_question(req.question, get_retriever())
    except RateLimitError:
        raise HTTPException(429, "The language-model provider is rate limiting requests. Try again shortly.")
    except APIConnectionError:
        raise HTTPException(502, "Could not reach the language-model provider. Try again shortly.")
    except APIStatusError as e:
        log.exception("Provider error")
        raise HTTPException(502, f"Language-model provider error (status {e.status_code}).")
    except RuntimeError as e:
        log.exception("Model output error")
        raise HTTPException(502, f"The model returned an unusable response: {str(e)[:200]}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")