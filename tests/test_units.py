import importlib

from app.config import _env
from app.critic import numeric_issues
from app.ingest import Chunk, load_corpus
from app.pipeline import resolve_citations


def _chunk(text="eGFR below 30 mL/min", section="4 CONTRAINDICATIONS"):
    return Chunk("dailymed-metformin.md", "metformin", "dailymed", section, text)


def test_corpus_loads_with_metadata():
    chunks = load_corpus()
    assert len(chunks) > 100
    assert {"warfarin", "metformin", "glossary"} <= {c.drug for c in chunks}
    assert all(c.section and c.text for c in chunks)


def test_section_paths_have_no_h1_prefix():
    assert not any("(DailyMed label)" in c.section for c in load_corpus())


def test_citations_resolve_and_unknown_ids_are_dropped():
    answer, cites = resolve_citations("Fact [E1] [E1]. Bogus [E9].", [_chunk()])
    assert "[dailymed-metformin.md › 4 CONTRAINDICATIONS]" in answer
    assert answer.count("[dailymed-metformin.md") == 1
    assert "E9" not in answer
    assert len(cites) == 1


def test_numeric_check_flags_invented_numbers():

    ev = [_chunk()]
    assert numeric_issues("Contraindicated below 30 [E1]", ev, "") == []
    assert numeric_issues("Contraindicated below 45 [E1]", ev, "")


def test_env_strips_quotes(monkeypatch):
    monkeypatch.setenv("X_TEST", "'https://example.com'")
    assert _env("X_TEST") == "https://example.com"


def test_base_url_read_from_provided_spelling(monkeypatch):
    """The provided .env spells the variable OPENAI_BASE_UR; it must be read as-is."""
    from app import config
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_UR", "'https://proxy.example.com'")
    try:
        importlib.reload(config)
        assert config.BASE_URL == "https://proxy.example.com"
    finally:
        monkeypatch.undo()
        importlib.reload(config)  