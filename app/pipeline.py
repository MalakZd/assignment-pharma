import re

from openai import BadRequestError

from .agent import run_agent
from .critic import has_blocking_issues, has_issues, review, revise
from .ingest import Chunk
from .retriever import Retriever
from .schemas import AskResponse, Citation, CriticReport

DISCLAIMER = ("This output is not medical advice. It is a technical exercise based on a "
              "limited set of FDA label and consumer-monograph excerpts. "
              "Consult a qualified healthcare professional.")


def resolve_citations(answer: str, evidence: list[Chunk]) -> tuple[str, list[Citation]]:
    """Turn [E#] markers into real file › section references; drop unknown IDs."""
    by_id = {f"E{i}": c for i, c in enumerate(evidence, 1)}
    seen: dict[tuple[str, str], Citation] = {}

    def sub(m: re.Match) -> str:
        c = by_id.get(m.group(1))
        if not c:
            return ""
        seen.setdefault((c.file, c.section), Citation(file=c.file, section=c.section))
        return f"[{c.file} › {c.section}]"

    resolved = re.sub(r"\[(E\d+)\]", sub, answer)
    # Collapse identical markers repeated back to back (several chunks, same section).
    resolved = re.sub(r"(\[[^\]]+\])(?:\s*\1)+", r"\1", resolved)
    return resolved, list(seen.values())


def _answer_question(question: str, retriever: Retriever) -> AskResponse:
    res = run_agent(question, retriever)
    draft = res.draft
    report = review(question, draft, res.evidence)

    if has_issues(report):
        draft = revise(question, draft, report, res.evidence)
        report2 = review(question, draft, res.evidence)
        report = report2.model_copy(update={
            "revised": True,
            "verdict": "flagged" if has_blocking_issues(report2) else "revised"})
    else:
        report = report.model_copy(update={"verdict": "pass"})

    answer, citations = resolve_citations(draft.answer, res.evidence)

    if report.verdict == "flagged":
        caveats = [f"- {c.claim} ({c.status.replace('_', ' ')}): {c.note}"
                   for c in report.claims if c.status != "supported"]
        caveats += [f"- {n}" for n in report.numeric_issues]
        answer += "\n\nCaveats flagged by the reviewer:\n" + "\n".join(caveats)

    return AskResponse(
        answer=answer, citations=citations, plan=res.plan.sub_questions,
        tool_calls=res.tool_calls, critic=report, missing_drugs=res.missing_drugs,
        disclaimer=DISCLAIMER)


def _blocked_response(reason: str) -> AskResponse:
    return AskResponse(
        answer=("I couldn't process this request because the language-model provider's safety "
                "filter blocked it. Please rephrase it as a question about the drugs in the "
                "corpus (for example, their contraindications, interactions or dosing)."),
        citations=[], plan=[], tool_calls=[], missing_drugs=[],
        critic=CriticReport(verdict="flagged", claims=[], numeric_issues=[reason], revised=False),
        disclaimer=DISCLAIMER)


def answer_question(question: str, retriever: Retriever) -> AskResponse:
    """Public entry point. Degrades gracefully when the provider's content filter blocks a prompt."""
    try:
        return _answer_question(question, retriever)
    except BadRequestError as e:
        msg = str(e).lower()
        if "content" in msg and ("policy" in msg or "filter" in msg):
            return _blocked_response("Request blocked by the provider's content filter")
        raise