import re

from .agent import format_evidence
from .ingest import Chunk
from .llm import json_call
from .schemas import CriticLLMReport, CriticReport, Draft

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
ID_RE = re.compile(r"\[E\d+\]")

CRITIC_SYSTEM = (
    "You are a strict clinical-grounding reviewer. Split the ANSWER into distinct factual claims "
    "(ignore disclaimers). For each, decide against the EVIDENCE only:\n"
    "- supported: the cited passages directly state it, OR it is a comparison/summary that follows "
    "directly from facts the passages state (e.g. both lists mention bleeding; one list has an item "
    "the other lacks). Synthesis is allowed when every fact it rests on is in the evidence.\n"
    "- unsupported: adds facts not in the evidence, or overreaches (invented frequencies, doses, "
    "thresholds, causal claims). Do NOT mark a claim unsupported just because it is phrased as an "
    "interpretation or uses different wording than the source.\n"
    "- clinically_ambiguous: technically in the text but vague, patient-dependent, or easily "
    "misread as personal medical advice.\n"
    "Check every number, unit and threshold character by character. "
    "Give a one-line note for any claim that is not supported."
)

REVISE_SYSTEM = (
    "Rewrite the ANSWER to fix the reviewer's issues. Remove or soften unsupported claims, "
    "state ambiguity explicitly, correct any numbers using the evidence. Use ONLY the evidence, "
    "keep [E#] citations on every factual sentence, and do not add new facts or new analysis. "
    "The result must be shorter than or equal in length to the original. "
    "Return JSON: {\"answer\": str, \"evidence_ids\": [str]}."
)

REVISE_SYSTEM = (
    "Rewrite the ANSWER to fix the reviewer's issues. Remove or soften unsupported claims, "
    "state ambiguity explicitly, correct any numbers using the evidence. Use ONLY the evidence, "
    "keep [E#] citations on every factual sentence, and do not add new facts. "
    "Return JSON: {\"answer\": str, \"evidence_ids\": [str]}."
)


def numeric_issues(answer: str, evidence: list[Chunk], question: str) -> list[str]:
    """Deterministic check: every number in the answer must appear in the evidence or question."""

    allowed = set(NUM_RE.findall(" ".join(c.text for c in evidence))) | set(NUM_RE.findall(question))
    found = set(NUM_RE.findall(ID_RE.sub("", answer)))
    return [f"Number '{n}' is not present in the retrieved evidence" for n in sorted(found - allowed)]


def review(question: str, draft: Draft, evidence: list[Chunk]) -> CriticReport:
    user = (f"QUESTION: {question}\n\nANSWER:\n{draft.answer}\n\n"
            f"EVIDENCE:\n{format_evidence(evidence) or '(none)'}")
    llm = json_call(CRITIC_SYSTEM, user, CriticLLMReport)
    return CriticReport(claims=llm.claims,
                        numeric_issues=numeric_issues(draft.answer, evidence, question))


def has_issues(report: CriticReport) -> bool:
    """Any non-supported claim or numeric mismatch triggers a revision."""
    return bool(report.numeric_issues) or any(c.status != "supported" for c in report.claims)


def has_blocking_issues(report: CriticReport) -> bool:
    """After revision, only unsupported claims or bad numbers get flagged to the user."""
    return bool(report.numeric_issues) or any(c.status == "unsupported" for c in report.claims)


def revise(question: str, draft: Draft, report: CriticReport, evidence: list[Chunk]) -> Draft:
    issues = [f"- ({c.status}) {c.claim}: {c.note}" for c in report.claims if c.status != "supported"]
    issues += [f"- (numeric) {n}" for n in report.numeric_issues]
    user = (f"QUESTION: {question}\n\nANSWER:\n{draft.answer}\n\nISSUES:\n" + "\n".join(issues)
            + f"\n\nEVIDENCE:\n{format_evidence(evidence)}")
    return json_call(REVISE_SYSTEM, user, Draft)