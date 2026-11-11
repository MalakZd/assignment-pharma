from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=6)
    drugs_involved: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    file: str
    section: str


class Draft(BaseModel):
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool: str
    args: dict
    n_results: int


class ClaimCheck(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "clinically_ambiguous"]
    note: str = ""



class CriticLLMReport(BaseModel):
    claims: list[ClaimCheck]


class CriticReport(BaseModel):
    verdict: Literal["pass", "revised", "flagged"] = "pass"
    claims: list[ClaimCheck] = Field(default_factory=list)
    numeric_issues: list[str] = Field(default_factory=list)
    revised: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    plan: list[str]
    tool_calls: list[ToolCallRecord]
    critic: CriticReport
    missing_drugs: list[str]
    disclaimer: str