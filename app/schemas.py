from pydantic import BaseModel, Field


class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=6)
    drugs_involved: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    file: str
    section: str


class Draft(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool: str
    args: dict
    n_results: int