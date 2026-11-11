import json

from .ingest import Chunk
from .retriever import Retriever
from .schemas import ToolCallRecord


def _fmt(chunk: Chunk) -> str:
    return f"[{chunk.file} › {chunk.section}]\n{chunk.text}"


class Toolbox:
    """Local tools the LLM can call. Records every call and all evidence it returns."""

    def __init__(self, retriever: Retriever):
        self.r = retriever
        self.drugs = sorted({c.drug for c in retriever.chunks if c.source_type != "glossary"})
        self.calls: list[ToolCallRecord] = []
        self.evidence: dict[tuple[str, str, int], Chunk] = {}

    # ---- tools ----
    def list_available_drugs(self) -> str:
        return json.dumps(self.drugs)

    def search_corpus(self, query: str, drug: str | None = None, k: int = 5) -> list[Chunk]:
        k = max(1, min(int(k), 8))
        return [c for c, _ in self.r.search(query, k=k, drug=drug)]

    def get_section(self, drug: str, section: str) -> list[Chunk]:
        drug, section = drug.lower().strip(), section.lower().strip()
        return [c for c in self.r.chunks
                if c.drug == drug and section in c.section.lower()]


    def glossary_lookup(self, term: str) -> list[Chunk]:
        term_l = term.lower().strip()
        hits = [c for c in self.r.chunks if c.drug == "glossary" and term_l in c.text.lower()]
        return hits[:3] or [c for c, _ in self.r.search(term, k=2, drug="glossary")]

    # ---- dispatch ----
    def run(self, name: str, args: dict) -> str:
        try:
            if name == "list_available_drugs":
                self._record(name, args, len(self.drugs))
                return self.list_available_drugs()
            if name == "search_corpus":
                chunks = self.search_corpus(**args)
            elif name == "get_section":
                chunks = self.get_section(**args)
                if not chunks:
                    sections = sorted({c.section for c in self.r.chunks
                                       if c.drug == args.get("drug", "").lower()})
                    self._record(name, args, 0)
                    return ("No section matched. Sections available for this drug: "
                            + json.dumps(sections) if sections
                            else "Unknown drug. Call list_available_drugs.")
            elif name == "glossary_lookup":
                chunks = self.glossary_lookup(**args)
            else:
                return f"Unknown tool: {name}"
        except TypeError as e:
            return f"Bad arguments for {name}: {e}"
        for i, c in enumerate(chunks):
            self.evidence[(c.file, c.section, hash(c.text))] = c
        self._record(name, args, len(chunks))

        return "\n\n---\n\n".join(_fmt(c) for c in chunks) or "No results."

    def _record(self, name, args, n):
        self.calls.append(ToolCallRecord(tool=name, args=args, n_results=n))


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_available_drugs",
        "description": "List the drugs that exist in the corpus. Use it to check whether a drug is covered.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_corpus",
        "description": "Semantic search over label and consumer-monograph passages. "
                       "Use for open questions or when you don't know the section name.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "drug": {"type": "string", "description": "Optional: restrict to one drug, lowercase."},
            "k": {"type": "integer", "description": "Passages to return (1-8)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_section",
        "description": "Exact lookup of a named section (e.g. 'contraindications', "
                       "'drug interactions', 'renal') for one drug. Prefer this for standard sections.",
        "parameters": {"type": "object", "properties": {
            "drug": {"type": "string"}, "section": {"type": "string"}},
            "required": ["drug", "section"]}}},
    {"type": "function", "function": {
        "name": "glossary_lookup",
        "description": "Look up a medical term in the glossary.",
        "parameters": {"type": "object", "properties": {"term": {"type": "string"}},
                       "required": ["term"]}}},

]