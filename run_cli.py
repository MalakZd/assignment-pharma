import sys

from app.agent import run_agent
from app.retriever import Retriever

DISCLAIMER = "Not medical advice. Technical exercise based on a limited corpus."

q = " ".join(sys.argv[1:]) or "What are the main contraindications for warfarin?"
res = run_agent(q, Retriever())

print("PLAN:", *res.plan.sub_questions, sep="\n  - ")
print("MISSING DRUGS:", res.missing_drugs or "none")
print("TOOL CALLS:")
for c in res.tool_calls:
    print(f"  - {c.tool}({c.args}) -> {c.n_results} passages")
print("\nANSWER:\n" + res.draft.answer)
print("\nCITATIONS:", [f"{c.file} › {c.section}" for c in res.draft.citations])
print("\n" + DISCLAIMER)