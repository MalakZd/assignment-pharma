import sys

from app.pipeline import answer_question
from app.retriever import Retriever

q = " ".join(sys.argv[1:]) or "What are the main contraindications for warfarin?"
r = answer_question(q, Retriever())

print("PLAN:", *r.plan, sep="\n  - ")
print("MISSING DRUGS:", r.missing_drugs or "none")
print("TOOL CALLS:")
for c in r.tool_calls:
    print(f"  - {c.tool}({c.args}) -> {c.n_results} passages")
print("\nANSWER:\n" + r.answer)
print("\nCITATIONS:")
for c in r.citations:
    print(f"  - {c.file} › {c.section}")
print(f"\nCRITIC: {r.critic.verdict} (revised={r.critic.revised})")
for c in r.critic.claims:
    print(f"  - [{c.status}] {c.claim[:90]}")
for n in r.critic.numeric_issues:
    print(f"  - [numeric] {n}")
print("\n" + r.disclaimer)