"""Run the public evaluation questions against a running API.

Usage:  python eval/run_eval.py [base_url]     (default http://localhost:8000)
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

CASES = [
    {"q": "What are the main contraindications for warfarin?",
     "files": ["dailymed-warfarin.md"]},
    {"q": "Compare the contraindications of warfarin and apixaban. What conditions are "
          "shared and which are unique to each drug?",
     "files": ["dailymed-warfarin.md", "dailymed-apixaban.md"]},
    {"q": "A patient currently on warfarin is prescribed sertraline. What is the risk of "
          "bleeding and what monitoring is recommended?",
     "files": ["dailymed-warfarin.md", "dailymed-sertraline.md"]},
    {"q": "A patient with an eGFR of 25 mL/min is being considered for metformin. Is "
          "metformin contraindicated at this level of renal function, and what is the threshold?",
     "files": ["dailymed-metformin.md"], "contains": "30"},
    {"q": "What are the contraindications of lisinopril?",
     "files": [], "contains": "not covered"},
]


def ask(question: str) -> dict:
    req = urllib.request.Request(
        f"{BASE}/ask", data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


passed = 0
for i, case in enumerate(CASES, 1):
    try:
        d = ask(case["q"])
    except Exception as e:
        print(f"[FAIL] Q{i}: request error: {e}")
        continue
    cited = {c["file"] for c in d["citations"]}
    missing = [f for f in case["files"] if f not in cited]
    text_ok = case.get("contains", "").lower() in d["answer"].lower()
    ok = not missing and text_ok and bool(d["disclaimer"])
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] Q{i}: critic={d['critic']['verdict']} "
          f"cited={sorted(cited)} tools={[c['tool'] for c in d['tool_calls']]}")
    if missing:
        print(f"        missing expected sources: {missing}")
    if not text_ok:
        print(f"        answer lacks expected text: {case['contains']!r}")

print(f"\n{passed}/{len(CASES)} passed")
sys.exit(0 if passed == len(CASES) else 1)