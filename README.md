# Pharma Research Agent

An agentic research assistant that answers compound questions about a local
pharmacology corpus (FDA DailyMed labels and MedlinePlus-style monographs).
It plans, retrieves, uses tools chosen by the model, reviews its own draft with a
critic, and returns a cited answer. **Technical exercise, not medical advice.**

## Run it

```bash
cp .env.example .env        # fill in OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
```

- UI: http://localhost:8000
- `GET /health` returns `{"status": "ok", "index_ready": ...}` immediately.
- `POST /ask` with `{"question": "..."}` returns the answer, citations, plan,
  tool calls and critic report.

The embedding index is built in the background at startup (a few seconds to a
minute). `/ask` waits for it if needed. Quoted values in `.env` are tolerated.

## Architecture

```
question
  -> PLAN      LLM returns validated sub-questions and the drugs involved
  -> EXECUTE   LLM chooses tools in a loop (max 6 rounds):
               search_corpus | get_section | glossary_lookup | list_available_drugs
  -> DRAFT     answer written only from numbered evidence [E1], [E2], ...
  -> CRITIC    LLM labels each claim supported / unsupported / clinically ambiguous;

               code checks every number in the answer appears in the evidence
  -> REVISE    one revision pass if issues found; unresolved ones are flagged
  -> ANSWER    [E#] markers resolved to file + section in code; disclaimer added in code
```

| File | Role |
|---|---|
| `app/ingest.py` | Chunk markdown by heading, with file/drug/source/section metadata |
| `app/retriever.py` | Embeddings + NumPy cosine search, cached on disk |
| `app/tools.py` | The four agent-callable tools and their schemas |
| `app/agent.py` | Planner, tool-calling loop, evidence-only drafting |
| `app/critic.py` | Claim verification, numeric check, revision |
| `app/pipeline.py` | Orchestration, citation resolution, content-filter handling |
| `app/main.py` | FastAPI app, validation and error mapping |
| `app/static/index.html` | Frontend with an agent-trace panel |
| `eval/run_eval.py` | Runs the public questions against a running API |

## Design decisions

- **Own agent loop on the OpenAI SDK** instead of a framework: fewer dependencies,
  every step is visible and explainable.
- **Citations by evidence ID.** The model cites `[E#]`; code maps IDs to real
  `file › section`, so citations cannot be misspelled or invented.
- **Two-layer verification.** An LLM critic judges grounding, and deterministic code
  checks numbers and units against the sources.
- **Out-of-corpus handling.** The agent checks the drug list and states when a drug
  is not covered instead of answering from model memory.
- **Prompt-injection hygiene.** Retrieved text and questions are treated as data.
- **Graceful errors.** Empty questions return 422; provider failures map to 429/502;
  a provider content-filter block returns a clear flagged response.

## Tests and evaluation


```bash
python -m pytest -q                 # unit tests, no network needed
python eval/run_eval.py             # public questions against http://localhost:8000
```

## Limitations and next steps

- Retrieval is embedding-only; hybrid keyword + embedding search and a reranker
  would help with exact terms.
- The critic is an LLM and can be wrong; the numeric check only verifies that
  numbers exist in the evidence, not that they are attached to the right claim.
- Cross-source conflicts (label vs consumer text) are not explicitly reconciled;
  the agent is instructed to prefer FDA label text.
- Answers take roughly 20 to 60 seconds because of several sequential LLM calls.

## AI assistance

Built with AI assistance (Claude). See `CLAUDE.md` for the conventions and prompts
used. All code was reviewed and run by the author.