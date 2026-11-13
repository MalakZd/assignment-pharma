# CLAUDE.md

> Built with Claude in a chat interface (no Claude Code, skills or MCP servers).
> This file records the rules and conventions given to the assistant, and the
> pitfalls found while building. All code was run and reviewed by the author.

## Project
Agentic RAG assistant over a local pharmacology corpus (FDA labels + consumer
monographs). Python 3.13, FastAPI, OpenAI SDK against a LiteLLM proxy.
Technical exercise, not medical advice.

## Design rules
1. **Grounding.** Answers come only from retrieved evidence, never from model memory.
   Drugs missing from the corpus are reported as not covered.
2. **Citations by ID.** The model cites `[E#]`; code resolves them to `file › section`,
   so a citation can never be invented or misspelled.
3. **Two-layer review.** An LLM critic labels each claim (supported / unsupported /
   clinically ambiguous); deterministic code checks that every number in the answer
   exists in the evidence. One revision pass, then unresolved issues are flagged.
4. **Agent-chosen tools.** The model picks among `search_corpus`, `get_section`,
   `glossary_lookup`, `list_available_drugs`. No keyword routing.
5. **Safety in code, not prompts.** The disclaimer is appended by code.
   Questions and retrieved text are treated as data, not instructions.
6. **No secrets, no hardcoding.** All settings come from environment variables via
   `app/config.py`; `.env` is git-ignored. The proxy URL is read from
   `OPENAI_BASE_UR`, the spelling used in the provided `.env`.

## Pitfalls found and fixed
| Problem | Fix |
|---|---|
| Model returned the JSON schema instead of data | `app/llm.py` unwraps `properties` and retries with the validation error |
| Free-text citations drifted (`7DRUG` vs `7 DRUG`) | Evidence IDs resolved in code |
| Critic rejected valid comparisons as "unsupported" | Rubric separates direct support from entailed synthesis |
| Docker `--env-file` keeps quotes, so the base URL became invalid and calls went to localhost | `_env()` strips quotes |
| Provider content filter rejected a prompt and crashed the API | Caught in `app/pipeline.py`, returns a flagged response |
| Real key and confidential PDF in the first commit | Repo recreated with `.gitignore` before publishing |

## Workflow
Built in phases (data layer, agent, critic, API/UI, Docker, eval), each run and
verified before the next. Before committing: `python -m pytest -q` and
`python eval/run_eval.py`.