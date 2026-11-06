---
name: implement-step
description: Implement one step of the local-rag project (scaffold, parse, chunk, embed, load, notebook) and update README.md. Use whenever the user asks to build or continue a project step.
---

# Implement one project step

1. Read `RAG-requirements.md` and the status table in `README.md`. Confirm which step is next.
2. Implement only that step. Keep config values in the config file and use the shared device helper (CUDA with CPU fallback).
3. Verify it runs inside `.venv` (tests or a small real run). Test against the separate test collection, not the main one.
4. Update `README.md`: flip the row in the status table, replace the matching "🚧 Not implemented yet" section with real instructions, and add any new commands.
5. Make sure nothing generated is tracked (`data/`, models, embeddings, `.venv/`).
6. Report what was done, what was verified, and what is not yet tested. Do not commit unless asked.
