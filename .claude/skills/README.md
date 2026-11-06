# Project skills

Each skill is a folder with a `SKILL.md`:

```
.claude/skills/<skill-name>/SKILL.md
```

`SKILL.md` starts with frontmatter (`name`, `description`), followed by the instructions. Keep skills short and specific to this project. Commit them to git.

| Skill | Purpose |
|---|---|
| `implement-step` | How to implement one project step and update the README |

## Plugins

`.claude/settings.json` (committed) declares the marketplaces and plugins below. After cloning the repo, trust the folder in Claude Code and accept the install prompt.

| Plugin | Marketplace | Use |
|---|---|---|
| `qdrant` | `qdrant/skills` | Hybrid search, search quality, SDK |

Not added: RAG Evaluation (no GitHub marketplace source found), Docker skills (not needed).

Candidates to add later: `docling-parsing`, `qdrant-hybrid-search`, `rag-eval`.
