# AGENTS.md — Natural Presenter AI

Read this first before making any code, prompt, or documentation changes in this repository.

## Mandatory Reading Order
1. `.ai/README.md` — Durable memory index and source-of-truth hierarchy
2. `.ai/current_state.md` — Current repo status, hardware, and active phase
3. `.ai/invariants.md` — Non-negotiable product & engineering rules
4. `.ai/architecture.md` & `.ai/decisions/` — Subsystem boundaries and accepted ADRs

## Standing Rules
- **Never violate `.ai/invariants.md`** (preserve authentic non-native accents, keep per-speaker individualized disfluency profiles, maintain 1:1 `raw_text` ↔ audio alignment, preserve natural breaths, never commit audio/model weights to Git).
- **Continuous Memory Rule:** No completed task, phase, architectural change, schema update, or training run is closed until the relevant `.ai/` files (e.g., `.ai/current_state.md`, `.ai/roadmap.md`, `.ai/experiments.md`, `.ai/lessons.md`, or a new ADR) are updated **in the same commit**. See `.ai/procedures/HOW_TO_UPDATE_PROJECT_MEMORY.md`.
- **Single-File Spec Boundary:** `.ai/` is the daily living ledger. Sync `docs/AI_AGENT_SPEC.md` (the single-file ChatGPT bundle) whenever schemas, pipeline contracts, or major phase milestones change.
- **ADR Immutability:** Never rewrite accepted ADRs; mark superseded ones as `Superseded by ADR-XXX` and create a new ADR.
