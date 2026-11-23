# Natural Presenter AI — Project Memory (`.ai/`)

This directory is the **durable engineering memory** for **Natural Presenter AI**.

## Purpose

The **repository** is the long-term source of truth for architecture, decisions, invariants, and procedures.  
Chat history is **working context only** — not durable memory. Any future AI agent (ChatGPT, Claude, Antigravity, Codex) or collaborator must be able to reconstruct the full state of the project from `.ai/` + code + tests + Git without needing prior chats.

## Source-of-Truth Hierarchy

```text
Tests / executable behavior
        +
Current code (src/, scripts/, app/)
        +
ADRs & Invariants (.ai/decisions/, .ai/invariants.md)
        +
Git history
        +
Current state & Roadmap (.ai/current_state.md, .ai/roadmap.md)
```

If documentation conflicts with tested code, **investigate the conflict** — do not blindly trust either side.

## Standing Rule (Continuous Documentation)

> **A completed task, phase, architectural decision, schema change, or operational fix is NOT closed until the relevant `.ai/` files (and `docs/AI_AGENT_SPEC.md` if contracts changed) are updated in the same commit.**

See [`procedures/HOW_TO_UPDATE_PROJECT_MEMORY.md`](procedures/HOW_TO_UPDATE_PROJECT_MEMORY.md) for the exact update matrix.

## Recommended Reading Order (For AI Agents & New Contributors)

1. Read [`current_state.md`](current_state.md) — current status, environment, what exists vs. what is next
2. Read [`invariants.md`](invariants.md) — non-negotiable rules that must never be broken
3. Read [`architecture.md`](architecture.md) — subsystem boundaries and data contracts
4. Read [`roadmap.md`](roadmap.md) — phased milestones
5. Read relevant ADRs in [`decisions/`](decisions/) before proposing architectural changes
6. Read [`../docs/AI_AGENT_SPEC.md`](../docs/AI_AGENT_SPEC.md) for full JSON schemas and pipeline details

## Index

| Path | Role |
|------|------|
| [current_state.md](current_state.md) | Start here — live snapshot of repo, environment, and active phase |
| [architecture.md](architecture.md) | Subsystem boundaries, dataflow, and speaker profile layout |
| [invariants.md](invariants.md) | Must-not-break rules for accents, disfluencies, data integrity, and Git safety |
| [roadmap.md](roadmap.md) | Phased execution plan (MVP -> Multi-Accent -> Product Polish) |
| [experiments.md](experiments.md) | Log of TTS LoRA fine-tuning runs and listening evaluations |
| [lessons.md](lessons.md) | Durable engineering lessons learned |
| [decisions/](decisions/) | Immutable Architecture Decision Records (`ADR-001` through `ADR-004`) |
| [procedures/HOW_TO_UPDATE_PROJECT_MEMORY.md](procedures/HOW_TO_UPDATE_PROJECT_MEMORY.md) | Rules for keeping `.ai/` continuously updated |
