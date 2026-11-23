# How to Update Project Memory (`.ai/`)

Update `.ai/` **continuously for every durable change** — never leave project memory behind the code.

**Standing rule:** A completed task, phase, architectural decision, schema update, or operational fix is **not closed** until the relevant `.ai/` files match the repository and verified behavior in the same commit. Chat history is not durable memory.

## When to Update What

| What Changed | Files to Update |
|---|---|
| Completed step, new script, environment setup, or blocker | [`current_state.md`](../current_state.md) + [`roadmap.md`](../roadmap.md) |
| Architectural boundary or pipeline flow | [`architecture.md`](../architecture.md) + new ADR in [`decisions/`](../decisions/) + [`docs/AI_AGENT_SPEC.md`](../../docs/AI_AGENT_SPEC.md) |
| Schema or JSON contract (`src/schemas.py`) | [`current_state.md`](../current_state.md) + [`docs/AI_AGENT_SPEC.md`](../../docs/AI_AGENT_SPEC.md) |
| Must-not-break rule added or revised | [`invariants.md`](../invariants.md) |
| Non-obvious bug, model quirk, or hardware gotcha solved | [`lessons.md`](../lessons.md) |
| Recording or training workflow changed | [`docs/RECORDING_GUIDE.md`](../../docs/RECORDING_GUIDE.md) or relevant procedure |

## ADR Immutability Rule

Do **not** rewrite old ADRs when a decision changes. Instead:
1. Change the old ADR's `Status` to **`Superseded by ADR-XXX`**.
2. Create a new **Accepted** ADR (`ADR-XXX`) explaining the new decision and rationale.

## What Does NOT Belong in `.ai/`
- Temporary debugging scratch notes or raw console dumps
- Raw chat transcripts
- Exhaustive per-session file diff lists (Git history already owns that)
