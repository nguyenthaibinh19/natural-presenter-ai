# ADR-003: Individualized Disfluency Learning via `style_profile.json`

- **Status:** Accepted
- **Date:** 2026-11-23

## Context
Different speakers and language backgrounds hesitate in completely different ways (e.g., `uh`/`um`/`so` vs. `eh`/`you know`/`like` vs. `erm`/`basically`, plus different pause rhythms). A generic global filler prompt would make all speakers sound unnatural and identical.

## Decision
1. Store paired `raw_text` (verbatim spoken with fillers) and `clean_text` (polished presentation English) in every speaker's `metadata.jsonl`.
2. Automatically extract a per-speaker `profiles/<speaker_id>/style_profile.json` containing speaking rate (WPM), pause cadence, normalized `filler_distribution`, and 10–15 real `(clean_text -> raw_text)` few-shot examples across `low`, `medium`, and `high` levels.
3. Inject `style_profile.json` into Stage 1 at runtime (with an optional future path to fine-tune a small local QLoRA LLM across all speakers' pairs).

## Consequences
- Stage 1 adapts immediately to any new speaker/nationality once their dataset is processed — zero LLM retraining required per new user.
- Requires strict 1:1 `raw_text` verification during dataset preparation (and disfluency-preserving Whisper prompts).
