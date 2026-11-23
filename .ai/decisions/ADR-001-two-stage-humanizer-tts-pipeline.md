# ADR-001: Two-Stage Decoupled Pipeline (Script Humanizer + Voice TTS)

- **Status:** Accepted
- **Date:** 2026-11-23

## Context
Users supply a polished, written presentation script and want natural spoken delivery with adjustable hesitation intensity (`none`, `low`, `medium`, `high`). Asking a single TTS model to simultaneously invent natural hesitations and synthesize speech gives poor controllability and risks unpredictable stuttering.

## Decision
Decouple synthesis into two explicit stages:
1. **Stage 1 (`src/humanizer.py`):** LLM-based text-to-text humanizer that transforms `clean_script` into `humanized_script` using the target speaker's `style_profile.json` and selected `filler_level`.
2. **Human Review Gate:** Users can inspect and manually edit `humanized_script` before synthesis.
3. **Stage 2 (`src/synthesizer.py`):** TTS strictly pronounces the final `humanized_script` without inventing unprompted disfluencies.

## Consequences
- Full transparency and control over filler placement and intensity.
- Simpler, cleaner TTS training because `raw_text` maps 1:1 to spoken audio.
- Allows Stage 1 and Stage 2 models to be upgraded independently.
