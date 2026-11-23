# ADR-002: G2P-Free F5-TTS Base Model with Per-Speaker LoRA Adapters

- **Status:** Accepted
- **Date:** 2026-11-23

## Context
The system must support multiple speakers with distinct non-native English accents (e.g., Vietnamese, Cypriot, Hungarian, Spanish) without normalizing pronunciation toward standard US/UK accents or causing accent bleed across users.

## Decision
1. Use **F5-TTS** (a flow-matching, G2P-free TTS model) as the shared Base Model, with **GPT-SoVITS** as the documented fallback.
2. Isolate every speaker as a lightweight **per-speaker LoRA adapter** (~30–100 MB) stored under `profiles/<speaker_id>/lora/`, alongside curated reference clips in `profiles/<speaker_id>/refs/`.

## Consequences
- Preserves authentic non-native consonant endings, vowel shifts, stress patterns, and natural breaths.
- Onboarding a new speaker takes 20–45 minutes of isolated LoRA training without touching any existing speaker.
- Fast hot-swapping on an 8GB VRAM local GPU (Base Model stays loaded; only LoRA weights swap).
- *Risk to monitor in Phase 1/2:* If LoRA underfits very strong phonetic shifts, we will benchmark per-speaker full fine-tune checkpoints and record a superseding ADR if needed.
