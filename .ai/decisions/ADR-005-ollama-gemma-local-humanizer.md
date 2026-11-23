# ADR-005: Local Ollama + Gemma as Default Stage 1 Script Humanizer

- **Status:** Accepted
- **Date:** 2026-11-23

## Context
Stage 1 (`src/humanizer.py`) converts clean presentation scripts into speaker-specific natural spoken scripts using `style_profile.json`. Relying on paid cloud LLM APIs adds recurring per-presentation cost and sends private presentation scripts externally.

## Decision
1. Default Stage 1 to **local Ollama running Gemma (`gemma3:4b`)** via Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`), configured in `.env.example`.
2. Retain automatic fallback to deterministic `humanize_offline()` if Ollama is not running, so the local Web UI never blocks.
3. Preserve the option to fine-tune a small Gemma/Qwen model via QLoRA on aggregated multi-speaker `(style_profile, filler_level, clean_text) -> raw_text` pairs if prompt-based humanization needs tighter adherence.

## Consequences
- Zero recurring API cost for Stage 1 humanization.
- 100% local, private presentation workflow alongside F5-TTS on the RTX 5060 Laptop (8GB VRAM).
- Easy future upgrade to cloud OpenAI/Gemini endpoints by changing `.env` without code changes.
