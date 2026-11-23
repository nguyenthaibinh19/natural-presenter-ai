# Architecture

## Subsystem Boundaries

Natural Presenter AI is strictly split into **three isolated layers** connected through file-based **Speaker Profiles** (`profiles/<speaker_id>/`):

### Layer 0 — Dataset & Profile Pipeline (`scripts/01..04`)
- **Input:** Raw speaker `.wav`/`.mp3`/`.m4a` recordings in `profiles/<speaker_id>/raw_recordings/`.
- **Output:**
  - `wavs/` — 24kHz mono 16-bit PCM clips (3–12s).
  - `metadata.jsonl` — Verbatim `raw_text` (for TTS) + polished `clean_text` (for Stage 1) + pause & filler metadata.
  - `style_profile.json` — Quantified speaking & disfluency fingerprint for Stage 1.
- **Boundary:** Never mutates `raw_recordings/`. All schemas are owned by `src/schemas.py`.

### Layer 1 — Personalized Script Humanizer (`src/humanizer.py`)
- **Input:** `HumanizeRequest` (`speaker_id`, `clean_script`, `filler_level`: `none|low|medium|high`, `tone`).
- **Reads:** `profiles/<speaker_id>/style_profile.json`.
- **Output:** `HumanizeResponse` containing an editable `humanized_script` using *that speaker's* own filler vocabulary, cadence, and punctuation conventions.
- **Boundary:** Pure text-to-text transformation; has zero dependency on TTS weights or GPU audio synthesis.

### Layer 2 — Accent-Preserving Voice Synthesizer (`src/synthesizer.py`)
- **Input:** `SynthesizeRequest` (`speaker_id`, `humanized_script`, `tone`, `speed`).
- **Reads:** Shared Base Model (`F5-TTS`) + `profiles/<speaker_id>/lora/` + `profiles/<speaker_id>/refs/`.
- **Output:** Stitched, loudness-normalized presentation audio (`.wav` / `.mp3`).
- **Boundary:** Strictly pronounces what is written in `humanized_script` using the target speaker's LoRA adapter; does not invent or inject extra disfluencies on its own.

### Presentation UI (`app/web_ui.py`)
- Thin local Gradio interface orchestrating Layer 1 → Human Review/Edit Gate → Layer 2.

---

## Multi-Speaker Isolation Model

```text
profiles/
├── vn_01/   (Vietnamese English - Speaker 1: own style_profile.json + own LoRA)
├── vn_02/   (Vietnamese English - Speaker 2: own style_profile.json + own LoRA)
├── es_01/   (Spanish English: own style_profile.json + own LoRA)
├── hu_01/   (Hungarian English: own style_profile.json + own LoRA)
└── cy_01/   (Cypriot English: own style_profile.json + own LoRA)
```

Every speaker is self-contained. Adding or retraining a speaker never modifies the shared Base Model or any other speaker's profile.
