# Current State

**Last Updated:** 2026-11-23  
**Active Phase:** Phase 1 — MVP Setup & First Speaker (`vn_01`) Dataset Preparation  
**Repository:** `https://github.com/nguyenthaibinh19/natural-presenter-ai` (Private)  
**Primary Local Path:** `D:\natural-presenter-ai`

---

## What Exists Today

### Documentation & Governance
- `.ai/` durable project memory system initialized (modeled on `EnglishApp` conventions), including `ADR-001` through `ADR-004`.
- `docs/PROJECT_GUIDE.md` — human-readable English guide for collaborators.
- `docs/AI_AGENT_SPEC.md` — comprehensive machine-readable technical spec for ChatGPT / AI agents.
- `docs/RECORDING_GUIDE.md` — speaker recording checklist, 60/40 spontaneous/scripted prompts, and CLI commands.

### Code Implemented
- `src/schemas.py` — Pydantic v2 data contracts (`DatasetClip`, `PauseEvent`, `StyleProfile`, `FewShotExample`, `HumanizeRequest`, `HumanizeResponse`, `SynthesizeRequest`) + JSONL/JSON persistence helpers.
- `scripts/01_preprocess_audio.py` — 24kHz mono WAV conversion, peak normalization (no aggressive noise gate), Silero VAD slicing (3–12s with 150ms padding).
- `scripts/02_transcribe_align.py` — `faster-whisper` transcription with disfluency-preserving prompt, word-level pause detection (`[pause_short]`, `[pause_long]`), and filler density classification.
- `scripts/03_build_dataset_labels.py` — LLM-assisted (OpenAI-compatible) + `--heuristic` offline fallback generator for `clean_text` pairs and `tone`.
- `scripts/04_extract_style_profile.py` — Extracts WPM, pause cadence, normalized `filler_distribution`, and few-shot examples into `style_profile.json`.
- `requirements.txt` — Core dependencies with RTX 5060 CUDA 12.8 note.

### Local Environment Status (Primary Dev Machine)
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (`8151 MiB` VRAM, Driver `610.71`, Blackwell `sm_120`).
- **Package Managers Available:** `conda 26.3.2`, `uv 0.11.25`, `winget`.
- **Not Yet Configured:** Dedicated Conda environment (`presenter-ai`: Python 3.11 + FFmpeg + PyTorch CUDA 12.8 + `requirements.txt`).
- **Local Speaker Folder Ready:** `profiles/vn_01/{raw_recordings,wavs,refs,lora}` created locally (gitignored).

---

## What Does NOT Exist Yet (Next Up)
1. Local Conda environment installation (`presenter-ai`) and end-to-end smoke test on real audio.
2. First speaker (`vn_01`) raw recordings (target: 20–30 min MVP test, then 60–90 min production).
3. `src/humanizer.py` — Stage 1 Personalized Script Humanizer (LLM + `style_profile.json`).
4. `scripts/05_train_lora_cloud.py` — Cloud RTX 4090 F5-TTS LoRA fine-tuning script/notebook.
5. `src/synthesizer.py` & `app/web_ui.py` — Stage 2 F5-TTS inference + local Gradio Web UI.
