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
- **Conda Environment Configured:** `presenter-ai` (`C:\Users\mrbin\miniconda3\envs\presenter-ai`) with Python 3.11.18, FFmpeg 9.1.0, `torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`, `faster-whisper==1.2.1`, `silero-vad==6.3`, `gradio==6.32.0`, `pydantic==2.14.4`. Verified `torch.cuda.is_available() == True`.
- **Initial Real Dataset Processed (from 2 presentation videos):**
  - `vn_01`: 40 WAV clips (6.0 min), `metadata.jsonl` & `style_profile.json` generated (`109.1 WPM`, fillers: `uh: 90.7%`, `like: 9.3%`, 10 few-shot examples).
  - `vn_02`: 30 WAV clips (4.5 min), `metadata.jsonl` & `style_profile.json` generated (`146.6 WPM`, fillers: `uh: 79.4%`, `eh: 11.8%`, `um: 8.8%`, 9 few-shot examples).
- **Additional Script Added:** `scripts/00_extract_diarize_video.py` (offline FFmpeg + Silero VAD + SpeechBrain ECAPA-TDNN speaker separation for future multi-speaker videos).

---

## What Does NOT Exist Yet (Next Up)
1. Human QA review of `vn_01`/`vn_02` `metadata.jsonl` + additional recording to reach 30–60 min for deep LoRA fine-tuning (current 6 min is great for MVP smoke tests and zero-shot F5-TTS testing).
2. `src/humanizer.py` — Stage 1 Personalized Script Humanizer (LLM + `style_profile.json`).
3. `scripts/05_train_lora_cloud.py` — Cloud RTX 4090 F5-TTS LoRA fine-tuning script/notebook.
4. `src/synthesizer.py` & `app/web_ui.py` — Stage 2 F5-TTS inference + local Gradio Web UI.
