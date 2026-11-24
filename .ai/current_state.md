# Current State

**Last Updated:** 2026-11-24  
**Active Phase:** Phase 1 — MVP Setup & First Speaker (`vn_01`) Dataset Preparation  
**Repository:** `https://github.com/nguyenthaibinh19/natural-presenter-ai` (Private)  
**Primary Local Path:** `D:\natural-presenter-ai`

---

## What Exists Today

### Documentation & Governance
- `.ai/` durable project memory system initialized (modeled on `EnglishApp` conventions), including `ADR-001` through `ADR-005` and `experiments.md` (`ZS-001` baseline rated 7–8/10).
- `AGENTS.md` — root auto-read entrypoint for AI coding agents.
- `docs/EGE_ONBOARDING_AND_AI_BRIEF.md` — onboarding brief, copy-paste AI prompt, and active work tracks for Ege & Ege's AI assistant.
- `docs/PROPOSALS_AND_SYNC.md` — shared Binh ↔ Ege proposal & approval board (`PROP-001`..`PROP-004` seeded).
- `docs/PROJECT_GUIDE.md` — human-readable English guide for collaborators.
- `docs/AI_AGENT_SPEC.md` — comprehensive machine-readable technical spec for ChatGPT / AI agents.
- `docs/RECORDING_GUIDE.md` — speaker recording checklist (including iPhone 16 Lossless ALAC + soft room tips), 8 Part A spontaneous prompts, 18 Part B conversational scripts (~11–13 min across 3 sessions), and `--append` CLI workflow.

### Code Implemented
- `src/schemas.py` — Pydantic v2 data contracts (`DatasetClip`, `PauseEvent`, `StyleProfile`, `FewShotExample`, `HumanizeRequest`, `HumanizeResponse`, `SynthesizeRequest`) + JSONL/JSON persistence helpers.
- `src/humanizer.py` — Stage 1 Personalized Script Humanizer defaulting to local **Ollama/Gemma (`gemma3:4b`)** (`http://localhost:11434/v1`) with graceful offline profile fallback (`ADR-005`).
- `src/synthesizer.py` — Stage 2 F5-TTS synthesizer supporting Zero-Shot reference clips (`refs/presentation_ref.wav`) and automatic per-speaker checkpoint/LoRA loading (`lora/`) with chunking & smooth audio stitching.
- `app/web_ui.py` — Local Gradio Web UI (`http://127.0.0.1:7860`) with two tabs: **Presentation Studio** (Stage 1 + Stage 2) and **Dataset QA Reviewer** (live verified progress counter, default `Only Unverified (TODO)` filter with `✅ OK` / `⬜ TODO` badges, one-click `✅ Verify & Next Clip`, and `Rebuild style_profile.json`).
- `scripts/05_train_cloud.py` — Prepares F5-TTS fine-tuning dataset (`metadata.csv`, `duration.json`) and invokes `f5-tts_finetune-cli` with character tokenizer (`--tokenizer char`) for Cloud RTX 4090 training.
- `.env.example` — Default Ollama/Gemma Stage 1 configuration template.
- `scripts/00_extract_diarize_video.py` — Offline FFmpeg + Silero VAD + SpeechBrain ECAPA-TDNN speaker separation for multi-speaker videos.
- `scripts/01_preprocess_audio.py` — 24kHz mono WAV conversion, peak normalization (no aggressive noise gate), Silero VAD slicing (3–12s with 150ms padding).
- `scripts/02_transcribe_align.py` — `faster-whisper` transcription with disfluency-preserving prompt, Windows CUDA DLL registration, word-level pause detection, and filler density classification.
- `scripts/03_build_dataset_labels.py` — LLM-assisted (OpenAI-compatible) + `--heuristic` offline fallback generator for `clean_text` pairs and `tone`.
- `scripts/04_extract_style_profile.py` — Extracts WPM, pause cadence, normalized `filler_distribution`, and few-shot examples into `style_profile.json`.
- `requirements.txt` — Core dependencies including `f5-tts` and RTX 5060 CUDA 12.8 note.

### Local Environment Status (Primary Dev Machine)
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (`8151 MiB` VRAM, Driver `610.71`, Blackwell `sm_120`).
- **Conda Environment Configured:** `presenter-ai` (`C:\Users\mrbin\miniconda3\envs\presenter-ai`) with Python 3.11.18, FFmpeg 9.1.0, `torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`, `f5-tts==1.1.22`, `faster-whisper==1.2.1`, `silero-vad==6.3`, `gradio==6.32.0`, `pydantic==2.14.4`. Verified `torch.cuda.is_available() == True`.
- **Initial Real Dataset Processed:**
  - `vn_01` (**Binh**): **174 WAV clips (~26.8 min total)** — 90 Part A spontaneous clips (89/90 Verified, 13.72 min) + 84 new Part B scripted clips (`0091..0174`, 12.9 min from 4 iPhone 16 Lossless ALAC files at 34.7–38.0 dB SNR, pending QA review). Curated ref clip `refs/presentation_ref.wav` (10.3s), `style_profile.json` (`102.7 WPM`, `9.0 words/pause`, fillers: `uh: 77.6%`, `like: 12.9%`, `um: 4.3%`, `actually: 4.3%`, `basically: 0.9%`). Once the 84 Part B clips are QA-verified, `vn_01` (~26–27 min clean balanced dataset) is ready for our first Cloud RTX 4090 fine-tune (`PROP-003`)!
  - `vn_02` (**Ege**): 30 WAV clips (4.5 min), curated ref clip `refs/presentation_ref.wav` (10.15s), `metadata.jsonl` & `style_profile.json` (`146.6 WPM`, fillers: `uh: 79.4%`, `eh: 11.8%`, `um: 8.8%`, 9 few-shot examples).

---

## What Does NOT Exist Yet (Next Up)
1. Listen-test Zero-Shot F5-TTS output for Binh (`vn_01`) and Ege (`vn_02`) via local Web UI (`http://127.0.0.1:7860`).
2. Human QA review of `vn_01`/`vn_02` `metadata.jsonl` + additional recording to reach 30–60 min if Zero-Shot accent/filler fidelity needs LoRA fine-tuning.
3. `scripts/05_train_lora_cloud.py` — Cloud RTX 4090 F5-TTS LoRA fine-tuning script/notebook.
