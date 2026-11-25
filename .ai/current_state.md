# Current State

**Last Updated:** 2026-11-25  
**Active Phase:** Phase 2 — Portfolio Showcase Ready + Multi-Speaker Expansion (`vn_02` / Ege)  
**Repository:** `https://github.com/nguyenthaibinh19/natural-presenter-ai`  
**Environment:** Local Windows dev laptop (RTX 5060 Laptop 8GB VRAM) + Cloud Colab fine-tuning

---

## What Exists Today

### Documentation & Governance
- `.ai/` durable project memory system initialized (modeled on `EnglishApp` conventions), including `ADR-001` through `ADR-006` and `experiments.md` (`FT-001` rated 9.5–10/10).
- `AGENTS.md` — root auto-read entrypoint for AI coding agents.
- `docs/EGE_ONBOARDING_AND_AI_BRIEF.md` — onboarding brief, copy-paste AI prompt, and active work tracks for Ege & Ege's AI assistant.
- `docs/PROPOSALS_AND_SYNC.md` — shared Binh ↔ Ege proposal & approval board (`PROP-001`..`PROP-004` seeded).
- `docs/PROJECT_GUIDE.md` — human-readable English guide for collaborators.
- `docs/AI_AGENT_SPEC.md` — comprehensive machine-readable technical spec for ChatGPT / AI agents.
- `docs/RECORDING_GUIDE.md` — speaker recording checklist (including iPhone 16 Lossless ALAC + soft room tips), 8 Part A spontaneous prompts, 18 Part B conversational scripts (~11–13 min across 3 sessions), and `--append` CLI workflow.
- `docs/CUE_SCRIPT_TEMPLATE.md` — copy-paste ChatGPT prompt + complete working multi-speaker Markdown Cue Script example (`## <ID> | speaker: ... | location: ...` + `POINT TO:` / `SAY:` / `NEXT:`) for Tab 4 import.

### Code Implemented
- `src/schemas.py` — Pydantic v2 data contracts (`DatasetClip`, `PauseEvent`, `StyleProfile`, `FewShotExample`, `HumanizeRequest`, `HumanizeResponse`, `SynthesizeRequest`) + JSONL/JSON persistence helpers.
- `src/humanizer.py` — Stage 1 Long-Form Personalized Script Humanizer defaulting to local **Ollama/Gemma (`gemma3:4b`)** (`ADR-005`) with paragraph-by-paragraph disfluency budgeting (diverse fillers + optional subtle self-repairs) and automatic acronym/pronunciation normalization (`config/pronunciation_map.json`).
- `src/synthesizer.py` — Stage 2 F5-TTS synthesizer supporting Zero-Shot and auto-detected per-speaker checkpoints (`lora/`), selectable **Reference Style** (`refs/*.wav` + `.txt`), pronunciation safety pass, and multi-paragraph/slide synthesis with configurable inter-paragraph pauses (`paragraph_pause_ms`).
- `src/cue_presenter.py` — Human-Driven Live Cue Presenter (`ADR-006`): parses Reading Copy PDFs (`A1..D1`) and annotated Jupyter notebooks (`# N4a | POINT TO ... # SAY: ...`), merges sentence fragments, batch-humanizes & pre-renders per-speaker `.wav` cues, and drives 0ms-latency live OBS playback via Windows global hotkeys (`F9`/`F8`/`F10`/`F7`/`F11`) + optional Mouse Side Buttons (`pynput` + `winsound`).
- `config/pronunciation_map.json` — Editable technical acronym/term lexicon (e.g., `MSE` → `M S E`, `CNN` → `C N N`, `ReLU` → `ray-loo`, `VRAM` → `V-ram`).
- `app/web_ui.py` — Local Gradio Web UI (`http://127.0.0.1:7860`) with four tabs: **Presentation Studio**, **Dataset QA Reviewer**, **Pronunciation Dictionary**, and **Cue Presenter (Live OBS)** (pre-loaded with 22 parsed cues from `lab2_demo`).
- `scripts/05_train_cloud.py` — Prepares F5-TTS fine-tuning dataset (`metadata.csv`, `duration.json`) and invokes `f5-tts_finetune-cli` with pretrained 2545-token vocab (`--tokenizer pinyin`, raw chars for English), `--num_warmup_updates 100`, and `--bnb_optimizer`.
- `scripts/06_export_humanizer_sft.py` — Exports verified `(clean_text -> raw_text)` pairs across speakers into ChatML JSONL (`profiles/humanizer_sft.jsonl`) for optional future QLoRA fine-tuning of Gemma/Qwen Stage 1 models.
- `scripts/07_generate_showcase_samples.py` — Generates 6 neutral self-recorded author voice (`vn_01`) portfolio audio comparison clips (`01`–`06`, including slide pause + self-repair) into `hf_space/samples/`; co-author handoff clip (`vn_02`) is strictly opt-in via `--include-coauthor` pending consent.
- `scripts/08_deploy_hf_space.py` — One-command Hugging Face **Static Space** (`sdk: static`, free forever) uploader with built-in Voice Consent Guard refusing co-author audio unless explicitly consented.
- `hf_space/` — Deployed free Static Space portfolio showcase (`index.html`, `README.md`, plus 6 pre-rendered `vn_01`-only `.wav` clips in `hf_space/samples/`) live (currently Private) at `https://huggingface.co/spaces/nguyenthaibinh19/natural-presenter-ai-showcase` with instant HTML5 A/B audio comparisons and a client-side Stage 1 Playground.
- `LICENSE` — MIT License; `README.md` upgraded to portfolio showcase format with Mermaid architecture diagram, verified benchmarks, and Responsible AI / Voice Consent section; `start_presenter.bat` made portable across machines.
- `notebooks/05_finetune_f5tts_colab.ipynb` — Ready-to-run Google Colab Pro notebook (NVIDIA L4 / Tesla T4) with Google Drive auto-checkpointing and Cell 6 slim FP16 `.safetensors` exporter (~643 MB).
- `notebooks/05_finetune_f5tts_jupyter_l40.ipynb` — Ready-to-run university JupyterLab notebook tuned for **NVIDIA L40 (44.4 GB VRAM, PyTorch 2.6+cu124)** (`batch_size=6400` frames, `bf16`, full AdamW, ~15–25 min total training, inline Zero-Shot vs. Fine-Tuned playback).
- `.env.example` — Default Ollama/Gemma Stage 1 configuration template.
- `scripts/00_extract_diarize_video.py` — Offline FFmpeg + Silero VAD + SpeechBrain ECAPA-TDNN speaker separation for multi-speaker videos.
- `scripts/01_preprocess_audio.py` — 24kHz mono WAV conversion, peak normalization (no aggressive noise gate), Silero VAD slicing (3–12s with 150ms padding).
- `scripts/02_transcribe_align.py` — `faster-whisper` transcription with disfluency-preserving prompt, Windows CUDA DLL registration, word-level pause detection, and filler density classification.
- `scripts/03_build_dataset_labels.py` — LLM-assisted (OpenAI-compatible) + `--heuristic` offline fallback generator for `clean_text` pairs and `tone`.
- `scripts/04_extract_style_profile.py` — Extracts WPM, pause cadence, normalized `filler_distribution`, and few-shot examples into `style_profile.json`.
- `requirements.txt` — Core dependencies including `f5-tts` and RTX 5060 CUDA 12.8 note.

### Local Environment Status (Primary Dev Machine)
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (`8151 MiB` VRAM, Driver `610.71`, Blackwell `sm_120`).
- **Conda Environment Configured:** `presenter-ai` (`%USERPROFILE%\miniconda3\envs\presenter-ai`) with Python 3.11.18, FFmpeg 9.1.0, `torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`, `f5-tts==1.1.22`, `faster-whisper==1.2.1`, `silero-vad==6.3`, `gradio==6.32.0`, `pydantic==2.14.4`, `pynput==1.8.2`, `pypdf==7.1.1`. Verified `torch.cuda.is_available() == True`.
- **Local LLM Configured (Stage 1):** **Ollama `0.40.18`** installed locally with **`gemma3:4b`** pulled and verified live at `http://localhost:11434/v1` (`.env` configured). Automatically expands symbols (`Q3` → `Q three`, `24%` → `24 percent`, `3x3` → `three by three`) and injects speaker-specific fillers from `style_profile.json`. `173` verified `vn_01` pairs exported to `profiles/humanizer_sft.jsonl` for future optional Gemma QLoRA fine-tuning.
- **Initial Real Dataset Processed:**
  - `vn_01` (**Binh**): **173 / 174 Verified clips (26.65 min clean, 0 integrity errors — only `vn_01_0090` excluded)**:
    - Part A (spontaneous): 89 clips (13.72 min)
    - Part B (scripted): 84 clips (12.93 min, iPhone 16 Lossless ALAC 34.7–38.0 dB SNR)
    - `style_profile.json` rebuilt (`102.7 WPM`, `9.0 words/pause`, fillers: `uh: 77.6%`, `like: 12.9%`, `um: 4.3%`, `actually: 4.3%`, `basically: 0.9%`, 12 verified few-shot examples).
    - **First Fine-Tuned Model Live Locally (`FT-001`):** `profiles/vn_01/lora/vn_01_model_4000_fp16.safetensors` (**643 MB**, pruned EMA FP16 from 4,020 updates / 60 epochs on Colab Tesla T4, final loss `0.408`). Verified on local RTX 5060: 6.5s audio synthesized in 5.3s (16 steps).
  - `vn_02` (**Ege**): 30 WAV clips (4.5 min), curated ref clip `refs/presentation_ref.wav` (10.15s), `metadata.jsonl` & `style_profile.json` (`146.6 WPM`, fillers: `uh: 79.4%`, `eh: 11.8%`, `um: 8.8%`, 9 few-shot examples). Zero-Shot active; awaiting Ege's QA & additional recordings (`PROP-001`).

---

## What Does NOT Exist Yet (Next Up — Phase 2)
1. Onboard Ege's expanded & QA-verified dataset (`vn_02` / `ege_01`, `PROP-001`) and fine-tune Ege's checkpoint (`FT-002`).
2. Benchmark/tune Stage 1 prompt or QLoRA fine-tune Gemma on multi-speaker `humanizer_sft.jsonl` once Ege + more speakers are verified (`PROP-002`).
3. Slide-by-slide multi-speaker presentation mode in Web UI (`PROP-004`).
