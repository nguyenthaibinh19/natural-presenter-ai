# Roadmap

## Phase 1 — MVP (Single Speaker `vn_01` on Multi-Speaker Architecture)
- [x] Private GitHub repo created (`nguyenthaibinh19/natural-presenter-ai`) & primary workspace moved to `D:\natural-presenter-ai`
- [x] Human guide (`docs/PROJECT_GUIDE.md`), AI spec (`docs/AI_AGENT_SPEC.md`), recording guide (`docs/RECORDING_GUIDE.md`), and `.ai/` memory established
- [x] Pydantic schemas (`src/schemas.py`) and dataset scripts (`scripts/00`–`04`) implemented
- [x] Local Conda environment (`presenter-ai`) set up with PyTorch CUDA 12.8 for RTX 5060 Laptop (verified GPU pipeline end-to-end)
- [x] Initial real presentation video data extracted & processed for `vn_01` (6.0 min, 40 clips) and `vn_02` (4.5 min, 30 clips) with individualized `style_profile.json`
- [x] Stage 1 (`src/humanizer.py`, Ollama/Gemma ready), Stage 2 (`src/synthesizer.py` Zero-Shot rated 7–8/10), local Gradio UI (`app/web_ui.py` with Presentation Studio + Dataset QA Reviewer), and cloud training script (`scripts/05_train_cloud.py`) implemented
- [x] Install local Ollama (`0.40.18`) + `gemma3:4b` on dev machine for Stage 1 LLM humanization + add `scripts/06_export_humanizer_sft.py` for future Stage 1 QLoRA fine-tuning (**Phase 1 MVP 100% Complete!**)
- [x] QA-verify `metadata.jsonl` + expand `vn_01` to **173 verified clips (26.65 min balanced Part A + Part B)** & package `vn_01_f5_dataset.zip` (61 MB) + JupyterLab L40 (44GB) & Colab L4 notebooks
- [x] Run first fine-tuning (`FT-001`) for `vn_01` (Colab Tesla T4, 60 epochs / 4,020 updates, loss `0.408`), pruned to 643 MB `vn_01_model_4000_fp16.safetensors`, and verified live on local RTX 5060 (faster than real time)

## Phase 2 — Multi-Accent Expansion
- [ ] Onboard `speaker_02` (collaborator) and validate zero accent/filler bleed between speakers
- [ ] Onboard additional accents (e.g., Spanish, Hungarian, Cypriot English) using identical `profiles/<speaker_id>/` pipeline
- [ ] Evaluate whether LoRA alone is sufficient for strong phonetic shifts or if per-speaker full fine-tune checkpoints are needed

## Phase 3 — Product Polish & Offline Scaling
- [x] Long-form presentation chunking & seamless inter-paragraph pause stitching (`src/synthesizer.py`)
- [x] Human-driven live Cue Presenter Studio (`ADR-006`, `src/cue_presenter.py` + Web UI Tab 4 with PDF/ipynb cue parser, pre-rendered audio, and global `F9`/`Mouse4` hotkeys for OBS recording)
- [ ] Optional QLoRA fine-tuned local LLM for 100% offline Stage 1 humanization
