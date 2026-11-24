# Roadmap

## Phase 1 — MVP (Single Speaker `vn_01` on Multi-Speaker Architecture)
- [x] Private GitHub repo created (`nguyenthaibinh19/natural-presenter-ai`) & primary workspace moved to `D:\natural-presenter-ai`
- [x] Human guide (`docs/PROJECT_GUIDE.md`), AI spec (`docs/AI_AGENT_SPEC.md`), recording guide (`docs/RECORDING_GUIDE.md`), and `.ai/` memory established
- [x] Pydantic schemas (`src/schemas.py`) and dataset scripts (`scripts/00`–`04`) implemented
- [x] Local Conda environment (`presenter-ai`) set up with PyTorch CUDA 12.8 for RTX 5060 Laptop (verified GPU pipeline end-to-end)
- [x] Initial real presentation video data extracted & processed for `vn_01` (6.0 min, 40 clips) and `vn_02` (4.5 min, 30 clips) with individualized `style_profile.json`
- [x] Stage 1 (`src/humanizer.py`, Ollama/Gemma ready), Stage 2 (`src/synthesizer.py` Zero-Shot rated 7–8/10), local Gradio UI (`app/web_ui.py` with Presentation Studio + Dataset QA Reviewer), and cloud training script (`scripts/05_train_cloud.py`) implemented
- [ ] Install local Ollama + `gemma3:4b` on dev machine for Stage 1 LLM humanization
- [x] QA-verify `metadata.jsonl` + expand `vn_01` to **173 verified clips (26.65 min balanced Part A + Part B)** & package `vn_01_f5_dataset.zip` (61 MB) + JupyterLab L40 (44GB) & Colab L4 notebooks
- [ ] Run first fine-tuning (`FT-001`) on university **NVIDIA L40 (44GB)** for `vn_01` and benchmark vs. Zero-Shot baseline (`ZS-001`)

## Phase 2 — Multi-Accent Expansion
- [ ] Onboard `speaker_02` (collaborator) and validate zero accent/filler bleed between speakers
- [ ] Onboard additional accents (e.g., Spanish, Hungarian, Cypriot English) using identical `profiles/<speaker_id>/` pipeline
- [ ] Evaluate whether LoRA alone is sufficient for strong phonetic shifts or if per-speaker full fine-tune checkpoints are needed

## Phase 3 — Product Polish & Offline Scaling
- [ ] Long-form presentation chunking & seamless crossfade stitching (10–30 min presentations)
- [ ] Slide-by-slide script management and pacing controls in Web UI
- [ ] Optional QLoRA fine-tuned local LLM for 100% offline Stage 1 humanization
