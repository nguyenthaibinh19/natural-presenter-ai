# Roadmap

## Phase 1 — MVP (Single Speaker `vn_01` on Multi-Speaker Architecture)
- [x] Private GitHub repo created (`nguyenthaibinh19/natural-presenter-ai`) & primary workspace moved to `D:\natural-presenter-ai`
- [x] Human guide (`docs/PROJECT_GUIDE.md`), AI spec (`docs/AI_AGENT_SPEC.md`), recording guide (`docs/RECORDING_GUIDE.md`), and `.ai/` memory established
- [x] Pydantic schemas (`src/schemas.py`) and dataset scripts (`scripts/01`–`04`) implemented
- [ ] Local Conda environment (`presenter-ai`) set up with PyTorch CUDA 12.8 for RTX 5060 Laptop
- [ ] First 20–30 min MVP recording (then 60–90 min full recording) for `vn_01` processed & QA-verified
- [ ] Cloud RTX 4090 fine-tuning script (`scripts/05_train_lora_cloud.py`) + first `vn_01` LoRA trained and validated
- [ ] Stage 1 (`src/humanizer.py`), Stage 2 (`src/synthesizer.py`), and local Gradio UI (`app/web_ui.py`) working end-to-end

## Phase 2 — Multi-Accent Expansion
- [ ] Onboard `speaker_02` (collaborator) and validate zero accent/filler bleed between speakers
- [ ] Onboard additional accents (e.g., Spanish, Hungarian, Cypriot English) using identical `profiles/<speaker_id>/` pipeline
- [ ] Evaluate whether LoRA alone is sufficient for strong phonetic shifts or if per-speaker full fine-tune checkpoints are needed

## Phase 3 — Product Polish & Offline Scaling
- [ ] Long-form presentation chunking & seamless crossfade stitching (10–30 min presentations)
- [ ] Slide-by-slide script management and pacing controls in Web UI
- [ ] Optional QLoRA fine-tuned local LLM for 100% offline Stage 1 humanization
