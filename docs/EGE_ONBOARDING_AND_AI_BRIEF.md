# Ege Onboarding & AI Assistant Brief

> **Audience:** Ege (Co-creator) and **Ege's AI Assistant** (ChatGPT, Claude, Antigravity, Codex, etc.).  
> **Purpose:** Give Ege and his AI immediate, complete context on what is already built, what Ege can work on next, and how Binh & Ege coordinate changes cleanly via GitHub.

---

## 1. Quick Copy-Paste Prompt for Ege's AI Assistant

If you are pasting docs into ChatGPT or starting a coding agent session, upload this file together with `docs/AI_AGENT_SPEC.md` and `.ai/current_state.md`, and use this prompt:

```text
You are assisting Ege, co-creator of "Natural Presenter AI" (a multi-speaker, accent-preserving English presentation AI built with Binh).
Read docs/EGE_ONBOARDING_AND_AI_BRIEF.md, .ai/invariants.md, .ai/current_state.md, docs/AI_AGENT_SPEC.md, and docs/PROPOSALS_AND_SYNC.md carefully.
Respect all invariants (never normalize non-native accents, never hardcode generic uh/um templates, never commit audio/model weights to Git).
Before implementing any new feature or architectural change, draft or update a proposal in docs/PROPOSALS_AND_SYNC.md so Binh and Ege can agree on it first.
When completing approved code/doc tasks, update .ai/ memory files in the same commit.
```

---

## 2. Where We Are Right Now (Project Snapshot)

Our working MVP is **already live locally** and our first Zero-Shot F5-TTS listening test scored **7–8/10**:

- **Architecture (2 Stages):**
  - **Stage 1 (`src/humanizer.py`):** Converts a clean presentation script into natural spoken English using *each speaker's own* `style_profile.json` (`uh`/`like` for Binh; `uh`/`eh`/`um` for Ege) at selectable intensity (`None` / `Low` / `Medium` / `High`). Defaults to local **Ollama + Gemma (`gemma3:4b`)** with an offline fallback.
  - **Stage 2 (`src/synthesizer.py`):** Uses **F5-TTS** (G2P-free, accent-preserving) in Zero-Shot mode today, and automatically upgrades to per-speaker fine-tuned checkpoints/LoRA once placed in `profiles/<speaker_id>/lora/`.
- **Local Web UI (`app/web_ui.py`):** Gradio app with two tabs:
  1. **Presentation Studio** — generate humanized scripts + synthesize speech.
  2. **Dataset QA Reviewer** — filter unverified (`TODO`) clips, play audio, fix transcripts, one-click `✅ Verify & Next`, and rebuild `style_profile.json`.
- **Current Real Data (stored locally, gitignored):**
  - **Binh (`vn_01`):** 90 clips (~13.9 min total — 40 clips from our Nov 7 & Nov 22 presentation videos with 39 verified + 50 new iPhone 16 Lossless clips currently being QA'd).
  - **Ege (`vn_02`):** 30 clips (~4.5 min extracted from our Nov 7 & Nov 22 presentation videos, `146.6 WPM`, fillers detected: `uh 79.4%`, `eh 11.8%`, `um 8.8%`). **Not QA-verified yet.** *(Note: we can rename `vn_02` to a custom speaker ID like `ege_01` anytime if you prefer!)*

---

## 3. What Ege Can Work On Right Now

You can pick from **Track A (Voice Data)**, **Track B (Engineering / AI)**, or both!

### Track A — Build & Clean Ege's Voice Dataset (`vn_02` / `ege_01`)
1. **Record 20–25 minutes of clean English speech** following [`docs/RECORDING_GUIDE.md`](RECORDING_GUIDE.md):
   - Use a phone (Lossless Voice Memos) or clean mic in a **soft-furnished room** (bedroom with curtains/bed — avoid empty echoey rooms).
   - **~60% spontaneous presenting** (walk through old slides, explain projects naturally with your real `uh`/`eh`/`um` pauses).
   - **~40% conversational script reading** (Part B in `docs/RECORDING_GUIDE.md`).
   - Share your raw `.wav`/`.m4a` files with Binh via Google Drive / shared folder (never push audio files to GitHub!).
2. **QA-Verify Ege's transcripts:**
   - Pull the repo, set up the Python environment (`requirements.txt`), open `python app/web_ui.py` → **Dataset QA Reviewer** tab → select **Ege** → fix any misheard technical terms in `raw_text` (keep all audible `uh`/`eh`/`um`!) and polish `clean_text`.

### Track B — Engineering & AI Tasks (Pick & Propose in `docs/PROPOSALS_AND_SYNC.md`)
1. **Stage 1 Prompt & Evaluation (`src/humanizer.py`):**
   - Benchmark local `gemma3:4b` vs `gemma3:12b` vs `qwen2.5:7b` on real presentation scripts for both Binh and Ege.
   - Improve handling of technical notation, numbers, equations, and slide transitions.
2. **Stage 2 Long-Form Audio Quality (`src/synthesizer.py`):**
   - Test 3–5 minute presentation scripts for pacing consistency, natural pauses between slides/paragraphs, and smooth crossfades.
   - Explore multi-reference selection by `tone` (`presentation_ref.wav`, `explaining_ref.wav`, `emphasis_ref.wav`).
3. **Cloud Fine-Tuning Workflow (`scripts/05_train_cloud.py`):**
   - Verify end-to-end RunPod / Vast.ai RTX 4090 training commands and add automated validation sample generation every 500 steps so we can compare checkpoints easily.
4. **Product & UI Ideas (`app/web_ui.py`):**
   - Slide-by-slide script editor, pause tag controls, export to `.mp3`, or anything else you envision!

---

## 4. How Binh & Ege Collaborate (Simple Rules)

1. **Propose Before Big Changes:** Whenever you (or your AI) want to add a feature, change an architecture decision, or pick up a workstream, add a short entry in **[`docs/PROPOSALS_AND_SYNC.md`](PROPOSALS_AND_SYNC.md)** with status `PROPOSED`. Once Binh marks it `APPROVED`, we build it together!
2. **Work on Branches or Clean Commits:** Pull `main` before starting; use a short feature branch (e.g., `ege/stage1-gemma-tuning`) or clean PRs when editing code so we never clash.
3. **Keep `.ai/` Continuously Updated (Standing Rule):** Every completed task, architectural decision, experiment result, or bug fix MUST update the relevant `.ai/` files (`.ai/current_state.md`, `.ai/experiments.md`, `.ai/lessons.md`, or a new ADR in `.ai/decisions/`) in the same commit.
4. **Never Commit Heavy Local Files:** `profiles/`, `*.wav`, `*.m4a`, `*.mp4`, `*.pt`, `*.safetensors` are gitignored. Share audio datasets and trained checkpoints via Google Drive or a private Hugging Face repo.

---

## 5. Mandatory Reading Order for Ege's AI Agent

1. [`AGENTS.md`](../AGENTS.md) — Root standing rules
2. [`.ai/README.md`](../.ai/README.md) — Durable project memory index
3. [`.ai/current_state.md`](../.ai/current_state.md) — Live status snapshot
4. [`.ai/invariants.md`](../.ai/invariants.md) — 8 must-not-break rules
5. [`docs/PROPOSALS_AND_SYNC.md`](PROPOSALS_AND_SYNC.md) — Active proposals & Binh ↔ Ege agreement board
6. [`docs/AI_AGENT_SPEC.md`](AI_AGENT_SPEC.md) — Full schemas & technical contracts
