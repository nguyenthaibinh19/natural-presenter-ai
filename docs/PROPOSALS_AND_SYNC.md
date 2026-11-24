# Proposals & Alignment Board (Binh ↔ Ege)

> **Purpose:** Shared space for **Binh** and **Ege** (and their AI assistants) to propose ideas, discuss scope, and formally agree on what to build next before investing time.

## How This Board Works
1. **Propose:** Anyone (Ege, Binh, or an AI assistant on their behalf) adds a new `PROP-XXX` section below with status `PROPOSED` and pushes/opens a PR.
2. **Discuss:** Leave comments/questions directly under the proposal's **Discussion Notes** section.
3. **Approve:** Once **both** `Binh` and `Ege` mark `✅ Approved`, change Status to **`APPROVED`** — the owner can start implementing immediately!
4. **Close:** When completed, update Status to **`DONE`** and mirror durable outcomes into `.ai/current_state.md` / `.ai/roadmap.md` (or a new ADR if architectural).

**Status Values:** `PROPOSED` | `DISCUSSING` | `APPROVED` | `IN PROGRESS` | `DONE` | `DEFERRED`

---

## Summary Table

| ID | Title | Proposed By | Status | Binh | Ege |
|---|---|---|---|---|---|
| `PROP-001` | Rename Ege's speaker ID & accent tag (optional) + build 25–30 min Ege dataset | Binh | `PROPOSED` | ✅ Approved | ⏳ Pending |
| `PROP-002` | Benchmark local Ollama/Gemma Stage 1 humanizer prompts | Binh | `PROPOSED` | ✅ Approved | ⏳ Pending |
| `PROP-003` | First Cloud RTX 4090 F5-TTS fine-tune run (`vn_01` ~30 min verified) | Binh | `PROPOSED` | ✅ Approved | ⏳ Pending |
| `PROP-004` | Slide-by-slide presentation mode in Web UI | Binh | `PROPOSED` | ⏳ Open to discuss | ⏳ Pending |

---

## Copy-Paste Template for New Proposals

```markdown
### PROP-XXX: <Short Imperative Title>
- **Proposed By:** Ege / Binh
- **Status:** `PROPOSED`
- **Decision:** Binh: ⏳ Pending | Ege: ⏳ Pending
- **Problem / Idea:** What are we solving or improving?
- **Proposed Approach:** Concrete technical or dataset steps.
- **Invariant Check (`.ai/invariants.md`):** Does this preserve accents, individualized fillers, and per-speaker isolation? (Yes / Explain)
- **Estimated Effort / Cost:** e.g., ~2 hours dev, $0
- **Discussion Notes:**
  - *Ege:* ...
  - *Binh:* ...
```

---

## Active Proposals

### PROP-001: Ege Speaker Profile Naming & 25–30 Min Dataset Expansion
- **Proposed By:** Binh
- **Status:** `PROPOSED`
- **Decision:** Binh: ✅ Approved | Ege: ⏳ Pending
- **Problem / Idea:** Ege currently has 30 clips (4.5 min) extracted from our Nov 7 & Nov 22 presentation videos under temporary ID `vn_02` (`vi-EN`). Before QA and fine-tuning, we should set Ege's preferred `speaker_id` and `accent` tag (e.g., `ege_01`, `tr-EN` or preferred tag) and expand the dataset to 25–30 minutes.
- **Proposed Approach:**
  1. Ege confirms desired `speaker_id` and `accent` label.
  2. Ege records ~20–25 min following `docs/RECORDING_GUIDE.md` (phone Lossless in a quiet soft-furnished room) and shares raw audio via Drive.
  3. Run `01`–`04` with `--append` and QA-verify in the Dataset QA Reviewer tab.
- **Invariant Check:** Yes — preserves Ege's distinct accent and individualized `uh/eh/um` filler profile.
- **Estimated Effort / Cost:** ~30 min recording + ~30 min QA, $0.
- **Discussion Notes:**
  - *Binh:* Let me know what `speaker_id` and accent tag you want, and feel free to add any topics you want included!
  - *Ege:* *(add your notes here)*

---

### PROP-002: Benchmark & Refine Stage 1 Prompt with Local Ollama + Gemma
- **Proposed By:** Binh
- **Status:** `PROPOSED`
- **Decision:** Binh: ✅ Approved | Ege: ⏳ Pending
- **Problem / Idea:** Stage 1 (`src/humanizer.py`) currently defaults to `gemma3:4b` via local Ollama (`ADR-005`). We should test how well `gemma3:4b` follows each speaker's `style_profile.json` across `None / Low / Medium / High` and refine the prompt (or test `gemma3:12b` / QLoRA fine-tuning if needed).
- **Proposed Approach:**
  1. Test 5 real presentation paragraphs for both Binh and Ege across all 4 filler levels.
  2. Verify semantic accuracy (zero hallucinated facts, clean expansion of numbers/acronyms like `Q3`, `3x3`, `%`).
  3. Log findings in `.ai/experiments.md` and tune `build_llm_system_prompt()` in `src/humanizer.py`.
- **Invariant Check:** Yes — strictly driven by per-speaker `style_profile.json`.
- **Estimated Effort / Cost:** ~2 hours, $0 (100% local via Ollama).
- **Discussion Notes:**
  - *Binh:* Great task if Ege (or Ege's AI) wants to experiment with prompt engineering & LLM evaluation!
  - *Ege:* *(add your notes here)*

---

### PROP-003: First Cloud RTX 4090 Fine-Tuning Run (`vn_01` ~30 Min Verified)
- **Proposed By:** Binh
- **Status:** `PROPOSED`
- **Decision:** Binh: ✅ Approved | Ege: ⏳ Pending
- **Problem / Idea:** Zero-Shot F5-TTS (`ZS-001`) already scored 7–8/10. Once Binh finishes QA-verifying ~30 min of clean recordings, we want to run our first Cloud RTX 4090 fine-tune (`scripts/05_train_cloud.py`) and measure how much accent + filler realism improves (target: 9–9.5/10).
- **Proposed Approach:**
  1. Export verified clips via `python scripts/05_train_cloud.py --speaker vn_01 --only-verified --prepare-only`.
  2. Rent 1x RTX 4090 on RunPod/Vast.ai (~1–2 hours, ~$1–$2), train with `--tokenizer char`, listen to checkpoints every 500 steps.
  3. Download best checkpoint to `profiles/vn_01/lora/`, terminate cloud instance, and log results as `FT-001` in `.ai/experiments.md`.
- **Invariant Check:** Yes — character tokenizer preserves non-native accent; per-speaker isolation preserved.
- **Estimated Effort / Cost:** ~2 hours, ~$1–$2 total cloud GPU cost.
- **Discussion Notes:**
  - *Binh:* Once `vn_01` proves the cloud workflow works cleanly, we run the exact same pipeline for Ege!
  - *Ege:* *(add your notes here)*

---

### PROP-004: Slide-by-Slide Presentation Script Mode in Web UI
- **Proposed By:** Binh
- **Status:** `PROPOSED`
- **Decision:** Binh: ⏳ Open to discuss | Ege: ⏳ Pending
- **Problem / Idea:** Real presentations are split by slides (Slide 1, Slide 2, ...) with slightly longer pauses between slides and sometimes different speakers presenting different slides (e.g., Ege presents Slides 1–4, Binh presents Slides 5–8!).
- **Proposed Approach:**
  - Allow inputting scripts separated by markers like `[SLIDE 1 | speaker=ege_01 | filler=medium]` and `[SLIDE 2 | speaker=vn_01 | filler=low]`, generating both per-slide `.wav` files and a stitched full-presentation audio file.
- **Invariant Check:** Yes — leverages existing multi-speaker profile isolation cleanly.
- **Estimated Effort / Cost:** ~3–4 hours dev (Phase 2/3), $0.
- **Discussion Notes:**
  - *Binh:* Could be super useful for joint presentations where both of us speak! Let me know if you want to prioritize this after our first fine-tune.
  - *Ege:* *(add your notes here)*
