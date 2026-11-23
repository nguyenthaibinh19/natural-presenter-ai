# AI Agent Technical Specification — Natural Presenter AI

> **Instructions for AI Assistants (ChatGPT / Claude / Coding Agents):**  
> This document is the authoritative technical specification for **Natural Presenter AI**. Treat it as your complete project context when helping the developers design, implement, debug, or extend any part of this system. Do not simplify away the core requirements around **non-native accent preservation** or **per-speaker individualized disfluency modeling**.

---

## 1. System Objective & Core Constraints

Build a multi-speaker, accent-preserving English presentation synthesis system that transforms a clean, polished presentation script into natural, human-sounding speech matching a specific target speaker.

### Hard Requirements (Non-Negotiable)
1. **Language:** English only (no Vietnamese or other target languages required at this stage).
2. **Authentic Non-Native Accent Preservation:** The TTS model MUST preserve the speaker's real English pronunciation, consonant endings, vowel shifts, stress patterns, and intonation (e.g., Vietnamese-accented English, Cypriot-accented English, Hungarian-accented English, Spanish-accented English). It MUST NOT "normalize" or "correct" speech toward standard US/UK accents.
3. **Individualized Disfluency & Rhythm Modeling:** Every speaker has unique hesitation patterns (`uh`, `um`, `eh`, `erm`, `you know`, `so,`, `actually,`, micro-pauses, breaths). The system MUST learn and reproduce **each speaker's own disfluency distribution and placement habits** — never apply a generic, hardcoded filler template across users.
4. **Controllable Naturalness Level:** At inference time, the user supplies a polished script and selects a disfluency intensity level: `none` | `low` | `medium` | `high`. The user MUST be able to inspect and manually edit the intermediate humanized script before audio synthesis.
5. **Multi-Speaker Scalability:** Adding a new speaker MUST NOT require retraining a monolithic multi-speaker model from scratch and MUST NOT cause accent bleed between speakers.
6. **Hardware Constraint:**
   - **Local machine (Inference & Dataset Prep):** Windows laptop with **NVIDIA RTX 5060 Laptop GPU (8GB VRAM, Blackwell architecture `sm_120`)**. Note: requires PyTorch >= 2.7 with CUDA 12.8+ support for Blackwell compatibility.
   - **Cloud machine (One-time Training per Speaker):** Rented GPU (e.g., RunPod / Vast.ai **RTX 4090 24GB VRAM**). Budget target: ~$1–$2 per speaker LoRA.

---

## 2. High-Level Architecture

The system is decoupled into two independent stages connected by a **Speaker Profile** directory (`profiles/<speaker_id>/`).

```text
Inputs:
  - clean_script: string
  - speaker_id: string (e.g., "vn_01", "es_01", "hu_01", "cy_01")
  - filler_level: "none" | "low" | "medium" | "high"
  - tone: "presentation" | "explaining" | "emphasis"
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Personalized Script Humanizer (LLM)                        │
│                                                                     │
│ Loads: profiles/<speaker_id>/style_profile.json                     │
│   - speaker's filler vocabulary & relative frequencies              │
│   - pause cadence stats (words per breath/pause group)              │
│   - placement rules (clause boundaries, topic shifts, before nouns) │
│   - 10-15 real (clean_text -> raw_text) few-shot pairs              │
│                                                                     │
│ Output: humanized_script (editable by user in UI)                   │
└─────────────────────────────────────────────────────────────────────┘
                 │
       [Human Review / Edit Gate]
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Accent-Preserving Voice Synthesizer (TTS)                  │
│                                                                     │
│ Base Model: F5-TTS (flow-matching, G2P-free text-to-audio)          │
│ Adapter:    profiles/<speaker_id>/lora/adapter.safetensors          │
│ Reference:  profiles/<speaker_id>/refs/<tone>_ref.wav + .txt        │
│                                                                     │
│ Process:                                                            │
│   1. Chunk humanized_script into 1-2 sentence segments (~5-15s)     │
│   2. Synthesize each chunk with speaker's LoRA + reference audio    │
│   3. Crossfade & stitch chunks with natural inter-sentence pauses   │
│                                                                     │
│ Output: presentation_output.wav / .mp3                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Architectural Rationale
- **Why F5-TTS (Primary) / GPT-SoVITS (Fallback):** Traditional TTS models (VITS, Piper, Tacotron) pass English text through a Grapheme-to-Phoneme (G2P) dictionary (e.g., CMUdict / espeak-ng), which forces textbook US/UK phonemes and destroys non-native accents. F5-TTS uses flow matching over character/byte-level representations without rigid G2P normalization, allowing fine-tuning to faithfully capture non-native phonetics, prosody, and breathing.
- **Why Per-Speaker LoRA Adapters:** Storing a ~30–100 MB LoRA adapter per speaker enables clean multi-tenant isolation, zero cross-accent interference, fast hot-swapping on an 8GB VRAM laptop, and low cloud training cost per new user.

---

## 3. Target Repository & Profile Directory Layout

```text
natural-presenter-ai/
├── README.md
├── .gitignore
├── docs/
│   ├── PROJECT_GUIDE.md
│   └── AI_AGENT_SPEC.md
├── scripts/
│   ├── 01_preprocess_audio.py       # Mono 24kHz WAV conversion, gentle denoise (preserve breaths)
│   ├── 02_transcribe_align.py       # WhisperX alignment with disfluency preservation + pause detection
│   ├── 03_build_dataset_labels.py   # LLM-assisted clean_text generation & filler_level labeling
│   ├── 04_extract_style_profile.py  # Aggregates metadata.jsonl -> style_profile.json
│   ├── 05_train_lora_cloud.py       # Training entrypoint for F5-TTS LoRA on Cloud RTX 4090
│   └── 06_infer_presentation.py     # CLI end-to-end inference (Stage 1 + Stage 2)
├── src/
│   ├── humanizer.py                 # Stage 1 LLM client + prompt builder + profile loader
│   ├── synthesizer.py               # Stage 2 F5-TTS loader + LoRA hot-swap + chunk stitching
│   └── schemas.py                   # Pydantic models for all JSON schemas
├── app/
│   └── web_ui.py                    # Gradio/Streamlit local UI
└── profiles/                        # GITIGNORED (local / cloud storage only)
    └── <speaker_id>/                # e.g., vn_01, es_01, hu_01, cy_01
        ├── raw_recordings/          # Uncut source WAV files
        ├── wavs/                    # Sliced 3-12s WAV clips (24kHz, mono, 16-bit PCM)
        ├── metadata.jsonl           # Master labeled dataset for this speaker
        ├── style_profile.json       # Extracted speaking & disfluency fingerprint
        ├── refs/                    # 2-4 curated 5-10s reference clips + transcripts by tone
        └── lora/
            └── model_lora.safetensors
```

---

## 4. Exact Data Schemas

### 4.1 `profiles/<speaker_id>/metadata.jsonl` (One JSON object per audio clip)

```json
{
  "clip_id": "vn_01_0042",
  "audio_path": "wavs/vn_01_0042.wav",
  "duration_sec": 7.42,
  "speaker_id": "vn_01",
  "accent": "vi-EN",
  "recording_type": "spontaneous",
  "raw_text": "So, um... when we look at the Q3 numbers, uh, you can see a clear shift in retention.",
  "clean_text": "When we look at the Q3 numbers, you can see a clear shift in retention.",
  "fillers_used": ["so", "um", "uh"],
  "filler_count": 3,
  "filler_level": "medium",
  "pause_events": [
    {"after_word": "um", "duration_ms": 480, "tag": "[pause_long]"},
    {"after_word": "numbers", "duration_ms": 260, "tag": "[pause_short]"}
  ],
  "tone": "presentation",
  "verified": true
}
```

**Critical Data Integrity Rules:**
- `raw_text` is the **ground truth for TTS training**. Every audible filler (`uh`, `um`, `eh`, `erm`) MUST appear in `raw_text` using consistent spelling per speaker. If an audio clip contains an audible filler that is omitted from `raw_text`, the clip MUST be flagged and corrected before training to prevent alignment drift or stuttering hallucinations.
- `clean_text` is paired with `raw_text` exclusively for **Stage 1 (Script Humanizer)** profile extraction and optional LLM fine-tuning.
- `filler_level` classification heuristic (per ~20 words):
  - `none`: 0 fillers
  - `low`: 1 filler
  - `medium`: 2–3 fillers
  - `high`: 4+ fillers or multiple long pauses

### 4.2 `profiles/<speaker_id>/style_profile.json`

```json
{
  "speaker_id": "vn_01",
  "display_name": "Speaker 01 (Vietnamese English)",
  "accent": "vi-EN",
  "speaking_rate_wpm": 138,
  "avg_words_per_pause": 9.4,
  "filler_distribution": {
    "uh": 0.45,
    "um": 0.30,
    "so": 0.15,
    "actually": 0.10
  },
  "common_filler_positions": [
    "sentence_start_transition",
    "before_technical_noun_or_metric",
    "after_introductory_clause"
  ],
  "punctuation_conventions": {
    "short_pause": ",",
    "hesitation_pause": "...",
    "clause_break": " — "
  },
  "filler_targets_per_100_words": {
    "none": 0,
    "low": 2,
    "medium": 5,
    "high": 9
  },
  "few_shot_examples": [
    {
      "filler_level": "medium",
      "clean_text": "The main challenge is that our onboarding flow takes too long for new users.",
      "raw_text": "So, the main challenge is that, uh, our onboarding flow... um, it takes too long for new users."
    },
    {
      "filler_level": "low",
      "clean_text": "Let's move on to the architecture diagram.",
      "raw_text": "Alright, uh, let's move on to the architecture diagram."
    }
  ]
}
```

---

## 5. Dataset Collection & Preprocessing Pipeline Specification

### 5.1 Recording Specification (Per Speaker)
- **Total Clean Duration:** 60–90 minutes (minimum 30 minutes for initial LoRA validation).
- **Content Mix:**
  - **~60% Spontaneous Presentation Speech:** Unscripted explanations of slides, projects, or familiar topics. Primary source of authentic accent, cadence, and real `uh`/`um`/`eh` habits.
  - **~40% Conversational Script Reading:** Varied English sentences covering numbers, technical terms, questions, lists, and emphasis.
- **Audio Format:** WAV, Mono, 24kHz (or 44.1kHz downsampled cleanly to 24kHz), 16-bit PCM, peak normalized to -3 dBFS.
- **Denoising Constraint:** Use light stationary noise reduction only (e.g., RNNoise / gentle spectral subtraction). **NEVER apply an aggressive noise gate** that strips subtle breath intakes between clauses.

### 5.2 Automated Preprocessing Workflow
1. **VAD Segmentation (`01_preprocess_audio.py`):** Split long recordings at natural silences (>350ms) into clips of **3.0 to 12.0 seconds**, keeping 100–150ms of natural padding at clip boundaries so starts/ends are never clipped.
2. **Disfluency-Preserving Transcription (`02_transcribe_align.py`):**
   - Run **WhisperX** (or `faster-whisper` with `initial_prompt` explicitly containing disfluencies, e.g.: `"Umm, uh, so, like, eh, you know, let's talk about this..."` which strongly biases Whisper to retain verbatim fillers rather than smoothing them out).
   - Extract word-level timestamps to detect internal silences (`200–400ms` → `[pause_short]`, `>400ms` → `[pause_long]`).
3. **Human Verification Pass (Targeted):** Prioritize manual review on clips flagged with low Whisper confidence or long internal pauses to ensure `uh`/`um`/`eh` tags in `raw_text` match audio 100%.
4. **Automated Pairing & Profile Extraction (`03_build_dataset_labels.py`, `04_extract_style_profile.py`):**
   - Pass each verified `raw_text` to an LLM to generate the corresponding polished `clean_text` and tag `tone`.
   - Compute `filler_distribution`, `avg_words_per_pause`, and select representative `few_shot_examples` across `low`, `medium`, and `high` levels into `style_profile.json`.

---

## 6. Stage 1 Technical Specification: Personalized Script Humanizer

### Implementation Options
- **Phase 1 (Fast & High Quality):** Prompt-engineered LLM (via API such as Gemini / OpenAI / Anthropic, or local Ollama with Qwen2.5-7B / Llama-3.1-8B) dynamically hydrated with `profiles/<speaker_id>/style_profile.json`.
- **Phase 2 (Optional Scaling):** Fine-tune a small open-weight LLM (e.g., Qwen2.5-3B/7B via QLoRA) on aggregated `(speaker_style_summary, filler_level, clean_text) -> raw_text` pairs across all onboarded speakers for 100% offline, ultra-consistent style transfer.

### Stage 1 Prompt Contract
The Stage 1 prompt MUST enforce:
1. **Strict Semantic Preservation:** Never alter facts, numbers, technical terms, or the core argument of `clean_script`.
2. **Speaker-Specific Vocabulary Only:** Only use filler tokens present in `style_profile.filler_distribution`, roughly matching their relative frequencies.
3. **Intensity Control:** Match `filler_targets_per_100_words[filler_level]`. When `filler_level == "none"`, insert zero fillers and only optimize punctuation for natural spoken phrasing.
4. **Natural Placement:** Place hesitations at realistic cognitive boundaries (topic transitions, before complex terms/examples, after discourse markers) — never mid-compound-noun or mechanically every N words.
5. **TTS-Friendly Formatting:** Expand ambiguous symbols/abbreviations into speakable English (e.g., `"$2.5M"` → `"two point five million dollars"`, `"Q3"` → `"Q three"`, `"e.g."` → `"for example"`).

---

## 7. Stage 2 Technical Specification: Multi-Speaker Voice Synthesis

### Base Model & Fine-Tuning
- **Base Model:** `F5-TTS` (e.g., `F5TTS_v1_Base`).
- **Fine-Tuning Strategy:** Per-speaker LoRA (or lightweight full fine-tune saved as separate speaker checkpoint if LoRA underfits strong non-native phonetic shifts — benchmark both on Speaker 01).
- **Training Hardware:** Cloud RTX 4090 (24GB VRAM, RunPod / Vast.ai), mixed precision `bf16`.
- **Starting Hyperparameters (F5-TTS Fine-Tune, ~60–90 min dataset):**
  - Learning rate: `1e-5` to `5e-5` (warmup ~100–500 steps, cosine decay)
  - Batch size: dynamic frame-based batching (~3200–6400 frames per GPU on 24GB VRAM)
  - Epochs / Steps: Monitor validation audio every 500 steps; stop once accent + fillers are crisp without overfitting/metallic artifacts (typically 30–90 minutes on RTX 4090).
- **Inference Hardware:** Local RTX 5060 Laptop (8GB VRAM).
  - Load `F5-TTS` in `fp16`/`bf16` (~2–3 GB VRAM).
  - Hot-swap speaker LoRA weights + select matching reference clip (`refs/<tone>_ref.wav`) for the chosen `speaker_id`.
- **Long-Form Presentation Stitching:**
  - Split `humanized_script` into chunks of ~1–2 sentences (max ~200–250 characters per chunk) to prevent drift or speed-up artifacts.
  - Synthesize chunks sequentially with consistent reference audio and seed control.
  - Concatenate with configurable natural inter-sentence silence (300–600ms) and short crossfades (15–30ms) + final loudness normalization (-16 LUFS for presentation audio).

---

## 8. Local Web UI Requirements (`app/web_ui.py`)

Framework: **Gradio** (recommended for rapid audio UI) or **Streamlit**.

Required UI Flow:
1. **Speaker Selector Dropdown:** Dynamically lists all folders in `profiles/` with display name & accent tag.
2. **Clean Script Input Box:** Multi-line text area for the user's polished presentation script.
3. **Controls:**
   - Filler Intensity Radio/Slider: `None` | `Low` | `Medium` | `High`
   - Tone Selector: `Presentation` | `Explaining` | `Emphasis`
   - Speed Multiplier: `0.85x` – `1.15x` (default `1.0x`)
4. **Button: "1. Humanize Script":** Calls Stage 1 and populates an **Editable Humanized Script** text box so the user can tweak any `uh`/`um`/pause manually.
5. **Button: "2. Generate Presentation Audio":** Calls Stage 2 on the edited script, displays progress per chunk, plays audio inline, and provides a `.wav`/`.mp3` download button.

---

## 9. Known Risks & Technical Mitigations

| Risk | Cause | Mitigation |
| :--- | :--- | :--- |
| **Whisper drops `uh`/`um` during dataset prep** | Whisper's default decoding smooths disfluencies | Pass disfluency-rich `initial_prompt` to WhisperX + run targeted QA review on clips with internal pauses |
| **Stuttering or repeated words at inference** | `raw_text` omitted audible fillers present in training WAVs | Enforce strict 1:1 `raw_text` ↔ audio match during dataset QA |
| **Voice sounds too "native" (accent lost)** | Too few fine-tuning steps or G2P-based model used | Use G2P-free F5-TTS; train sufficient steps on speaker's dataset; evaluate accent fidelity every 500 steps |
| **Robotic/dead silence between clauses** | Over-aggressive noise gating removed natural breaths | Use gentle stationary denoising only; keep natural breath intakes |
| **RTX 5060 Laptop CUDA incompatibility** | Blackwell (`sm_120`) requires newer CUDA/PyTorch | Pin PyTorch >= 2.7 + CUDA 12.8 in local environment setup |

---

## 10. Suggested Next Implementation Tasks (For AI Agents)

When asked to start coding, execute in this order:
1. Create `src/schemas.py` with Pydantic models for `DatasetClip`, `StyleProfile`, `HumanizeRequest`, and `SynthesizeRequest`.
2. Build `scripts/01_preprocess_audio.py` and `scripts/02_transcribe_align.py` (VAD slicing + WhisperX disfluency-preserving transcription + pause measurement).
3. Build `scripts/03_build_dataset_labels.py` and `scripts/04_extract_style_profile.py` to generate `clean_text` pairs and `style_profile.json`.
4. Build `src/humanizer.py` implementing Stage 1 with `style_profile.json` injection and intensity controls (`none`/`low`/`medium`/`high`).
5. Create `scripts/05_train_lora_cloud.py` (or Jupyter notebook) for RunPod/Vast.ai F5-TTS fine-tuning.
6. Build `src/synthesizer.py` and `app/web_ui.py` for local RTX 5060 inference and presentation audio export.
