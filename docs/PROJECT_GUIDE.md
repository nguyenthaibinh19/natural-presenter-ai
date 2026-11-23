# Natural Presenter AI — Project Guide

> **Audience:** Project collaborators and contributors.  
> If you want to feed this project to ChatGPT or another AI assistant for deep technical help or coding, use **[AI_AGENT_SPEC.md](AI_AGENT_SPEC.md)** instead.

---

## 1. Vision & Problem Statement

We want to build an AI system that can deliver English presentations on behalf of real people — **sounding genuinely like that specific human**, not like a generic synthetic voice.

Standard commercial or open-source TTS tools have two major flaws for this use case:
1. **They "over-correct" accents:** They force text through standard American or British phonetic dictionaries, erasing a speaker's real accent (such as Vietnamese, Cypriot, Hungarian, or Spanish English).
2. **They read too cleanly:** Real humans presenting ideas naturally pause, breathe, emphasize, and use subtle hesitation markers (`uh`, `um`, `eh`, `so...`). Without these, long presentations sound robotic.

**Our goal:** Allow a user to paste a clean, perfected presentation script, choose how natural/conversational they want it (`None`, `Low`, `Medium`, `High` hesitation level), and generate an audio presentation in **their own voice, accent, intonation, and personal speaking rhythm**.

---

## 2. How the System Works (2-Stage Pipeline)

Instead of asking a single model to do everything, we split the problem cleanly into two stages:

```text
[Clean Presentation Script] + [Speaker Selection] + [Filler Level: None / Low / Medium / High]
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │  STAGE 1: Personalized Script Humanizer      │
                 │  (LLM + Speaker's style_profile.json)        │
                 │  Converts clean script -> natural spoken     │
                 │  script using THAT speaker's filler habits   │
                 └──────────────────────────────────────────────┘
                                        │
                      (User can review & tweak the script here)
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │  STAGE 2: Accent-Preserving Voice TTS        │
                 │  (F5-TTS Base + Speaker's LoRA Adapter)      │
                 │  Synthesizes audio with authentic accent,    │
                 │  breaths, pauses, and intonation             │
                 └──────────────────────────────────────────────┘
                                        │
                                        ▼
                          [Final Presentation Audio]
```

### Why Separate Stage 1 and Stage 2?
- **Full user control:** You can inspect and edit where `uh`/`um` or pauses appear before generating audio.
- **Adjustable naturalness:** The same presentation script can be rendered completely formal (`None`) for an executive pitch or relaxed and conversational (`Medium`/`High`) for a team walkthrough.
- **Cleaner training:** The voice model only has to focus on *how* to pronounce what is written, rather than guessing when to hesitate.

---

## 3. Multi-Speaker & Multi-Accent Personalization

Because we want multiple people to use this system — including speakers with very different English accents (different Vietnamese speakers, Cypriot, Hungarian, Spanish, etc.) — **one-size-fits-all templates will not work**.

Every speaker has their own **Speaker Profile** (`profiles/<speaker_id>/`), containing two personalized components:

### A. Personal Speaking Fingerprint (`style_profile.json`)
Different people hesitate differently:
- A Vietnamese English speaker might use `uh...`, `um...`, `so,`, `actually,`.
- A Spanish English speaker might use `eh...`, `you know,`, `like,`.
- A Hungarian or Cypriot speaker might use `erm,`, `basically,`, or pause before technical nouns.

When we process a speaker's raw recordings, we extract their exact filler vocabulary, filler frequency, average pause rhythm, and 10–15 real before/after sentence examples (`clean_text` → `raw_text`). Stage 1 injects this profile into the LLM so it imitates **that exact person's** natural speaking style.

### B. Personal Voice Adapter (LoRA Weights)
- Rather than mixing all speakers into one monolithic retrained model (which causes accents to bleed into each other and requires expensive retraining for every new user), we use **one shared Base Model (F5-TTS)** plus **one lightweight LoRA adapter per speaker** (~30–100 MB each).
- Adding a new speaker only requires training their individual LoRA (~20–45 minutes on a cloud GPU) without touching any other user's voice.
- At runtime, the app loads the Base Model once and hot-swaps the selected speaker's LoRA in seconds.

### Why F5-TTS (with GPT-SoVITS as Fallback)?
Older TTS systems rely on rigid Grapheme-to-Phoneme (G2P) converters that map English words to textbook US/UK phonemes — destroying non-native accents. **F5-TTS** uses a modern flow-matching architecture trained directly on text-to-audio representations, so when fine-tuned on a speaker's dataset, it faithfully learns *their* exact pronunciation, consonant endings, stress patterns, and natural breathing.

---

## 4. Dataset & Labeling Strategy

Audio quality and transcript accuracy account for **70–80%** of the final realism.

### Recording Guidelines (Per Speaker)
- **Target Duration:** 1 to 1.5 hours of clean speech (30 minutes minimum for a first test).
- **The 60/40 Split (Crucial for Realism):**
  - **~60% Spontaneous Presentation Speech:** Talk naturally about familiar topics, explain slides, or tell stories as if presenting to colleagues. This captures authentic accent, real pauses, and natural `uh`/`um`/`eh` habits.
  - **~40% Conversational Script Reading:** Read varied presentation-style scripts to cover diverse vocabulary, numbers, questions, and emphasis.
- **Audio Specs:** `.wav`, Mono, 24kHz or 44.1kHz, quiet room with minimal echo, consistent mic distance.
- **Keep Natural Breaths:** Avoid aggressive noise gates that chop off light inhalations between sentences — breaths are a huge part of sounding human.

### The Golden Rule of Transcription
**The transcript (`raw_text`) MUST match 100% of what was actually spoken.**
- If the speaker said *"So, um... we need to, uh, look at this chart"*, `raw_text` must explicitly include `um...` and `uh,`.
- Never delete fillers from the TTS training text while leaving them in the audio — doing so confuses text-audio alignment and causes stuttering/hallucinations at inference time.
- *Note:* Standard ASR tools like Whisper automatically strip out `uh`/`um` by default. Our preprocessing pipeline will prompt/configure WhisperX to preserve disfluencies, followed by a quick human review pass.

### Structured Dataset Labels
Each 3–12 second audio clip will be labeled with:
- `speaker_id` & `accent` (e.g., `vn_user01`, `vi-EN`)
- `raw_text` (exact spoken words including fillers & pauses — used to train TTS)
- `clean_text` (polished version of the same sentence — paired with `raw_text` to teach Stage 1 how this speaker turns clean text into natural speech)
- `filler_level` (`none` / `low` / `medium` / `high`)
- `tone` (`presentation`, `explaining`, `emphasis`)

*(Most of these columns will be auto-generated by our preprocessing script using WhisperX + an LLM, so human effort is limited to quick verification.)*

---

## 5. Hardware & Cost Plan

We designed the workflow to stay fast and nearly free:

| Task | Where It Runs | Cost & Reasoning |
| :--- | :--- | :--- |
| **Daily Inference** (Generating presentations, running Web UI) | **Local Laptop** (RTX 5060 Laptop, 8GB VRAM) | **Free.** 8GB VRAM is more than enough for fast F5-TTS inference and swapping LoRA adapters. |
| **Dataset Prep** (WhisperX alignment & slicing) | **Local Laptop** | **Free.** Runs quickly on local GPU. |
| **LoRA Fine-Tuning** (One-time per speaker) | **Cloud GPU** (RunPod / Vast.ai RTX 4090, 24GB VRAM) | **~$1–$2 per speaker** (~$0.35–$0.50/hr for 1–2 hrs). Prevents laptop overheating, avoids 8GB VRAM limits, and produces a small checkpoint to download and run locally forever. |

---

## 6. Development Roadmap

### Phase 1 — MVP (Single Speaker, Multi-Speaker Architecture)
1. Build the folder structure and automated dataset pipeline (audio slicing, WhisperX disfluency-preserving alignment, pause detection, `clean_text` pairing, `style_profile.json` extraction).
2. Record 1–1.5 hours for `vn_user01` and verify dataset quality.
3. Fine-tune the first F5-TTS LoRA on a cloud RTX 4090 and download the weights.
4. Build a simple local Web UI (Gradio or Streamlit): paste script → select speaker → select filler level (`None/Low/Medium/High`) → edit humanized script → synthesize & download audio.

### Phase 2 — Multi-Accent Expansion
1. Onboard Speaker 02 and Speaker 03 (e.g., a second Vietnamese speaker + a European accent such as Spanish, Hungarian, or Cypriot).
2. Verify that each speaker retains their distinct accent and personal filler style with zero bleed-over.
3. Refine `style_profile.json` extraction and LLM prompting based on real listening tests.

### Phase 3 — Product Polish
1. Long-form presentation chunking & seamless audio stitching (handling 10–30 minute talks without quality drift).
2. Optional slide-by-slide script management and pacing controls.
3. Optional fine-tuned local LLM for 100% offline Stage 1 humanization.

---

## 7. How We Can Collaborate

- **Audio & Dataset Side:** Recording diverse presentation samples, reviewing `raw_text` accuracy (especially `uh`/`um`/`eh` tags), and testing listening realism.
- **Engineering Side:** Building the preprocessing scripts (`scripts/`), running cloud LoRA training notebooks/scripts, tuning Stage 1 LLM prompts, and building the local Web UI (`app/`).
- **Important Git Rule:** Never commit raw `.wav` audio datasets or `.pt`/`.safetensors` model checkpoints to GitHub (they are already ignored in `.gitignore`). Share datasets/weights via Google Drive, Hugging Face private repos, or cloud storage.
