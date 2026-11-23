# Speaker Recording Guide & Sample Presentation Scripts

Use this guide whenever onboarding a new speaker (`vn_01`, `es_01`, `hu_01`, `cy_01`, etc.).  
**Remember:** Audio realism is 70–80% of the final result. Speak **exactly like your real self** — do NOT force a fake American/British accent, and do NOT try to eliminate your natural pauses or `uh`/`um`/`eh` habits.

---

## 1. Recording Setup Checklist

- **Format:** `.wav` (Uncompressed PCM, Mono, 24kHz or 44.1kHz). *(Recommended free tool: Audacity or OBS)*
- **Environment:** Quiet room with curtains/carpet to reduce echo. Turn off fans or AC if loud.
- **Mic Consistency:** Keep the same microphone and same distance (~15–20 cm) across all recording sessions.
- **Posture:** Sit upright or stand as if actually presenting to colleagues or an audience.
- **NO Aggressive Noise Gate:** Do NOT apply heavy noise gating or "auto-silence removal" before saving. Subtle breaths between sentences are essential for sounding human.
- **Target Duration:**
  - **Quick MVP Test:** 20–30 minutes of speech.
  - **Production Quality:** 60–90 minutes of speech (can be split into multiple 10–15 minute `.wav` files).

Save all raw files into:
```text
profiles/<speaker_id>/raw_recordings/
```
*(Example: `profiles/vn_01/raw_recordings/session_01_spontaneous.wav`)*

---

## 2. Part A — Spontaneous Presentation Speech (~60% of Dataset)

**Goal:** Capture your authentic accent, real speaking rhythm, and natural hesitation habits (`uh`, `um`, `eh`, `so...`).  
Pick 6–10 prompts below and speak freely for **3–5 minutes each** as if explaining to a teammate or presenting slides. Do NOT read from a word-for-word script during Part A.

1. **Introduce your current project or job:** What problem are you solving, why does it matter, and who uses it?
2. **Walk through a past presentation or slide deck:** Open any old slides you have and present them out loud naturally.
3. **Explain a technical or complex concept simply:** Pretend you are explaining it to a smart colleague from another department.
4. **Compare two options / tools / strategies:** Discuss pros, cons, trade-offs, and why you recommend one over the other.
5. **Tell the story of a problem that went wrong and how you fixed it:** Describe what happened, what you tried first, and the final lesson learned.
6. **Pitch a new feature or business idea:** Explain the problem, your proposed solution, expected impact, and next steps.
7. **Walk through numbers or results:** Make up or recall a quarterly review — talk through growth, bottlenecks, percentages, and targets.
8. **Answer imaginary Q&A questions:** Ask yourself 5 tough audience questions and answer them on the spot (great for natural thinking pauses!).

---

## 3. Part B — Conversational Presentation Scripts (~40% of Dataset)

**Goal:** Cover diverse English sounds, numbers, transitions, emphasis, and questioning intonation while keeping a natural presentation delivery. Read these **conversationally**, not like a stiff news anchor — feel free to add slight natural pauses or your own `uh`/`um` where it feels right.

### Script 1 — Opening & Framing a Presentation
> Hello everyone, thanks for joining today. In the next fifteen minutes, I want to walk you through our plan for the upcoming quarter, show you what worked well over the last three months, and highlight two key bottlenecks we need to solve together. Feel free to jump in with questions at any point, or we can save them for the Q&A at the end.

### Script 2 — Explaining Data, Metrics & Trade-offs
> If we look closely at the numbers from Q3, user sign-ups grew by twenty-four percent, which is great. However, retention after day fourteen dropped from sixty-two percent down to fifty-one percent. Why did that happen? Basically, the onboarding flow has five separate steps, and almost a third of new users drop off before they even reach the main dashboard.

### Script 3 — Walking Through Architecture & Process
> Let's break the system down into two simple stages. First, we take a clean written document and adapt its pacing so it sounds like natural spoken English. Second, we pass that adapted text into our personalized voice model. By separating these two steps, every user gets full control over the script before generating a single second of audio.

### Script 4 — Emphasizing Key Takeaways & Decisions
> Now, this next point is really important, so let's pause here for a second. We don't need to build the biggest or most expensive model on day one. What actually matters is consistency, speed, and reliability. If we ship a clean version this month and test it with five real users, we'll learn ten times more than spending six months guessing in the dark.

### Script 5 — Natural Transitions & Examples (With Light Hesitation)
> So, um... how does this look in practice? Let's take a real example. Imagine you have a ten-minute client presentation tomorrow morning, and, uh, you already wrote the perfect script, right? Instead of recording twenty takes because you stumbled over a sentence, you just paste your script, choose how relaxed you want the delivery to feel, and export the final audio in a couple of minutes.

### Script 6 — Handling Questions & Closing
> That's a really good question. To be honest, we haven't tested that exact edge case yet, but our current assumption is that it will scale linearly as we add more profiles. Let me double-check the benchmarks with the team and follow up with you by Friday afternoon. Alright, to wrap up: our three priorities are dataset quality, fast iteration, and keeping the workflow simple. Thanks a lot everyone — let's open it up for questions!

---

## 4. After Recording — Running the Automated Pipeline

Once your `.wav` files are placed in `profiles/<speaker_id>/raw_recordings/`, run:

```powershell
# 1. Slice into 3-12s clips (24kHz mono WAV)
python scripts/01_preprocess_audio.py --speaker vn_01

# 2. Transcribe with disfluency preservation + pause timings
python scripts/02_transcribe_align.py --speaker vn_01 --accent vi-EN

# 3. (Recommended) Quick review of profiles/vn_01/metadata.jsonl to verify uh/um accuracy

# 4. Generate clean_text pairs & tone labels (use --heuristic if no LLM API key yet)
python scripts/03_build_dataset_labels.py --speaker vn_01

# 5. Extract individualized style_profile.json
python scripts/04_extract_style_profile.py --speaker vn_01 --display-name "Speaker 01 (Vietnamese English)"
```
