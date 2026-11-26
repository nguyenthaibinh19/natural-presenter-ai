# Speaker Recording Guide & Sample Presentation Scripts

Use this guide whenever onboarding a new speaker (`vn_01`, `speaker_02`, `es_01`, `hu_01`, `cy_01`, etc.).  
**Remember:** Audio realism is 70–80% of the final result. Speak **exactly like your real self** — do NOT force a fake American/British accent, and do NOT try to eliminate your natural pauses or `uh`/`um`/`eh` habits.

---

## 1. Recording Setup Checklist

- **Format:**
  - **Computer (OBS / Audacity):** `.wav` (Uncompressed PCM, Mono, 24kHz or 44.1kHz).
  - **Phone (Recommended — iPhone Voice Memos):** Go to **Settings → Apps → Voice Memos → Audio Quality → `Lossless`** (ALAC 48kHz — proven up to **42.6 dB SNR** in our benchmarks!).
- **Room Acoustics (CRITICAL):**
  - **Avoid empty rooms** (hard flat walls create echo/reverb that gets baked into the AI voice).
  - Record in a **soft-furnished room** (bedroom with bed, blankets, curtains, or facing an open closet) for dry, studio-like clarity.
- **Mic Placement:** Keep the mic/phone **15–20 cm** from your mouth, angled slightly **30–45° off-axis** to prevent breath pops on `p`/`b`/`t`. Place the phone on a stable stand/desk rather than holding it in your hand.
- **Posture:** Sit upright or stand as if actually presenting to colleagues or an audience.
- **NO Aggressive Noise Gate:** Do NOT turn on "Enhance Recording" or heavy silence-stripping filters — subtle natural breaths between clauses are essential for human realism.
- **Target Duration (Per Speaker):**
  - **First Fine-Tune Target:** **25–30 minutes** of verified speech (~15–18 min Part A + ~10–12 min Part B).
  - **Production Quality:** **60 minutes** of verified speech.

Save raw files into:
```text
profiles/<speaker_id>/raw_recordings/
```

---

## 2. Part A — Spontaneous Presentation Speech (~60% of Dataset)

**Goal:** Capture your authentic accent, real speaking rhythm, and natural hesitation habits (`uh`, `um`, `eh`, `like`, `so...`).  
Pick 6–8 prompts below and speak freely for **3–5 minutes each** as if explaining to a teammate or presenting slides. Do NOT read from a word-for-word script during Part A.

1. **Introduce your current project or job:** What problem are you solving, why does it matter, and who uses it?
2. **Walk through a past presentation or slide deck:** Open any old slides you have and present them out loud naturally.
3. **Explain a technical or complex concept simply:** Pretend you are explaining it to a smart colleague from another department.
4. **Compare two options / tools / strategies:** Discuss pros, cons, trade-offs, and why you recommend one over the other.
5. **Tell the story of a problem that went wrong and how you fixed it:** Describe what happened, what you tried first, and the final lesson learned.
6. **Pitch a new feature or business idea:** Explain the problem, your proposed solution, expected impact, and next steps.
7. **Walk through numbers or results:** Recall a benchmark or quarterly review — talk through growth, bottlenecks, percentages, and targets.
8. **Answer imaginary Q&A questions:** Ask yourself 5 tough audience questions and answer them on the spot (great for natural thinking pauses!).

---

## 3. Part B — Conversational Presentation Scripts (~40% of Dataset)

**Goal:** Cover diverse English phonemes, technical terms, numbers, dates, percentages, rising question intonation, contrasts, and emphasis.  
**Total Reading Time:** **~11–13 minutes** at a natural presentation pace (~110–130 WPM).  
**Delivery Tip:** Read **conversationally** as if presenting to a live room — pause naturally at commas/periods, emphasize bold ideas, and don't worry if you naturally add a light `uh` or pause (we will keep exact `raw_text` during QA). You can record all 3 sessions in one file or split them into 3 short ~4-minute recordings.

---

### Session 1 — Core Presentations (~4 minutes)

#### Script 1 — Opening & Framing a Presentation
> Hello everyone, thanks for joining today. In the next fifteen minutes, I want to walk you through our plan for the upcoming quarter, show you what worked well over the last three months, and highlight two key bottlenecks we need to solve together. Feel free to jump in with questions at any point, or we can save them for the Q and A at the end.

#### Script 2 — Explaining Data, Metrics & Trade-offs
> If we look closely at the numbers from Q three, user sign-ups grew by twenty-four percent, which is great. However, retention after day fourteen dropped from sixty-two percent down to fifty-one percent. Why did that happen? Basically, the onboarding flow has five separate steps, and almost a third of new users drop off before they even reach the main dashboard.

#### Script 3 — Walking Through Architecture & Process
> Let's break the system down into two simple stages. First, we take a clean written document and adapt its pacing so it sounds like natural spoken English. Second, we pass that adapted text into our personalized voice model. By separating these two steps, every user gets full control over the script before generating a single second of audio.

#### Script 4 — Emphasizing Key Takeaways & Decisions
> Now, this next point is really important, so let's pause here for a second. We don't need to build the biggest or most expensive model on day one. What actually matters is consistency, speed, and reliability. If we ship a clean version this month and test it with five real users, we'll learn ten times more than spending six months guessing in the dark.

#### Script 5 — Natural Transitions & Examples (With Light Hesitation)
> So, um... how does this look in practice? Let's take a real example. Imagine you have a ten-minute client presentation tomorrow morning, and, uh, you already wrote the perfect script, right? Instead of recording twenty takes because you stumbled over a sentence, you just paste your script, choose how relaxed you want the delivery to feel, and export the final audio in a couple of minutes.

#### Script 6 — Handling Questions & Closing
> That's a really good question. To be honest, we haven't tested that exact edge case yet, but our current assumption is that it will scale linearly as we add more profiles. Let me double-check the benchmarks with the team and follow up with you by Friday afternoon. Alright, to wrap up: our three priorities are dataset quality, fast iteration, and keeping the workflow simple. Thanks a lot everyone — let's open it up for questions!

---

### Session 2 — Technical, AI & Research Presentations (~4 minutes)

#### Script 7 — Experimental Setup & Methodology
> On this slide, you can see our experimental setup. We trained three different variants on the exact same dataset, keeping the learning rate fixed at one times ten to the minus five, and evaluating every five hundred steps. Each run took roughly ninety minutes on a single RTX four thousand ninety GPU. To make the comparison fair, we fixed the random seed across all eighty trials.

#### Script 8 — Interpreting Curves, Overfitting & Ablations
> Look at the blue line on the left graph. During the first twenty epochs, the validation loss drops sharply from zero point eight four down to zero point two one. After epoch forty, though, the training loss keeps falling while validation starts creeping back up. That's a classic sign of overfitting — which tells us early stopping around step fifteen hundred gives the cleanest generalization.

#### Script 9 — Debugging a Surprising Failure Case
> Initially, we hit a really strange bug in our audio pipeline. The transcripts looked clean on paper, yet the synthesized voice kept stuttering at random pauses. When we inspected the raw waveforms, we realized the speech recognizer was silently deleting every single hesitation sound from the labels. Once we forced it to keep verbatim fillers, the alignment issue disappeared almost overnight.

#### Script 10 — Latency, Memory & Hardware Trade-offs
> Let's talk briefly about hardware and cost. Running inference locally on an eight gigabyte laptop GPU takes about two seconds per sentence, uses under four gigabytes of memory, and costs zero dollars per month. Meanwhile, heavy fine-tuning only happens once per speaker, so renting a cloud GPU for ninety minutes costs less than two dollars total. That hybrid split keeps the whole system both fast and affordable.

#### Script 11 — Introducing a Live Demo
> Instead of just showing static charts, let me switch over to the live interface so you can hear the difference directly. Here on the top left, I select the speaker profile and set the hesitation slider to medium. Notice how the script automatically adapts its rhythm before we generate any audio. Let's click synthesize and listen closely to the pacing on the second sentence.

#### Script 12 — Limitations & Future Research Directions
> Of course, there are still a few clear limitations we need to address. First, very short datasets under ten minutes don't capture enough phonetic variety for rare technical words. Second, background echo in empty rooms degrades voice sharpness noticeably. In the next phase, we plan to test slide-by-slide pacing controls and evaluate whether a small fine-tuned language model improves script naturalness even further.

---

### Session 3 — Business, Storytelling, Numbers & High Phoneme Variety (~4 minutes)

#### Script 13 — Problem-Solution Storytelling
> Think back to the last time you had to record a five-minute product walkthrough in English. Maybe your ideas were crystal clear, yet you still spent two hours re-recording the same three slides because you tripped over a complicated phrase or forgot a transition. Our goal isn't to replace your identity with a fake radio announcer — it's to let your authentic voice deliver your best work without the stress.

#### Script 14 — Comparing Alternatives Clearly
> Why didn't we just use an off-the-shelf voice cloning service? Well, there are three big reasons. First, commercial APIs charge every single time you regenerate a paragraph, which gets expensive fast. Second, they tend to flatten non-native accents into a generic American tone. And third, keeping everything local gives users full privacy over their own voice data and internal company presentations.

#### Script 15 — Numbers, Dates, Currencies & Acronyms
> Let's review the budget and timeline for twenty twenty-six. Phase one kicks off on November twenty-fourth with a hardware budget of fifty dollars for cloud experiments. By January fifteenth, we aim to onboard five test speakers from Vietnam, Cyprus, Hungary, and Spain. If user satisfaction stays above eighty-five percent — say, four point three out of five — we'll move straight into our Q two beta release with a target cost under zero point zero five dollars per minute.

#### Script 16 — Asking Engaging Rhetorical Questions
> Now, you might be wondering: won't adding pauses and filler words make a presenter sound less confident? Actually, it's the exact opposite! When a voice speaks non-stop with zero hesitation and zero breath, listeners subconsciously recognize it as artificial within five seconds. Subtle, well-placed pauses signal thoughtful reflection — they give the audience a moment to absorb your point before you move on.

#### Script 17 — Navigating Difficult Vocabulary & Consonant Clusters
> To stress-test pronunciation accuracy, let's walk through a few dense technical terms in context. Specifically, we measure spectral smoothness, threshold stability, cross-lingual generalization, and real-time throughput across distributed clusters. Whether someone says architecture, throughput, visualization, or synchronization, the model should reflect their genuine speaking rhythm rather than stumbling or skipping syllables.

#### Script 18 — Strong Executive Summary & Call to Action
> Let me leave you with three simple takeaways before we finish. One: authentic voices build far more trust than overly polished synthetic speech. Two: separating script adaptation from voice synthesis gives every presenter full creative control. And three: our architecture scales cleanly from one person to dozens of international accents without ever mixing voices together. Thank you so much for your time, your feedback, and your support — I'd love to hear your thoughts!

---

## 4. After Recording — Running the Incremental Pipeline (`--append`)

Always use **`--append`** when adding new recordings so your previously QA-verified clips are **never overwritten**:

```powershell
# 0. If recording from iPhone (.m4a ALAC), decode to 24kHz mono WAV first:
#    ffmpeg -y -i profiles/vn_01/raw_recordings/new_session.m4a -ac 1 -ar 24000 -c:a pcm_s16le profiles/vn_01/raw_recordings/new_session.wav
#    (Move the original .m4a into raw_recordings/_originals/ afterwards)

# 1. Slice ONLY the newly added WAV file(s) and append new clip IDs
python scripts/01_preprocess_audio.py --speaker vn_01 --append --files new_session.wav

# 2. Transcribe ONLY newly added clips (preserving all verified clips!)
python scripts/02_transcribe_align.py --speaker vn_01 --accent vi-EN --recording-type scripted --append

# 3. Auto-fill clean_text for new clips (existing clean_text is preserved unless --overwrite)
python scripts/03_build_dataset_labels.py --speaker vn_01 --heuristic

# 4. Open http://127.0.0.1:7860 -> Dataset QA Reviewer (filter: Only Unverified TODO)
#    Review only the new clips, click "✅ Verify & Next Clip", then click "Rebuild style_profile.json" once at the end!
```
