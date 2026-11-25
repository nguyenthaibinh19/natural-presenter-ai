# Natural Presenter AI

**Authentic, multi-accent AI voice presentation system** that turns polished presentation scripts into natural, human-sounding speech — preserving each speaker's real voice, non-native English accent (e.g., Vietnamese, Cypriot, Hungarian, Spanish), personal cadence, breathing, and individualized hesitation patterns (`uh`, `um`, `eh`, pauses).

---

## Documentation

This repository contains two complementary documents tailored to different audiences:

1. **[Project Memory Index (`.ai/README.md`)](.ai/README.md)** — **Durable living memory** (current state, invariants, ADRs, roadmap, experiments, lessons). Updated continuously on every decision or milestone.
2. **[Ege Onboarding & AI Assistant Brief](docs/EGE_ONBOARDING_AND_AI_BRIEF.md)** — **Start here if you are Ege (or Ege's AI assistant)** — quick copy-paste AI prompt, current snapshot, and concrete tasks ready to pick up.
3. **[Proposals & Alignment Board (Binh ↔ Ege)](docs/PROPOSALS_AND_SYNC.md)** — Shared space where Ege or Binh propose ideas, discuss trade-offs, and approve what to build together.
4. **[Project Guide (For Humans)](docs/PROJECT_GUIDE.md)** — A clear, readable overview of the product vision, two-stage architecture, dataset strategy, hardware plan, and development roadmap.
5. **[AI Agent Technical Specification (For ChatGPT / Coding Agents)](docs/AI_AGENT_SPEC.md)** — Single-file technical spec with exact JSON schemas, dataset pipeline, model choices, and implementation milestones.
6. **[Speaker Recording Guide & Sample Scripts](docs/RECORDING_GUIDE.md)** — Setup checklist, 60/40 spontaneous + scripted presentation prompts, and commands to run the automated dataset pipeline.
7. **[Cue Script Template & ChatGPT Prompt](docs/CUE_SCRIPT_TEMPLATE.md)** — Copy-paste ChatGPT prompt + working multi-speaker Markdown Cue Script example for **Tab 4: Cue Presenter (Live OBS)**.

---

## Core Idea in 30 Seconds

Most Text-to-Speech (TTS) systems try to make everyone sound like a polished native news anchor. For real presentations, that often sounds unnatural or noticeably fake.

**Natural Presenter AI** solves this with a **two-stage pipeline** built around **Speaker Profiles**:

1. **Stage 1 — Personalized Script Humanizer (LLM):**
   - Input: A clean, polished presentation script + desired naturalness level (`None`, `Low`, `Medium`, `High`).
   - Uses each speaker's personal `style_profile.json` (extracted from their real speech habits) to insert *their* specific fillers (`uh`, `um`, `eh`, `you know`, etc.) and pacing pauses — never a one-size-fits-all template.
   - Output: An editable, natural-sounding spoken script.
2. **Stage 2 — Accent-Preserving Voice Synthesis (TTS + Per-Speaker LoRA):**
   - Uses a non-G2P modern TTS base model (**F5-TTS**) paired with a lightweight **per-speaker LoRA adapter** (~30–100 MB per person).
   - Preserves each user's authentic pronunciation, non-native English accent, intonation, and breathing without forcing native American/British phonetics.

---

## Planned Repository Layout

```text
natural-presenter-ai/
├── README.md
├── .gitignore
├── docs/
│   ├── PROJECT_GUIDE.md          # Human-readable project guide
│   └── AI_AGENT_SPEC.md          # Full technical spec for AI agents (ChatGPT, etc.)
├── scripts/                      # Dataset prep, WhisperX alignment, training & inference (planned)
├── app/                          # Local Web UI for script humanization & voice generation (planned)
└── profiles/                     # Local-only speaker profiles, datasets & LoRA weights (gitignored)
```

---

## Current Status

- [x] Architecture & multi-speaker design finalized
- [x] Documentation for collaborators and AI agents published
- [ ] Phase 1 (MVP): Build dataset pipeline + train first speaker (`vn_user01`)
- [ ] Phase 2: Validate multi-accent scaling (additional Vietnamese, Cypriot, Hungarian, Spanish speakers)
- [ ] Phase 3: Local Web UI with speaker selector & filler intensity controls
