# ADR-006: Human-Driven Cue Soundboard Presenter (Pre-Rendered Audio + Global Hotkeys)

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
To record full multi-speaker technical presentations (switching between a clean PDF report and a clean Jupyter notebook while pointing the cursor at specific tables, plots, and code lines), we evaluated two approaches:
1. **Computer-Use Vision Agent (AI moves mouse/scrolls):** Slow (2–5s latency per screenshot/action), jerky cursor motion, prone to mis-scrolling or mis-pointing during live OBS screen recording.
2. **Human-Driven Screen Control + AI Voice:** Human smoothly moves the cursor, scrolls, and switches windows (PDF ↔ Jupyter) on the clean deliverables while AI handles 100% of the spoken delivery. Within Approach 2, pure hover-hotspot triggers risk accidental playback whenever scroll offset/zoom shifts or the mouse crosses another element.

## Decision
Implement a **Pre-Rendered Cue Soundboard & Global Hotkey Presenter (`src/cue_presenter.py`)**:
1. **Cue Parsing:** Automatically parse structured presenter cues (`ID`, `POINT TO`, `SAY`, `Transition`, `Speaker`) from annotated Jupyter notebooks (`# N4a | POINT TO ... # SAY: ...`), annotated Reading Copy PDFs (`A1`, `B1`, `C1`, `D1`), or JSON cue sheets.
2. **Two-Stage Pre-Rendering:** Before recording, Stage 1 (`src/humanizer.py`) humanizes each cue using the assigned speaker's `style_profile.json` + `pronunciation_map.json`, and Stage 2 (`src/synthesizer.py`) pre-renders every cue to a local `.wav` file (`presentations/<project>/audio/<cue_id>.wav`).
3. **Zero-Latency Live OBS Recording Control:** A lightweight global hotkey/mouse listener (`pynput` + Windows `winsound`) lets the presenter control playback even while focused on the clean PDF or Jupyter window:
   - `F9` (or optional side mouse button `Mouse4`): Play current cue immediately (0ms latency) and advance pointer to next cue.
   - `F8`: Replay previous cue.
   - `F10`: Stop audio immediately.
   - `F7` / `F11`: Step back / skip forward without playing.
4. **Multi-Speaker Joint Presentations:** A single human operator can record a seamless two-speaker presentation (e.g., Speaker A + Speaker B) in one continuous take, with each cue speaking in its assigned speaker's voice.

## Consequences
- Zero inference latency during live recording; 100% natural cursor motion and scrolling.
- Zero accidental hover triggers on scroll/zoom changes.
- Clean deliverables remain untouched on the recorded screen; `POINT TO` instructions display in the off-screen Cue Controller UI.
