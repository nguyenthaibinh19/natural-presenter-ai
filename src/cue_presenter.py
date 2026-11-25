"""Human-Driven Cue Presenter Engine (ADR-006).

Parses presenter cues from:
  1. Annotated Jupyter notebooks (`# N4a | POINT TO ... # SAY: ...`)
  2. Annotated Reading Copy PDFs (`READING COPY | REPORT PAGE X`, `A1`, `B1`, `POINT TO`, `SAY`)

Pre-renders every cue through Stage 1 (Gemma 3 4B + pronunciation map) and Stage 2 (F5-TTS),
then provides a zero-latency global hotkey / side-mouse-button soundboard (`pynput` + `winsound`)
so the human smoothly controls cursor, scrolling, and window switching during OBS recording.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf

from src.humanizer import humanize_script
from src.schemas import CueDeck, CueItem, FillerLevel, HumanizeRequest, SynthesizeRequest
from src.synthesizer import PresentationSynthesizer

# Default ordering heuristic matching standard report+notebook presentation flow
CUE_GROUP_ORDER = {
    "N1": 10,
    "N2": 20,
    "N3": 30,
    "N4": 40,
    "A1": 50,
    "A2": 60,
    "B1": 70,
    "N5": 80,
    "B2": 90,
    "B3": 100,
    "B4": 110,
    "C1": 120,
    "C2": 130,
    "C3": 140,
    "C4": 150,
    "C5": 160,
    "C6": 170,
    "D1": 180,
}


def _clean_ocr_text(text: str) -> str:
    text = text.replace("?s", "'s").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _infer_default_speaker(cue_id: str, primary_speaker: str = "vn_01", secondary_speaker: str = "vn_02") -> str:
    """Assign Binh (vn_01) to B* and N5* cues; Ege (vn_02) to A*, C*, D*, N1..N4 by default."""
    cid = cue_id.upper()
    if cid.startswith("B") or cid.startswith("N5"):
        return primary_speaker
    return secondary_speaker


def _cue_sort_key(cue_id: str) -> Tuple[int, str]:
    base = re.match(r"([A-Za-z]+\d+)", cue_id)
    prefix = base.group(1).upper() if base else cue_id.upper()
    return (CUE_GROUP_ORDER.get(prefix, 999), cue_id)


def parse_notebook_cues(
    ipynb_path: Path,
    merge_fragments: bool = True,
    primary_speaker: str = "vn_01",
    secondary_speaker: str = "vn_02",
) -> List[CueItem]:
    """Parse `# <ID> | POINT TO ... # SAY: ...` annotations from a presenter Jupyter notebook."""
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    raw_cues: List[CueItem] = []

    cue_header_re = re.compile(r"^#\s*([A-Za-z]+\d+[a-z]?)\s*\|\s*POINT TO\s*(?:\(([^)]+)\))?\s*:\s*(.*)$")

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        lines = "".join(cell.get("source", [])).splitlines()
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            m = cue_header_re.match(stripped)
            if not m:
                i += 1
                continue

            cue_id = m.group(1)
            loc_hint = m.group(2) or f"cell {cell_idx + 1}"
            point_parts = [m.group(3).strip()]
            say_parts: List[str] = []
            mode = "point"
            i += 1

            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt.startswith("#") or cue_header_re.match(nxt):
                    break
                content = nxt.lstrip("#").strip()
                if content.upper().startswith("SAY:"):
                    mode = "say"
                    say_parts.append(content[4:].strip())
                elif mode == "point":
                    point_parts.append(content)
                else:
                    say_parts.append(content)
                i += 1

            point_text = _clean_ocr_text(" ".join(point_parts))
            say_text = _clean_ocr_text(" ".join(say_parts))
            if say_text:
                raw_cues.append(
                    CueItem(
                        cue_id=cue_id,
                        source=f"Notebook ({loc_hint})",
                        speaker_id=_infer_default_speaker(cue_id, primary_speaker, secondary_speaker),
                        point_to=point_text,
                        clean_text=say_text,
                    )
                )

    if not merge_fragments or not raw_cues:
        return raw_cues

    # Merge sentence fragments (e.g., N4a + N4b, N5a + N5b + N5c) so TTS speaks full natural sentences
    merged: List[CueItem] = []
    for cue in raw_cues:
        if (
            merged
            and not merged[-1].clean_text.endswith((".", "!", "?"))
            and merged[-1].speaker_id == cue.speaker_id
        ):
            prev = merged[-1]
            # Combine ID like N5a + N5b -> N5a-b
            base_prev = prev.cue_id.split("-")[0]
            prev.cue_id = f"{base_prev}-{cue.cue_id[-1]}"
            prev.point_to = f"{prev.point_to} -> THEN: {cue.point_to}"
            prev.clean_text = f"{prev.clean_text} {cue.clean_text}"
        else:
            merged.append(cue)

    return merged


def parse_reading_copy_pdf(
    pdf_path: Path,
    primary_speaker: str = "vn_01",
    secondary_speaker: str = "vn_02",
) -> List[CueItem]:
    """Parse `READING COPY | REPORT PAGE X` margin cues (`POINT TO` / `SAY`) from a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    cues: List[CueItem] = []

    cue_block_re = re.compile(
        r"\b([A-D]\d+)\s+POINT TO\s+(.*?)\s+SAY\s+(.*?)(?=\b[A-D]\d+\s+POINT TO|\bAfter\s+[A-D]\d+|\bContinue to|\bEnd after|$)",
        re.DOTALL,
    )
    trans_re = re.compile(r"((?:After\s+[A-D]\d+|Continue to|End after)[^\n]*(?:\n[^\n]*)?)", re.IGNORECASE)

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        marker_idx = text.find("READING COPY")
        if marker_idx == -1:
            continue
        margin_text = text[marker_idx:]
        # Find transition note at bottom of margin if present
        t_match = trans_re.search(margin_text)
        page_transition = _clean_ocr_text(t_match.group(1)) if t_match else ""

        page_cues: List[CueItem] = []
        for m in cue_block_re.finditer(margin_text):
            cue_id = m.group(1).strip()
            point_to = _clean_ocr_text(m.group(2))
            say_text = _clean_ocr_text(m.group(3))
            if say_text:
                page_cues.append(
                    CueItem(
                        cue_id=cue_id,
                        source=f"Report Page {page_num}",
                        speaker_id=_infer_default_speaker(cue_id, primary_speaker, secondary_speaker),
                        point_to=point_to,
                        clean_text=say_text,
                    )
                )

        if page_cues and page_transition:
            page_cues[-1].transition_note = page_transition
        cues.extend(page_cues)

    return cues


def build_presentation_deck(
    project_name: str,
    pdf_path: Optional[Path] = None,
    ipynb_path: Optional[Path] = None,
    presentations_root: Path = Path("presentations"),
    merge_fragments: bool = True,
    primary_speaker: str = "vn_01",
    secondary_speaker: str = "vn_02",
) -> Tuple[CueDeck, Path]:
    """Parse PDF and/or notebook cues, sort in presentation order, and save deck.json."""
    all_cues: List[CueItem] = []
    if ipynb_path and ipynb_path.exists():
        all_cues.extend(
            parse_notebook_cues(
                ipynb_path,
                merge_fragments=merge_fragments,
                primary_speaker=primary_speaker,
                secondary_speaker=secondary_speaker,
            )
        )
    if pdf_path and pdf_path.exists():
        all_cues.extend(
            parse_reading_copy_pdf(
                pdf_path,
                primary_speaker=primary_speaker,
                secondary_speaker=secondary_speaker,
            )
        )

    all_cues.sort(key=lambda c: _cue_sort_key(c.cue_id))
    deck = CueDeck(project_name=project_name, cues=all_cues)

    deck_path = presentations_root / project_name / "deck.json"
    save_cue_deck(deck, deck_path)
    return deck, deck_path


def save_cue_deck(deck: CueDeck, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(deck.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_cue_deck(path: Path) -> CueDeck:
    return CueDeck.model_validate_json(path.read_text(encoding="utf-8"))


def humanize_cue_deck(
    deck: CueDeck,
    deck_path: Path,
    filler_level: FillerLevel = "medium",
    include_self_repairs: bool = True,
    profiles_root: Path = Path("profiles"),
    llm_model: Optional[str] = None,
) -> CueDeck:
    """Run Stage 1 humanization + acronym normalization on every cue in the deck."""
    for cue in deck.cues:
        req = HumanizeRequest(
            speaker_id=cue.speaker_id,
            clean_script=cue.clean_text,
            filler_level=filler_level,
            include_self_repairs=include_self_repairs,
        )
        resp, _ = humanize_script(req, profiles_root=profiles_root, llm_model=llm_model)
        cue.humanized_text = resp.humanized_script
    save_cue_deck(deck, deck_path)
    return deck


def prerender_cue_deck(
    deck: CueDeck,
    deck_path: Path,
    synthesizer: PresentationSynthesizer,
    speed: float = 0.90,
    nfe_step: int = 16,
    only_missing: bool = False,
) -> CueDeck:
    """Pre-render every cue to presentations/<project>/audio/<cue_id>.wav for 0ms live playback."""
    audio_dir = deck_path.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for cue in deck.cues:
        target_wav = audio_dir / f"{cue.cue_id}.wav"
        if only_missing and target_wav.exists() and cue.audio_path:
            continue
        text_to_speak = (cue.humanized_text or cue.clean_text).strip()
        req = SynthesizeRequest(
            speaker_id=cue.speaker_id,
            humanized_script=text_to_speak,
            ref_name=cue.ref_name,
            speed=speed,
            nfe_step=nfe_step,
        )
        tmp_wav, _ = synthesizer.synthesize(req)
        data, sr = sf.read(str(tmp_wav))
        sf.write(str(target_wav), data, sr, subtype="PCM_16")
        cue.audio_path = str(target_wav)
        cue.duration_sec = round(len(data) / float(sr), 2)

    save_cue_deck(deck, deck_path)
    return deck


class LiveCueController:
    """Zero-latency Windows global hotkey + synced teleprompter controller for OBS recording.

    Uses Win32 GetAsyncKeyState polling (no hook drops, no admin required, works across all windows)
    and keeps the displayed cue strictly synced with the currently playing audio until it finishes.
    """

    def __init__(self) -> None:
        self.deck: Optional[CueDeck] = None
        self.current_idx: int = 0
        self.playing_idx: Optional[int] = None
        self.last_played_idx: Optional[int] = None
        self.is_playing: bool = False
        self.play_end_time: float = 0.0
        self.active: bool = False
        self.enable_mouse_trigger: bool = False
        self.last_event: str = "Idle — Load & Pre-render a Cue Deck, then click ARM GLOBAL HOTKEYS."
        self._lock = threading.Lock()
        self._stop_poll = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._advance_timer: Optional[threading.Timer] = None
        self._play_token: int = 0

    def load_deck(self, deck: CueDeck) -> None:
        with self._lock:
            self._cancel_timer_locked()
            self.deck = deck
            self.current_idx = 0
            self.playing_idx = None
            self.last_played_idx = None
            self.is_playing = False
            self.last_event = f"Loaded deck '{deck.project_name}' ({len(deck.cues)} cues)."

    def _cancel_timer_locked(self) -> None:
        if self._advance_timer is not None:
            self._advance_timer.cancel()
            self._advance_timer = None

    def _play_wav_async(self, wav_path: str) -> None:
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, 0)
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def _on_playback_finished(self, token: int, finished_idx: int, advance_after: bool) -> None:
        with self._lock:
            if token != self._play_token:
                return
            self.is_playing = False
            self.playing_idx = None
            self._advance_timer = None
            if advance_after and self.deck and finished_idx == self.current_idx:
                self.current_idx = min(self.current_idx + 1, len(self.deck.cues))
            if self.deck and self.current_idx < len(self.deck.cues):
                nxt = self.deck.cues[self.current_idx]
                self.last_event = f"✅ Finished [{self.deck.cues[finished_idx].cue_id}] — Ready for [{nxt.cue_id}] ({nxt.speaker_id})"
            else:
                self.last_event = "🎉 Finished final cue!"

    def stop_audio(self) -> str:
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, 0)
        with self._lock:
            self._play_token += 1
            self._cancel_timer_locked()
            self.is_playing = False
            self.playing_idx = None
            self.last_event = "⏹ Stopped playback (staying on current cue so you can retry with F9 / Alt+Space)."
            return self.last_event

    def _start_cue_locked(self, idx: int, advance_after: bool, prefix: str) -> str:
        import time

        assert self.deck is not None
        cue = self.deck.cues[idx]
        if not cue.audio_path or not Path(cue.audio_path).exists():
            self.last_event = f"⚠️ Cue {cue.cue_id} has no pre-rendered audio yet!"
            return self.last_event

        self._cancel_timer_locked()
        self._play_token += 1
        token = self._play_token

        self._play_wav_async(cue.audio_path)
        self.is_playing = True
        self.playing_idx = idx
        self.last_played_idx = idx
        dur = max(0.5, float(cue.duration_sec or 3.0))
        self.play_end_time = time.monotonic() + dur

        self._advance_timer = threading.Timer(
            dur + 0.15,
            self._on_playback_finished,
            args=(token, idx, advance_after),
        )
        self._advance_timer.daemon = True
        self._advance_timer.start()

        self.last_event = f"{prefix} [{cue.cue_id}] ({cue.speaker_id}, {dur:.1f}s) — screen stays synced until audio ends!"
        return self.last_event

    def play_current_and_advance(self) -> str:
        """Play current cue while keeping screen locked on it; auto-advance only when audio finishes.

        If pressed while a cue is ALREADY playing, immediately skips to and plays the next cue.
        """
        with self._lock:
            if not self.deck or not self.deck.cues:
                self.last_event = "No deck loaded."
                return self.last_event

            if self.is_playing and self.playing_idx is not None:
                # User pressed F9 early to jump straight into the next cue
                next_idx = self.playing_idx + 1
                if next_idx >= len(self.deck.cues):
                    self.current_idx = len(self.deck.cues)
                    self.last_event = "Already at end of presentation!"
                    return self.last_event
                self.current_idx = next_idx
                return self._start_cue_locked(self.current_idx, advance_after=True, prefix="▶ Playing")

            if self.current_idx >= len(self.deck.cues):
                self.last_event = "End of presentation! Press F8 / Alt+R to replay last cue or click Reset."
                return self.last_event

            return self._start_cue_locked(self.current_idx, advance_after=True, prefix="▶ Playing")

    def replay_last(self) -> str:
        with self._lock:
            if not self.deck or not self.deck.cues:
                return "No deck loaded."
            if self.is_playing and self.playing_idx is not None:
                idx = self.playing_idx
                advance = True
            elif self.last_played_idx is not None:
                idx = self.last_played_idx
                advance = False
            else:
                idx = max(0, min(self.current_idx, len(self.deck.cues) - 1))
                advance = False
            return self._start_cue_locked(idx, advance_after=advance, prefix="🔁 Replaying")

    def step_index(self, delta: int) -> str:
        with self._lock:
            if not self.deck or not self.deck.cues:
                return "No deck loaded."
            self.current_idx = max(0, min(len(self.deck.cues) - 1, self.current_idx + delta))
            cue = self.deck.cues[self.current_idx]
            self.last_event = f"Moved cursor to [{cue.cue_id}] ({cue.speaker_id})"
            return self.last_event

    def reset_to_start(self) -> str:
        self.stop_audio()
        with self._lock:
            self.current_idx = 0
            self.last_played_idx = None
            self.last_event = "↺ Reset to Cue #1."
            return self.last_event

    def start_listeners(self, enable_mouse_x2: bool = False) -> str:
        """Start native Win32 GetAsyncKeyState polling thread (reliable across all windows & Fn-Lock)."""
        self.stop_listeners()
        self.enable_mouse_trigger = enable_mouse_x2
        self._stop_poll.clear()

        def poll_win32_keys():
            import ctypes
            import time

            user32 = ctypes.windll.user32

            def is_down(vk: int) -> bool:
                return bool(user32.GetAsyncKeyState(vk) & 0x8000)

            # Virtual-key codes
            VK_MENU = 0x12      # Alt
            VK_SPACE = 0x20
            VK_LEFT = 0x25
            VK_RIGHT = 0x27
            VK_R = 0x52
            VK_S = 0x53
            VK_F7 = 0x76
            VK_F8 = 0x77
            VK_F9 = 0x78
            VK_F10 = 0x79
            VK_F11 = 0x7A
            VK_XBUTTON1 = 0x05  # Mouse Back side button
            VK_XBUTTON2 = 0x06  # Mouse Forward side button

            prev_state: Dict[str, bool] = {}
            last_fire: Dict[str, float] = {}
            debounce_sec = 0.28

            while not self._stop_poll.is_set():
                try:
                    alt = is_down(VK_MENU)
                    combos = {
                        "play": is_down(VK_F9) or (alt and is_down(VK_SPACE)) or (enable_mouse_x2 and is_down(VK_XBUTTON2)),
                        "replay": is_down(VK_F8) or (alt and is_down(VK_R)) or (enable_mouse_x2 and is_down(VK_XBUTTON1)),
                        "stop": is_down(VK_F10) or (alt and is_down(VK_S)),
                        "prev": is_down(VK_F7) or (alt and is_down(VK_LEFT)),
                        "next": is_down(VK_F11) or (alt and is_down(VK_RIGHT)),
                    }
                    now = time.monotonic()
                    for action, down in combos.items():
                        was_down = prev_state.get(action, False)
                        if down and not was_down and (now - last_fire.get(action, 0.0) >= debounce_sec):
                            last_fire[action] = now
                            if action == "play":
                                self.play_current_and_advance()
                            elif action == "replay":
                                self.replay_last()
                            elif action == "stop":
                                self.stop_audio()
                            elif action == "prev":
                                self.step_index(-1)
                            elif action == "next":
                                self.step_index(1)
                        prev_state[action] = down
                except Exception as exc:
                    self.last_event = f"Hotkey poll error: {exc}"
                time.sleep(0.03)

        self._poll_thread = threading.Thread(target=poll_win32_keys, daemon=True)
        self._poll_thread.start()
        self.active = True
        mouse_note = " + Mouse Side Buttons" if enable_mouse_x2 else ""
        self.last_event = (
            f"🟢 ARMED (Win32 Global): F9 or Alt+Space = Play | F8 or Alt+R = Replay | "
            f"F10 or Alt+S = Stop | F7/F11 or Alt+←/→ = Skip{mouse_note}"
        )
        return self.last_event

    def stop_listeners(self) -> str:
        self._stop_poll.set()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=0.5)
        self._poll_thread = None
        self.active = False
        self.last_event = "⚪ Hotkeys disarmed."
        return self.last_event

    def render_teleprompter_markdown(self) -> str:
        import time

        with self._lock:
            state = (
                "🟢 **ARMED** (`F9` / `Alt+Space` = Play | `F8` / `Alt+R` = Replay | `F10` / `Alt+S` = Stop)"
                if self.active
                else "⚪ **DISARMED** *(Click `🟢 ARM GLOBAL HOTKEYS` to enable keyboard shortcuts)*"
            )
            if not self.deck or not self.deck.cues:
                return f"{state}\n\n_No presentation cue deck loaded yet._"

            total = len(self.deck.cues)
            rendered = sum(1 for c in self.deck.cues if c.audio_path and Path(c.audio_path).exists())
            header = (
                f"{state}  \n"
                f"**Deck:** `{self.deck.project_name}` | **Pre-rendered:** `{rendered}/{total}` | **Status:** {self.last_event}\n\n---\n"
            )

            # While speaking, keep the main panel LOCKED on the currently speaking cue!
            if self.is_playing and self.playing_idx is not None and self.playing_idx < total:
                cur = self.deck.cues[self.playing_idx]
                rem = max(0.0, self.play_end_time - time.monotonic())
                nxt = self.deck.cues[self.playing_idx + 1] if self.playing_idx + 1 < total else None
                body = (
                    f"## 🔊 NOW SPEAKING (`{self.playing_idx + 1}/{total}`): `[{cur.cue_id}]` — **`{cur.speaker_id}`** "
                    f"*(⏳ {rem:.1f}s remaining)*\n"
                    f"**📄 Location:** `{cur.source}`\n\n"
                    f"### 👉 POINT TO NOW:\n> **{cur.point_to}**\n\n"
                    f"### 🗣️ SPEAKING NOW:\n> {cur.humanized_text or cur.clean_text}\n"
                )
                if cur.transition_note:
                    body += f"\n**➡️ AFTER THIS FINISHES:** `{cur.transition_note}`\n"
                if nxt:
                    body += f"\n---\n**⏭ Up Next (auto-switches when voice ends):** `[{nxt.cue_id}]` (`{nxt.speaker_id}`) — `{nxt.source}` — _{nxt.point_to[:110]}_"
                return header + body

            if self.current_idx >= total:
                return header + "## 🎉 Presentation Complete!\nPress **F8 / Alt+R** to replay the final cue or click **Reset to #1**."

            cur = self.deck.cues[self.current_idx]
            nxt = self.deck.cues[self.current_idx + 1] if self.current_idx + 1 < total else None
            ready_badge = (
                f"🔊 READY (`{cur.duration_sec}s`)"
                if (cur.audio_path and Path(cur.audio_path).exists())
                else "⚠️ NOT RENDERED YET"
            )

            body = (
                f"## 🟢 READY TO PLAY (`{self.current_idx + 1}/{total}`): `[{cur.cue_id}]` — **`{cur.speaker_id}`** ({ready_badge})\n"
                f"**📄 Location:** `{cur.source}`\n\n"
                f"### 👉 MOVE CURSOR TO:\n> **{cur.point_to}**\n\n"
                f"### 🗣️ WILL SAY (Press `F9` or `Alt+Space`):\n> {cur.humanized_text or cur.clean_text}\n"
            )
            if cur.transition_note:
                body += f"\n**➡️ AFTER THIS CUE:** `{cur.transition_note}`\n"
            if nxt:
                body += f"\n---\n**⏭ Up Next:** `[{nxt.cue_id}]` (`{nxt.speaker_id}`) — `{nxt.source}` — _{nxt.point_to[:110]}_"
            return header + body
