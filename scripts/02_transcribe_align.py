"""Step 2: Transcribe sliced WAV clips while preserving uh/um/eh fillers & pause timings.

Usage:
    python scripts/02_transcribe_align.py --speaker vn_01 --accent vi-EN
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
from tqdm import tqdm

# Allow running directly as `python scripts/02_transcribe_align.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schemas import DatasetClip, FillerLevel, PauseEvent, RecordingType, save_metadata

# Bias Whisper to retain verbatim disfluencies instead of smoothing them out.
DISFLUENCY_PROMPT = (
    "Umm, uh, eh, erm, ah, so, like, you know, actually, basically, "
    "well... let's, uh, talk about this presentation."
)

FILLER_PATTERNS = [
    ("uh", re.compile(r"\b(uh+|uhh+)\b", re.IGNORECASE)),
    ("um", re.compile(r"\b(um+|umm+)\b", re.IGNORECASE)),
    ("eh", re.compile(r"\b(eh+|ehh+)\b", re.IGNORECASE)),
    ("erm", re.compile(r"\b(erm+)\b", re.IGNORECASE)),
    ("you know", re.compile(r"\byou know\b", re.IGNORECASE)),
    ("like", re.compile(r"(?:^|,\s*)like(?=,|\.\.\.|\s+uh|\s+um)", re.IGNORECASE)),
    ("actually", re.compile(r"\bactually\b", re.IGNORECASE)),
    ("basically", re.compile(r"\bbasically\b", re.IGNORECASE)),
]


def detect_fillers(text: str) -> Tuple[List[str], int, FillerLevel]:
    """Detect filler occurrences and estimate filler density level."""
    found: List[str] = []
    for canonical, pattern in FILLER_PATTERNS:
        matches = pattern.findall(text)
        found.extend([canonical] * len(matches))

    count = len(found)
    words = max(1, len(text.split()))
    per_20_words = (count / words) * 20.0

    if count == 0:
        level: FillerLevel = "none"
    elif per_20_words < 1.5:
        level = "low"
    elif per_20_words < 3.5:
        level = "medium"
    else:
        level = "high"

    return found, count, level


def _register_cuda_dlls_windows() -> None:
    """Ensure ctranslate2/faster-whisper can locate cublas64_12.dll bundled with PyTorch on Windows."""
    if sys.platform != "win32":
        return
    import os

    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def transcribe_speaker(
    speaker_id: str,
    accent: str,
    recording_type: RecordingType,
    profiles_root: Path,
    model_size: str,
    device: str,
    compute_type: str,
) -> None:
    if device == "cuda":
        _register_cuda_dlls_windows()
    from faster_whisper import WhisperModel

    speaker_dir = profiles_root / speaker_id
    wavs_dir = speaker_dir / "wavs"
    wav_files = sorted(wavs_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV clips found in {wavs_dir}. Run 01_preprocess_audio.py first.")

    print(f"Loading faster-whisper '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    clips: List[DatasetClip] = []
    for wav_path in tqdm(wav_files, desc=f"Transcribing {speaker_id}"):
        info = sf.info(str(wav_path))
        duration_sec = round(info.duration, 2)

        segments, _ = model.transcribe(
            str(wav_path),
            language="en",
            initial_prompt=DISFLUENCY_PROMPT,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        text_parts: List[str] = []
        pause_events: List[PauseEvent] = []
        prev_word_end: float | None = None
        prev_word_text: str = ""

        for seg in segments:
            text_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    if prev_word_end is not None:
                        gap_ms = int((w.start - prev_word_end) * 1000)
                        if gap_ms >= 400:
                            pause_events.append(
                                PauseEvent(
                                    after_word=prev_word_text.strip(" ,.!?;:"),
                                    duration_ms=gap_ms,
                                    tag="[pause_long]",
                                )
                            )
                        elif gap_ms >= 200:
                            pause_events.append(
                                PauseEvent(
                                    after_word=prev_word_text.strip(" ,.!?;:"),
                                    duration_ms=gap_ms,
                                    tag="[pause_short]",
                                )
                            )
                    prev_word_end = w.end
                    prev_word_text = w.word

        raw_text = " ".join(p for p in text_parts if p).strip()
        if not raw_text:
            continue

        fillers_used, filler_count, filler_level = detect_fillers(raw_text)

        clips.append(
            DatasetClip(
                clip_id=wav_path.stem,
                audio_path=f"wavs/{wav_path.name}",
                duration_sec=duration_sec,
                speaker_id=speaker_id,
                accent=accent,
                recording_type=recording_type,
                raw_text=raw_text,
                clean_text=None,
                fillers_used=fillers_used,
                filler_count=filler_count,
                filler_level=filler_level,
                pause_events=pause_events,
                tone="presentation",
                verified=False,
            )
        )

    out_path = speaker_dir / "metadata.jsonl"
    save_metadata(clips, out_path)
    print(f"Saved {len(clips)} transcripts to {out_path}")
    print("Next step: Review clips with long pauses/fillers so raw_text matches audio 100%.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe speaker clips preserving disfluencies.")
    parser.add_argument("--speaker", required=True, help="Speaker ID, e.g., vn_01")
    parser.add_argument("--accent", default="vi-EN", help="Accent tag, e.g., vi-EN, es-EN, hu-EN, cy-EN")
    parser.add_argument("--recording-type", choices=["spontaneous", "scripted"], default="spontaneous")
    parser.add_argument("--profiles-dir", default="profiles")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16", help="float16 for GPU, int8 for CPU")
    args = parser.parse_args()

    transcribe_speaker(
        speaker_id=args.speaker,
        accent=args.accent,
        recording_type=args.recording_type,
        profiles_root=Path(args.profiles_dir),
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()
