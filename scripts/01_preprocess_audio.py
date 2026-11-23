"""Step 1: Convert raw speaker recordings to 24kHz mono WAV and slice into 3-12s clips.

Usage:
    python scripts/01_preprocess_audio.py --speaker vn_01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

TARGET_SR = 24000
MIN_SEC = 3.0
MAX_SEC = 12.0
PAD_SEC = 0.15
PEAK_DBFS = -3.0
SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def peak_normalize(audio: np.ndarray, target_dbfs: float = PEAK_DBFS) -> np.ndarray:
    """Peak-normalize audio without applying aggressive gating (preserves natural breaths)."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    target_amp = 10 ** (target_dbfs / 20.0)
    return (audio / peak * target_amp).astype(np.float32)


def get_speech_segments_silero(audio_24k: np.ndarray) -> List[Tuple[int, int]]:
    """Use Silero VAD at 16kHz to find speech timestamps, mapped back to 24kHz samples."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    audio_16k = librosa.resample(audio_24k, orig_sr=TARGET_SR, target_sr=16000)
    tensor_16k = torch.from_numpy(audio_16k)

    timestamps = get_speech_timestamps(
        tensor_16k,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=400,
        min_silence_duration_ms=350,
        speech_pad_ms=150,
    )
    ratio = TARGET_SR / 16000.0
    return [(int(t["start"] * ratio), int(t["end"] * ratio)) for t in timestamps]


def merge_and_split_segments(
    segments: List[Tuple[int, int]],
    total_samples: int,
    sr: int = TARGET_SR,
) -> List[Tuple[int, int]]:
    """Group short VAD speech chunks into natural 3-12s clips with light boundary padding."""
    if not segments:
        return []

    min_samples = int(MIN_SEC * sr)
    max_samples = int(MAX_SEC * sr)
    pad_samples = int(PAD_SEC * sr)

    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = segments[0]

    for start, end in segments[1:]:
        proposed_len = end - cur_start
        if proposed_len <= max_samples:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    final_clips: List[Tuple[int, int]] = []
    for start, end in merged:
        length = end - start
        if length < min_samples:
            continue
        if length <= max_samples:
            s = max(0, start - pad_samples)
            e = min(total_samples, end + pad_samples)
            final_clips.append((s, e))
        else:
            # Split overly long continuous speech cleanly into ~10s windows
            chunk_len = int(10.0 * sr)
            pos = start
            while pos < end:
                chunk_end = min(end, pos + chunk_len)
                if chunk_end - pos >= min_samples:
                    s = max(0, pos - pad_samples)
                    e = min(total_samples, chunk_end + pad_samples)
                    final_clips.append((s, e))
                pos = chunk_end

    return final_clips


def process_speaker(speaker_id: str, profiles_root: Path) -> None:
    speaker_dir = profiles_root / speaker_id
    raw_dir = speaker_dir / "raw_recordings"
    wavs_dir = speaker_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not raw_files:
        raise FileNotFoundError(
            f"No audio files found in {raw_dir}. Place your .wav/.mp3/.m4a recordings there first."
        )

    clip_idx = 1
    total_dur = 0.0

    for raw_path in tqdm(raw_files, desc=f"Preprocessing {speaker_id}"):
        audio, _ = librosa.load(str(raw_path), sr=TARGET_SR, mono=True)
        audio = peak_normalize(audio)

        segments = get_speech_segments_silero(audio)
        clips = merge_and_split_segments(segments, len(audio), sr=TARGET_SR)

        for start, end in clips:
            clip_audio = peak_normalize(audio[start:end])
            clip_name = f"{speaker_id}_{clip_idx:04d}.wav"
            out_path = wavs_dir / clip_name
            sf.write(str(out_path), clip_audio, TARGET_SR, subtype="PCM_16")
            total_dur += len(clip_audio) / TARGET_SR
            clip_idx += 1

    print(
        f"Done! Created {clip_idx - 1} clips ({total_dur / 60:.1f} minutes total) in {wavs_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and VAD-slice speaker recordings.")
    parser.add_argument("--speaker", required=True, help="Speaker ID, e.g., vn_01")
    parser.add_argument("--profiles-dir", default="profiles", help="Root profiles directory")
    args = parser.parse_args()

    process_speaker(args.speaker, Path(args.profiles_dir))


if __name__ == "__main__":
    main()
