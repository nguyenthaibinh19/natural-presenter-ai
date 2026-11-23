"""Step 0 (Optional): Extract audio from multi-speaker videos and separate speakers offline.

Uses FFmpeg + Silero VAD + SpeechBrain ECAPA-TDNN embeddings + Agglomerative Clustering
(100% offline/open, no gated HuggingFace token required).

Usage:
    # 1. Extract and separate 2 speakers from videos in profiles/raw_videos/
    python scripts/00_extract_diarize_video.py --input profiles/raw_videos --num-speakers 2

    # 2. Listen to profiles/diarized/preview_SPEAKER_00.wav and preview_SPEAKER_01.wav,
    #    then assign them to speaker profile folders:
    python scripts/00_extract_diarize_video.py --assign SPEAKER_00=vn_01 SPEAKER_01=vn_02
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

TARGET_SR = 24000
EMBED_SR = 16000
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".wav", ".mp3", ".m4a"}


def extract_wav_ffmpeg(src_path: Path, out_wav: Path) -> None:
    """Extract 24kHz mono 16-bit PCM WAV from a video/audio file using FFmpeg."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def peak_normalize(audio: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    return (audio / peak * (10 ** (target_dbfs / 20.0))).astype(np.float32)


def vad_segments(audio_24k: np.ndarray) -> List[Tuple[int, int]]:
    """Detect speech turns with Silero VAD."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad_model = load_silero_vad()
    audio_16k = librosa.resample(audio_24k, orig_sr=TARGET_SR, target_sr=EMBED_SR)
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio_16k),
        vad_model,
        sampling_rate=EMBED_SR,
        min_speech_duration_ms=800,
        min_silence_duration_ms=400,
        speech_pad_ms=120,
    )
    ratio = TARGET_SR / float(EMBED_SR)
    segments: List[Tuple[int, int]] = []
    for t in timestamps:
        s = int(t["start"] * ratio)
        e = int(t["end"] * ratio)
        if (e - s) / TARGET_SR >= 1.2:
            segments.append((s, e))
    return segments


def diarize_videos(input_path: Path, output_dir: Path, num_speakers: int) -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )

    if not files:
        raise FileNotFoundError(
            f"No video/audio files found in {input_path}. Drop your presentation videos there first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SpeechBrain ECAPA-TDNN speaker encoder on {device}...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    temp_dir = output_dir / "_extracted_wavs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_clips: List[Tuple[str, np.ndarray]] = []
    embeddings: List[np.ndarray] = []

    for fpath in files:
        print(f"Extracting audio from {fpath.name}...")
        wav_path = temp_dir / f"{fpath.stem}.wav"
        extract_wav_ffmpeg(fpath, wav_path)

        audio, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        audio = peak_normalize(audio)
        segs = vad_segments(audio)

        for idx, (s, e) in enumerate(tqdm(segs, desc=f"Embedding {fpath.stem}")):
            clip_24k = audio[s:e]
            clip_16k = librosa.resample(clip_24k, orig_sr=TARGET_SR, target_sr=EMBED_SR)
            sig = torch.from_numpy(clip_16k).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = classifier.encode_batch(sig).squeeze().cpu().numpy()
            # L2 normalize for cosine clustering
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            all_clips.append((f"{fpath.stem}_{idx + 1:04d}", clip_24k))
            embeddings.append(emb)

    if len(embeddings) < num_speakers:
        raise RuntimeError(
            f"Only found {len(embeddings)} speech segments — too few to cluster into {num_speakers} speakers."
        )

    X = np.vstack(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=num_speakers,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X)

    # Save separated clips and generate a preview WAV per speaker
    speaker_totals = {i: 0.0 for i in range(num_speakers)}
    speaker_clips: dict[int, List[Tuple[str, np.ndarray]]] = {i: [] for i in range(num_speakers)}

    for (clip_name, clip_audio), label in zip(all_clips, labels):
        spk_name = f"SPEAKER_{label:02d}"
        spk_dir = output_dir / spk_name
        spk_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(spk_dir / f"{clip_name}.wav"), peak_normalize(clip_audio), TARGET_SR, subtype="PCM_16")
        speaker_totals[label] += len(clip_audio) / TARGET_SR
        speaker_clips[label].append((clip_name, clip_audio))

    print("\n=== Diarization Complete ===")
    for label in range(num_speakers):
        spk_name = f"SPEAKER_{label:02d}"
        clips_list = speaker_clips[label]
        # Pick a clean 5-10s clip (or the longest clip) as preview
        ideal = [c for c in clips_list if 4.5 <= len(c[1]) / TARGET_SR <= 10.0]
        preview_clip = max(ideal or clips_list, key=lambda c: len(c[1]))
        preview_path = output_dir / f"preview_{spk_name}.wav"
        sf.write(str(preview_path), peak_normalize(preview_clip[1]), TARGET_SR, subtype="PCM_16")
        print(
            f"  {spk_name}: {len(clips_list)} clips | {speaker_totals[label] / 60:.2f} min | Preview: {preview_path}"
        )

    print(
        "\nListen to the preview_*.wav files, then run:\n"
        "  python scripts/00_extract_diarize_video.py --assign SPEAKER_00=vn_01 SPEAKER_01=vn_02"
    )


def assign_speakers(assignments: List[str], diarized_dir: Path, profiles_root: Path) -> None:
    """Copy diarized speaker folders into profiles/<speaker_id>/raw_recordings/."""
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"Invalid assignment '{item}'. Use format SPEAKER_00=vn_01")
        src_name, target_speaker = item.split("=", 1)
        src_dir = diarized_dir / src_name
        if not src_dir.exists():
            raise FileNotFoundError(f"Diarized folder not found: {src_dir}")

        dst_dir = profiles_root / target_speaker / "raw_recordings"
        dst_dir.mkdir(parents=True, exist_ok=True)

        wavs = sorted(src_dir.glob("*.wav"))
        for w in wavs:
            shutil.copy2(w, dst_dir / f"video_{w.name}")
        print(f"Copied {len(wavs)} clips from {src_name} -> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio from videos and separate speakers.")
    parser.add_argument("--input", default="profiles/raw_videos", help="Video file or folder")
    parser.add_argument("--output", default="profiles/diarized", help="Output folder for separated speakers")
    parser.add_argument("--num-speakers", type=int, default=2, help="Number of speakers to separate")
    parser.add_argument(
        "--assign",
        nargs="+",
        help="Assign diarized clusters to speaker profiles, e.g., SPEAKER_00=vn_01 SPEAKER_01=vn_02",
    )
    parser.add_argument("--profiles-dir", default="profiles")
    args = parser.parse_args()

    if args.assign:
        assign_speakers(args.assign, Path(args.output), Path(args.profiles_dir))
    else:
        diarize_videos(Path(args.input), Path(args.output), args.num_speakers)


if __name__ == "__main__":
    main()
