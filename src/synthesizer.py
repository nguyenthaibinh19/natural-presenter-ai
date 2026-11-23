"""Stage 2: Accent-Preserving Voice Synthesizer (F5-TTS).

Supports:
  - Zero-Shot Reference Mode (using profiles/<speaker_id>/refs/presentation_ref.wav + .txt)
  - Per-Speaker Fine-Tuned Checkpoint/LoRA Mode (automatically detected in profiles/<speaker_id>/lora/)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch

from src.schemas import SynthesizeRequest

TARGET_SR = 24000


def split_into_chunks(text: str, max_chars: int = 220) -> List[str]:
    """Split a presentation script into natural sentence-aligned chunks for stable TTS."""
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if not sent:
            continue
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chars:
            current = f"{current} {sent}"
        else:
            chunks.append(current)
            current = sent

    if current:
        chunks.append(current)
    return chunks


def stitch_audio(chunks: List[np.ndarray], sr: int, pause_ms: int = 350, fade_ms: int = 20) -> np.ndarray:
    """Concatenate synthesized chunks with natural inter-sentence pauses and short fades."""
    if not chunks:
        return np.zeros(int(sr * 0.5), dtype=np.float32)

    fade_samples = min(int(sr * fade_ms / 1000), 480)
    pause = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)

    parts: List[np.ndarray] = []
    for idx, wav in enumerate(chunks):
        wav = np.asarray(wav, dtype=np.float32).flatten()
        if len(wav) > 2 * fade_samples:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            wav[:fade_samples] *= fade_in
            wav[-fade_samples:] *= fade_out
        parts.append(wav)
        if idx < len(chunks) - 1:
            parts.append(pause)

    combined = np.concatenate(parts)
    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = combined / peak * 0.92
    return combined


class PresentationSynthesizer:
    """Lazy-loading F5-TTS wrapper that caches the base model on RTX 5060 GPU."""

    def __init__(self, profiles_root: Path = Path("profiles"), outputs_dir: Path = Path("outputs")):
        self.profiles_root = profiles_root
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._tts = None
        self._loaded_ckpt: str | None = None

    def _get_model(self, custom_ckpt: Path | None = None):
        from f5_tts.api import F5TTS

        ckpt_str = str(custom_ckpt) if custom_ckpt else "DEFAULT_BASE"
        if self._tts is None or self._loaded_ckpt != ckpt_str:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if custom_ckpt and custom_ckpt.exists():
                self._tts = F5TTS(ckpt_file=str(custom_ckpt), device=device)
            else:
                self._tts = F5TTS(device=device)
            self._loaded_ckpt = ckpt_str
        return self._tts

    def _find_speaker_ckpt(self, speaker_id: str) -> Path | None:
        lora_dir = self.profiles_root / speaker_id / "lora"
        if not lora_dir.exists():
            return None
        for ext in ("*.safetensors", "*.pt", "*.ckpt"):
            found = sorted(lora_dir.glob(ext))
            if found:
                return found[0]
        return None

    def synthesize(self, req: SynthesizeRequest) -> Tuple[Path, str]:
        """Synthesize presentation audio and return (output_wav_path, mode_info)."""
        spk_dir = self.profiles_root / req.speaker_id
        ref_wav = spk_dir / "refs" / "presentation_ref.wav"
        ref_txt_path = spk_dir / "refs" / "presentation_ref.txt"

        if not ref_wav.exists() or not ref_txt_path.exists():
            raise FileNotFoundError(
                f"Reference clip missing for '{req.speaker_id}'. Expected {ref_wav} and {ref_txt_path}."
            )

        ref_text = ref_txt_path.read_text(encoding="utf-8").strip()
        custom_ckpt = self._find_speaker_ckpt(req.speaker_id)
        tts = self._get_model(custom_ckpt)

        mode_info = (
            f"Fine-tuned checkpoint ({custom_ckpt.name})"
            if custom_ckpt
            else f"Zero-Shot F5-TTS (ref: {ref_wav.name})"
        )

        chunks = split_into_chunks(req.humanized_script)
        audio_chunks: List[np.ndarray] = []
        sr = TARGET_SR

        for chunk in chunks:
            wav, sr, _ = tts.infer(
                ref_file=str(ref_wav),
                ref_text=ref_text,
                gen_text=chunk,
                speed=req.speed,
                remove_silence=False,
            )
            audio_chunks.append(wav)

        final_audio = stitch_audio(audio_chunks, sr=sr)
        out_name = f"{req.speaker_id}_{int(time.time())}.wav"
        out_path = self.outputs_dir / out_name
        sf.write(str(out_path), final_audio, sr, subtype="PCM_16")
        return out_path, f"{mode_info} | {len(chunks)} chunk(s) | {len(final_audio) / sr:.1f}s"
