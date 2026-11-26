"""Stage 2: Accent-Preserving Long-Form Voice Synthesizer (F5-TTS).

Supports:
  - Zero-Shot Reference Mode (using profiles/<speaker_id>/refs/presentation_ref.wav + .txt)
  - Per-Speaker Fine-Tuned Checkpoint/LoRA Mode (auto-detected in profiles/<speaker_id>/lora/)
  - Automatic pronunciation lexicon safety pass (MSE -> M S E, CNN -> C N N, etc.)
  - Long-form paragraph/slide synthesis with natural inter-paragraph pauses & smooth fades.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch

from src.humanizer import apply_pronunciation_map
from src.schemas import SynthesizeRequest

TARGET_SR = 24000


def stitch_paragraphs(
    paragraph_wavs: List[np.ndarray],
    sr: int,
    paragraph_pause_ms: int = 650,
    fade_ms: int = 20,
) -> np.ndarray:
    """Concatenate synthesized paragraphs with natural slide/section pauses and soft fades."""
    if not paragraph_wavs:
        return np.zeros(int(sr * 0.5), dtype=np.float32)

    fade_samples = min(int(sr * fade_ms / 1000), 480)
    pause = np.zeros(int(sr * paragraph_pause_ms / 1000), dtype=np.float32)

    parts: List[np.ndarray] = []
    for idx, wav in enumerate(paragraph_wavs):
        wav = np.asarray(wav, dtype=np.float32).flatten()
        if len(wav) > 2 * fade_samples:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            wav[:fade_samples] *= fade_in
            wav[-fade_samples:] *= fade_out
        parts.append(wav)
        if idx < len(paragraph_wavs) - 1:
            parts.append(pause)

    combined = np.concatenate(parts)
    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = combined / peak * 0.92
    return combined


def _sanitize_tts_punctuation(text: str) -> str:
    """Normalize typographic punctuation (em-dashes, smart quotes) so F5-TTS chunks cleanly.

    Em-dashes (`—`) are out-of-distribution in character-level TTS training sets and stop
    F5-TTS from splitting long clauses, which causes stretched attention and metallic/robotic buzz.
    """
    out = text.replace("\u2014", ", ").replace("\u2013", ", ").replace(" -- ", ", ")
    out = out.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    out = re.sub(r"\s*,\s*,+", ", ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


class PresentationSynthesizer:
    """Lazy-loading F5-TTS wrapper that caches the active model on local RTX 5060 GPU."""

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

    def list_references(self, speaker_id: str) -> List[str]:
        """List available reference clip stems inside profiles/<speaker_id>/refs/."""
        refs_dir = self.profiles_root / speaker_id / "refs"
        if not refs_dir.exists():
            return []
        stems = [
            p.stem
            for p in sorted(refs_dir.glob("*.wav"))
            if (refs_dir / f"{p.stem}.txt").exists()
        ]
        return stems

    def synthesize(self, req: SynthesizeRequest) -> Tuple[Path, str]:
        """Synthesize short or long multi-paragraph presentation audio."""
        spk_dir = self.profiles_root / req.speaker_id
        refs_dir = spk_dir / "refs"

        chosen_ref = req.ref_name
        if not chosen_ref:
            avail = self.list_references(req.speaker_id)
            chosen_ref = (
                "partA_natural_slow_ref"
                if "partA_natural_slow_ref" in avail
                else (avail[0] if avail else "presentation_ref")
            )

        ref_wav = refs_dir / f"{chosen_ref}.wav"
        ref_txt_path = refs_dir / f"{chosen_ref}.txt"

        if not ref_wav.exists() or not ref_txt_path.exists():
            raise FileNotFoundError(
                f"Reference clip missing for '{req.speaker_id}'. Expected {ref_wav} and {ref_txt_path}."
            )

        ref_text = ref_txt_path.read_text(encoding="utf-8").strip()
        custom_ckpt = self._find_speaker_ckpt(req.speaker_id)
        tts = self._get_model(custom_ckpt)

        mode_info = (
            f"Fine-tuned ({custom_ckpt.name}) | ref: {chosen_ref}"
            if custom_ckpt
            else f"Zero-Shot F5-TTS | ref: {chosen_ref}"
        )

        # Split by blank lines to insert natural slide/paragraph pauses, sanitize punctuation, and apply pronunciation pass
        paragraphs = [
            _sanitize_tts_punctuation(apply_pronunciation_map(p.strip()))
            for p in re.split(r"\n\s*\n", req.humanized_script.strip())
            if p.strip()
        ]
        if not paragraphs:
            paragraphs = [_sanitize_tts_punctuation(apply_pronunciation_map(req.humanized_script.strip()))]

        t0 = time.perf_counter()
        para_wavs: List[np.ndarray] = []
        sr = TARGET_SR

        for para in paragraphs:
            wav, sr, _ = tts.infer(
                ref_file=str(ref_wav),
                ref_text=ref_text,
                gen_text=para,
                speed=req.speed,
                nfe_step=req.nfe_step,
                cross_fade_duration=0.15,
                remove_silence=False,
            )
            para_wavs.append(wav)

        elapsed = time.perf_counter() - t0
        final_audio = stitch_paragraphs(
            para_wavs, sr=sr, paragraph_pause_ms=req.paragraph_pause_ms
        )

        out_name = f"{req.speaker_id}_{int(time.time())}.wav"
        out_path = self.outputs_dir / out_name
        sf.write(str(out_path), final_audio, sr, subtype="PCM_16")
        return (
            out_path,
            f"{mode_info} | {len(paragraphs)} para(s) | {req.nfe_step} steps | "
            f"audio: {len(final_audio) / sr:.1f}s | generated in {elapsed:.1f}s",
        )
