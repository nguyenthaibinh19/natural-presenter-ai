"""Step 7 (Optional): Generate 6 neutral portfolio audio samples for the free-CPU Hugging Face Space.

By default, generates ONLY self-recorded author voice (`vn_01`) clips to strictly honor
our Responsible AI & Voice Consent policy. Pass `--include-coauthor` only after receiving
explicit co-author consent.

Saves clips into `hf_space/samples/` (gitignored in GitHub via `*.wav`, ready to upload to HF Spaces).

Usage:
    python scripts/07_generate_showcase_samples.py
    python scripts/07_generate_showcase_samples.py --only 06
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schemas import SynthesizeRequest
from src.synthesizer import PresentationSynthesizer, stitch_paragraphs

SAMPLES_DIR = ROOT_DIR / "hf_space" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def render_to(
    synth: PresentationSynthesizer,
    out_name: str,
    speaker_id: str,
    text: str,
    ref_name: str | None = None,
    speed: float = 0.90,
    nfe_step: int = 32,
    force_zeroshot: bool = False,
) -> None:
    if force_zeroshot:
        # Temporarily bypass lora/ checkpoint to produce true Zero-Shot baseline
        orig_find = synth._find_speaker_ckpt
        synth._find_speaker_ckpt = lambda _spk: None  # type: ignore[method-assign]
    try:
        tmp_wav, info = synth.synthesize(
            SynthesizeRequest(
                speaker_id=speaker_id,
                humanized_script=text,
                ref_name=ref_name,
                speed=speed,
                nfe_step=nfe_step,
            )
        )
    finally:
        if force_zeroshot:
            synth._find_speaker_ckpt = orig_find  # type: ignore[method-assign]

    data, sr = sf.read(str(tmp_wav))
    out_path = SAMPLES_DIR / out_name
    sf.write(str(out_path), data, sr, subtype="PCM_16")
    print(f"Saved {out_path.name} ({len(data) / sr:.1f}s) -> {info}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate neutral portfolio showcase WAV clips for hf_space/samples/.")
    parser.add_argument("--only", default="", help="Optional filename prefix filter (e.g. '06' to render only clip 06).")
    parser.add_argument(
        "--include-coauthor",
        action="store_true",
        help="Opt-in: also render 06_multi_speaker_handoff.wav with vn_02 (requires explicit co-author consent).",
    )
    args = parser.parse_args()

    def should_run(name: str) -> bool:
        return not args.only or name.startswith(args.only)

    synth = PresentationSynthesizer(
        profiles_root=ROOT_DIR / "profiles",
        outputs_dir=ROOT_DIR / "outputs",
    )

    demo_clean = (
        "Hello everyone, today I want to walk you through our convolutional autoencoder results. "
        "Replacing max pooling with stride-two convolutions reduced test MSE by twenty-five percent across all three seeds."
    )
    demo_medium = (
        "Hello everyone, today I want to, uh, walk you through our convolutional autoencoder results. "
        "Replacing max pooling with, like, stride-two convolutions reduced test M S E by twenty-five percent across all three seeds."
    )
    demo_acronyms = (
        "On a single G P U, our C N N autoencoder uses ray-loo activations and reached a test M S E of 0.0023 in F P sixteen."
    )

    # 1. Zero-Shot Baseline vs 2. Fine-Tuned FT-001
    if should_run("01_zeroshot_baseline.wav"):
        render_to(
            synth,
            "01_zeroshot_baseline.wav",
            "vn_01",
            demo_medium,
            ref_name="partA_natural_slow_ref",
            force_zeroshot=True,
        )
    if should_run("02_finetuned_ft001.wav"):
        render_to(
            synth,
            "02_finetuned_ft001.wav",
            "vn_01",
            demo_medium,
            ref_name="partA_natural_slow_ref",
            force_zeroshot=False,
        )

    # 3. Filler None vs 4. Filler Medium
    if should_run("03_filler_none.wav"):
        render_to(
            synth,
            "03_filler_none.wav",
            "vn_01",
            demo_clean,
            ref_name="partB_studio_clear_ref",
        )
    if should_run("04_filler_medium_natural.wav"):
        render_to(
            synth,
            "04_filler_medium_natural.wav",
            "vn_01",
            demo_medium,
            ref_name="partA_natural_slow_ref",
        )

    # 5. Acronym Lexicon Fix
    if should_run("05_acronym_mse_fixed.wav"):
        render_to(
            synth,
            "05_acronym_mse_fixed.wav",
            "vn_01",
            demo_acronyms,
            ref_name="partA_natural_slow_ref",
        )

    # 6. Long-Form Slide Transition Pause + Subtle Self-Repair (100% vn_01 self-recorded voice)
    if should_run("06_self_repair_and_slide_pause.wav"):
        p1 = (
            "So, uh, looking at the model compression step, our raw training checkpoint was three point three eight gigabytes."
        )
        p2 = (
            "However, after stripping the optimizer state, we, we pruned the F P sixteen weights down to six hundred forty-three megabytes for local G P U inference."
        )
        w1, _ = synth.synthesize(
            SynthesizeRequest(
                speaker_id="vn_01",
                humanized_script=p1,
                ref_name="partA_natural_slow_ref",
                speed=0.90,
                nfe_step=32,
            )
        )
        w2, _ = synth.synthesize(
            SynthesizeRequest(
                speaker_id="vn_01",
                humanized_script=p2,
                ref_name="partA_natural_slow_ref",
                speed=0.90,
                nfe_step=32,
            )
        )
        d1, sr1 = sf.read(str(w1))
        d2, sr2 = sf.read(str(w2))
        assert sr1 == sr2
        combined = stitch_paragraphs(
            [np.asarray(d1, dtype=np.float32), np.asarray(d2, dtype=np.float32)],
            sr=sr1,
            paragraph_pause_ms=750,
        )
        out6 = SAMPLES_DIR / "06_self_repair_and_slide_pause.wav"
        sf.write(str(out6), combined, sr1, subtype="PCM_16")
        print(f"Saved {out6.name} ({len(combined) / sr1:.1f}s) -> Slide transition pause + self-repair")

    # Optional opt-in co-author handoff (disabled by default until explicit consent)
    if args.include_coauthor and should_run("06_multi_speaker_handoff.wav"):
        w1, _ = synth.synthesize(
            SynthesizeRequest(
                speaker_id="vn_01",
                humanized_script="So, uh, looking at Exercise two, spatial sixteen lowered test M S E by fifty-nine percent with the exact same parameters. Now I will hand over to Ege for colourisation.",
                ref_name="partA_natural_slow_ref",
                speed=0.90,
                nfe_step=32,
            )
        )
        w2, _ = synth.synthesize(
            SynthesizeRequest(
                speaker_id="vn_02",
                humanized_script="Thanks! Uh, for colourisation, predicting chroma while reusing input luminance clearly outperformed direct R G B.",
                speed=0.90,
                nfe_step=32,
            )
        )
        d1, sr1 = sf.read(str(w1))
        d2, sr2 = sf.read(str(w2))
        assert sr1 == sr2
        combined = stitch_paragraphs(
            [np.asarray(d1, dtype=np.float32), np.asarray(d2, dtype=np.float32)],
            sr=sr1,
            paragraph_pause_ms=700,
        )
        out_co = SAMPLES_DIR / "06_multi_speaker_handoff.wav"
        sf.write(str(out_co), combined, sr1, subtype="PCM_16")
        print(f"Saved {out_co.name} ({len(combined) / sr1:.1f}s) -> Multi-speaker handoff (consented)")

    print(f"\nShowcase samples ready in {SAMPLES_DIR}! Upload hf_space/ to Hugging Face Spaces.")


if __name__ == "__main__":
    main()
