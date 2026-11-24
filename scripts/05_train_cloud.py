"""Step 5: Prepare F5-TTS fine-tuning dataset and launch per-speaker fine-tuning on Cloud GPU (RTX 4090).

Converts profiles/<speaker_id>/metadata.jsonl + wavs/ into F5-TTS's expected format
(metadata.csv + arrow dataset) and invokes `f5-tts_finetune-cli`.

Usage:
    # 1. Prepare & verify dataset locally before uploading to RunPod / Vast.ai:
    python scripts/05_train_cloud.py --speaker vn_01 --prepare-only

    # 2. Run fine-tuning on Cloud RTX 4090 (24GB VRAM):
    python scripts/05_train_cloud.py --speaker vn_01 --epochs 80 --lr 1e-5 --batch-frames 3200

After training finishes, copy the best `.safetensors` / `.pt` checkpoint into:
    profiles/<speaker_id>/lora/
The local Web UI (src/synthesizer.py) will automatically load it instead of Zero-Shot!
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schemas import load_metadata


def prepare_f5_dataset(
    speaker_id: str,
    profiles_root: Path,
    only_verified: bool = False,
) -> Path:
    """Export speaker metadata.jsonl + wavs/ into F5-TTS fine-tuning directory."""
    spk_dir = profiles_root / speaker_id
    meta_path = spk_dir / "metadata.jsonl"
    clips = load_metadata(meta_path)
    if not clips:
        raise FileNotFoundError(f"No metadata found at {meta_path}")

    if only_verified:
        clips = [c for c in clips if c.verified]
        if not clips:
            raise ValueError("No clips marked verified=True yet. Review clips in Dataset QA tab first.")

    export_dir = spk_dir / "f5_dataset"
    wavs_out = export_dir / "wavs"
    wavs_out.mkdir(parents=True, exist_ok=True)

    csv_lines = ["audio_file|text"]
    durations = []
    vocab_chars = set()

    for clip in clips:
        src_wav = spk_dir / clip.audio_path
        if not src_wav.exists():
            continue
        dst_wav = wavs_out / src_wav.name
        if not dst_wav.exists():
            shutil.copy2(src_wav, dst_wav)

        # Clean newlines/pipes from raw_text while strictly preserving uh/um/eh
        text = clip.raw_text.replace("|", " ").replace("\n", " ").strip()
        if not text:
            continue

        csv_lines.append(f"wavs/{src_wav.name}|{text}")
        durations.append(clip.duration_sec)
        vocab_chars.update(text)

    (export_dir / "metadata.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (export_dir / "duration.json").write_text(
        json.dumps({"duration": durations}, indent=2), encoding="utf-8"
    )

    total_min = sum(durations) / 60.0
    print(f"Prepared F5-TTS dataset at {export_dir}")
    print(f"  Clips: {len(durations)} | Total duration: {total_min:.2f} minutes")
    if total_min < 20.0:
        print(
            "  [TIP] Dataset is under 20 minutes. Fine-tuning works best with 30-60 minutes "
            "to capture full accent & disfluency stability."
        )
    return export_dir


def run_f5_finetune(
    speaker_id: str,
    dataset_dir: Path,
    epochs: int,
    lr: float,
    batch_frames: int,
    save_per_updates: int,
) -> None:
    """Invoke F5-TTS official fine-tune CLI with pretrained 2545-token vocab (raw chars for English)."""
    cmd = [
        "f5-tts_finetune-cli",
        "--exp_name",
        "F5TTS_v1_Base",
        "--dataset_name",
        speaker_id,
        "--learning_rate",
        str(lr),
        "--batch_size_per_gpu",
        str(batch_frames),
        "--batch_size_type",
        "frame",
        "--epochs",
        str(epochs),
        "--num_warmup_updates",
        "100",
        "--save_per_updates",
        str(save_per_updates),
        "--last_per_updates",
        "250",
        "--keep_last_n_checkpoints",
        "4",
        "--finetune",
        "--tokenizer",
        "pinyin",
        "--bnb_optimizer",
    ]
    print("\nLaunching F5-TTS fine-tuning:")
    print("  " + " ".join(cmd))
    print(f"  (Ensure dataset arrow files from {dataset_dir} are registered for F5-TTS data dir.)")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run F5-TTS fine-tuning on Cloud GPU.")
    parser.add_argument("--speaker", required=True, help="Speaker ID, e.g., vn_01 or vn_02")
    parser.add_argument("--profiles-dir", default="profiles")
    parser.add_argument("--only-verified", action="store_true", help="Only include QA-verified clips")
    parser.add_argument("--prepare-only", action="store_true", help="Export dataset only without training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-frames", type=int, default=3200)
    parser.add_argument("--save-per-updates", type=int, default=500)
    args = parser.parse_args()

    dataset_dir = prepare_f5_dataset(
        speaker_id=args.speaker,
        profiles_root=Path(args.profiles_dir),
        only_verified=args.only_verified,
    )

    if not args.prepare_only:
        run_f5_finetune(
            speaker_id=args.speaker,
            dataset_dir=dataset_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_frames=args.batch_frames,
            save_per_updates=args.save_per_updates,
        )


if __name__ == "__main__":
    main()
