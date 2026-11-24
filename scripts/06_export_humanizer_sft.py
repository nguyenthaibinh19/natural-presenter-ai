"""Step 6 (Optional): Export verified (clean_text -> raw_text) pairs across speakers for Stage 1 LLM fine-tuning.

Generates a ChatML JSONL file (`profiles/humanizer_sft.jsonl`) suitable for QLoRA fine-tuning
Gemma / Qwen models (e.g., via Unsloth or TRL) when prompt-based Stage 1 needs tighter adherence
to each speaker's unique filler & pause rhythm.

Usage:
    python scripts/06_export_humanizer_sft.py
    python scripts/06_export_humanizer_sft.py --speakers vn_01 vn_02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.humanizer import build_llm_system_prompt
from src.schemas import load_metadata, load_style_profile


def export_sft_dataset(
    profiles_root: Path,
    speaker_ids: List[str] | None,
    out_path: Path,
    only_verified: bool = True,
) -> None:
    if speaker_ids:
        spk_dirs = [profiles_root / sid for sid in speaker_ids]
    else:
        spk_dirs = sorted(p.parent for p in profiles_root.glob("*/style_profile.json"))

    records = []
    per_speaker_counts = {}

    for spk_dir in spk_dirs:
        meta_path = spk_dir / "metadata.jsonl"
        prof_path = spk_dir / "style_profile.json"
        if not meta_path.exists() or not prof_path.exists():
            continue

        profile = load_style_profile(prof_path)
        clips = load_metadata(meta_path)
        count = 0

        for clip in clips:
            if only_verified and not clip.verified:
                continue
            if not clip.clean_text or not clip.raw_text:
                continue

            sys_prompt = build_llm_system_prompt(profile, clip.filler_level, clip.tone)
            records.append(
                {
                    "speaker_id": clip.speaker_id,
                    "accent": clip.accent,
                    "filler_level": clip.filler_level,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": clip.clean_text.strip()},
                        {"role": "assistant", "content": clip.raw_text.strip()},
                    ],
                }
            )
            count += 1
        per_speaker_counts[profile.speaker_id] = count

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Exported {len(records)} Stage 1 SFT pairs to {out_path}")
    print(f"Per-speaker breakdown: {per_speaker_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage 1 LLM fine-tuning dataset (ChatML JSONL).")
    parser.add_argument("--profiles-dir", default="profiles")
    parser.add_argument("--speakers", nargs="+", help="Optional speaker IDs to include")
    parser.add_argument("--out", default="profiles/humanizer_sft.jsonl")
    parser.add_argument("--include-unverified", action="store_true")
    args = parser.parse_args()

    export_sft_dataset(
        profiles_root=Path(args.profiles_dir),
        speaker_ids=args.speakers,
        out_path=Path(args.out),
        only_verified=not args.include_unverified,
    )


if __name__ == "__main__":
    main()
