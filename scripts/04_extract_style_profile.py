"""Step 4: Extract a speaker's individualized StyleProfile (style_profile.json) from metadata.jsonl.

Usage:
    python scripts/04_extract_style_profile.py --speaker vn_01 --display-name "Binh (Vietnamese English)"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schemas import (
    FewShotExample,
    FillerLevel,
    StyleProfile,
    load_metadata,
    save_style_profile,
)


def extract_profile(
    speaker_id: str,
    display_name: str,
    profiles_root: Path,
    max_examples: int = 12,
) -> None:
    speaker_dir = profiles_root / speaker_id
    meta_path = speaker_dir / "metadata.jsonl"
    clips = load_metadata(meta_path)
    if not clips:
        raise FileNotFoundError(f"No metadata found at {meta_path}")

    accent = clips[0].accent
    total_words = 0
    total_duration_sec = 0.0
    total_pauses = 0
    filler_counter: Counter[str] = Counter()

    for clip in clips:
        words = len(clip.raw_text.split())
        total_words += words
        total_duration_sec += clip.duration_sec
        total_pauses += len(clip.pause_events)
        filler_counter.update(clip.fillers_used)

    wpm = round((total_words / max(1.0, total_duration_sec)) * 60.0, 1)
    avg_words_per_pause = round(total_words / max(1, total_pauses), 1)

    total_fillers = sum(filler_counter.values())
    filler_distribution: Dict[str, float] = {}
    if total_fillers > 0:
        for token, count in filler_counter.most_common():
            filler_distribution[token] = round(count / total_fillers, 3)

    # Select balanced real few-shot examples across low, medium, and high filler levels
    few_shots: List[FewShotExample] = []
    per_level_quota = max(1, max_examples // 3)
    target_levels: List[FillerLevel] = ["low", "medium", "high"]

    for level in target_levels:
        candidates = [
            c
            for c in clips
            if c.filler_level == level
            and c.clean_text
            and c.clean_text.strip() != c.raw_text.strip()
            and 8 <= len(c.raw_text.split()) <= 35
        ]
        # Prefer verified clips first
        candidates.sort(key=lambda c: (not c.verified, -len(c.fillers_used)))
        for c in candidates[:per_level_quota]:
            few_shots.append(
                FewShotExample(
                    filler_level=level,
                    clean_text=c.clean_text or c.raw_text,
                    raw_text=c.raw_text,
                )
            )

    profile = StyleProfile(
        speaker_id=speaker_id,
        display_name=display_name or f"{speaker_id} ({accent})",
        accent=accent,
        speaking_rate_wpm=wpm,
        avg_words_per_pause=avg_words_per_pause,
        filler_distribution=filler_distribution,
        few_shot_examples=few_shots,
    )

    out_path = speaker_dir / "style_profile.json"
    save_style_profile(profile, out_path)
    print(f"Saved style profile to {out_path}")
    print(f"  WPM: {wpm} | Avg words/pause: {avg_words_per_pause}")
    print(f"  Filler distribution: {filler_distribution}")
    print(f"  Few-shot examples selected: {len(few_shots)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract individualized speaker style_profile.json.")
    parser.add_argument("--speaker", required=True, help="Speaker ID, e.g., vn_01")
    parser.add_argument("--display-name", default="", help="Human-friendly speaker display name")
    parser.add_argument("--profiles-dir", default="profiles")
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()

    extract_profile(
        speaker_id=args.speaker,
        display_name=args.display_name,
        profiles_root=Path(args.profiles_dir),
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
