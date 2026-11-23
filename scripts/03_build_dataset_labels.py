"""Step 3: Generate clean_text pairs and tone labels for Stage 1 learning.

Supports either an OpenAI-compatible LLM API (OpenAI, Gemini, local Ollama)
or a fast offline --heuristic mode for testing.

Usage:
    python scripts/03_build_dataset_labels.py --speaker vn_01
    python scripts/03_build_dataset_labels.py --speaker vn_01 --heuristic
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schemas import ToneType, load_metadata, save_metadata

CLEANUP_SYSTEM_PROMPT = (
    "You convert verbatim spoken English presentation transcripts into polished, "
    "clean presentation English while preserving the exact meaning and facts. "
    "Remove spoken fillers (uh, um, eh, erm, you know, redundant 'so'/'like') and false starts. "
    "Also classify tone as one of: presentation, explaining, emphasis. "
    'Return ONLY valid JSON: {"clean_text": "...", "tone": "presentation"}'
)


def clean_heuristic(raw_text: str) -> tuple[str, ToneType]:
    """Simple offline fallback cleaner when no LLM API is configured."""
    cleaned = re.sub(
        r"\b(uh+|um+|eh+|erm+|you know)\b[,]?\s*",
        "",
        raw_text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\.\.\.", ",", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned, "presentation"


def clean_with_llm(client, model: str, raw_text: str) -> tuple[str, ToneType]:
    """Use an OpenAI-compatible LLM to generate clean_text and tone."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
    )
    payload = json.loads(resp.choices[0].message.content)
    clean_text = payload.get("clean_text", raw_text).strip()
    tone = payload.get("tone", "presentation")
    if tone not in ("presentation", "explaining", "emphasis"):
        tone = "presentation"
    return clean_text, tone


def build_labels(
    speaker_id: str,
    profiles_root: Path,
    use_heuristic: bool,
    model: str,
    overwrite: bool,
) -> None:
    load_dotenv()
    meta_path = profiles_root / speaker_id / "metadata.jsonl"
    clips = load_metadata(meta_path)
    if not clips:
        raise FileNotFoundError(f"No metadata found at {meta_path}. Run 02_transcribe_align.py first.")

    client = None
    if not use_heuristic:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key and not base_url:
            raise EnvironmentError(
                "Set OPENAI_API_KEY (and optional OPENAI_BASE_URL) in .env, or pass --heuristic."
            )
        client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

    updated = 0
    for clip in tqdm(clips, desc=f"Labeling {speaker_id}"):
        if clip.clean_text and not overwrite:
            continue
        if use_heuristic:
            clean_text, tone = clean_heuristic(clip.raw_text)
        else:
            clean_text, tone = clean_with_llm(client, model, clip.raw_text)
        clip.clean_text = clean_text
        clip.tone = tone
        updated += 1

    save_metadata(clips, meta_path)
    print(f"Updated {updated} clips in {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean_text and tone labels.")
    parser.add_argument("--speaker", required=True, help="Speaker ID, e.g., vn_01")
    parser.add_argument("--profiles-dir", default="profiles")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--heuristic", action="store_true", help="Offline regex cleaner (no API key)")
    parser.add_argument("--overwrite", action="store_true", help="Re-label clips that already have clean_text")
    args = parser.parse_args()

    build_labels(
        speaker_id=args.speaker,
        profiles_root=Path(args.profiles_dir),
        use_heuristic=args.heuristic,
        model=args.model,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
