"""Stage 1: Personalized Script Humanizer.

Transforms a clean presentation script into a natural spoken script using a specific
speaker's individualized StyleProfile (filler vocabulary, cadence, and few-shot examples).

Supports:
  - LLM mode (OpenAI-compatible API: OpenAI, Gemini, or local Ollama via .env)
  - Zero-setup Offline Profile mode (uses speaker's exact filler_distribution when no API key is set)
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from src.schemas import FillerLevel, HumanizeRequest, HumanizeResponse, StyleProfile, load_style_profile

load_dotenv()


def build_llm_system_prompt(profile: StyleProfile, filler_level: FillerLevel, tone: str) -> str:
    """Build a speaker-specific system prompt from style_profile.json."""
    target_per_100 = profile.filler_targets_per_100_words.get(filler_level, 4)
    fillers_str = ", ".join(
        f'"{tok}" ({int(prob * 100)}%)' for tok, prob in profile.filler_distribution.items()
    ) or '"uh" (70%), "um" (30%)'

    examples_block = ""
    matching = [ex for ex in profile.few_shot_examples if ex.filler_level == filler_level]
    chosen = (matching or profile.few_shot_examples)[:6]
    if chosen:
        lines = []
        for idx, ex in enumerate(chosen, 1):
            lines.append(f"Example {idx} ({ex.filler_level}):")
            lines.append(f"  Clean:   {ex.clean_text}")
            lines.append(f"  Natural: {ex.raw_text}")
        examples_block = "\n".join(lines)

    return f"""You are a Personalized Presentation Script Humanizer for speaker "{profile.display_name}" (accent: {profile.accent}).
Your job is to adapt a clean written presentation script into natural spoken English that matches THIS EXACT SPEAKER's personal speaking rhythm and hesitation habits.

Rules (STRICT):
1. Preserve 100% of the facts, numbers, technical terms, and core meaning.
2. Expand symbols/abbreviations into speakable English (e.g., "$2.5M" -> "two point five million dollars", "Q3" -> "Q three", "%" -> "percent").
3. Desired hesitation intensity: "{filler_level}" (target roughly {target_per_100} fillers per 100 words).
   - If "none": insert ZERO fillers; only optimize punctuation for natural spoken delivery.
4. ONLY use this speaker's personal filler vocabulary in roughly these proportions: {fillers_str}.
5. Place hesitations ONLY at natural cognitive boundaries (sentence transitions, after introductory clauses, before key technical metrics/terms) — never awkwardly inside a compound noun.
6. Tone: {tone}. Average speaker pace: {profile.speaking_rate_wpm} WPM (~{profile.avg_words_per_pause} words per natural pause).
7. Return ONLY the rewritten spoken script text — no commentary, no markdown quotes.

Real examples of how this speaker talks:
{examples_block}
"""


def humanize_offline(
    clean_script: str,
    profile: StyleProfile,
    filler_level: FillerLevel,
    seed: int = 42,
) -> str:
    """Deterministic offline humanizer driven strictly by the speaker's style_profile.json."""
    text = re.sub(r"\s+", " ", clean_script.strip())
    if filler_level == "none":
        return text

    target_per_100 = profile.filler_targets_per_100_words.get(filler_level, 4)
    words = text.split()
    if not words or target_per_100 <= 0:
        return text

    desired_count = max(1, round((len(words) / 100.0) * target_per_100))

    # Build weighted filler list from speaker's own distribution
    dist = profile.filler_distribution or {"uh": 0.7, "um": 0.3}
    tokens = list(dist.keys())
    weights = list(dist.values())

    rng = random.Random(seed + len(words))

    # Split into sentences and insert fillers at natural cognitive boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Candidate insertion slots: (sentence_idx, kind)
    inserted = 0
    out_sentences: List[str] = []

    transition_words = {
        "and", "but", "so", "however,", "therefore,", "basically,", "first,", "second,",
        "finally,", "because", "when", "if", "then", "also", "now,",
    }

    for s_idx, sent in enumerate(sentences):
        if inserted >= desired_count:
            out_sentences.append(sent)
            continue

        s_words = sent.split()
        if len(s_words) < 5:
            out_sentences.append(sent)
            continue

        filler = rng.choices(tokens, weights=weights, k=1)[0]

        # Prefer inserting after a comma or transition word in the middle of the sentence
        placed = False
        for i in range(2, len(s_words) - 2):
            w_lower = s_words[i].lower()
            if s_words[i].endswith(",") or w_lower in transition_words:
                s_words.insert(i + 1, f"{filler},")
                placed = True
                inserted += 1
                break

        # Fallback: insert near a natural midpoint (before a longer content word)
        if not placed and (s_idx % 2 == 0 or filler_level in ("medium", "high")):
            mid = max(2, min(len(s_words) - 2, len(s_words) // 2))
            s_words.insert(mid, f"{filler},")
            inserted += 1

        out_sentences.append(" ".join(s_words))

    return " ".join(out_sentences)


def humanize_script(
    req: HumanizeRequest,
    profiles_root: Path = Path("profiles"),
    llm_model: str | None = None,
) -> Tuple[HumanizeResponse, str]:
    """Run Stage 1 humanization. Prefers local Ollama/Gemma or configured LLM; falls back cleanly."""
    profile_path = profiles_root / req.speaker_id / "style_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Speaker style profile not found: {profile_path}")

    profile = load_style_profile(profile_path)

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model_name = llm_model or os.getenv("HUMANIZER_MODEL", "gemma3:4b")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        sys_prompt = build_llm_system_prompt(profile, req.filler_level, req.tone)
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.4,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": req.clean_script},
            ],
        )
        humanized = (resp.choices[0].message.content or "").strip()
        if humanized:
            mode = f"LLM ({model_name} @ {base_url}) + {profile.display_name} style_profile.json"
            return (
                HumanizeResponse(
                    speaker_id=req.speaker_id,
                    filler_level=req.filler_level,
                    humanized_script=humanized,
                ),
                mode,
            )
    except Exception as exc:
        fallback_reason = f"LLM unavailable ({type(exc).__name__}); using Offline Profile Mode"
    else:
        fallback_reason = "Empty LLM response; using Offline Profile Mode"

    humanized = humanize_offline(req.clean_script, profile, req.filler_level)
    mode = f"{fallback_reason} ({profile.display_name} filler distribution)"
    return (
        HumanizeResponse(
            speaker_id=req.speaker_id,
            filler_level=req.filler_level,
            humanized_script=humanized,
        ),
        mode,
    )
