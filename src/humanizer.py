"""Stage 1: Personalized Long-Form Script Humanizer & Pronunciation Normalizer.

Features:
  1. Editable acronym/technical pronunciation lexicon (`config/pronunciation_map.json`)
     so terms like MSE -> "M S E", CNN -> "C N N", ReLU -> "ray-loo" are spoken cleanly.
  2. Paragraph-aware long-form disfluency planning:
     - Uses each speaker's individualized `style_profile.json` filler vocabulary & proportions.
     - Varies disfluency types across long presentations (fillers, light word repetitions,
       and optional subtle self-repairs like "the accuracy— sorry, the validation error")
       without ever overdoing it or repeating the same filler mechanically.
  3. Defaults to local Ollama/Gemma (`gemma3:4b`) with graceful offline fallback.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from src.schemas import FillerLevel, HumanizeRequest, HumanizeResponse, StyleProfile, load_style_profile

load_dotenv()

DEFAULT_PRONUNCIATION_PATH = Path(__file__).resolve().parent.parent / "config" / "pronunciation_map.json"


def load_pronunciation_map(path: Path = DEFAULT_PRONUNCIATION_PATH) -> Dict[str, str]:
    """Load editable pronunciation dictionary (ignoring keys starting with '_')."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, str)}


def save_pronunciation_map(mapping: Dict[str, str], path: Path = DEFAULT_PRONUNCIATION_PATH) -> None:
    """Save pronunciation dictionary while preserving header comment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Editable pronunciation dictionary for F5-TTS. Maps technical acronyms/terms to "
            "spoken letter/phonetic forms so F5-TTS pronounces them cleanly (e.g., MSE -> M S E)."
        ),
        **mapping,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def apply_pronunciation_map(text: str, mapping: Dict[str, str] | None = None) -> str:
    """Replace technical acronyms/terms with TTS-friendly spoken forms (whole-word match)."""
    lexicon = mapping if mapping is not None else load_pronunciation_map()
    if not lexicon:
        return text

    # Sort longest keys first so "RMSE" matches before "MSE", "LLMs" before "LLM", etc.
    out = text
    for term in sorted(lexicon.keys(), key=len, reverse=True):
        spoken = lexicon[term]
        # Match standalone token (allow punctuation around it, including Q&A)
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])")
        out = pattern.sub(spoken, out)
    return out


def build_llm_system_prompt(
    profile: StyleProfile,
    filler_level: FillerLevel,
    tone: str,
    allow_self_repair_here: bool = False,
    paragraph_index: int = 1,
    total_paragraphs: int = 1,
) -> str:
    """Build a speaker-specific system prompt for a single paragraph of a presentation."""
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

    if filler_level == "none":
        disfluency_rules = (
            "3. Hesitation intensity is NONE: insert ZERO fillers (no uh/um/like/eh), "
            "ZERO repetitions, and ZERO self-repairs. Only normalize numbers/symbols/acronyms "
            "and optimize punctuation for smooth spoken delivery."
        )
    else:
        repair_instruction = (
            "   - Subtle self-repair ALLOWED in this paragraph (at most ONCE): you may include ONE "
            "very natural human self-correction or false start, e.g., restarting a clause "
            '("we tested— uh, we actually ran..."), refining a word ("the accuracy— sorry, the '
            'validation error..."), or lightly repeating a connector ("and, and...", "in the, the..."). '
            "Keep it completely natural and never alter the final factual meaning!"
            if allow_self_repair_here
            else "   - Do NOT add a self-repair in this paragraph (keep it clean with natural fillers/pauses only)."
        )
        disfluency_rules = (
            f'3. Hesitation intensity: "{filler_level}" (target roughly {target_per_100} fillers per 100 words).\n'
            f"   - ONLY use this speaker's personal filler vocabulary in roughly these proportions: {fillers_str}.\n"
            f"   - VARY your phrasing — do NOT repeat the exact same filler in back-to-back clauses or mechanically every sentence.\n"
            f"   - Place hesitations ONLY at natural cognitive boundaries (topic transitions, after introductory clauses, before key technical metrics/terms).\n"
            f"{repair_instruction}"
        )

    return f"""You are a Personalized Presentation Script Humanizer for speaker "{profile.display_name}" (accent: {profile.accent}).
You are adapting paragraph {paragraph_index} of {total_paragraphs} from a clean written presentation script into natural spoken English matching THIS EXACT SPEAKER's personal speaking rhythm.

Rules (STRICT):
1. Preserve 100% of ALL numbers, digits, dimensions, metrics, and facts EXACTLY (NEVER change "32 by 32" to "8 by 8" or misread "1,024" — keep numbers like 32, 255, 1,024, 0.0056 as exact digits so you NEVER make arithmetic mistakes).
2. Make symbols and acronyms TTS-ready:
   - Spell out letter-by-letter acronyms with spaces (e.g., "MSE" -> "M S E", "CNN" -> "C N N", "GPU" -> "G P U", "RGB" -> "R G B", "API" -> "A P I").
   - Expand symbols ("%" -> "percent", "3x3" -> "3 by 3", "Q3" -> "Q three", "$" -> "dollars"). Do NOT rewrite multi-digit numbers into wrong words.
{disfluency_rules}
4. Tone: {tone}. Speaker pace: {profile.speaking_rate_wpm} WPM (~{profile.avg_words_per_pause} words per natural pause).
5. Return ONLY the rewritten spoken paragraph text — no commentary, no labels, no markdown quotes.

Real examples of how this speaker actually talks:
{examples_block}
"""


def _plan_self_repair_paragraphs(
    paragraphs: List[str], filler_level: FillerLevel, include_self_repairs: bool
) -> set[int]:
    """Choose at most ~1 paragraph every ~200 words for a subtle human self-repair."""
    if not include_self_repairs or filler_level == "none":
        return set()

    # Budget: low -> at most 1 in long talks; medium -> ~1 per 200 words; high -> ~1 per 120 words
    words_per_repair = {"low": 280, "medium": 180, "high": 110}[filler_level]
    total_words = sum(len(p.split()) for p in paragraphs)
    if total_words < 25:
        return set()

    budget = max(1, round(total_words / words_per_repair))
    if filler_level == "low" and total_words < 90:
        budget = 0
    budget = min(budget, len(paragraphs))
    if budget <= 0:
        return set()

    # Spread repairs evenly across the presentation (avoiding the very first opening sentence if multi-para)
    candidate_indices = list(range(len(paragraphs)))
    if len(paragraphs) > 2:
        candidate_indices = candidate_indices[1:]
    rng = random.Random(total_words + len(paragraphs))
    chosen = set(rng.sample(candidate_indices, k=min(budget, len(candidate_indices))))
    return chosen


def humanize_offline_paragraph(
    paragraph: str,
    profile: StyleProfile,
    filler_level: FillerLevel,
    allow_self_repair: bool = False,
    seed: int = 42,
) -> str:
    """Deterministic offline humanizer driven by speaker's style_profile.json."""
    text = re.sub(r"\s+", " ", paragraph.strip())
    if filler_level == "none":
        return text

    target_per_100 = profile.filler_targets_per_100_words.get(filler_level, 4)
    words = text.split()
    if not words or target_per_100 <= 0:
        return text

    desired_count = max(1, round((len(words) / 100.0) * target_per_100))
    dist = profile.filler_distribution or {"uh": 0.7, "um": 0.3}
    tokens = list(dist.keys())
    weights = list(dist.values())
    rng = random.Random(seed + len(words))

    sentences = re.split(r"(?<=[.!?])\s+", text)
    inserted = 0
    did_repair = False
    out_sentences: List[str] = []
    transition_words = {
        "and", "but", "so", "however,", "therefore,", "basically,", "first,", "second,",
        "finally,", "because", "when", "if", "then", "also", "now,",
    }

    for s_idx, sent in enumerate(sentences):
        s_words = sent.split()
        if len(s_words) < 5:
            out_sentences.append(sent)
            continue

        # Optional subtle connector repetition (e.g., "and, and" or "the, the")
        if allow_self_repair and not did_repair and s_idx == len(sentences) // 2:
            for i in range(1, len(s_words) - 2):
                if s_words[i].lower() in {"the", "we", "and", "our"}:
                    s_words.insert(i, f"{s_words[i]},")
                    did_repair = True
                    break

        if inserted < desired_count:
            filler = rng.choices(tokens, weights=weights, k=1)[0]
            placed = False
            for i in range(2, len(s_words) - 2):
                if s_words[i].endswith(",") or s_words[i].lower() in transition_words:
                    s_words.insert(i + 1, f"{filler},")
                    placed = True
                    inserted += 1
                    break
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
    pronunciation_path: Path = DEFAULT_PRONUNCIATION_PATH,
) -> Tuple[HumanizeResponse, str]:
    """Run Stage 1 paragraph-by-paragraph humanization + acronym pronunciation normalization."""
    profile_path = profiles_root / req.speaker_id / "style_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Speaker style profile not found: {profile_path}")

    profile = load_style_profile(profile_path)
    lexicon = load_pronunciation_map(pronunciation_path)

    # Split into paragraphs separated by blank lines so long presentations preserve slide/section structure
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", req.clean_script.strip()) if p.strip()]
    if not paragraphs:
        paragraphs = [req.clean_script.strip()]

    repair_indices = _plan_self_repair_paragraphs(
        paragraphs, req.filler_level, req.include_self_repairs
    )

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model_name = llm_model or os.getenv("HUMANIZER_MODEL", "gemma3:4b")

    out_paragraphs: List[str] = []
    used_llm = False
    fallback_reason = ""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0)
        for idx, para in enumerate(paragraphs):
            # Pre-normalize known acronyms so the LLM keeps them clean and doesn't mangle them
            pre_para = apply_pronunciation_map(para, lexicon)
            sys_prompt = build_llm_system_prompt(
                profile=profile,
                filler_level=req.filler_level,
                tone=req.tone,
                allow_self_repair_here=(idx in repair_indices),
                paragraph_index=idx + 1,
                total_paragraphs=len(paragraphs),
            )
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.45,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": pre_para},
                ],
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            if not rewritten:
                raise RuntimeError("Empty LLM response")
            # Safety pass on acronyms in case LLM reverted "M S E" back to "MSE"
            out_paragraphs.append(apply_pronunciation_map(rewritten, lexicon))
        used_llm = True
    except Exception as exc:
        fallback_reason = f"LLM unavailable ({type(exc).__name__}); using Offline Profile Mode"
        out_paragraphs = []
        for idx, para in enumerate(paragraphs):
            pre_para = apply_pronunciation_map(para, lexicon)
            out_paragraphs.append(
                humanize_offline_paragraph(
                    pre_para,
                    profile,
                    req.filler_level,
                    allow_self_repair=(idx in repair_indices),
                    seed=42 + idx * 17,
                )
            )

    final_script = "\n\n".join(out_paragraphs)
    repair_note = f" | self-repairs planned in {len(repair_indices)}/{len(paragraphs)} para(s)" if req.include_self_repairs and req.filler_level != "none" else ""
    if used_llm:
        mode = f"LLM ({model_name}) + {profile.display_name} ({len(paragraphs)} para(s){repair_note})"
    else:
        mode = f"{fallback_reason} ({profile.display_name}, {len(paragraphs)} para(s){repair_note})"

    return (
        HumanizeResponse(
            speaker_id=req.speaker_id,
            filler_level=req.filler_level,
            humanized_script=final_script,
        ),
        mode,
    )
