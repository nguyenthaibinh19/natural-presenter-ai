"""Pydantic schemas for Natural Presenter AI dataset, speaker profiles, and pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

FillerLevel = Literal["none", "low", "medium", "high"]
RecordingType = Literal["spontaneous", "scripted"]
ToneType = Literal["presentation", "explaining", "emphasis"]
PauseTag = Literal["[pause_short]", "[pause_long]", "[breath]"]


class PauseEvent(BaseModel):
    """Represents a measured internal pause inside an audio clip."""

    after_word: str = Field(..., description="Word immediately preceding the pause.")
    duration_ms: int = Field(..., ge=0, description="Duration of the pause in milliseconds.")
    tag: PauseTag = Field(..., description="Categorized pause marker.")


class DatasetClip(BaseModel):
    """Single labeled 3-12s clip entry stored in profiles/<speaker_id>/metadata.jsonl."""

    clip_id: str = Field(..., description="Unique clip identifier, e.g., vn_01_0042.")
    audio_path: str = Field(..., description="Relative path to WAV inside speaker profile.")
    duration_sec: float = Field(..., gt=0, description="Audio duration in seconds.")
    speaker_id: str = Field(..., description="Speaker profile ID, e.g., vn_01.")
    accent: str = Field(..., description="Accent tag, e.g., vi-EN, es-EN, hu-EN, cy-EN.")
    recording_type: RecordingType = Field(default="spontaneous")
    raw_text: str = Field(
        ...,
        description="Exact verbatim transcript matching audio 1:1, including uh/um/eh fillers.",
    )
    clean_text: Optional[str] = Field(
        default=None,
        description="Polished presentation version of raw_text (used for Stage 1 learning).",
    )
    fillers_used: List[str] = Field(default_factory=list)
    filler_count: int = Field(default=0, ge=0)
    filler_level: FillerLevel = Field(default="none")
    pause_events: List[PauseEvent] = Field(default_factory=list)
    tone: ToneType = Field(default="presentation")
    verified: bool = Field(
        default=False,
        description="True once a human has verified raw_text matches the audio.",
    )


class FewShotExample(BaseModel):
    """Real clean -> natural spoken example extracted from a speaker's own dataset."""

    filler_level: FillerLevel
    clean_text: str
    raw_text: str


class StyleProfile(BaseModel):
    """Individualized speaking & disfluency fingerprint stored in style_profile.json."""

    speaker_id: str
    display_name: str
    accent: str
    speaking_rate_wpm: float = Field(default=135.0, gt=0)
    avg_words_per_pause: float = Field(default=10.0, gt=0)
    filler_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized relative frequency of each filler token for this speaker.",
    )
    common_filler_positions: List[str] = Field(
        default_factory=lambda: [
            "sentence_start_transition",
            "before_technical_noun_or_metric",
            "after_introductory_clause",
        ]
    )
    punctuation_conventions: Dict[str, str] = Field(
        default_factory=lambda: {
            "short_pause": ",",
            "hesitation_pause": "...",
            "clause_break": " — ",
        }
    )
    filler_targets_per_100_words: Dict[FillerLevel, int] = Field(
        default_factory=lambda: {
            "none": 0,
            "low": 2,
            "medium": 5,
            "high": 9,
        }
    )
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)


class HumanizeRequest(BaseModel):
    """Input payload for Stage 1 (Personalized Script Humanizer)."""

    speaker_id: str
    clean_script: str = Field(..., min_length=1)
    filler_level: FillerLevel = Field(default="medium")
    tone: ToneType = Field(default="presentation")
    include_self_repairs: bool = Field(
        default=True,
        description="Allow subtle natural self-repairs / light repetitions in longer scripts.",
    )


class HumanizeResponse(BaseModel):
    """Output payload from Stage 1."""

    speaker_id: str
    filler_level: FillerLevel
    humanized_script: str


class SynthesizeRequest(BaseModel):
    """Input payload for Stage 2 (Accent-Preserving TTS)."""

    speaker_id: str
    humanized_script: str = Field(..., min_length=1)
    tone: ToneType = Field(default="presentation")
    speed: float = Field(default=1.0, ge=0.75, le=1.25)
    nfe_step: int = Field(default=16, ge=8, le=64)
    paragraph_pause_ms: int = Field(
        default=650,
        ge=150,
        le=2000,
        description="Pause duration in milliseconds inserted between paragraphs / slide breaks.",
    )


def load_metadata(path: Path) -> List[DatasetClip]:
    """Load all DatasetClip records from a metadata.jsonl file."""
    if not path.exists():
        return []
    clips: List[DatasetClip] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                clips.append(DatasetClip.model_validate_json(line))
    return clips


def save_metadata(clips: List[DatasetClip], path: Path) -> None:
    """Write DatasetClip records to a metadata.jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for clip in clips:
            f.write(clip.model_dump_json() + "\n")


def load_style_profile(path: Path) -> StyleProfile:
    """Load a speaker's style_profile.json."""
    return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_style_profile(profile: StyleProfile, path: Path) -> None:
    """Save a speaker's style_profile.json with readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
