"""Local Gradio Web UI for Natural Presenter AI.

Tabs:
  1. Presentation Studio: Stage 1 (Ollama/Gemma Humanizer) + Stage 2 (F5-TTS Voice Synthesis)
  2. Dataset QA Reviewer: Listen to sliced clips, fix raw_text/clean_text, mark verified,
     and refresh style_profile.json without touching raw JSON files.

Usage:
    python app/web_ui.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.humanizer import humanize_script
from src.schemas import (
    HumanizeRequest,
    SynthesizeRequest,
    load_metadata,
    load_style_profile,
    save_metadata,
)
from src.synthesizer import PresentationSynthesizer

PROFILES_DIR = ROOT_DIR / "profiles"
OUTPUTS_DIR = ROOT_DIR / "outputs"

SAMPLE_SCRIPT = (
    "Hello everyone, today I want to walk you through our experiment results. "
    "We tested two model architectures across eighty trials, and we noticed a clear drop "
    "in baseline error after tuning the encoder resolution. "
    "Next, let's look at the main trade-off between speed and accuracy."
)

synthesizer = PresentationSynthesizer(profiles_root=PROFILES_DIR, outputs_dir=OUTPUTS_DIR)


def discover_speakers() -> Dict[str, str]:
    """Return mapping of Display Name -> speaker_id for all valid speaker profiles."""
    speakers: Dict[str, str] = {}
    if not PROFILES_DIR.exists():
        return speakers
    for prof_file in sorted(PROFILES_DIR.glob("*/style_profile.json")):
        prof = load_style_profile(prof_file)
        speakers[prof.display_name] = prof.speaker_id
    return speakers


def get_speaker_summary(display_name: str) -> str:
    speakers = discover_speakers()
    spk_id = speakers.get(display_name)
    if not spk_id:
        return "No speaker selected."
    prof = load_style_profile(PROFILES_DIR / spk_id / "style_profile.json")
    fillers = ", ".join(f"`{k}` ({int(v * 100)}%)" for k, v in prof.filler_distribution.items())
    return (
        f"**Speaker ID:** `{prof.speaker_id}` | **Accent:** `{prof.accent}` | "
        f"**Pace:** `{prof.speaking_rate_wpm} WPM` | **Fillers:** {fillers or 'none detected'}"
    )


def on_humanize(
    display_name: str, clean_script: str, filler_level: str, tone: str, llm_model: str
) -> Tuple[str, str]:
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    req = HumanizeRequest(
        speaker_id=spk_id,
        clean_script=clean_script,
        filler_level=filler_level.lower(),
        tone=tone.lower(),
    )
    resp, mode = humanize_script(
        req, profiles_root=PROFILES_DIR, llm_model=llm_model.strip() or None
    )
    return resp.humanized_script, f"Stage 1 complete — {mode}"


def on_synthesize(
    display_name: str, humanized_script: str, tone: str, speed: float
) -> Tuple[str, str]:
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    req = SynthesizeRequest(
        speaker_id=spk_id,
        humanized_script=humanized_script,
        tone=tone.lower(),
        speed=speed,
    )
    out_path, info = synthesizer.synthesize(req)
    return str(out_path), f"Stage 2 complete — {info}"


# --- Dataset QA Reviewer helpers ---


def qa_list_clips(display_name: str) -> List[str]:
    speakers = discover_speakers()
    spk_id = speakers.get(display_name)
    if not spk_id:
        return []
    clips = load_metadata(PROFILES_DIR / spk_id / "metadata.jsonl")
    return [f"{c.clip_id} ({'OK' if c.verified else 'TODO'} | {c.duration_sec}s)" for c in clips]


def _parse_clip_id(label: str) -> str:
    return label.split(" ", 1)[0].strip() if label else ""


def qa_load_clip(display_name: str, clip_label: str):
    speakers = discover_speakers()
    spk_id = speakers.get(display_name)
    clip_id = _parse_clip_id(clip_label)
    if not spk_id or not clip_id:
        return None, "", "", False, "Select a clip to review."

    clips = load_metadata(PROFILES_DIR / spk_id / "metadata.jsonl")
    for c in clips:
        if c.clip_id == clip_id:
            wav_path = PROFILES_DIR / spk_id / c.audio_path
            info = (
                f"Clip `{c.clip_id}` | Duration: `{c.duration_sec}s` | "
                f"Fillers: `{c.fillers_used}` (`{c.filler_level}`)"
            )
            return str(wav_path), c.raw_text, c.clean_text or "", c.verified, info
    return None, "", "", False, f"Clip {clip_id} not found."


def qa_on_speaker_change(display_name: str):
    choices = qa_list_clips(display_name)
    first = choices[0] if choices else None
    audio, raw_t, clean_t, verified, info = qa_load_clip(display_name, first or "")
    return gr.Dropdown(choices=choices, value=first), audio, raw_t, clean_t, verified, info


def qa_save_clip(
    display_name: str, clip_label: str, raw_text: str, clean_text: str, verified: bool
):
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    clip_id = _parse_clip_id(clip_label)
    meta_path = PROFILES_DIR / spk_id / "metadata.jsonl"
    clips = load_metadata(meta_path)

    next_idx = 0
    for idx, c in enumerate(clips):
        if c.clip_id == clip_id:
            c.raw_text = raw_text.strip()
            c.clean_text = clean_text.strip() or None
            c.verified = verified
            next_idx = min(idx + 1, len(clips) - 1)
            break

    save_metadata(clips, meta_path)
    choices = qa_list_clips(display_name)
    next_label = choices[next_idx] if choices else None
    audio, raw_t, clean_t, ver, _ = qa_load_clip(display_name, next_label or "")
    verified_count = sum(1 for c in clips if c.verified)
    status = f"Saved `{clip_id}`! Verified: {verified_count}/{len(clips)} clips."
    return gr.Dropdown(choices=choices, value=next_label), audio, raw_t, clean_t, ver, status


def qa_refresh_profile(display_name: str) -> str:
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "04_extract_style_profile.py"),
        "--speaker",
        spk_id,
        "--display-name",
        display_name,
    ]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        return f"Error refreshing profile:\n{res.stderr}"
    return f"Style profile refreshed!\n{res.stdout.strip()}"


def build_ui() -> gr.Blocks:
    speakers = discover_speakers()
    speaker_names = list(speakers.keys())
    default_speaker = speaker_names[0] if speaker_names else None

    with gr.Blocks(title="Natural Presenter AI") as demo:
        gr.Markdown(
            "# Natural Presenter AI\n"
            "Multi-speaker, accent-preserving English presentation studio powered by **Ollama/Gemma** + **F5-TTS**."
        )

        with gr.Tabs():
            # --- TAB 1: PRESENTATION STUDIO ---
            with gr.Tab("Presentation Studio"):
                with gr.Row():
                    speaker_dd = gr.Dropdown(
                        choices=speaker_names,
                        value=default_speaker,
                        label="Select Speaker Profile",
                    )
                    filler_radio = gr.Radio(
                        choices=["None", "Low", "Medium", "High"],
                        value="Medium",
                        label="Hesitation / Filler Level (uh, um, eh)",
                    )
                    tone_dd = gr.Dropdown(
                        choices=["Presentation", "Explaining", "Emphasis"],
                        value="Presentation",
                        label="Tone",
                    )
                    llm_box = gr.Textbox(
                        value="gemma3:4b",
                        label="Stage 1 Ollama/LLM Model",
                    )

                profile_md = gr.Markdown(
                    get_speaker_summary(default_speaker) if default_speaker else ""
                )
                speaker_dd.change(fn=get_speaker_summary, inputs=speaker_dd, outputs=profile_md)

                clean_box = gr.Textbox(
                    label="1. Clean Presentation Script (Input)",
                    value=SAMPLE_SCRIPT,
                    lines=5,
                    placeholder="Paste your polished English presentation script here...",
                )

                humanize_btn = gr.Button(
                    "Step 1: Humanize Script (Add Speaker's Natural Rhythm)", variant="primary"
                )
                stage1_status = gr.Textbox(label="Stage 1 Status", interactive=False)

                humanized_box = gr.Textbox(
                    label="2. Humanized Spoken Script (Editable before generating voice)",
                    lines=5,
                    placeholder="Click 'Step 1: Humanize Script' or type directly here...",
                )

                with gr.Row():
                    speed_slider = gr.Slider(
                        minimum=0.80,
                        maximum=1.20,
                        value=1.0,
                        step=0.05,
                        label="Speaking Speed",
                    )
                    synth_btn = gr.Button(
                        "Step 2: Generate Presentation Audio (F5-TTS)", variant="primary"
                    )

                stage2_status = gr.Textbox(label="Stage 2 Status", interactive=False)
                audio_out = gr.Audio(label="Generated Presentation Audio", type="filepath")

                humanize_btn.click(
                    fn=on_humanize,
                    inputs=[speaker_dd, clean_box, filler_radio, tone_dd, llm_box],
                    outputs=[humanized_box, stage1_status],
                )
                synth_btn.click(
                    fn=on_synthesize,
                    inputs=[speaker_dd, humanized_box, tone_dd, speed_slider],
                    outputs=[audio_out, stage2_status],
                )

            # --- TAB 2: DATASET QA REVIEWER ---
            with gr.Tab("Dataset QA Reviewer"):
                gr.Markdown(
                    "Listen to each sliced clip, fix any misheard technical terms in **Verbatim `raw_text`** "
                    "(keep all audible `uh`/`um`/`eh`!), and click **Save & Next** before cloud fine-tuning."
                )
                initial_clips = qa_list_clips(default_speaker) if default_speaker else []
                first_clip = initial_clips[0] if initial_clips else None
                init_audio, init_raw, init_clean, init_ver, init_info = (
                    qa_load_clip(default_speaker, first_clip)
                    if (default_speaker and first_clip)
                    else (None, "", "", False, "No clips loaded.")
                )

                with gr.Row():
                    qa_speaker_dd = gr.Dropdown(
                        choices=speaker_names,
                        value=default_speaker,
                        label="Speaker Dataset",
                    )
                    qa_clip_dd = gr.Dropdown(
                        choices=initial_clips,
                        value=first_clip,
                        label="Select Clip",
                    )

                qa_audio = gr.Audio(value=init_audio, label="Clip Audio", type="filepath")
                qa_raw_box = gr.Textbox(
                    value=init_raw,
                    label="Verbatim raw_text (MUST match audio 1:1, keep uh/um/eh!)",
                    lines=3,
                )
                qa_clean_box = gr.Textbox(
                    value=init_clean,
                    label="Polished clean_text (for Stage 1 learning)",
                    lines=3,
                )
                with gr.Row():
                    qa_verified_chk = gr.Checkbox(value=init_ver, label="Verified (1:1 accurate)")
                    qa_save_btn = gr.Button("Save & Next Clip", variant="primary")
                    qa_refresh_btn = gr.Button("Rebuild style_profile.json")

                qa_status = gr.Textbox(value=init_info, label="QA Status", interactive=False)

                qa_speaker_dd.change(
                    fn=qa_on_speaker_change,
                    inputs=qa_speaker_dd,
                    outputs=[
                        qa_clip_dd,
                        qa_audio,
                        qa_raw_box,
                        qa_clean_box,
                        qa_verified_chk,
                        qa_status,
                    ],
                )
                qa_clip_dd.change(
                    fn=qa_load_clip,
                    inputs=[qa_speaker_dd, qa_clip_dd],
                    outputs=[qa_audio, qa_raw_box, qa_clean_box, qa_verified_chk, qa_status],
                )
                qa_save_btn.click(
                    fn=qa_save_clip,
                    inputs=[
                        qa_speaker_dd,
                        qa_clip_dd,
                        qa_raw_box,
                        qa_clean_box,
                        qa_verified_chk,
                    ],
                    outputs=[
                        qa_clip_dd,
                        qa_audio,
                        qa_raw_box,
                        qa_clean_box,
                        qa_verified_chk,
                        qa_status,
                    ],
                )
                qa_refresh_btn.click(
                    fn=qa_refresh_profile,
                    inputs=qa_speaker_dd,
                    outputs=qa_status,
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
