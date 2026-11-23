"""Local Gradio Web UI for Natural Presenter AI (Stage 1 Humanizer + Stage 2 F5-TTS).

Usage:
    python app/web_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.humanizer import humanize_script
from src.schemas import HumanizeRequest, SynthesizeRequest, load_style_profile
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


def on_humanize(display_name: str, clean_script: str, filler_level: str, tone: str) -> Tuple[str, str]:
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    req = HumanizeRequest(
        speaker_id=spk_id,
        clean_script=clean_script,
        filler_level=filler_level.lower(),
        tone=tone.lower(),
    )
    resp, mode = humanize_script(req, profiles_root=PROFILES_DIR)
    return resp.humanized_script, f"Stage 1 complete — {mode}"


def on_synthesize(display_name: str, humanized_script: str, tone: str, speed: float) -> Tuple[str, str]:
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


def build_ui() -> gr.Blocks:
    speakers = discover_speakers()
    speaker_names = list(speakers.keys())
    default_speaker = speaker_names[0] if speaker_names else None

    with gr.Blocks(title="Natural Presenter AI") as demo:
        gr.Markdown(
            "# Natural Presenter AI\n"
            "**Stage 1** adapts your polished script with *each speaker's* natural `uh/um/eh` rhythm. "
            "**Stage 2** speaks it in their authentic voice & accent on your local RTX 5060 GPU."
        )

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

        profile_md = gr.Markdown(get_speaker_summary(default_speaker) if default_speaker else "")
        speaker_dd.change(fn=get_speaker_summary, inputs=speaker_dd, outputs=profile_md)

        clean_box = gr.Textbox(
            label="1. Clean Presentation Script (Input)",
            value=SAMPLE_SCRIPT,
            lines=5,
            placeholder="Paste your polished English presentation script here...",
        )

        humanize_btn = gr.Button("Step 1: Humanize Script (Add Speaker's Natural Rhythm)", variant="primary")
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
            synth_btn = gr.Button("Step 2: Generate Presentation Audio (F5-TTS)", variant="primary")

        stage2_status = gr.Textbox(label="Stage 2 Status", interactive=False)
        audio_out = gr.Audio(label="Generated Presentation Audio", type="filepath")

        humanize_btn.click(
            fn=on_humanize,
            inputs=[speaker_dd, clean_box, filler_radio, tone_dd],
            outputs=[humanized_box, stage1_status],
        )
        synth_btn.click(
            fn=on_synthesize,
            inputs=[speaker_dd, humanized_box, tone_dd, speed_slider],
            outputs=[audio_out, stage2_status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
