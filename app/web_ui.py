"""Local Gradio Web UI for Natural Presenter AI.

Tabs:
  1. Presentation Studio: Stage 1 (Ollama/Gemma Long-Form Humanizer + self-repairs + acronyms)
     + Stage 2 (F5-TTS Voice Synthesis with inter-paragraph pauses)
  2. Dataset QA Reviewer: Filter by TODO/OK/All, live verified progress, one-click Verify & Next
  3. Pronunciation Dictionary: Edit/add acronyms & technical terms (e.g., MSE -> M S E)

Usage:
    python app/web_ui.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.cue_presenter import (
    LiveCueController,
    build_presentation_deck,
    humanize_cue_deck,
    load_cue_deck,
    prerender_cue_deck,
    save_cue_deck,
)
from src.humanizer import (
    DEFAULT_PRONUNCIATION_PATH,
    humanize_script,
    load_pronunciation_map,
    save_pronunciation_map,
)
from src.schemas import (
    DatasetClip,
    HumanizeRequest,
    SynthesizeRequest,
    load_metadata,
    load_style_profile,
    save_metadata,
)
from src.synthesizer import PresentationSynthesizer

PROFILES_DIR = ROOT_DIR / "profiles"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PRESENTATIONS_DIR = ROOT_DIR / "presentations"

SAMPLE_SCRIPT = (
    "Hello everyone, today I want to walk you through our experiment results. "
    "We trained a CNN autoencoder across eighty trials on a single GPU, and we measured reconstruction quality using MSE.\n\n"
    "On the next slide, you can see that our MSE dropped by 24% after switching to ReLU activations and tuning the encoder resolution. "
    "However, latency increased slightly, so let's discuss the trade-off between speed and accuracy."
)

QA_FILTERS = ["Only Unverified (TODO)", "All Clips", "Only Verified (OK)"]

synthesizer = PresentationSynthesizer(profiles_root=PROFILES_DIR, outputs_dir=OUTPUTS_DIR)
live_controller = LiveCueController()


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
    display_name: str,
    clean_script: str,
    filler_level: str,
    tone: str,
    llm_model: str,
    include_self_repairs: bool,
) -> Tuple[str, str]:
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    req = HumanizeRequest(
        speaker_id=spk_id,
        clean_script=clean_script,
        filler_level=filler_level.lower(),
        tone=tone.lower(),
        include_self_repairs=include_self_repairs,
    )
    resp, mode = humanize_script(
        req, profiles_root=PROFILES_DIR, llm_model=llm_model.strip() or None
    )
    return resp.humanized_script, f"Stage 1 complete — {mode}"


def list_speaker_refs(display_name: str | None) -> List[str]:
    if not display_name:
        return []
    spk_id = discover_speakers().get(display_name)
    if not spk_id:
        return []
    return synthesizer.list_references(spk_id)


def default_ref_for_speaker(display_name: str | None) -> str | None:
    refs = list_speaker_refs(display_name)
    if not refs:
        return None
    if "partA_natural_slow_ref" in refs:
        return "partA_natural_slow_ref"
    return refs[0]


def get_ref_preview(display_name: str | None, ref_name: str | None):
    if not display_name or not ref_name:
        return None, "No reference clip selected."
    spk_id = discover_speakers().get(display_name)
    if not spk_id:
        return None, "Speaker not found."
    refs_dir = PROFILES_DIR / spk_id / "refs"
    wav_p = refs_dir / f"{ref_name}.wav"
    txt_p = refs_dir / f"{ref_name}.txt"
    if not wav_p.exists():
        return None, f"Missing {wav_p.name}"
    txt = txt_p.read_text(encoding="utf-8").strip() if txt_p.exists() else ""
    return str(wav_p), f"**Reference transcript (`{ref_name}`):** _{txt}_"


def on_studio_speaker_change(display_name: str):
    summary = get_speaker_summary(display_name)
    refs = list_speaker_refs(display_name)
    chosen = default_ref_for_speaker(display_name)
    ref_wav, ref_md = get_ref_preview(display_name, chosen)
    return summary, gr.Dropdown(choices=refs, value=chosen), ref_wav, ref_md


def on_synthesize(
    display_name: str,
    humanized_script: str,
    tone: str,
    ref_name: str | None,
    speed: float,
    quality_mode: str,
    paragraph_pause_ms: int,
) -> Tuple[str, str]:
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    nfe_step = 16 if "Fast" in quality_mode else 32
    req = SynthesizeRequest(
        speaker_id=spk_id,
        humanized_script=humanized_script,
        tone=tone.lower(),
        ref_name=ref_name,
        speed=speed,
        nfe_step=nfe_step,
        paragraph_pause_ms=int(paragraph_pause_ms),
    )
    out_path, info = synthesizer.synthesize(req)
    return str(out_path), f"Stage 2 complete — {info}"


# --- Dataset QA Reviewer helpers ---


def _format_clip_label(clip: DatasetClip) -> str:
    badge = "✅ OK" if clip.verified else "⬜ TODO"
    return f"{clip.clip_id} | {badge} | {clip.duration_sec:.1f}s"


def _parse_clip_id(label: str | None) -> str:
    if not label:
        return ""
    return label.split("|", 1)[0].strip()


def qa_progress_markdown(display_name: str) -> str:
    speakers = discover_speakers()
    spk_id = speakers.get(display_name)
    if not spk_id:
        return "No speaker selected."
    clips = load_metadata(PROFILES_DIR / spk_id / "metadata.jsonl")
    if not clips:
        return f"**{display_name}:** No clips found."

    verified_clips = [c for c in clips if c.verified]
    todo_clips = [c for c in clips if not c.verified]
    total_min = sum(c.duration_sec for c in clips) / 60.0
    ver_min = sum(c.duration_sec for c in verified_clips) / 60.0
    pct = (len(verified_clips) / len(clips)) * 100.0
    next_todo = todo_clips[0].clip_id if todo_clips else "None (All Done!)"

    return (
        f"### QA Progress — {display_name}\n"
        f"- **Verified (`✅ OK`):** `{len(verified_clips)} / {len(clips)}` clips (**{pct:.1f}%**) "
        f"| **Verified Audio:** `{ver_min:.2f} / {total_min:.2f} min`\n"
        f"- **Remaining (`⬜ TODO`):** `{len(todo_clips)}` clips | **Next TODO Clip:** `{next_todo}`"
    )


def qa_list_clips(display_name: str, filter_mode: str = "Only Unverified (TODO)") -> List[str]:
    speakers = discover_speakers()
    spk_id = speakers.get(display_name)
    if not spk_id:
        return []
    clips = load_metadata(PROFILES_DIR / spk_id / "metadata.jsonl")
    if filter_mode == "Only Unverified (TODO)":
        clips = [c for c in clips if not c.verified]
    elif filter_mode == "Only Verified (OK)":
        clips = [c for c in clips if c.verified]
    return [_format_clip_label(c) for c in clips]


def qa_load_clip(display_name: str, clip_label: str | None):
    speakers = discover_speakers()
    spk_id = speakers.get(display_name)
    clip_id = _parse_clip_id(clip_label)
    if not spk_id or not clip_id:
        return None, "", "", False, "No clips match the current filter (all done or empty)."

    clips = load_metadata(PROFILES_DIR / spk_id / "metadata.jsonl")
    for c in clips:
        if c.clip_id == clip_id:
            wav_path = PROFILES_DIR / spk_id / c.audio_path
            badge = "VERIFIED (OK)" if c.verified else "UNVERIFIED (TODO)"
            info = (
                f"[{badge}] Clip `{c.clip_id}` | Duration: `{c.duration_sec}s` | "
                f"Fillers: `{c.fillers_used}` (`{c.filler_level}`)"
            )
            return str(wav_path), c.raw_text, c.clean_text or "", c.verified, info
    return None, "", "", False, f"Clip {clip_id} not found."


def qa_on_speaker_or_filter_change(display_name: str, filter_mode: str):
    progress = qa_progress_markdown(display_name)
    choices = qa_list_clips(display_name, filter_mode)
    first = choices[0] if choices else None
    audio, raw_t, clean_t, verified, info = qa_load_clip(display_name, first)
    return progress, gr.Dropdown(choices=choices, value=first), audio, raw_t, clean_t, verified, info


def _save_clip_core(
    display_name: str,
    filter_mode: str,
    clip_label: str | None,
    raw_text: str,
    clean_text: str,
    verified: bool,
    action_label: str,
):
    speakers = discover_speakers()
    spk_id = speakers[display_name]
    clip_id = _parse_clip_id(clip_label)
    meta_path = PROFILES_DIR / spk_id / "metadata.jsonl"
    clips = load_metadata(meta_path)

    old_choices = qa_list_clips(display_name, filter_mode)
    old_ids = [_parse_clip_id(lbl) for lbl in old_choices]
    old_pos = old_ids.index(clip_id) if clip_id in old_ids else 0

    for c in clips:
        if c.clip_id == clip_id:
            c.raw_text = raw_text.strip()
            c.clean_text = clean_text.strip() or None
            c.verified = verified
            break

    save_metadata(clips, meta_path)

    progress = qa_progress_markdown(display_name)
    new_choices = qa_list_clips(display_name, filter_mode)

    if not new_choices:
        next_label = None
    elif filter_mode == "Only Unverified (TODO)" and verified:
        next_idx = min(old_pos, len(new_choices) - 1)
        next_label = new_choices[next_idx]
    else:
        next_idx = min(old_pos + 1, len(new_choices) - 1)
        next_label = new_choices[next_idx]

    audio, raw_t, clean_t, ver, _ = qa_load_clip(display_name, next_label)
    verified_count = sum(1 for c in clips if c.verified)
    status = f"{action_label} `{clip_id}` | Total Verified: {verified_count}/{len(clips)}"
    return progress, gr.Dropdown(choices=new_choices, value=next_label), audio, raw_t, clean_t, ver, status


def qa_verify_and_next(
    display_name: str, filter_mode: str, clip_label: str | None, raw_text: str, clean_text: str
):
    return _save_clip_core(
        display_name, filter_mode, clip_label, raw_text, clean_text, True, "✅ Verified & Saved"
    )


def qa_save_current(
    display_name: str,
    filter_mode: str,
    clip_label: str | None,
    raw_text: str,
    clean_text: str,
    verified: bool,
):
    return _save_clip_core(
        display_name, filter_mode, clip_label, raw_text, clean_text, verified, "Saved"
    )


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


def qa_save_as_reference(display_name: str, clip_label: str | None, custom_ref_name: str) -> str:
    spk_id = discover_speakers().get(display_name)
    clip_id = _parse_clip_id(clip_label)
    if not spk_id or not clip_id:
        return "Select a clip first."
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in custom_ref_name.strip())
    if not safe_name:
        safe_name = f"{clip_id}_ref"
    if not safe_name.endswith("_ref"):
        safe_name = f"{safe_name}_ref"

    spk_dir = PROFILES_DIR / spk_id
    clips = load_metadata(spk_dir / "metadata.jsonl")
    for c in clips:
        if c.clip_id == clip_id:
            refs_dir = spk_dir / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spk_dir / c.audio_path, refs_dir / f"{safe_name}.wav")
            (refs_dir / f"{safe_name}.txt").write_text(c.raw_text.strip(), encoding="utf-8")
            return f"Saved `{clip_id}` as Reference Style `{safe_name}`! Switch to Presentation Studio (re-select speaker to refresh list)."
    return f"Clip `{clip_id}` not found."


# --- Pronunciation Dictionary helpers ---


def pron_load_json() -> str:
    return json.dumps(load_pronunciation_map(DEFAULT_PRONUNCIATION_PATH), indent=2, ensure_ascii=False)


def pron_add_term(term: str, spoken: str) -> Tuple[str, str]:
    term = term.strip()
    spoken = spoken.strip()
    if not term or not spoken:
        return pron_load_json(), "Please enter both a written term (e.g., MSE) and spoken form (e.g., M S E)."
    mapping = load_pronunciation_map(DEFAULT_PRONUNCIATION_PATH)
    mapping[term] = spoken
    save_pronunciation_map(mapping, DEFAULT_PRONUNCIATION_PATH)
    return pron_load_json(), f"Saved rule: `{term}` → `{spoken}` (takes effect immediately!)"


def pron_save_raw_json(raw_json: str) -> str:
    try:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
        ):
            return "Error: Must be a JSON object mapping string -> string."
        save_pronunciation_map(parsed, DEFAULT_PRONUNCIATION_PATH)
        return f"Saved {len(parsed)} pronunciation rules to config/pronunciation_map.json!"
    except Exception as exc:
        return f"Invalid JSON: {exc}"


# --- Live Cue Presenter helpers ---


def cue_list_decks() -> List[str]:
    if not PRESENTATIONS_DIR.exists():
        return []
    return sorted(p.parent.name for p in PRESENTATIONS_DIR.glob("*/deck.json"))


def _deck_path(project_name: str) -> Path:
    return PRESENTATIONS_DIR / project_name / "deck.json"


def _cue_choices_for_deck(project_name: str | None) -> List[str]:
    if not project_name or not _deck_path(project_name).exists():
        return []
    deck = load_cue_deck(_deck_path(project_name))
    out = []
    for idx, c in enumerate(deck.cues, 1):
        badge = "🔊" if (c.audio_path and Path(c.audio_path).exists()) else "⬜"
        out.append(f"{idx:02d}. {c.cue_id} | {badge} | {c.speaker_id} | {c.source}")
    return out


def _parse_cue_id_from_label(label: str | None) -> str:
    if not label:
        return ""
    parts = label.split("|")
    if not parts:
        return ""
    first = parts[0].strip()
    return first.split(" ", 1)[1].strip() if " " in first else first


def cue_load_deck_action(project_name: str | None):
    if not project_name or not _deck_path(project_name).exists():
        return (
            gr.Dropdown(choices=[], value=None),
            live_controller.render_teleprompter_markdown(),
            "No deck found.",
        )
    deck = load_cue_deck(_deck_path(project_name))
    live_controller.load_deck(deck)
    choices = _cue_choices_for_deck(project_name)
    return (
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        live_controller.render_teleprompter_markdown(),
        f"Loaded `{project_name}` ({len(deck.cues)} cues).",
    )


def cue_import_files(project_name: str, pdf_file, ipynb_file, merge_fragments: bool):
    name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in (project_name or "").strip())
    if not name:
        name = "my_presentation"
    pdf_p = Path(pdf_file) if pdf_file else None
    nb_p = Path(ipynb_file) if ipynb_file else None
    if not pdf_p and not nb_p:
        return gr.Dropdown(choices=cue_list_decks()), gr.Dropdown(), live_controller.render_teleprompter_markdown(), "Upload at least a Reading Copy PDF or Presenter .ipynb file."
    deck, _ = build_presentation_deck(
        project_name=name,
        pdf_path=pdf_p,
        ipynb_path=nb_p,
        presentations_root=PRESENTATIONS_DIR,
        merge_fragments=merge_fragments,
    )
    live_controller.load_deck(deck)
    decks = cue_list_decks()
    choices = _cue_choices_for_deck(name)
    return (
        gr.Dropdown(choices=decks, value=name),
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        live_controller.render_teleprompter_markdown(),
        f"Imported `{name}` with {len(deck.cues)} cues!",
    )


def cue_select_for_edit(project_name: str | None, cue_label: str | None):
    if not project_name or not cue_label or not _deck_path(project_name).exists():
        return "vn_01", "", "", "", None
    cid = _parse_cue_id_from_label(cue_label)
    deck = load_cue_deck(_deck_path(project_name))
    for c in deck.cues:
        if c.cue_id == cid:
            wav = c.audio_path if (c.audio_path and Path(c.audio_path).exists()) else None
            return c.speaker_id, c.point_to, c.clean_text, c.humanized_text or "", wav
    return "vn_01", "", "", "", None


def cue_save_and_rerender_one(
    project_name: str,
    cue_label: str,
    speaker_id: str,
    point_to: str,
    clean_text: str,
    humanized_text: str,
    speed: float,
    quality_mode: str,
):
    dp = _deck_path(project_name)
    deck = load_cue_deck(dp)
    cid = _parse_cue_id_from_label(cue_label)
    nfe_step = 16 if "Fast" in quality_mode else 32
    audio_out = None

    for c in deck.cues:
        if c.cue_id == cid:
            c.speaker_id = speaker_id.strip()
            c.point_to = point_to.strip()
            c.clean_text = clean_text.strip()
            c.humanized_text = humanized_text.strip() or None
            # Re-render this single cue immediately
            target_wav = dp.parent / "audio" / f"{c.cue_id}.wav"
            target_wav.parent.mkdir(parents=True, exist_ok=True)
            req = SynthesizeRequest(
                speaker_id=c.speaker_id,
                humanized_script=c.humanized_text or c.clean_text,
                speed=speed,
                nfe_step=nfe_step,
            )
            tmp_wav, _ = synthesizer.synthesize(req)
            import soundfile as sf

            data, sr = sf.read(str(tmp_wav))
            sf.write(str(target_wav), data, sr, subtype="PCM_16")
            c.audio_path = str(target_wav)
            c.duration_sec = round(len(data) / float(sr), 2)
            audio_out = str(target_wav)
            break

    save_cue_deck(deck, dp)
    live_controller.load_deck(deck)
    choices = _cue_choices_for_deck(project_name)
    return (
        gr.Dropdown(choices=choices),
        audio_out,
        live_controller.render_teleprompter_markdown(),
        f"Saved & re-rendered cue `{cid}`!",
    )


def cue_humanize_all_action(
    project_name: str, filler_level: str, include_self_repairs: bool, llm_model: str
):
    dp = _deck_path(project_name)
    deck = load_cue_deck(dp)
    deck = humanize_cue_deck(
        deck,
        dp,
        filler_level=filler_level.lower(),
        include_self_repairs=include_self_repairs,
        profiles_root=PROFILES_DIR,
        llm_model=llm_model.strip() or None,
    )
    live_controller.load_deck(deck)
    return (
        live_controller.render_teleprompter_markdown(),
        f"Stage 1 humanized all {len(deck.cues)} cues in `{project_name}`!",
    )


def cue_prerender_all_action(
    project_name: str, speed: float, quality_mode: str, only_missing: bool
):
    dp = _deck_path(project_name)
    deck = load_cue_deck(dp)
    nfe_step = 16 if "Fast" in quality_mode else 32
    deck = prerender_cue_deck(
        deck,
        dp,
        synthesizer=synthesizer,
        speed=speed,
        nfe_step=nfe_step,
        only_missing=only_missing,
    )
    live_controller.load_deck(deck)
    total_sec = sum(c.duration_sec for c in deck.cues)
    choices = _cue_choices_for_deck(project_name)
    return (
        gr.Dropdown(choices=choices),
        live_controller.render_teleprompter_markdown(),
        f"Pre-rendered all {len(deck.cues)} cues ({total_sec / 60:.1f} min total audio) — ready for 0ms live OBS recording!",
    )


def build_ui() -> gr.Blocks:
    speakers = discover_speakers()
    speaker_names = list(speakers.keys())
    default_speaker = speaker_names[0] if speaker_names else None
    default_filter = "Only Unverified (TODO)"

    with gr.Blocks(title="Natural Presenter AI") as demo:
        gr.Markdown(
            "# Natural Presenter AI\n"
            "Multi-speaker, accent-preserving English presentation studio powered by **Ollama/Gemma** + **Fine-Tuned F5-TTS**."
        )

        with gr.Tabs():
            # --- TAB 1: PRESENTATION STUDIO ---
            with gr.Tab("Presentation Studio"):
                init_refs = list_speaker_refs(default_speaker)
                init_ref = default_ref_for_speaker(default_speaker)
                init_ref_wav, init_ref_md = get_ref_preview(default_speaker, init_ref)

                with gr.Row():
                    speaker_dd = gr.Dropdown(
                        choices=speaker_names,
                        value=default_speaker,
                        label="Select Speaker Profile",
                    )
                    ref_style_dd = gr.Dropdown(
                        choices=init_refs,
                        value=init_ref,
                        label="Reference Style (Voice Vibe & Pace)",
                    )
                    filler_radio = gr.Radio(
                        choices=["None", "Low", "Medium", "High"],
                        value="Medium",
                        label="Hesitation / Filler Level (uh, um, like, eh)",
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

                with gr.Row():
                    profile_md = gr.Markdown(
                        get_speaker_summary(default_speaker) if default_speaker else ""
                    )
                    self_repair_chk = gr.Checkbox(
                        value=True,
                        label="Include subtle self-repairs & varied hesitations in long scripts (e.g., 'accuracy— sorry, error')",
                    )

                with gr.Accordion("Preview Selected Reference Clip (click to listen)", open=False):
                    ref_preview_md = gr.Markdown(init_ref_md)
                    ref_preview_audio = gr.Audio(
                        value=init_ref_wav, label="Reference Clip Audio", type="filepath"
                    )

                speaker_dd.change(
                    fn=on_studio_speaker_change,
                    inputs=speaker_dd,
                    outputs=[profile_md, ref_style_dd, ref_preview_audio, ref_preview_md],
                )
                ref_style_dd.change(
                    fn=get_ref_preview,
                    inputs=[speaker_dd, ref_style_dd],
                    outputs=[ref_preview_audio, ref_preview_md],
                )

                clean_box = gr.Textbox(
                    label="1. Clean Presentation Script (Separate slides/sections with a blank line for natural pauses)",
                    value=SAMPLE_SCRIPT,
                    lines=7,
                    placeholder="Paste your polished English presentation script here (use blank lines between slides/paragraphs)...",
                )

                humanize_btn = gr.Button(
                    "Step 1: Humanize Script (Apply Speaker Rhythm + Acronyms + Self-Repairs)",
                    variant="primary",
                )
                stage1_status = gr.Textbox(label="Stage 1 Status", interactive=False)

                humanized_box = gr.Textbox(
                    label="2. Humanized Spoken Script (Editable before generating voice)",
                    lines=7,
                    placeholder="Click 'Step 1: Humanize Script' or type directly here...",
                )

                with gr.Row():
                    speed_slider = gr.Slider(
                        minimum=0.80,
                        maximum=1.20,
                        value=0.90,
                        step=0.05,
                        label="Speaking Speed (0.85–0.90 recommended)",
                    )
                    para_pause_slider = gr.Slider(
                        minimum=200,
                        maximum=1500,
                        value=650,
                        step=50,
                        label="Slide / Paragraph Pause (ms)",
                    )
                    quality_radio = gr.Radio(
                        choices=["Fast Preview (16 steps — 2x faster)", "High Quality (32 steps)"],
                        value="Fast Preview (16 steps — 2x faster)",
                        label="TTS Quality / Speed",
                    )
                    synth_btn = gr.Button(
                        "Step 2: Generate Presentation Audio (F5-TTS)", variant="primary"
                    )

                stage2_status = gr.Textbox(label="Stage 2 Status", interactive=False)
                audio_out = gr.Audio(label="Generated Presentation Audio", type="filepath")

                humanize_btn.click(
                    fn=on_humanize,
                    inputs=[
                        speaker_dd,
                        clean_box,
                        filler_radio,
                        tone_dd,
                        llm_box,
                        self_repair_chk,
                    ],
                    outputs=[humanized_box, stage1_status],
                )
                synth_btn.click(
                    fn=on_synthesize,
                    inputs=[
                        speaker_dd,
                        humanized_box,
                        tone_dd,
                        ref_style_dd,
                        speed_slider,
                        quality_radio,
                        para_pause_slider,
                    ],
                    outputs=[audio_out, stage2_status],
                )

            # --- TAB 2: DATASET QA REVIEWER ---
            with gr.Tab("Dataset QA Reviewer"):
                initial_progress = (
                    qa_progress_markdown(default_speaker) if default_speaker else "No speaker found."
                )
                initial_clips = (
                    qa_list_clips(default_speaker, default_filter) if default_speaker else []
                )
                first_clip = initial_clips[0] if initial_clips else None
                init_audio, init_raw, init_clean, init_ver, init_info = (
                    qa_load_clip(default_speaker, first_clip)
                    if (default_speaker and first_clip)
                    else (None, "", "", False, "All clips verified or none loaded!")
                )

                qa_progress_md = gr.Markdown(initial_progress)

                with gr.Row():
                    qa_speaker_dd = gr.Dropdown(
                        choices=speaker_names,
                        value=default_speaker,
                        label="Speaker Dataset",
                    )
                    qa_filter_radio = gr.Radio(
                        choices=QA_FILTERS,
                        value=default_filter,
                        label="Filter Clips (default hides already-reviewed clips)",
                    )

                qa_clip_dd = gr.Dropdown(
                    choices=initial_clips,
                    value=first_clip,
                    label="Select Clip (✅ OK = already reviewed, ⬜ TODO = needs review)",
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
                    qa_verify_next_btn = gr.Button(
                        "✅ Verify & Next Clip (1-click)", variant="primary"
                    )
                    qa_save_btn = gr.Button("Save (Keep Checkbox As-Is) & Next")
                    qa_refresh_btn = gr.Button("Rebuild style_profile.json (Run once at end)")

                with gr.Row():
                    qa_ref_name_box = gr.Textbox(
                        label="Save Current Clip as Reference Style (optional)",
                        placeholder="e.g., relaxed_partA_ref",
                    )
                    qa_save_ref_btn = gr.Button("⭐ Save Clip as Reference Style")

                qa_status = gr.Textbox(value=init_info, label="QA Status", interactive=False)

                qa_speaker_dd.change(
                    fn=qa_on_speaker_or_filter_change,
                    inputs=[qa_speaker_dd, qa_filter_radio],
                    outputs=[
                        qa_progress_md,
                        qa_clip_dd,
                        qa_audio,
                        qa_raw_box,
                        qa_clean_box,
                        qa_verified_chk,
                        qa_status,
                    ],
                )
                qa_filter_radio.change(
                    fn=qa_on_speaker_or_filter_change,
                    inputs=[qa_speaker_dd, qa_filter_radio],
                    outputs=[
                        qa_progress_md,
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
                qa_verify_next_btn.click(
                    fn=qa_verify_and_next,
                    inputs=[
                        qa_speaker_dd,
                        qa_filter_radio,
                        qa_clip_dd,
                        qa_raw_box,
                        qa_clean_box,
                    ],
                    outputs=[
                        qa_progress_md,
                        qa_clip_dd,
                        qa_audio,
                        qa_raw_box,
                        qa_clean_box,
                        qa_verified_chk,
                        qa_status,
                    ],
                )
                qa_save_btn.click(
                    fn=qa_save_current,
                    inputs=[
                        qa_speaker_dd,
                        qa_filter_radio,
                        qa_clip_dd,
                        qa_raw_box,
                        qa_clean_box,
                        qa_verified_chk,
                    ],
                    outputs=[
                        qa_progress_md,
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
                qa_save_ref_btn.click(
                    fn=qa_save_as_reference,
                    inputs=[qa_speaker_dd, qa_clip_dd, qa_ref_name_box],
                    outputs=qa_status,
                )

            # --- TAB 3: PRONUNCIATION DICTIONARY ---
            with gr.Tab("Pronunciation Dictionary"):
                gr.Markdown(
                    "### Technical Acronyms & Custom Pronunciations (`config/pronunciation_map.json`)\n"
                    "Applied automatically in **both Stage 1 and Stage 2** so acronyms like `MSE` → `M S E` "
                    "are never misread as a single word."
                )
                with gr.Row():
                    pron_term_box = gr.Textbox(label="Written Term (exact match)", placeholder="e.g., MSE")
                    pron_spoken_box = gr.Textbox(label="Spoken Form for F5-TTS", placeholder="e.g., M S E")
                    pron_add_btn = gr.Button("Add / Update Rule", variant="primary")

                pron_status = gr.Textbox(label="Dictionary Status", interactive=False)
                pron_json_box = gr.Code(
                    value=pron_load_json(),
                    language="json",
                    label="Full Pronunciation Map (JSON)",
                    lines=16,
                )
                pron_save_json_btn = gr.Button("Save Full JSON")

                pron_add_btn.click(
                    fn=pron_add_term,
                    inputs=[pron_term_box, pron_spoken_box],
                    outputs=[pron_json_box, pron_status],
                )
                pron_save_json_btn.click(
                    fn=pron_save_raw_json,
                    inputs=pron_json_box,
                    outputs=pron_status,
                )

            # --- TAB 4: LIVE CUE PRESENTER (OBS STUDIO) ---
            with gr.Tab("Cue Presenter (Live OBS)"):
                gr.Markdown(
                    "### Human-Driven Live Cue Soundboard (`ADR-006`)\n"
                    "You smoothly move the mouse, scroll, and switch between your clean **Report PDF** & **Jupyter Notebook** in OBS. "
                    "Press **`F9`** (or optional **Mouse Side Button**) whenever your cursor reaches the target to play the pre-rendered AI voice with **0ms latency**!"
                )

                init_decks = cue_list_decks()
                init_deck = init_decks[0] if init_decks else None
                if init_deck and live_controller.deck is None:
                    live_controller.load_deck(load_cue_deck(_deck_path(init_deck)))
                init_cue_choices = _cue_choices_for_deck(init_deck)

                with gr.Row():
                    cue_deck_dd = gr.Dropdown(
                        choices=init_decks,
                        value=init_deck,
                        label="Active Presentation Deck",
                    )
                    cue_load_btn = gr.Button("Load / Refresh Deck")
                    cue_mouse_chk = gr.Checkbox(
                        value=False,
                        label="Also trigger with Mouse Side Buttons (Forward=Play, Back=Replay)",
                    )
                    cue_arm_btn = gr.Button("🟢 ARM GLOBAL HOTKEYS (F9 / Alt+Space)", variant="primary")
                    cue_disarm_btn = gr.Button("⚪ Disarm Hotkeys")

                teleprompter_md = gr.Markdown(live_controller.render_teleprompter_markdown())

                with gr.Row():
                    btn_prev = gr.Button("⏮ Prev (F7 / Alt+←)")
                    btn_play = gr.Button("▶ Play Current (F9 / Alt+Space)", variant="primary")
                    btn_replay = gr.Button("🔁 Replay (F8 / Alt+R)")
                    btn_stop = gr.Button("⏹ Stop (F10 / Alt+S)")
                    btn_next = gr.Button("⏭ Next (F11 / Alt+→)")
                    btn_reset = gr.Button("↺ Reset to #1")

                cue_action_status = gr.Textbox(label="Presenter Status", interactive=False)

                # Live 0.5s timer keeps teleprompter synced when user presses F9 while focused on PDF/Jupyter
                live_timer = gr.Timer(0.5)
                live_timer.tick(fn=live_controller.render_teleprompter_markdown, outputs=teleprompter_md)

                with gr.Accordion("Prepare Deck: Batch Humanize & Pre-Render All Audio", open=False):
                    with gr.Row():
                        cue_filler_radio = gr.Radio(
                            choices=["None", "Low", "Medium", "High"],
                            value="Medium",
                            label="Cue Filler Level",
                        )
                        cue_repair_chk = gr.Checkbox(value=False, label="Include subtle self-repairs")
                        cue_llm_box = gr.Textbox(value="gemma3:4b", label="Stage 1 Model")
                        cue_speed_slider = gr.Slider(0.80, 1.15, value=0.90, step=0.05, label="Speed")
                        cue_qual_radio = gr.Radio(
                            choices=["Fast Preview (16 steps)", "High Quality (32 steps)"],
                            value="Fast Preview (16 steps)",
                            label="Quality",
                        )
                        cue_only_missing_chk = gr.Checkbox(value=True, label="Only render missing WAVs")
                    with gr.Row():
                        cue_humanize_all_btn = gr.Button("1. Humanize All Cues (Gemma 3 4B)")
                        cue_prerender_all_btn = gr.Button(
                            "2. Pre-Render All Cue WAVs (F5-TTS)", variant="primary"
                        )

                with gr.Accordion("Inspect / Edit Individual Cue & Re-Render", open=False):
                    cue_item_dd = gr.Dropdown(
                        choices=init_cue_choices,
                        value=init_cue_choices[0] if init_cue_choices else None,
                        label="Select Cue",
                    )
                    with gr.Row():
                        cue_edit_spk = gr.Textbox(label="Speaker ID (e.g., vn_01 or vn_02)")
                        cue_edit_point = gr.Textbox(label="POINT TO (Visual Cursor Instruction)")
                    cue_edit_clean = gr.Textbox(label="Original SAY (Clean)", lines=2)
                    cue_edit_hum = gr.Textbox(label="Humanized Spoken Script (TTS Input)", lines=2)
                    cue_edit_audio = gr.Audio(label="Pre-Rendered Cue Audio", type="filepath")
                    cue_save_one_btn = gr.Button("Save & Re-Render This Cue", variant="primary")

                with gr.Accordion("Import New Presentation (Reading Copy PDF + Presenter .ipynb)", open=False):
                    with gr.Row():
                        imp_name = gr.Textbox(label="Project Name", placeholder="e.g., lab3_presentation")
                        imp_merge = gr.Checkbox(value=True, label="Merge sentence fragments (e.g., N5a+N5b+N5c)")
                    with gr.Row():
                        imp_pdf = gr.File(label="Reading Copy PDF (.pdf)", file_types=[".pdf"], type="filepath")
                        imp_nb = gr.File(label="Presenter Notebook (.ipynb)", file_types=[".ipynb"], type="filepath")
                    imp_btn = gr.Button("Import & Parse Presentation Cues", variant="primary")

                # Wire Tab 4 callbacks
                cue_load_btn.click(
                    fn=cue_load_deck_action,
                    inputs=cue_deck_dd,
                    outputs=[cue_item_dd, teleprompter_md, cue_action_status],
                )
                cue_arm_btn.click(
                    fn=lambda m: (live_controller.start_listeners(m), live_controller.render_teleprompter_markdown()),
                    inputs=cue_mouse_chk,
                    outputs=[cue_action_status, teleprompter_md],
                )
                cue_disarm_btn.click(
                    fn=lambda: (live_controller.stop_listeners(), live_controller.render_teleprompter_markdown()),
                    outputs=[cue_action_status, teleprompter_md],
                )
                btn_play.click(
                    fn=lambda: (live_controller.play_current_and_advance(), live_controller.render_teleprompter_markdown()),
                    outputs=[cue_action_status, teleprompter_md],
                )
                btn_replay.click(
                    fn=lambda: (live_controller.replay_last(), live_controller.render_teleprompter_markdown()),
                    outputs=[cue_action_status, teleprompter_md],
                )
                btn_stop.click(
                    fn=lambda: (live_controller.stop_audio(), live_controller.render_teleprompter_markdown()),
                    outputs=[cue_action_status, teleprompter_md],
                )
                btn_prev.click(
                    fn=lambda: (live_controller.step_index(-1), live_controller.render_teleprompter_markdown()),
                    outputs=[cue_action_status, teleprompter_md],
                )
                btn_next.click(
                    fn=lambda: (live_controller.step_index(1), live_controller.render_teleprompter_markdown()),
                    outputs=[cue_action_status, teleprompter_md],
                )
                btn_reset.click(
                    fn=lambda: (live_controller.reset_to_start(), live_controller.render_teleprompter_markdown()),
                    outputs=[cue_action_status, teleprompter_md],
                )
                cue_humanize_all_btn.click(
                    fn=cue_humanize_all_action,
                    inputs=[cue_deck_dd, cue_filler_radio, cue_repair_chk, cue_llm_box],
                    outputs=[teleprompter_md, cue_action_status],
                )
                cue_prerender_all_btn.click(
                    fn=cue_prerender_all_action,
                    inputs=[cue_deck_dd, cue_speed_slider, cue_qual_radio, cue_only_missing_chk],
                    outputs=[cue_item_dd, teleprompter_md, cue_action_status],
                )
                cue_item_dd.change(
                    fn=cue_select_for_edit,
                    inputs=[cue_deck_dd, cue_item_dd],
                    outputs=[cue_edit_spk, cue_edit_point, cue_edit_clean, cue_edit_hum, cue_edit_audio],
                )
                cue_save_one_btn.click(
                    fn=cue_save_and_rerender_one,
                    inputs=[
                        cue_deck_dd,
                        cue_item_dd,
                        cue_edit_spk,
                        cue_edit_point,
                        cue_edit_clean,
                        cue_edit_hum,
                        cue_speed_slider,
                        cue_qual_radio,
                    ],
                    outputs=[cue_item_dd, cue_edit_audio, teleprompter_md, cue_action_status],
                )
                imp_btn.click(
                    fn=cue_import_files,
                    inputs=[imp_name, imp_pdf, imp_nb, imp_merge],
                    outputs=[cue_deck_dd, cue_item_dd, teleprompter_md, cue_action_status],
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
