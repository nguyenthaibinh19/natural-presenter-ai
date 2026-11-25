"""Free CPU-Only Hugging Face Spaces Portfolio Showcase for Natural Presenter AI.

Demonstrates:
  1. Pre-rendered Audio Comparisons (Zero-Shot vs Fine-Tuned FT-001, Filler Levels, Acronym Fix, Multi-Speaker Handoff)
  2. Live Stage 1 Playground (Acronym Normalization + Individualized Speaker Style Adaptation — runs on CPU!)
  3. Engineering Benchmarks & Architecture Overview
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List

import gradio as gr

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

PRONUNCIATION_MAP: Dict[str, str] = {
    "RMSE": "R M S E",
    "MSE": "M S E",
    "MAE": "M A E",
    "CNN": "C N N",
    "RNN": "R N N",
    "LSTM": "L S T M",
    "LLM": "L L M",
    "NLP": "N L P",
    "TTS": "T T S",
    "ASR": "A S R",
    "GPU": "G P U",
    "CPU": "C P U",
    "VRAM": "V-ram",
    "API": "A P I",
    "SQL": "S Q L",
    "PCA": "P C A",
    "ReLU": "ray-loo",
    "CUDA": "kooda",
    "LoRA": "low-rah",
    "FP16": "F P sixteen",
    "BF16": "B F sixteen",
    "Q3": "Q three",
}

DEMO_PROFILES = {
    "Speaker 1 — Vietnamese English (102.7 WPM | uh 78%, like 13%, um 4%)": {
        "fillers": {"uh": 0.78, "like": 0.13, "um": 0.05, "actually": 0.04},
        "targets": {"none": 0, "low": 2, "medium": 5, "high": 9},
    },
    "Speaker 2 — Turkish English (146.6 WPM | uh 79%, eh 12%, um 9%)": {
        "fillers": {"uh": 0.79, "eh": 0.12, "um": 0.09},
        "targets": {"none": 0, "low": 2, "medium": 5, "high": 9},
    },
}

DEFAULT_DEMO_TEXT = (
    "Hello everyone, today I want to walk you through our CNN autoencoder results on a single GPU. "
    "After switching to ReLU activations, test MSE dropped by 24% in Q3.\n\n"
    "However, latency increased slightly, so we pruned the FP16 weights from 3.38 GB down to 643 MB for local inference."
)


def apply_pronunciation(text: str) -> str:
    out = text
    for term in sorted(PRONUNCIATION_MAP.keys(), key=len, reverse=True):
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])")
        out = pat.sub(PRONUNCIATION_MAP[term], out)
    out = re.sub(r"(\d+)\s*%", r"\1 percent", out)
    return out


def cpu_humanize_demo(speaker_label: str, clean_text: str, level: str, self_repairs: bool) -> str:
    prof = DEMO_PROFILES[speaker_label]
    lvl = level.lower()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text.strip()) if p.strip()]
    if not paragraphs:
        return ""

    tokens = list(prof["fillers"].keys())
    weights = list(prof["fillers"].values())
    target_per_100 = prof["targets"][lvl]
    transitions = {"and", "but", "so", "however,", "after", "when", "if", "because"}

    out_paras: List[str] = []
    for p_idx, para in enumerate(paragraphs):
        norm = apply_pronunciation(para)
        if lvl == "none":
            out_paras.append(norm)
            continue

        words = norm.split()
        desired = max(1, round((len(words) / 100.0) * target_per_100))
        rng = random.Random(101 + p_idx * 17 + len(words))
        sentences = re.split(r"(?<=[.!?])\s+", norm)
        inserted = 0
        did_repair = False
        new_sents: List[str] = []

        for s_idx, sent in enumerate(sentences):
            sw = sent.split()
            if len(sw) < 5:
                new_sents.append(sent)
                continue
            if self_repairs and not did_repair and p_idx == len(paragraphs) - 1 and s_idx == 0:
                for i in range(1, len(sw) - 2):
                    if sw[i].lower() in {"we", "the", "our", "and"}:
                        sw.insert(i, f"{sw[i]},")
                        did_repair = True
                        break
            if inserted < desired:
                filler = rng.choices(tokens, weights=weights, k=1)[0]
                placed = False
                for i in range(2, len(sw) - 2):
                    if sw[i].endswith(",") or sw[i].lower() in transitions:
                        sw.insert(i + 1, f"{filler},")
                        placed = True
                        inserted += 1
                        break
                if not placed:
                    sw.insert(len(sw) // 2, f"{filler},")
                    inserted += 1
            new_sents.append(" ".join(sw))
        out_paras.append(" ".join(new_sents))

    return "\n\n".join(out_paras)


def sample_path(filename: str) -> str | None:
    p = SAMPLES_DIR / filename
    return str(p) if p.exists() else None


def build_showcase() -> gr.Blocks:
    with gr.Blocks(title="Natural Presenter AI — Portfolio Showcase") as demo:
        gr.Markdown(
            "# 🎙️ Natural Presenter AI — Portfolio Showcase\n"
            "**100% Local, Accent-Preserving, Multi-Speaker AI Voice Presentation Studio**  \n"
            "Built by **Thai Binh Nguyen & Ege Deniz** | "
            "[GitHub Repository](https://github.com/nguyenthaibinh19/natural-presenter-ai)\n\n"
            "> This Space runs on **free CPU** with pre-rendered audio benchmarks and a live Stage 1 playground. "
            "Full local synthesis runs faster than real time on an 8GB laptop GPU (RTX 5060)."
        )

        with gr.Tabs():
            with gr.Tab("1. Audio Benchmarks (Listen)"):
                gr.Markdown(
                    "### Side-by-Side Listening Comparisons\n"
                    "Compare the **Zero-Shot baseline (7–8/10)** against our **26.65-minute Fine-Tuned F5-TTS checkpoint (`FT-001`, 9.5–10/10)**, "
                    "individualized hesitation intensity, acronym pronunciation (`MSE` → `M S E`), and a two-speaker presentation handoff."
                )

                with gr.Row():
                    gr.Audio(
                        value=sample_path("01_zeroshot_baseline.wav"),
                        label="A. Zero-Shot F5-TTS Base (10s ref only — 7–8/10)",
                        type="filepath",
                    )
                    gr.Audio(
                        value=sample_path("02_finetuned_ft001.wav"),
                        label="B. Fine-Tuned FT-001 (643 MB FP16, 4020 updates — 9.5–10/10)",
                        type="filepath",
                    )

                with gr.Row():
                    gr.Audio(
                        value=sample_path("03_filler_none.wav"),
                        label="C. Filler Level: NONE (Formal Polished Delivery)",
                        type="filepath",
                    )
                    gr.Audio(
                        value=sample_path("04_filler_medium_natural.wav"),
                        label="D. Filler Level: MEDIUM (Speaker's Authentic uh / like Rhythm)",
                        type="filepath",
                    )

                with gr.Row():
                    gr.Audio(
                        value=sample_path("05_acronym_mse_fixed.wav"),
                        label="E. Acronym Lexicon Fix (MSE, CNN, GPU, ReLU)",
                        type="filepath",
                    )
                    gr.Audio(
                        value=sample_path("06_multi_speaker_handoff.wav"),
                        label="F. Multi-Speaker Joint Presentation (Speaker 1 + Speaker 2 Handoff)",
                        type="filepath",
                    )

            with gr.Tab("2. Live Stage 1 Playground (CPU)"):
                gr.Markdown(
                    "### Try Personalized Script Humanization & Acronym Normalization\n"
                    "See how the same polished presentation script adapts differently to each speaker's "
                    "measured disfluency profile (`style_profile.json`) while normalizing technical acronyms for G2P-free TTS."
                )
                with gr.Row():
                    spk_dd = gr.Dropdown(
                        choices=list(DEMO_PROFILES.keys()),
                        value=list(DEMO_PROFILES.keys())[0],
                        label="Speaker Profile",
                    )
                    lvl_radio = gr.Radio(
                        choices=["None", "Low", "Medium", "High"],
                        value="Medium",
                        label="Hesitation Intensity",
                    )
                    rep_chk = gr.Checkbox(value=True, label="Include subtle self-repair / repetition")

                in_box = gr.Textbox(value=DEFAULT_DEMO_TEXT, label="Clean Presentation Script", lines=5)
                run_btn = gr.Button("Humanize & Normalize Acronyms", variant="primary")
                out_box = gr.Textbox(label="TTS-Ready Spoken Script", lines=6)

                run_btn.click(
                    fn=cpu_humanize_demo,
                    inputs=[spk_dd, in_box, lvl_radio, rep_chk],
                    outputs=out_box,
                )

            with gr.Tab("3. Architecture & Metrics"):
                gr.Markdown(
                    """### Key Engineering Decisions & Results

| Milestone | Engineering Detail | Outcome |
|---|---|---|
| **Data Curation & QA** | 60/40 Spontaneous + Scripted split, iPhone 16 Lossless ALAC (34.7–42.6 dB SNR), Silero VAD, disfluency-preserving `faster-whisper`, custom Gradio QA Reviewer | **173 verified clips (26.65 min)**, 0 integrity errors |
| **Accent-Preserving TTS** | Fine-tuned `F5TTS_v1_Base` (G2P-free Flow Matching) with pretrained 2545-token vocab (`warmup=100`, `lr=1e-5`, 60 epochs / 4,020 updates) | Final loss **0.408**, rated **9.5–10/10** with authentic Vietnamese-English accent preserved |
| **Model Compression** | Pruned training checkpoint (`model_4000.pt`) to EMA FP16 `safetensors` | **3.38 GB → 643 MB (5.2× smaller)** |
| **Local Edge Inference** | Single-pass F5-TTS batching + 16-step Fast Preview + Local Ollama `gemma3:4b` on RTX 5060 Laptop (8GB VRAM) | **5.3s for 6.5s audio** (faster than real time, \$0 API cost) |
| **Live OBS Cue Presenter** | Native Win32 `GetAsyncKeyState` global hotkeys (`F9` / `Alt+Space` / Mouse Side Button) + pre-rendered cue WAVs + synced `NOW SPEAKING` teleprompter | **0 ms latency**, smooth human cursor control over clean PDF/Jupyter deliverables |

### Responsible AI & Privacy
All voice models are trained exclusively on self-recorded, explicitly consented speech. No personal voice datasets or trained voice checkpoints are distributed in the public GitHub repository.
"""
                )

    return demo


if __name__ == "__main__":
    build_showcase().launch()
