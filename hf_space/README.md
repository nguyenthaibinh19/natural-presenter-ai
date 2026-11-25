---
title: Natural Presenter AI — Showcase
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
---

# Natural Presenter AI — Interactive Portfolio Showcase (Free CPU)

Zero-GPU interactive showcase for **[Natural Presenter AI](https://github.com/nguyenthaibinh19/natural-presenter-ai)** — a 100% local, privacy-first, multi-accent AI voice presentation studio built by **Thai Binh Nguyen** and **Ege Deniz**.

## Deploy to Hugging Face Spaces (Free Forever)
1. Run `python scripts/07_generate_showcase_samples.py` locally on your RTX GPU to render the 6 neutral demo `.wav` clips into `hf_space/samples/`.
2. Create a new **Gradio Space** (`CPU basic — Free`) on https://huggingface.co/spaces.
3. Upload the contents of `hf_space/` (`README.md`, `app.py`, `requirements.txt`, and `samples/*.wav`).
4. Paste the live Space URL into your GitHub `README.md` and CV!
