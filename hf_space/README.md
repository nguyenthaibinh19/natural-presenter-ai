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
1. Run `python scripts/07_generate_showcase_samples.py` locally on your RTX GPU to render the 6 self-recorded demo `.wav` clips (`01`–`06`, `vn_01` only) into `hf_space/samples/`.
2. Authenticate once with `hf auth login` (or set `HF_TOKEN`), then run:
   ```powershell
   python scripts/08_deploy_hf_space.py --repo-id <your-hf-username>/natural-presenter-ai-showcase
   ```
   *(Or manually create a **Gradio** Space on `CPU basic — Free` at https://huggingface.co/spaces and upload `hf_space/`.)*
3. Paste the live Space URL into your GitHub `README.md` and CV!
