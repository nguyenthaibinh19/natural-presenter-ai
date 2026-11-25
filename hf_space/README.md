---
title: Natural Presenter AI — Showcase
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Natural Presenter AI — Interactive Portfolio Showcase (Static Space — Free Forever)

Zero-cost, instant-loading **Static Space** showcase for **[Natural Presenter AI](https://github.com/nguyenthaibinh19/natural-presenter-ai)** — a 100% local, privacy-first, multi-accent AI voice presentation studio built by **Thai Binh Nguyen** and **Ege Deniz**.

## Deploy to Hugging Face Spaces (Free Forever)
1. Run `python scripts/07_generate_showcase_samples.py` locally on your RTX GPU to render the 6 self-recorded demo `.wav` clips (`01`–`06`, `vn_01` only) into `hf_space/samples/`.
2. Authenticate once with `hf auth login`, then run:
   ```powershell
   python scripts/08_deploy_hf_space.py --repo-id <your-hf-username>/natural-presenter-ai-showcase
   ```
3. Paste the live Space URL into your GitHub `README.md` and CV!
