# Engineering Lessons Learned

Durable lessons captured during development so future contributors and AI agents do not repeat solved pitfalls.

1. **Whisper Strips Disfluencies by Default (2026-11-23):**  
   Standard Whisper decoding silently deletes spoken `uh`, `um`, `eh`, `erm`. Always pass a disfluency-rich `initial_prompt` in `scripts/02_transcribe_align.py` and perform a targeted QA review on clips with internal pauses.

2. **RTX 5060 Laptop (Blackwell `sm_120`) Requires CUDA 12.8+ PyTorch (2026-11-23):**  
   Older PyTorch/CUDA wheels fail or fall back on Blackwell GPUs. Always install PyTorch via `--index-url https://download.pytorch.org/whl/cu128` inside the dedicated Conda environment.

3. **Windows Cross-Drive `Move-Item` on `.git` Folders (2026-11-23):**  
   Moving a Git repo across drives (`C:\` → `D:\`) in PowerShell may copy cleanly but fail to delete read-only objects inside the source `.git` directory; remove the old copy explicitly with `Remove-Item -Recurse -Force` after verifying the destination.
