# Engineering Lessons Learned

Durable lessons captured during development so future contributors and AI agents do not repeat solved pitfalls.

1. **Whisper Strips Disfluencies by Default (2026-11-23):**  
   Standard Whisper decoding silently deletes spoken `uh`, `um`, `eh`, `erm`. Always pass a disfluency-rich `initial_prompt` in `scripts/02_transcribe_align.py` and perform a targeted QA review on clips with internal pauses.

2. **RTX 5060 Laptop (Blackwell `sm_120`) Requires CUDA 12.8+ PyTorch (2026-11-23):**  
   Older PyTorch/CUDA wheels fail or fall back on Blackwell GPUs. Always install PyTorch via `--index-url https://download.pytorch.org/whl/cu128` inside the dedicated Conda environment.

3. **Windows Cross-Drive `Move-Item` on `.git` Folders (2026-11-23):**  
   Moving a Git repo across drives (`C:\` → `D:\`) in PowerShell may copy cleanly but fail to delete read-only objects inside the source `.git` directory; remove the old copy explicitly with `Remove-Item -Recurse -Force` after verifying the destination.

4. **`faster-whisper` (`ctranslate2`) Missing `cublas64_12.dll` on Windows (2026-11-23):**  
   PyTorch bundles `cublas64_12.dll` inside `torch/lib`, which is not on Windows' default DLL search path. `scripts/02_transcribe_align.py` registers `torch/lib` via `os.add_dll_directory()` before importing `faster_whisper`, resolving CUDA execution cleanly without manual PATH edits.

5. **PowerShell `-c:a` Parsing with `conda run ffmpeg` (2026-11-23):**  
   PowerShell splits unquoted `-c:a` into `-c:` and `a` when passed through `conda run`. Always invoke `ffmpeg.exe` directly or quote `"-c:a"` in PowerShell commands.

6. **iPhone 16 Lossless `.m4a` (ALAC) Decoding on Windows (2026-11-23):**  
   `libsndfile`/`librosa` cannot open Apple ALAC `.m4a` directly without an FFmpeg backend. Decode `.m4a` to 24kHz mono PCM `.wav` using `ffmpeg.exe` first, and move the `.m4a` originals to `raw_recordings/_originals/` to prevent duplicate slicing. Use `--append` on `01_preprocess_audio.py` and `02_transcribe_align.py` so previously QA-verified clips are never overwritten.

7. **RTX 5060 Laptop Battery Power Cap & F5-TTS Inference Speed (2026-11-24):**  
   On battery / power-saver mode, the RTX 5060 Laptop throttles to `P4` (`15W / 31W`), slowing F5-TTS by 4–6x (~40s warmup + ~11.5s/batch at 32 steps). Plugging in AC power (Performance mode) + removing redundant double-chunking in `src/synthesizer.py` + defaulting to `nfe_step=16` (`Fast Preview`) cuts generation time dramatically.
