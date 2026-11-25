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

8. **F5-TTS Fine-Tuning Tokenizer & Warmup Pitfalls (2026-11-24):**  
   `F5TTS_v1_Base` (`model_1250000.safetensors`) uses a 2545-token pretrained `vocab.txt` under `--tokenizer pinyin` (which keeps English text as raw characters with zero G2P normalization). Fine-tuning MUST reuse that exact pretrained `vocab.txt` + `--tokenizer pinyin` so text embeddings load cleanly, and MUST override `--num_warmup_updates` from the default `20000` down to `100` for ~30-minute datasets (~1,500–2,000 total updates).

9. **University JupyterHub Pod OOMKill from F5-TTS `num_workers=16` (2026-11-24):**  
   F5-TTS `trainer.py` hardcodes `DataLoader(num_workers=16, persistent_workers=True, pin_memory=True)`, which forks 16 Python processes and exhausts container CPU RAM / `/dev/shm` on Kubernetes JupyterHub pods (triggering `Server unavailable or unreachable`), even when GPU VRAM (NVIDIA L40 44GB) is nearly empty. Patching `trainer.py` to `num_workers=0, persistent_workers=False, pin_memory=False` eliminates the pod OOMKill cleanly.

10. **Acronym Pronunciation in G2P-Free F5-TTS (`MSE` → *"m-se"*) (2026-11-25):**  
    Because F5-TTS reads raw characters without a G2P dictionary, uppercase acronyms like `MSE` or `CNN` can be mispronounced as single words. Applying a dual-layer editable lexicon (`config/pronunciation_map.json`, mapping `MSE` → `M S E`, `ReLU` → `ray-loo`, etc.) in both Stage 1 and Stage 2 fixes this cleanly without retraining.
