# Training & Evaluation Experiments Log

Track every TTS fine-tuning run and Stage 1 humanizer evaluation here so we never repeat failed hyperparameters or lose track of which checkpoint sounds most natural.

## F5-TTS Per-Speaker LoRA Runs

| Run ID | Date | Speaker | Dataset (min / clips) | Base Model | Key Hyperparams (LR / Steps / Batch) | Cloud GPU & Cost | Accent Fidelity (1–5) | Filler & Breath Realism (1–5) | Verdict & Notes |
|---|---|---|---|---|---|---|---|---|---|
| `ZS-001` | 2026-11-23 | `vn_01` (Binh) & `vn_02` (Ege) | 10s ref clip each (Zero-Shot) | `F5TTS_v1_Base` | Zero-Shot (no training) | Local RTX 5060 (\$0) | 4 / 5 | 4 / 5 | **User rated 7–8/10 overall!** Strong baseline; next step is 30–60 min dataset + Cloud fine-tune to push to 9–9.5/10. |
| `FT-001` | 2026-11-24 | `vn_01` (Binh) | 26.65 min / 173 verified clips (51.5% Part A, 48.5% Part B) | `F5TTS_v1_Base` | `lr=1e-5`, `batch=1800` frames x `grad_acc=2`, `epochs=60` (4020 updates), `fp16`, `bnb_optimizer`, `warmup=100`, final loss `0.408` | Colab Pro Tesla T4 15GB (\$0 extra) | **5 / 5** | **5 / 5** | **User rated 9.5–10/10 for voice & pausing!** Pruned `model_4000.pt` (3.38 GB) → `vn_01_model_4000_fp16.safetensors` (**643 MB**). Local RTX 5060: 6.5s audio in **5.3s** (16 steps). |

## Stage 1 Script Humanizer Evaluations

| Eval ID | Date | Model | Speakers Tested | Notes & Verdict |
|---|---|---|---|---|
| `S1-001` | 2026-11-24 | Local Ollama `gemma3:4b` | `vn_01` (Binh), `vn_02` (Ege) | Clean symbol expansion (`Q3` → `Q three`, `24%` → `24 percent`, `3x3` → `three by three`) and natural `uh,` placement. Exported 173 `vn_01` pairs to `profiles/humanizer_sft.jsonl` for future optional QLoRA fine-tuning once Ege + additional speakers are added. |

## Evaluation Criteria
- **Accent Fidelity (1–5):** `5` = indistinguishable from speaker's authentic non-native English accent; `1` = noticeably normalized toward generic US/UK accent.
- **Filler & Breath Realism (1–5):** `5` = natural `uh`/`um`/`eh` transitions and subtle breaths with zero stuttering; `1` = robotic, clipped, or hallucinated repetitions.
