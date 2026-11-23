# Training & Evaluation Experiments Log

Track every TTS fine-tuning run and Stage 1 humanizer evaluation here so we never repeat failed hyperparameters or lose track of which checkpoint sounds most natural.

## F5-TTS Per-Speaker LoRA Runs

| Run ID | Date | Speaker | Dataset (min / clips) | Base Model | Key Hyperparams (LR / Steps / Batch) | Cloud GPU & Cost | Accent Fidelity (1–5) | Filler & Breath Realism (1–5) | Verdict & Notes |
|---|---|---|---|---|---|---|---|---|---|
| *(none yet)* | — | `vn_01` | — | `F5TTS_v1_Base` | — | — | — | — | Pending first dataset recording |

## Evaluation Criteria
- **Accent Fidelity (1–5):** `5` = indistinguishable from speaker's authentic non-native English accent; `1` = noticeably normalized toward generic US/UK accent.
- **Filler & Breath Realism (1–5):** `5` = natural `uh`/`um`/`eh` transitions and subtle breaths with zero stuttering; `1` = robotic, clipped, or hallucinated repetitions.
