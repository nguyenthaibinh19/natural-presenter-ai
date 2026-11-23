# Invariants (Must-Not-Break Rules)

These rules define the core product integrity of Natural Presenter AI. Any code, prompt, or model change that violates an invariant must be rejected or require an explicit superseding ADR.

1. **No Accent Normalization:**  
   The TTS layer MUST preserve each speaker's authentic non-native English pronunciation, consonant endings, and intonation. Do NOT introduce rigid US/UK G2P (Grapheme-to-Phoneme) normalization into the primary TTS pipeline.

2. **No Generic Filler Templates Across Speakers:**  
   Stage 1 (Script Humanizer) MUST derive filler vocabulary, frequency, and placement from the selected speaker's `profiles/<speaker_id>/style_profile.json`. Never hardcode a single global `uh/um` style for all users.

3. **Strict 1:1 `raw_text` ↔ Audio Alignment:**  
   In `metadata.jsonl`, `raw_text` is the ground truth for TTS training. Every audible filler (`uh`, `um`, `eh`, `erm`) in a training WAV MUST appear in `raw_text`, and `raw_text` MUST NOT contain fillers that were not spoken.

4. **Preserve Natural Breaths:**  
   Audio preprocessing (`01_preprocess_audio.py`) MUST NOT apply aggressive noise gates or silence strippers that remove subtle breath intakes between clauses.

5. **Human Review Gate Between Stage 1 and Stage 2:**  
   The UI and workflow MUST allow the user to inspect and edit the humanized script before triggering TTS audio generation.

6. **Per-Speaker Isolation:**  
   Speaker voice adapters (LoRA) and style profiles MUST remain isolated under `profiles/<speaker_id>/` so onboarding a new accent never degrades or alters existing speakers.

7. **No Audio Datasets or Model Weights in Git:**  
   `profiles/`, `*.wav`, `*.mp3`, `*.pt`, `*.safetensors`, and `*.ckpt` MUST remain gitignored and never committed to GitHub.

8. **Continuous Memory Rule:**  
   No completed phase, architectural change, schema update, or operational fix is closed until `.ai/` (and `docs/AI_AGENT_SPEC.md` when contracts change) is updated in the same commit.
