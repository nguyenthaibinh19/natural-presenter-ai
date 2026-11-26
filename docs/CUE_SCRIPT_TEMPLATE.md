# Cue Script Template & ChatGPT Prompt (`Natural Presenter AI`)

Use this template to turn **any clean presentation report (PDF/Slides) + code notebook (`.ipynb`)** into a ready-to-import **Cue Script (`.md`)** for **Tab 4: `Cue Presenter (Live OBS)`**.

Next time you have a new presentation, just upload your clean `report.pdf` + `code.ipynb` + **this file** to ChatGPT (or any AI assistant) with the prompt below, then paste or upload the output directly into **Natural Presenter AI**!

---

## 1. Copy-Paste Prompt for ChatGPT / AI Assistant

```text
You are preparing a live screen-recorded presentation script for "Natural Presenter AI" (Tab 4: Cue Presenter).
I am attaching my clean report/slides and/or code notebook.
Follow the exact Markdown Cue Script format defined in docs/CUE_SCRIPT_TEMPLATE.md.

Presentation details:
- Speakers & parts:
  - speaker_02 (vn_02): Introduction, data/code setup, Exercise 1, Exercise 3, Conclusion
  - binh (vn_01): Bottleneck analysis, Exercise 2 (code + results)
- Target total speaking length: ~5 minutes (keep each cue crisp: 1–3 sentences, ~15–45 words).

STRICT FORMAT RULES:
1. Every cue MUST start with a header line:
   ## <CUE_ID> | speaker: <speaker_id_or_name> | location: <Document Page or Cell>
   (Optional: add `| ref: <ref_name>` to override reference voice style, e.g., `ref: partA_natural_slow_ref`.)
2. Immediately below the header, include:
   POINT TO: <Exact visual target where the human should move the mouse cursor on the CLEAN document>
   SAY: <Polished, clean spoken English — NO markdown bold/italics/bullets, NO filler words (Stage 1 adds uh/um/like automatically!), keep ALL numbers/metrics 100% exact>
   NEXT: <Optional transition instruction whenever scrolling to a new page/cell, switching PDF <-> Code, or handing over to another speaker>
3. Order cues in the EXACT chronological speaking order from start to finish.
4. Return ONLY the Markdown Cue Script inside a single ```markdown code block so I can copy-paste or save it directly as `script.md`.
```

---

## 2. Format Reference

| Field | Required? | Description |
| :--- | :--- | :--- |
| `## <CUE_ID> \| speaker: ... \| location: ...` | **Yes** | Unique short ID (`01`, `N2`, `B1`, etc.), speaker ID or display name (`vn_01` / `binh`, `speaker_02`, `es_01`, etc.), optional `ref: <ref_name>`, and screen location (`Report Page 1`, `Notebook Cell 10`). |
| `POINT TO:` | **Yes** | Shown in giant bold text on your off-screen Teleprompter so you know exactly where to point the mouse while AI speaks. |
| `SAY:` | **Yes** | Clean polished English text (1–3 sentences). Do NOT manually add `uh`/`um` here — Stage 1 (Gemma) injects each speaker's personal hesitation style automatically! Acronyms like `MSE`, `CNN`, `GPU`, `ReLU` are auto-normalized by `config/pronunciation_map.json`. |
| `NEXT:` | Optional | Shown right after the cue to remind you to scroll, switch windows (`Report <-> Notebook`), or hand over to the next speaker. |

---

## 3. Complete Working Example (Copy-Paste Ready to Test in Tab 4)

```markdown
## 01 | speaker: speaker_02 | location: Notebook Cell 4
POINT TO: Point to the saved output line "Train / validation / test: 48000 6000 6000" below cell 4, then move up to "/ 255.0".
SAY: Hello everyone. We split the sixty thousand CIFAR-10 images eighty-ten-ten into training, validation, and test sets, and scaled pixel values to zero through one by dividing by 255.
NEXT: Scroll down to Notebook Cell 6.

## 02 | speaker: speaker_02 | location: Notebook Cell 6
POINT TO: Trace lines 14–19 in encoder_layers (Conv2d, pool1, Conv2d, pool2, latent Conv2d), then stop on nn.ReLU() and nn.Sigmoid().
SAY: The encoder uses 3 by 3 convolutions and max pooling to reduce the image from 32 by 32 down to 8 by 8, ending in a central layer with 16 channels, or 1,024 values. Hidden layers use ReLU, and the decoder upsamples back to RGB with a sigmoid output.
NEXT: Scroll down to Notebook Cell 8.

## 03 | speaker: speaker_02 | location: Notebook Cell 8
POINT TO: Point to nn.MSELoss() on line 1, move to torch.optim.Adam on line 23, then point to best_state on line 47.
SAY: We train with MSE and Adam for twenty epochs per run across three random seeds, keeping the checkpoint with the lowest validation error.
NEXT: Switch from Notebook to Report Page 2 (Figure 1).

## 04 | speaker: speaker_02 | location: Report Page 2
POINT TO: Trace the baseline curve in the top-left panel of Figure 1, then point to test MSE 0.005629 above the plot.
SAY: The baseline error drops quickly in the first three epochs, then improves more slowly while training and validation curves stay close. Mean test MSE across three seeds is 0.0056.
NEXT: Hand over to Binh and jump to Report Page 1 (Table 1).

## 05 | speaker: binh | ref: partA_natural_slow_ref | location: Report Page 1
POINT TO: Point to the two adjacent values in Table 1: 768 (Max pooling 2x2) and 1,024 (latent space).
SAY: Looking at the bottleneck in Table 1, the central code has 1,024 values, while the preceding pooling layer is actually narrower at 768.
NEXT: Switch to Notebook Cell 10 (reconstruction_configs).

## 06 | speaker: binh | ref: partA_natural_slow_ref | location: Notebook Cell 10
POINT TO: Sweep lines 2–4 (narrow, baseline, wide), move to lines 5–6 (latent_side 4 and 16), then sweep lines 7–9 (strided, deeper, kernel5).
SAY: In Exercise 2, we compared width, spatial resolution, and structural changes at a fixed latent size.
NEXT: Switch back to Report Page 4 (Table 3 and Figure 3).

## 07 | speaker: binh | ref: partA_natural_slow_ref | location: Report Page 4
POINT TO: Point to spatial4, baseline, and spatial16 in the Test MSE column of Table 3, then point to strided.
SAY: Spatial16 achieved 59 percent lower test MSE than baseline while keeping the exact same 4,911 parameters. At the same latent size, replacing max pooling with stride-two convolutions reduced error by about 25 percent across all three seeds.
NEXT: Hand over to Speaker 2 and continue to Report Page 5.

## 08 | speaker: speaker_02 | location: Report Page 7 (Conclusion)
POINT TO: Point to the reconstruction and colourisation summary paragraphs in Section 5.
SAY: Overall, we learned that both latent spatial resolution and how we downsample strongly affect reconstruction quality. For colourisation, predicting chroma while reusing input luminance clearly outperformed direct RGB. Thank you for listening.
NEXT: Stop OBS recording.
```

---

## 4. How to Run This in Natural Presenter AI
1. Double-click **`start_presenter.bat`** (ensures global hotkeys work on your real Windows desktop).
2. Open **Tab 4: `Cue Presenter (Live OBS)`** → expand **`Import Cue Script (.md or Paste Text)`**.
3. Give your project a name (e.g., `lab3_presentation`), **paste the Markdown Cue Script** (or upload `.md`), and click **`Import Cue Script`**.
4. Expand **`Prepare Deck`**: click **`1. Humanize All Cues`** (Gemma adds each speaker's unique `uh`/`um`/`like` rhythm + expands `MSE` → `M S E`), verify the cue table, then click **`2. Pre-Render All Cue WAVs`**.
5. Click **`🟢 ARM GLOBAL HOTKEYS`** → open your clean PDF/Notebook in OBS → move your mouse and press **`F9` / `Alt+Space`** (or Mouse Side Button) on each cue!
