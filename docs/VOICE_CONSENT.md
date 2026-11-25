# Speaker Voice Consent & Responsible AI Record

**Project:** Natural Presenter AI  
**Policy Effective Date:** 2026-11-25  

Natural Presenter AI is strictly a **consensual, local-first voice presentation tool** designed for self-cloning and explicitly authorized team collaboration. Because voice prints constitute sensitive biometric data under EU GDPR and ethical AI guidelines, this document records the explicit consent scope for every speaker whose synthetic voice appears in our public portfolio showcase or demo video.

---

## Consented Showcase Speakers

| Speaker ID | Name & Role | Consent Scope | Status |
| :--- | :--- | :--- | :--- |
| `vn_01` | **Thai Binh Nguyen** (Co-Creator) | Self-recorded audio used for local fine-tuning (`FT-001`), public Hugging Face Space audio benchmarks (`01`–`07`), and portfolio demo video narration. | **Consented (Self)** |
| `vn_02` | **Ege Deniz** (Co-Creator) | Self-recorded presentation audio used for Zero-Shot synthesis in public Hugging Face Space multi-speaker benchmarks (`07`–`08`) and portfolio demo video narration. | **Consented (Co-Creator, 2026-11-25)** |

*(Optional provenance note: Co-creators may append a signed Git commit or GitHub PR approval referencing this file at any time.)*

---

## Strict Privacy & Distribution Boundaries

1. **No Raw Voice Data or Model Weights Distributed:** Raw recordings (`profiles/`), verified clips (`*.wav`), and trained model checkpoints (`*.safetensors`, `*.pt`) are strictly `.gitignore`d and never uploaded to GitHub or Hugging Face.
2. **Showcase Clips Only:** Public artifacts contain only short, neutral, technical portfolio demonstration clips explicitly approved by the respective speaker(s).
3. **Revocation Right:** Either speaker may request removal of their showcase audio clips at any time; clips will be removed immediately from the Hugging Face Space and any hosted demo assets.
4. **Prohibited Use:** Users of this open-source codebase must never clone, synthesize, or impersonate any individual's voice without explicit, informed consent.
