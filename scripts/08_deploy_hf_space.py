"""Step 8 (Optional): One-command deploy of `hf_space/` to a FREE Hugging Face Static Space.

Static Spaces (`sdk: static`, `index.html` + `samples/*.wav`) are 100% free forever on
Hugging Face without requiring a PRO subscription, load instantly with zero cold start,
and run the Stage 1 playground directly in the visitor's browser.

Safety Guard:
  By default, refuses to upload any co-author audio (`06_multi_speaker_handoff.wav`)
  unless `--allow-coauthor-consent` is explicitly passed.

Usage:
    # 1. Authenticate once with `hf auth login`
    hf auth login

    # 2. Deploy the free Static Space showcase
    python scripts/08_deploy_hf_space.py --repo-id <your-hf-username>/natural-presenter-ai-showcase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalTokenNotFoundError

ROOT_DIR = Path(__file__).resolve().parent.parent
HF_SPACE_DIR = ROOT_DIR / "hf_space"
CONSENT_DOC = ROOT_DIR / "docs" / "VOICE_CONSENT.md"
COAUTHOR_FILES = {"07_multi_speaker_handoff.wav", "08_ege_cadence_demo.wav"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy hf_space/ to a free CPU Hugging Face Gradio Space.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Space ID in the form '<hf-username>/natural-presenter-ai-showcase'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/keep the Space private instead of public.",
    )
    parser.add_argument(
        "--allow-coauthor-consent",
        action="store_true",
        help="Allow uploading co-author audio samples (only after explicit consent).",
    )
    args = parser.parse_args()

    samples_dir = HF_SPACE_DIR / "samples"
    wavs = sorted(samples_dir.glob("*.wav")) if samples_dir.exists() else []
    if not wavs:
        print("ERROR: No .wav files found in hf_space/samples/. Run scripts/07_generate_showcase_samples.py first.")
        sys.exit(1)

    # Voice Consent Guard: require docs/VOICE_CONSENT.md to exist (or --allow-coauthor-consent)
    has_consent_record = CONSENT_DOC.is_file() and "Consented (Co-Creator" in CONSENT_DOC.read_text(encoding="utf-8")
    blocked = [
        p.name
        for p in wavs
        if p.name in COAUTHOR_FILES and not (args.allow_coauthor_consent or has_consent_record)
    ]
    if blocked:
        print(f"BLOCKED by Voice Consent Guard: {blocked}")
        print("Document explicit co-author consent in docs/VOICE_CONSENT.md or pass --allow-coauthor-consent.")
        sys.exit(1)

    api = HfApi()
    try:
        user_info = api.whoami()
    except LocalTokenNotFoundError:
        print("ERROR: Not logged in to Hugging Face.")
        print("Run `hf auth login` (with a Write token from https://huggingface.co/settings/tokens) and retry.")
        sys.exit(1)

    print(f"Authenticated on Hugging Face as: {user_info.get('name', 'unknown')}")
    print(f"Preparing Space: https://huggingface.co/spaces/{args.repo_id} (private={args.private})")
    print(f"Audio samples to upload ({len(wavs)}): {[p.name for p in wavs]}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(HF_SPACE_DIR),
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store", "app.py", "requirements.txt"],
        commit_message="Deploy Natural Presenter AI portfolio showcase (Static Space, consented Binh & Ege benchmarks)",
    )

    print(f"\nDEPLOY_OK: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
