"""Step 8 (Optional): One-command deploy of `hf_space/` to a FREE Hugging Face Static Space.

Static Spaces (`sdk: static`, `index.html` + `samples/*.wav`) are 100% free forever on
Hugging Face without requiring a PRO subscription, load instantly with zero cold start,
and run the Stage 1 playground directly in the visitor's browser.

Strict Voice Consent Guard:
  Allows ONLY the 6 self-recorded author (`vn_01`) clips (`01`–`06`). Refuses to upload
  any unapproved audio file, and automatically purges any stale remote audio clips from
  the Hugging Face Space via `delete_patterns`.

Usage:
    hf auth login
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
ALLOWED_SAMPLES = {
    "01_zeroshot_baseline.wav",
    "02_finetuned_ft001.wav",
    "03_filler_none.wav",
    "04_filler_medium_natural.wav",
    "05_acronym_mse_fixed.wav",
    "06_self_repair_and_slide_pause.wav",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy hf_space/ to a free Hugging Face Static Space.")
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
    args = parser.parse_args()

    samples_dir = HF_SPACE_DIR / "samples"
    wavs = sorted(samples_dir.glob("*.wav")) if samples_dir.exists() else []
    if not wavs:
        print("ERROR: No .wav files found in hf_space/samples/. Run scripts/07_generate_showcase_samples.py first.")
        sys.exit(1)

    # Strict allowlist Voice Consent Guard (vn_01 self-recorded samples only)
    disallowed = [p.name for p in wavs if p.name not in ALLOWED_SAMPLES]
    if disallowed:
        print(f"BLOCKED by Voice Consent Guard — non-allowlisted audio found: {disallowed}")
        print("Remove unconsented clips from hf_space/samples/ before deploying.")
        sys.exit(1)

    api = HfApi()
    try:
        user_info = api.whoami()
    except LocalTokenNotFoundError:
        print("ERROR: Not logged in to Hugging Face.")
        print("Run `hf auth login` and retry.")
        sys.exit(1)

    print(f"Authenticated on Hugging Face as: {user_info.get('name', 'unknown')}")
    print(f"Preparing Space: https://huggingface.co/spaces/{args.repo_id} (private={args.private})")
    print(f"Allowlisted self-recorded audio samples ({len(wavs)}): {[p.name for p in wavs]}")

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
        delete_patterns=["samples/*.wav"],
        commit_message="Deploy Natural Presenter AI portfolio showcase (Static Space, vn_01 self-recorded voice only)",
    )

    remote_wavs = sorted(f for f in api.list_repo_files(repo_id=args.repo_id, repo_type="space") if f.endswith(".wav"))
    print(f"Verified remote Space audio files ({len(remote_wavs)}): {remote_wavs}")
    print(f"\nDEPLOY_OK: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
