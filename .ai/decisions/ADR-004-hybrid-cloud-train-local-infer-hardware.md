# ADR-004: Hybrid Hardware Strategy (Cloud One-Time Training + Local RTX 5060 Inference)

- **Status:** Accepted
- **Date:** 2026-11-23

## Context
The primary development machine is a Windows laptop with an **NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, Blackwell `sm_120`)**. While 8GB VRAM is fast for daily inference and Whisper dataset prep, multi-hour TTS fine-tuning on an 8GB laptop risks thermal throttling and out-of-memory limits.

## Decision
1. Run **dataset preprocessing (`scripts/01..04`) and daily inference (`app/web_ui.py`) 100% locally** on the RTX 5060 Laptop (free, private, fast).
2. Rent a **cloud RTX 4090 (24GB VRAM)** on RunPod / Vast.ai (~$0.35–$0.50/hr) for one-time per-speaker LoRA training (~1–2 hours, ~$1–$2 per speaker), then download the lightweight adapter locally and terminate the cloud instance.

## Consequences
- Near-zero ongoing operational cost.
- Fast, reliable training without thermal wear on the laptop.
- Local Python environment must use PyTorch >= 2.7 with CUDA 12.8+ for RTX 5060 Blackwell support.
