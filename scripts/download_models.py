"""Download the two local Hugging Face models into ./models (one-time, ~3.3 GB).

Usage:
    python scripts/download_models.py

After this, the application runs fully offline (HF_HUB_OFFLINE=1).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "Qwen/Qwen3-4B-MLX-4bit": ROOT / "models" / "qwen3-4b-mlx-4bit",
    "Qwen/Qwen3-Embedding-0.6B": ROOT / "models" / "qwen3-embedding-0.6b",
}


def main() -> int:
    os.environ["HF_HUB_OFFLINE"] = "0"  # downloading requires network
    from huggingface_hub import snapshot_download

    for repo, dest in MODELS.items():
        if (dest / "config.json").exists() and any(dest.glob("*.safetensors")):
            print(f"[skip] {repo} already in {dest.relative_to(ROOT)}")
            continue
        print(f"[get ] {repo} -> {dest.relative_to(ROOT)}")
        snapshot_download(repo_id=repo, local_dir=str(dest))
    print("Done. Verify with: python scripts/check_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
