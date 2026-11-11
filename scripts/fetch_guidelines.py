"""Download guideline PDFs listed in data/guidelines/sources.json and verify checksums.

Usage:
    python scripts/fetch_guidelines.py           # download missing files
    python scripts/fetch_guidelines.py --force   # re-download everything
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.rag.ingestion import load_manifest, sha256_file  # noqa: E402

USER_AGENT = "Mozilla/5.0 (Macintosh; DetoxRAG research prototype)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = ap.parse_args()

    s = get_settings()
    s.guidelines_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for doc in load_manifest(s.sources_manifest):
        dest = s.guidelines_dir / doc.file
        if dest.exists() and not args.force:
            print(f"[skip] {doc.document_id}: {dest.name} already present")
        else:
            print(f"[get ] {doc.document_id}: {doc.source_url}")
            req = urllib.request.Request(doc.source_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as f:
                f.write(resp.read())
        actual = sha256_file(dest)
        if doc.sha256 and actual != doc.sha256:
            ok = False
            print(
                f"[FAIL] {doc.document_id}: checksum mismatch\n"
                f"       expected {doc.sha256}\n       actual   {actual}\n"
                "       The publisher may have updated the document. Review it, then update sources.json."
            )
        else:
            print(f"[ ok ] {doc.document_id}: sha256 {actual[:16]}…")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
