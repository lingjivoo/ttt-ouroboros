#!/usr/bin/env python3
"""Create or verify the release's file-level SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_MANIFEST.json"
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "results",
    "tmp",
}


def inventory() -> list[dict[str, object]]:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path == OUTPUT
            or EXCLUDED_PARTS.intersection(path.parts)
            or (path.parent.name == "figures" and path.suffix in {".pdf", ".png"})
        ):
            continue
        data = path.read_bytes()
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = {"schema_version": 1, "files": inventory()}
    if args.check:
        recorded = json.loads(OUTPUT.read_text())
        if recorded != current:
            raise SystemExit("SOURCE_MANIFEST.json is stale")
        print(f"verified {len(current['files'])} files")
        return
    OUTPUT.write_text(json.dumps(current, indent=2) + "\n")
    print(f"wrote {OUTPUT} with {len(current['files'])} files")


if __name__ == "__main__":
    main()
