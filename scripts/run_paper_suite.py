#!/usr/bin/env python3
"""Run an ordered collection of paper manifests.

The smoke mode preserves all policy logic but reduces the canonical language
suite to one seed, one physical row and 16 chunks. It is an integration test,
not a paper result.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

if __package__:
    from .run_config import ROOT, build_command
else:
    from run_config import ROOT, build_command


def load_suite(path: Path) -> list[tuple[Path, dict]]:
    suite = yaml.safe_load(path.read_text())
    if suite.get("schema_version") != 1:
        raise ValueError("suite schema_version must be 1")
    rows = []
    for item in suite.get("runs", []):
        manifest_path = (path.parent / item).resolve()
        manifest = yaml.safe_load(manifest_path.read_text())
        rows.append((manifest_path, manifest))
    if not rows:
        raise ValueError(f"suite has no runs: {path}")
    return rows


def smoke_manifest(manifest: dict, output_root: Path) -> dict:
    out = copy.deepcopy(manifest)
    args = out.setdefault("args", {})
    if out.get("entrypoint") != "scripts/horizon.py":
        raise ValueError(f"smoke overrides are undefined for {out.get('entrypoint')}")
    args.update(
        n_chunks=16,
        n_seqs=1,
        seeds="42",
        book_offset=2,
        initial_probe=True,
        out=str(output_root / f"{out['name']}.json"),
    )
    out["name"] = f"smoke-{out['name']}"
    out.pop("analysis", None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "configs/suites/main_125m.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--smoke-out", type=Path, default=ROOT / "results/smoke")
    args = parser.parse_args()

    failures = []
    for path, original in load_suite(args.suite.resolve()):
        manifest = smoke_manifest(original, args.smoke_out) if args.smoke else original
        command = build_command(manifest)
        unresolved = [part for part in command if "${" in part]
        if unresolved:
            raise SystemExit(f"unresolved environment variables in {path}: {unresolved}")
        metadata = {"manifest": str(path), "name": manifest.get("name"), "command": command}
        print(json.dumps(metadata, indent=2), flush=True)
        if args.dry_run:
            continue
        output = manifest.get("args", {}).get("out")
        if output:
            Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((str(path), exc.returncode))
            if not args.keep_going:
                break

    if failures:
        print(json.dumps({"failed": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
