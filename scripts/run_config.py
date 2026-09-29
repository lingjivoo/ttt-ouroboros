#!/usr/bin/env python3
"""Run one versioned experiment manifest.

The manifest is intentionally a thin, inspectable layer over an experiment's
argparse interface. It records the exact entry point and arguments without
creating a second implementation of the protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(item) for item in value]
    return value


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_command(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    entrypoint = ROOT / manifest["entrypoint"]
    if not entrypoint.is_file() or ROOT not in entrypoint.resolve().parents:
        raise ValueError(f"invalid entrypoint: {entrypoint}")
    command = [sys.executable, str(entrypoint)]
    for name, raw in manifest.get("args", {}).items():
        value = expand(raw)
        if value is False or value is None:
            continue
        command.append(flag(name))
        if value is True:
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        command.append(str(value))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="print command metadata as JSON")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    command = build_command(manifest)
    env = os.environ.copy()
    env.update({key: str(expand(value)) for key, value in manifest.get("environment", {}).items()})
    unresolved = [token for token in command if "${" in token]
    if unresolved:
        raise SystemExit(f"unresolved environment variable in: {unresolved}")

    metadata = {
        "name": manifest.get("name", args.manifest.stem),
        "manifest": str(args.manifest.resolve()),
        "command": command,
        "cwd": str(ROOT),
    }
    if args.json:
        print(json.dumps(metadata, indent=2))
    else:
        print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
