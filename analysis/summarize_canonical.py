#!/usr/bin/env python3
"""Summarize the paired canonical Closed/Writes-Off/Fixed suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = {"closed": "Closed Loop", "masked": "Writes Off", "open": "Fixed Generation"}


def load(path: Path, expected_mode: str) -> tuple[dict, list[np.ndarray], list[int]]:
    payload = json.loads(path.read_text())
    status = payload.pop("status", None)
    if status != "passed":
        raise ValueError(f"{path}: expected passed result, got {status!r}")
    config = payload.pop("_config")
    if config["mode"] != expected_mode:
        raise ValueError(f"{path}: expected {expected_mode}, got {config['mode']}")
    if not config.get("initial_probe"):
        raise ValueError(f"{path}: missing the prefill-end initial probe")
    rows, positions = [], None
    for key in sorted(payload):
        record = payload[key]
        probes = record.get("probes_book")
        if not probes:
            raise ValueError(f"{path}:{key}: no per-book probes")
        pos = [int(item[0]) for item in probes]
        values = np.asarray([item[1] for item in probes], dtype=float).T
        if positions is None:
            positions = pos
        elif positions != pos:
            raise ValueError(f"{path}:{key}: probe positions differ")
        rows.append(values)
    return config, rows, positions or []


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed", type=Path, required=True)
    parser.add_argument("--writes-off", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--books", default="2,3,4,5,6,7")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selected = [int(x) for x in args.books.split(",")]
    loaded = {
        "closed": load(args.closed, "closed"),
        "masked": load(args.writes_off, "masked"),
        "open": load(args.fixed, "open"),
    }
    reference_cfg = loaded["masked"][0]
    for mode, (cfg, rows, positions) in loaded.items():
        for field in (
            "n_chunks", "n_seqs", "book_offset", "preset", "val", "initial_probe",
            "seeds", "checkpoint_sha256", "validation_sha256", "book_indices", "book_bounds",
        ):
            if field not in cfg or field not in reference_cfg:
                raise ValueError(f"{mode}: signed result is missing {field}")
            if cfg[field] != reference_cfg[field]:
                raise ValueError(f"{mode}: mismatched {field}")
        if len(rows) != 5 or len(positions) != 16:
            raise ValueError(f"{mode}: expected 5 seeds and 16 probes")
    if reference_cfg["n_seqs"] != 8 or reference_cfg["book_offset"] != 0:
        raise ValueError("canonical suite must run physical width 8 over books 0-7")
    if sorted(int(seed) for seed in reference_cfg["seeds"].split(",")) != [1, 2, 3, 7, 42]:
        raise ValueError("canonical suite must use seeds 42,1,7,2,3")
    if any(book < 0 or book >= 8 for book in selected):
        raise ValueError("reported book indices must be physical rows 0-7")

    curves = {}
    changes = {}
    for mode, (_, seed_rows, positions) in loaded.items():
        cube = np.stack(seed_rows)[:, selected, :]
        curves[mode] = cube.mean(axis=(0, 1))
        changes[mode] = (cube[:, :, -1] - cube[:, :, 0]).mean(axis=0)
    rng = np.random.default_rng(20260925)
    rows = []
    for mode in ("masked", "open", "closed"):
        harm = changes[mode] - changes["masked"]
        mean, lo, hi = interval(harm, rng)
        rows.append((MODES[mode], mean, lo, hi))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    machine = {
        "books": selected,
        "seeds": 5,
        "physical_width": 8,
        "probe_positions": loaded["closed"][2],
        "curves": {key: value.tolist() for key, value in curves.items()},
        "harm": {name: {"mean": mean, "ci95": [lo, hi]} for name, mean, lo, hi in rows},
    }
    args.out.with_suffix(".json").write_text(json.dumps(machine, indent=2) + "\n")
    lines = [
        "# Canonical 125M paired summary",
        "",
        "Physical width 8; reported PG-19 books 2–7; five seeds; 16 branch-only probes.",
        "",
        "| Policy | Excess first-to-last NLL H [95% book bootstrap CI] |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {mean:.4f} [{lo:.4f}, {hi:.4f}] |" for name, mean, lo, hi in rows)
    args.out.write_text("\n".join(lines) + "\n")
    print(args.out.read_text(), end="")


if __name__ == "__main__":
    main()
