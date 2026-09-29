from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_config import ROOT, build_command
from scripts.run_paper_suite import load_suite, smoke_manifest


def test_main_suite_and_smoke_manifests(monkeypatch, tmp_path):
    monkeypatch.setenv("TTT_CKPT", "/checkpoints")
    monkeypatch.setenv("TTT_DATA", "/data")
    monkeypatch.setenv("TTT_OUT", "/results")
    rows = load_suite(ROOT / "configs/suites/main_125m.yaml")
    assert [row[1]["args"]["mode"] for row in rows] == ["masked", "closed", "open"]
    for _, manifest in rows:
        assert manifest["args"]["n_seqs"] == 8
        assert manifest["args"]["book_offset"] == 0
        assert manifest["args"]["initial_probe"] is True
        assert manifest["analysis"]["report_book_indices"] == [2, 3, 4, 5, 6, 7]
        smoke = smoke_manifest(manifest, tmp_path)
        command = build_command(smoke)
        assert "--n-chunks" in command and command[command.index("--n-chunks") + 1] == "16"
        assert "--n-seqs" in command and command[command.index("--n-seqs") + 1] == "1"


def _result(path: Path, mode: str, delta: float):
    config = {
        "mode": mode,
        "n_chunks": 128,
        "n_seqs": 8,
        "book_offset": 0,
        "preset": "125m-e2e-ext32k",
        "val": "/data/pg19/val.npy",
        "initial_probe": True,
        "seeds": "42,1,7,2,3",
        "checkpoint_sha256": "checkpoint",
        "validation_sha256": "validation",
        "book_indices": list(range(8)),
        "book_bounds": [[book * 100, (book + 1) * 100] for book in range(8)],
    }
    payload = {"_config": config, "status": "passed"}
    for seed in (1, 2, 3, 7, 42):
        probes = []
        for probe in range(16):
            values = [3.0 + 0.01 * book + delta * probe / 15 for book in range(8)]
            probes.append([(probe + 1) * 8, values])
        payload[f"{mode}_s{seed}"] = {"probes_book": probes}
    path.write_text(json.dumps(payload))


def test_canonical_summary_selects_six_reported_books(tmp_path):
    closed, off, fixed = (tmp_path / name for name in ("closed.json", "off.json", "fixed.json"))
    _result(closed, "closed", 2.0)
    _result(off, "masked", 0.0)
    _result(fixed, "open", 0.1)
    out = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis/summarize_canonical.py"),
            "--closed", str(closed),
            "--writes-off", str(off),
            "--fixed", str(fixed),
            "--out", str(out),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(out.with_suffix(".json").read_text())
    assert result["books"] == [2, 3, 4, 5, 6, 7]
    assert result["physical_width"] == 8
    assert abs(result["harm"]["Closed Loop"]["mean"] - 2.0) < 1e-12
    assert abs(result["harm"]["Fixed Generation"]["mean"] - 0.1) < 1e-12
