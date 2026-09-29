from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.run_config import ROOT, build_command


def test_all_manifests_build(monkeypatch):
    monkeypatch.setenv("TTT_CKPT", "/artifacts/checkpoints")
    monkeypatch.setenv("TTT_DATA", "/artifacts/data")
    monkeypatch.setenv("TTT_OUT", "/artifacts/results")
    monkeypatch.setenv("ALFWORLD_DATA", "/artifacts/alfworld")
    monkeypatch.setenv("AGENTBENCH_ROOT", "/opt/AgentBench-v0.2")
    manifests = sorted((ROOT / "configs").glob("*.yaml"))
    assert manifests
    for path in manifests:
        command = build_command(yaml.safe_load(path.read_text()))
        assert command[0]
        assert Path(command[1]).is_file()
        assert not any("${" in token for token in command)


def test_result_schema_is_valid_json():
    schema = json.loads((ROOT / "schemas/result.schema.json").read_text())
    assert schema["required"] == ["status"]
    assert set(schema["properties"]["status"]["enum"]) == {"running", "complete", "error"}
