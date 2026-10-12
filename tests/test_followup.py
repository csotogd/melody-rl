"""Protect the held-out test gate and isolate the reward-ablation variable."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_followup


def prepare(tmp_path, monkeypatch, diverse_changes=None):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    base = {"reward": "graded", "rl_steps": 128, "rl_lr": 5e-5}
    config = tmp_path / "base.json"
    config.write_text(json.dumps(base))
    diverse = tmp_path / "diverse.json"
    diverse.write_text(json.dumps({**base, "reward": "diverse", **(diverse_changes or {})}))
    run = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_followup", "--run", str(run), "--sft-adapter", str(adapter),
                                      "--config", str(config), "--diversity-config", str(diverse)])
    return run, config, diverse


def test_all_policies_frozen_before_test_and_shared_selector(tmp_path, monkeypatch):
    run, config, diverse = prepare(tmp_path, monkeypatch)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_followup.subprocess, "run", fake_run)
    run_followup.main()
    training = [i for i, c in enumerate(commands) if "rl" in c]
    testing = [i for i, c in enumerate(commands) if "--split" in c and "test_id" in c]
    validation = [i for i, c in enumerate(commands) if "evaluate" in c and "--split" not in c]
    assert len(training) == 2 and len(testing) == 6
    assert min(testing) > max(training + validation)
    for c in commands:
        if "evaluate" in c:
            assert c[c.index("--config") + 1] == str(config)
    train_configs = [commands[i][commands[i].index("--config") + 1] for i in training]
    assert train_configs == [str(config), str(diverse)]
    assert (run / "experiment.json").exists()


def test_ablation_cannot_silently_change_learning_rate(tmp_path, monkeypatch):
    run, _, _ = prepare(tmp_path, monkeypatch, {"rl_lr": .001})
    with pytest.raises(SystemExit):
        run_followup.main()
    assert not run.exists()
