"""Execute the fixed follow-up: reuse SFT, train once, evaluate validation and ID test.

Every stage runs in a separate process. No score-dependent branching or checkpoint
selection. Existing runs are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/extended-study.json")
    args = parser.parse_args()
    run, adapter, config = args.run.resolve(), args.sft_adapter.resolve(), args.config.resolve()
    if run.exists():
        parser.error("Choose a new run directory")
    if not (adapter / "adapter_config.json").exists():
        parser.error("SFT adapter not found")
    run.mkdir(parents=True)
    (run / "logs").mkdir()
    base = [sys.executable, "-m", "melody_rl.cli", "--config", str(config), "--data", str(run / "data")]
    rl = run / "grpo/adapter"
    stages = [("data", ["data"])]
    for name, extra in (("rules", ["--rules"]), ("sft-eval", []), ("bestof4", ["--best-of", "4"])):
        stages.append((name, ["evaluate", "--sft-adapter", str(adapter), "--limit", "0",
                              "--output", str(run / name)] + extra))
    stages.append(("grpo", ["rl", "--sft-adapter", str(adapter), "--output", str(run / "grpo")]))
    for name, extra in (("grpo-eval", []), ("grpo-sampled", ["--best-of", "4"])):
        stages.append((name, ["evaluate", "--sft-adapter", str(adapter), "--rl-adapter", str(rl),
                              "--limit", "0", "--output", str(run / name)] + extra))
    for policy in ("sft", "grpo"):
        for mode, sampling in (("greedy", []), ("sampled", ["--best-of", "4"])):
            name = f"test-id-{policy}-{mode}"
            command = ["evaluate", "--sft-adapter", str(adapter), "--split", "test_id", "--limit", "0",
                       "--output", str(run / name)] + sampling
            if policy == "grpo":
                command.extend(["--rl-adapter", str(rl)])
            stages.append((name, command))
    manifest = {"started_unix": time.time(), "config": json.loads(config.read_text()),
                "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root / "melody_rl").glob("*.py"))},
                "sft_adapter_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in adapter.glob("*.safetensors")},
                "commands": {name: base + command for name, command in stages}}
    (run / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, command in stages:
        print(f"Running {name}", flush=True)
        with (run / "logs" / f"{name}.log").open("w") as handle:
            result = subprocess.run(base + command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{name} failed; inspect its log. No subsequent stages ran.")
    for label, before, after in (("validation-greedy", "sft-eval", "grpo-eval"),
                                  ("validation-sampled", "bestof4", "grpo-sampled"),
                                  ("test-id-greedy", "test-id-sft-greedy", "test-id-grpo-greedy"),
                                  ("test-id-sampled", "test-id-sft-sampled", "test-id-grpo-sampled")):
        subprocess.run([sys.executable, str(root / "scripts/compare_evaluations.py"),
                        "--before", str(run / before / "predictions.jsonl"),
                        "--after", str(run / after / "predictions.jsonl"),
                        "--output", str(run / f"{label}-comparison.json")], check=True, cwd=root)
    subprocess.run([sys.executable, str(root / "scripts/build_report.py"), "--run", str(run),
                    "--output", str(run / "report.html")], check=True, cwd=root)


if __name__ == "__main__":
    main()
