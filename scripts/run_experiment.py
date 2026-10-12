"""Run a complete experiment in separate processes, releasing model memory each time.

No cluster, scheduler or paid service. Existing output directories are never reused.
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
    parser.add_argument("--config", type=Path, default=root / "configs" / "pilot.json")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--best-of", type=int, default=0, help="Optional SFT selection baseline; 0 skips")
    args = parser.parse_args()
    if args.best_of < 0:
        parser.error("--best-of must be nonnegative")
    run = args.run.resolve()
    if run.exists():
        parser.error("Choose a new run directory; existing runs will not be overwritten")
    run.mkdir(parents=True)
    logdir = run / "logs"
    logdir.mkdir()
    config = args.config.resolve()
    data = run / "data"
    base = [sys.executable, "-m", "melody_rl.cli", "--config", str(config), "--data", str(data)]
    sft = run / "sft" / "adapter"
    rl = run / "grpo" / "adapter"
    stages = [
        ("data", ["data"]),
        ("base", ["evaluate", "--output", str(run / "base")]),
        ("rules", ["evaluate", "--rules", "--output", str(run / "rules")]),
        ("sft", ["sft", "--output", str(run / "sft")]),
        ("sft-eval", ["evaluate", "--sft-adapter", str(sft), "--output", str(run / "sft-eval")]),
        ("grpo", ["rl", "--sft-adapter", str(sft), "--output", str(run / "grpo")]),
        ("grpo-eval", ["evaluate", "--sft-adapter", str(sft), "--rl-adapter", str(rl), "--output", str(run / "grpo-eval")]),
    ]
    if args.best_of:
        stages.append((f"bestof{args.best_of}", ["evaluate", "--sft-adapter", str(sft),
                       "--best-of", str(args.best_of), "--output", str(run / f"bestof{args.best_of}")]))
    manifest = {"config": json.loads(config.read_text()), "started_unix": time.time(),
                "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root / "melody_rl").glob("*.py"))},
                "commands": {name: base + command for name, command in stages}}
    (run / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, command in stages:
        print(f"Running {name}; log: {logdir / (name + '.log')}", flush=True)
        with (logdir / f"{name}.log").open("w") as handle:
            result = subprocess.run(base + command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"Stage {name} failed. Inspect its log. No subsequent stages were run.")
    subprocess.run([sys.executable, str(root / "scripts" / "build_report.py"), "--run", str(run),
                    "--output", str(run / "report.html")], check=True, cwd=root)


if __name__ == "__main__":
    main()
