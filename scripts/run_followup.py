"""Reuse SFT and train fixed policy variants before evaluating the ID test.

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
    parser.add_argument("--diversity-config", type=Path, help="Optional controlled reward ablation, trained before any ID test")
    args = parser.parse_args()
    run, adapter, config = args.run.resolve(), args.sft_adapter.resolve(), args.config.resolve()
    if run.exists():
        parser.error("Choose a new run directory")
    if not (adapter / "adapter_config.json").exists():
        parser.error("SFT adapter not found")
    if args.diversity_config:
        args.diversity_config = args.diversity_config.resolve()
        expected = json.loads(config.read_text())
        expected["reward"] = "diverse"
        if json.loads(args.diversity_config.read_text()) != expected:
            parser.error("The diversity config must differ only in reward='diverse'")
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
    policies = {"sft": None, "grpo": rl}
    if args.diversity_config:
        diverse = run / "diverse/adapter"
        policies["diverse"] = diverse
        stages.append(("diverse", ["rl", "--sft-adapter", str(adapter), "--output", str(run / "diverse")]))
        for name, extra in (("diverse-eval", []), ("diverse-sampled", ["--best-of", "4"])):
            stages.append((name, ["evaluate", "--sft-adapter", str(adapter), "--rl-adapter", str(diverse),
                                  "--limit", "0", "--output", str(run / name)] + extra))
    for policy, rl_adapter in policies.items():
        for mode, sampling in (("greedy", []), ("sampled", ["--best-of", "4"])):
            name = f"test-id-{policy}-{mode}"
            command = ["evaluate", "--sft-adapter", str(adapter), "--split", "test_id", "--limit", "0",
                       "--output", str(run / name)] + sampling
            if rl_adapter:
                command.extend(["--rl-adapter", str(rl_adapter)])
            stages.append((name, command))
    def command_for(name, command):
        prefix = list(base)
        if name == "diverse":
            prefix[prefix.index("--config") + 1] = str(args.diversity_config)
        return prefix + command
    manifest = {"started_unix": time.time(), "config": json.loads(config.read_text()),
                "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root / "melody_rl").glob("*.py"))},
                "sft_adapter_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in adapter.glob("*.safetensors")},
                "diversity_config": json.loads(args.diversity_config.read_text()) if args.diversity_config else None,
                "commands": {name: command_for(name, command) for name, command in stages}}
    (run / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, command in stages:
        print(f"Running {name}", flush=True)
        with (run / "logs" / f"{name}.log").open("w") as handle:
            result = subprocess.run(command_for(name, command), cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{name} failed; inspect its log. No subsequent stages ran.")
    comparisons = [("validation-greedy", "sft-eval", "grpo-eval"),
                                  ("validation-sampled", "bestof4", "grpo-sampled"),
                                  ("test-id-greedy", "test-id-sft-greedy", "test-id-grpo-greedy"),
                                  ("test-id-sampled", "test-id-sft-sampled", "test-id-grpo-sampled")]
    if args.diversity_config:
        comparisons.extend([
            ("validation-diverse-greedy", "sft-eval", "diverse-eval"),
            ("validation-diverse-sampled", "bestof4", "diverse-sampled"),
            ("test-id-diverse-greedy", "test-id-sft-greedy", "test-id-diverse-greedy"),
            ("test-id-diverse-sampled", "test-id-sft-sampled", "test-id-diverse-sampled"),
            ("validation-reward-ablation", "grpo-sampled", "diverse-sampled"),
            ("test-id-reward-ablation", "test-id-grpo-sampled", "test-id-diverse-sampled"),
        ])
    for label, before, after in comparisons:
        subprocess.run([sys.executable, str(root / "scripts/compare_evaluations.py"),
                        "--before", str(run / before / "predictions.jsonl"),
                        "--after", str(run / after / "predictions.jsonl"),
                        "--output", str(run / f"{label}-comparison.json")], check=True, cwd=root)
    subprocess.run([sys.executable, str(root / "scripts/build_report.py"), "--run", str(run),
                    "--output", str(run / "report.html")], check=True, cwd=root)


if __name__ == "__main__":
    main()
