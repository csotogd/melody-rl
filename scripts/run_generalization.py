"""Evaluate frozen policies on both OOD splits without training or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melody_rl.analysis import paired_comparison, task_metrics
from melody_rl.data import read_rows
from scripts.build_report import build

SPLITS = {"test_composition": 54, "test_length": 162}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_adapters(root, audit):
    """Fail before any evaluation if an adapter changed since policy freeze."""
    for files in audit["frozen_adapter_sha256"].values():
        for relative, expected in files.items():
            if sha256(root / relative) != expected:
                raise ValueError(f"Frozen adapter checksum mismatch: {relative}")


def summarize_policy(greedy, sampled):
    metrics = [task_metrics(row) for row in sampled]
    candidates = [c for row in sampled for c in row["candidates"]]
    repeats = [m["candidate_repetition"] for m in metrics if m["candidate_repetition"] is not None]
    return {
        "tasks": len(greedy),
        "greedy_successes": sum(r["metrics"]["all_satisfied"] for r in greedy),
        "sampled_successes": sum(c["metrics"]["all_satisfied"] for c in candidates),
        "sampled_completions": len(candidates),
        "sampled_success_rate": sum(m["candidate_success"] for m in metrics) / len(metrics),
        "sampled_repetition_mean_per_task": sum(repeats) / len(repeats) if repeats else None,
        "sampled_validity": sum(c["metrics"]["valid"] for c in candidates) / len(candidates),
        "sampled_truncated": sum(c["truncated"] for c in candidates),
        "best_of_four_successes": sum(r["metrics"]["all_satisfied"] for r in sampled),
        "sampled_constraint_pass_rates": {
            k: sum(c["metrics"]["checks"][k] for c in candidates) / len(candidates)
            for k in ("bars", "meter", "key", "range", "final", "leap")},
        "greedy_constraint_failures": {
            k: sum(not r["metrics"]["checks"][k] for r in greedy)
            for k in ("bars", "meter", "key", "range", "final", "leap")},
    }


def export_split(run, destination, split):
    prefix = split.replace("_", "-")
    loaded, summary = {}, {}
    for policy in ("sft", "grpo", "diverse"):
        loaded[policy] = [read_rows(run / f"{prefix}-{policy}-{mode}/predictions.jsonl")
                          for mode in ("greedy", "sampled")]
        greedy, sampled = loaded[policy]
        assert len(greedy) == len(sampled) == SPLITS[split]
        assert [r["id"] for r in greedy] == [r["id"] for r in sampled]
        assert all(len(r["candidates"]) == 1 for r in greedy)
        assert all(len(r["candidates"]) == 4 for r in sampled)
        summary[policy] = summarize_policy(greedy, sampled)
    for policy, label in (("grpo", ""), ("diverse", "diverse-")):
        for index, mode in enumerate(("greedy", "sampled")):
            comparison = paired_comparison(loaded["sft"][index], loaded[policy][index])
            (run / f"{prefix}-{label}{mode}-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    comparison = paired_comparison(loaded["grpo"][1], loaded["diverse"][1])
    (run / f"{prefix}-reward-ablation-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    rules = json.loads((run / f"{prefix}-rules/summary.json").read_text())
    summary["rules"] = {"tasks": rules["n"], "successes": rules["successes"]}
    build(run, destination / f"generalization-{split.removeprefix('test_')}.html", split)
    return summary


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=root / "runs/extended-study/data")
    parser.add_argument("--config", type=Path, default=root / "configs/extended-study.json")
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--graded-adapter", type=Path, required=True)
    parser.add_argument("--diverse-adapter", type=Path, required=True)
    parser.add_argument("--frozen-audit", type=Path)
    args = parser.parse_args()
    run, destination = args.run.resolve(), args.output.resolve()
    if run.exists():
        parser.error("Choose a new run directory; interrupted evaluations are not overwritten")
    config = json.loads(args.config.read_text())
    if config["reward"] != "graded":
        parser.error("All policies must use the same graded selection criterion")
    adapters = {"sft": args.sft_adapter.resolve(), "grpo": args.graded_adapter.resolve(),
                "diverse": args.diverse_adapter.resolve()}
    hashes = {name: {p.name: sha256(p) for p in folder.glob('*.safetensors')}
              for name, folder in adapters.items()}
    if not all(hashes.values()):
        parser.error("Every adapter must contain saved weights")
    if args.frozen_audit:
        audit = json.loads(args.frozen_audit.read_text())
        check_frozen_adapters(root, audit)
        for policy, name in (("sft", "sft"), ("grpo", "graded"), ("diverse", "diverse")):
            if set(hashes[policy].values()) != set(audit['frozen_adapter_sha256'][name].values()):
                parser.error("Provided adapters do not match the frozen audit")
    data = args.data.resolve()
    train_ids = {r['id'] for r in read_rows(data / 'train.jsonl')}
    seen = set(train_ids)
    for split, count in SPLITS.items():
        rows = read_rows(data / f'{split}.jsonl')
        ids = {r['id'] for r in rows}
        if len(rows) != count or len(ids) != count or ids & seen:
            parser.error(f"Unexpected or overlapping tasks in {split}")
        seen.update(ids)
    run.mkdir(parents=True)
    (run / 'logs').mkdir()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'started_unix': time.time(), 'config': config, 'adapter_sha256': hashes,
                'source_sha256': {str(p.relative_to(root)): sha256(p) for p in sorted((root/'melody_rl').glob('*.py'))},
                'data_sha256': {split: sha256(data / f'{split}.jsonl') for split in SPLITS},
                'commands': [], 'note': 'Evaluation only. No training, tuning or checkpoint selection. All OOD tests are consumed after this run.'}
    result = {'protocol': 'results/diagnosis/generalization-protocol.json', 'paid_compute_eur': 0,
              'note': 'One training seed; paired task-bootstrap intervals do not measure training variability. Both OOD endpoints are reported without multiplicity adjustment.', 'splits': {}}
    for split in SPLITS:
        prefix = split.replace('_', '-')
        for policy, mode, extras in [('rules', 'rules', ['--rules'])] + [
                (policy, mode, ['--sft-adapter', str(adapters['sft'])] +
                 ([] if policy == 'sft' else ['--rl-adapter', str(adapters[policy])]) +
                 ([] if mode == 'greedy' else ['--best-of', '4']))
                for policy in adapters for mode in ('greedy', 'sampled')]:
            name = f'{prefix}-rules' if policy == 'rules' else f'{prefix}-{policy}-{mode}'
            command = [sys.executable, '-m', 'melody_rl.cli', '--config', str(args.config.resolve()),
                       '--data', str(data), 'evaluate', '--split', split, '--limit', '0',
                       '--output', str(run / name)] + extras
            manifest['commands'].append([part.replace(str(root) + '/', '') for part in command])
            (run/'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(f'Running {name}', flush=True)
            with (run/'logs'/f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        result['splits'][split] = export_split(run, destination, split)
        (destination/'generalization-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    after = {name: {p.name: sha256(p) for p in folder.glob('*.safetensors')} for name, folder in adapters.items()}
    if after != hashes:
        raise RuntimeError('Adapter changed during evaluation')
    manifest['completed_unix'] = time.time()
    manifest['adapter_hashes_unchanged_after_evaluation'] = True
    (destination/'generalization-experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Completed both OOD splits; adapters unchanged.', flush=True)


if __name__ == '__main__':
    main()
