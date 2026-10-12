import argparse
import json
import os
from pathlib import Path


def main():
    # All downloads remain inside the project; no inference API keys are needed.
    root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description="Local verifiable-reward music post-training")
    parser.add_argument("--config", type=Path, default=root / "configs" / "pilot.json")
    parser.add_argument("--data", type=Path, default=root / "data")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data")
    sub.add_parser("doctor")
    for command in ("sft", "rl", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command != "sft":
            p.add_argument("--sft-adapter", type=Path, required=command == "rl")
        if command in ("sft", "rl"):
            p.add_argument("--steps", type=int)
        else:
            p.add_argument("--rl-adapter", type=Path)
            p.add_argument("--split", choices=("validation", "test_id", "test_composition", "test_length"), default="validation")
            p.add_argument("--limit", type=int, help="0 means full split")
            p.add_argument("--rules", action="store_true")
            p.add_argument("--best-of", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "data":
        from .data import write_data
        result = write_data(args.data, config.get("dataset_seed", 42))
    elif args.command == "doctor":
        import torch
        from .model import choose_device
        result = {"torch": torch.__version__, "mps_built": torch.backends.mps.is_built(),
                  "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
                  "selected_device": str(choose_device(config["device"])), "cloud_spend_eur": 0,
                  "model": config["model"], "revision": config["revision"]}
    elif args.command == "sft":
        from .train import train_sft
        if args.steps is not None:
            config["sft_steps"] = args.steps
        result = train_sft(config, args.data, args.output)
    elif args.command == "rl":
        from .train import train_rl
        if args.steps is not None:
            config["rl_steps"] = args.steps
        result = train_rl(config, args.data, args.sft_adapter, args.output)
    else:
        from .evaluate import evaluate
        if args.rl_adapter and not args.sft_adapter:
            parser.error("--rl-adapter requires its original --sft-adapter")
        result = evaluate(config, args.data, args.output, split=args.split, sft_adapter=args.sft_adapter,
                          rl_adapter=args.rl_adapter, rule_based=args.rules, best_of=args.best_of, limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
