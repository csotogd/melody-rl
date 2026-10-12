# melody-rl

Local-first research on **verifiable-reward post-training for symbolic music**.
The first experiment uses Qwen3-0.6B, LoRA, supervised fine-tuning (SFT), and an
explicit GRPO implementation. No paid APIs, cloud accounts, or robot hardware.

**Research question:** can group-relative RL improve compositional instruction
following while avoiding repetitive solutions that exploit the reward?

This is a small, auditable research project, not a reproduction of a published
leaderboard or a full audio-generation system. The current score language contains
quarter-note melodies only. Compliance, diversity proxies, and musical quality
are different things.

## What is implemented

- A strict score parser and six independently reported constraint checks.
- Procedural SFT demonstrations and deterministic, disjoint task splits.
- Base, rule-based, SFT, GRPO, and best-of-N evaluation paths.
- LoRA training on a single MPS, CPU, or CUDA device.
- On-policy GRPO with group advantages, clipped ratios, and an SFT-reference KL.
- Raw rollout logs, zero-variance diagnostics, frozen model revision and environment lock.
- MIDI and WAV previews for valid outputs, plus a shareable HTML report.
- Tests for reward exploits, held-out splits, and objective gradient behavior.

**Distributed execution is not implemented or validated.** Model inference,
verification, data construction, and the objective are separate modules so future
rollout workers can reuse the same task contract. A multi-GPU configuration alone
would not constitute evidence of distributed training.

## Quick start

Python 3.12+ is required. On Apple Silicon, run from a normal local terminal so
PyTorch can access Metal; restricted environments can report MPS unavailable.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m melody_rl.cli doctor
python -m pytest -q
python -m melody_rl.cli data
```

`requirements-lock.txt` records the exact first-pilot environment. It is a snapshot
of an Apple Silicon/Python 3.12 environment, not a guarantee for every platform.
Model files download to `.cache/huggingface/` in this repository. The pilot model
download is approximately 1.5 GB; allow additional space for dependencies. No API
key is needed for this public model. After the first download,
`HF_HUB_OFFLINE=1` prevents network access from the Hugging Face loader.

Run each stage in a separate process to release GPU memory between stages:

```bash
python -m melody_rl.cli evaluate --output runs/pilot/base
python -m melody_rl.cli evaluate --rules --output runs/pilot/rules
python -m melody_rl.cli sft --output runs/pilot/sft
python -m melody_rl.cli evaluate --sft-adapter runs/pilot/sft/adapter --output runs/pilot/sft-eval
python -m melody_rl.cli rl --sft-adapter runs/pilot/sft/adapter --output runs/pilot/grpo
python -m melody_rl.cli evaluate --sft-adapter runs/pilot/sft/adapter --rl-adapter runs/pilot/grpo/adapter --output runs/pilot/grpo-eval
python scripts/build_report.py --run runs/pilot --output results/pilot.html
```

Existing run directories are protected against overwriting. Use another run name
to repeat an experiment. Training saves final adapters; interrupted training is
currently restarted, not resumed with optimizer state. A saved GRPO adapter must
be loaded on top of **its original merged SFT adapter**, as shown above.

Defaults are in `configs/pilot.json`: 24 SFT updates, 4 RL groups with four samples
each, 96 generated tokens maximum, and four validation tasks. These defaults are
a feasibility pilot, **not enough to establish an improvement**. Equal-reward
groups are logged and skipped; the reported update count may be lower than the
requested number of groups.

For automatic sequential execution with separate logs and a source-hash manifest:

```bash
python scripts/run_experiment.py --config configs/pilot.json --run runs/new-pilot
```

`configs/local-study.json` expands this to two passes over the 174 SFT examples
(348 updates), 32 RL groups, and 24 validation tasks. `--best-of 4` adds the
sampling-and-selection baseline. It is still a single-seed development study;
the test splits are not used by this runner.

## Task and reward

Example output (custom format, **not ABC**):

```text
C4 D4 E4 | G4 E4 C4
```

Each pitch is a quarter note. `|` separates bars; rests, chords, prose, code fences,
and trailing separators are invalid. Pitch spelling uses sharps only (A# is the
same pitch class as Bb). Requested constraints are:

1. Number of bars.
2. Notes per bar (3/4 or 4/4).
3. Membership in a major scale, assessed on notes rather than a claimed key label.
4. Inclusive pitch range.
5. Exact final pitch.
6. Maximum adjacent pitch interval in semitones.

Invalid or token-truncated outputs receive zero reward. Three reward configurations:

| Variant | Reward |
| --- | --- |
| `binary` | 1 only when every constraint holds |
| `graded` | 0.75 × fraction of constraints satisfied + 0.25 × all-satisfied |
| `diverse` | 0.9 × graded + 0.1 × all-satisfied × diversity proxy |

The diversity proxy combines up to three distinct pitches and avoidance of
consecutive repetition. It is deliberately simple, exploitable, and **not a
measure of musical quality**. A constant tonic can perfectly satisfy the other
constraints; an explicit test preserves this known counterexample.

## GRPO details

For each task, sample G completions from the current policy. Standardize rewards
within the group using population standard deviation plus epsilon. Optimize the
mean of each sequence's mean token loss, including EOS and excluding the prompt:

```text
A_i = (r_i - mean(r)) / (std_population(r) + epsilon)
ratio_t = exp(log pi_theta_t - log pi_old_t)
KL_t = exp(log pi_ref_t - log pi_theta_t) - (log pi_ref_t - log pi_theta_t) - 1
loss_i = mean_t[-min(ratio_t*A_i, clip(ratio_t,1-e,1+e)*A_i) + beta*KL_t]
```

This is a baseline GRPO objective, not Dr. GRPO or a claim to the best current
algorithm. Sampling uses temperature 1, top-p 1, top-k 0 and no repetition penalty
so recomputed log-probabilities describe the behavior distribution. No dropout is
used. Old and reference log-probabilities are detached. The SFT adapter is merged
into the frozen base; a fresh RL adapter is trained, and disabling it gives the
exact SFT reference without another resident base model.

Each group makes one optimizer update. Ratios therefore start at approximately
one and clipping is generally inactive in this pilot; the code does not claim
that it studied multi-epoch policy drift. Equal-reward groups are skipped and
reported, including all-zero groups.

## Evaluation protocol

The split is by the **full tuple of constraints**, not just different random seeds:

| Split | Tasks | Purpose |
| --- | ---: | --- |
| train | 174 | SFT teacher targets and RL prompts |
| validation | 48 | Development and checkpoint decisions |
| test_id | 48 | Unseen tuples from the training families |
| test_composition | 54 | G major combined with 4/4, withheld from training |
| test_length | 162 | Three-bar melodies, absent from training |

Test JSONL files contain no teacher targets. Do not tune on test results. The
pilot evaluates only four validation tasks; the test sets remain reserved.
`dataset_seed` fixes all task splits independently of the training `seed`. Keep
it fixed when comparing training seeds or algorithms.

Report all-satisfied rate, individual constraints, validity, truncation, repetition
among valid outputs, unique-output fraction, generated tokens, and latency.
Wilson intervals describe binomial uncertainty in the small sample; multiple
training seeds are still required for research conclusions. Neither the evaluator
nor the reward uses an LLM judge.

The rule-based backward sampler satisfies every feasible task by construction.
It is a mandatory baseline, not model output. If exact compliance is the only
product requirement, that baseline is the right tool. The neural research question
concerns learning, generalization, and the limitations of reward optimization.

Best-of-N selects using the same training reward and logs all candidates:

```bash
python -m melody_rl.cli evaluate --sft-adapter runs/pilot/sft/adapter --best-of 4 --output runs/pilot/bestof4
```

Compare its extra generated tokens and latency, not just its success rate.

## Next experiments

1. Measure the local feasibility pilot and audit raw trajectories.
2. Establish a stable SFT baseline on validation before increasing RL steps.
3. Compare binary, graded, and diversity-aware rewards at equal sampling budgets.
4. Compare RL against extra SFT and best-of-N at documented compute budgets.
5. Run multiple seeds, freeze decisions, then evaluate held-out splits once.
6. Add richer rhythm, phrasing, and a separately evaluated musicality criterion.

No claim of novel methodology is made at this stage. Novelty requires checking the
literature and a result beyond the reference methods.

## Sources and attribution

- [MusicRLVR (2026)](https://arxiv.org/abs/2609.23665): inspiration for verifiable
  musical constraints. This project uses a smaller custom task and is not an
  implementation of MusicConstraintBench; no benchmark results are comparable.
- [NotaGen (2025)](https://arxiv.org/abs/2502.18008) and
  [code](https://github.com/ElectricAlexis/NotaGen): related symbolic music
  post-training work; its weights and datasets are not used in this pilot.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): pretrained model,
  Apache-2.0. Exact revision is pinned in the configuration.
- [Hugging Face GRPO documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
  and [PEFT](https://github.com/huggingface/peft): algorithm and adapter references.

Original code is MIT licensed. Procedural tasks contain no scraped compositions.
Model weights retain their own license and are not committed to Git.
