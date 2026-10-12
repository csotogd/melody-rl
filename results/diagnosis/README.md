# Diagnosing the first RL plateau

The [audit](pilot-audit.json) is computed from saved predictions and training logs
by `scripts/diagnose_pilot.py`. The first pilot and all its failures remain public.

## What the evidence says

- Increasing greedy validation from 24 to all 48 tasks gives **40/48 for both
  SFT and the original GRPO policy**. A larger evaluation alone does not train
  a better model.
- The first RL run sampled 128 completions on just **32 of 174 training tasks**
  and made **28 updates**. Four groups had identical rewards; zero-variance
  groups are a real issue but do not explain the whole plateau here.
- The SFT model's stochastic performance is much weaker than its greedy score:
  **30/96 successful samples** across the original 24 validation tasks, versus
  21/24 greedy and 18/24 selected best-of-four. These are different metrics.
- The 48-task greedy SFT failures are two note-count failures and six excessive
  pitch-interval failures. Every task passes key, range, final pitch and bar count.
- Finite, nonzero gradients and nonzero reference KL show that updates occurred.
  The loss can be near zero because standardized advantages sum to zero when
  on-policy ratios are one; this does not imply a zero gradient. Unit tests cover
  token alignment, EOS masking, clipping and gradient direction. These checks
  reduce obvious implementation risks, but are not proof that no bug exists.
- The graded reward has **no repetition penalty**. More constraint compliance
  alone cannot establish better music.

## Fixed follow-up

The [protocol](protocol.json) was committed before follow-up training. It reuses
the same SFT adapter, uses 128 groups and a learning rate of 5e-5, and evaluates
only the final policy. Both training budget and learning rate change, so the
comparison cannot isolate their separate causal effects.

The primary metric is the mean fraction of four temperature-1 samples per task
that satisfy every constraint, **before selection**. Greedy and best-of-four
success, repetition, validity and compute are reported separately. Sampling is
batched identically for both policies; training still makes one update per group.

All 48 validation tasks and then all 48 untouched ID test tasks are evaluated for
both fixed policies, regardless of whether validation improves. Uncertainty uses
a paired task-level bootstrap: four completions do not become four independent
tasks. One training seed cannot establish robustness across training runs.

Reproduce after creating the original SFT adapter:

```bash
HF_HUB_OFFLINE=1 python scripts/run_followup.py \
  --sft-adapter runs/local-study/sft/adapter --run runs/new-followup
```

The runner saves source hashes, the original adapter checksum, all commands,
all candidates, generated audio and both positive and negative results. It stops
if a stage fails. There is no score-based search or selection inside the runner.

For context on the algorithm's zero-reward-variance and loss diagnostics, see the
[TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer). This repo
implements its own small GRPO loop and does not claim equivalence to every TRL option.
