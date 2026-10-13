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
- **47 of 48 greedy SFT outputs begin with A4 C5.** This is strong prefix
  concentration in deterministic decoding, not proof that the entire stochastic
  distribution has collapsed. The task allows a common template to satisfy many
  prompts. Neither the current reward nor adjacent repetition alone captures
  this lack of variety across instructions.
- C, D, E, G and A belong to **all three supported keys**. The current task only
  forbids out-of-key notes; it does not require key-characteristic notes. A model
  can therefore pass key checks using a shared pitch vocabulary, which limits
  claims about learning tonal conditioning. A future task revision should require
  characteristic notes and evaluate transposition or melodic contour separately.
  That revision is deliberately not mixed into the fixed follow-up comparison.

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

## The longer run exposed a reward exploit

On full validation, the longer graded-reward GRPO run raised greedy compliance
from **40/48 to 44/48**, but adjacent repetition rose from **10.63% to 72.12%**.
Its dominant greedy prefix became A4 A4 (44/48 tasks), replacing SFT's A4 C5.

Across all four stochastic samples per task, compliance rose from **69/192 to
158/192**. Mean per-task adjacent repetition among valid samples rose from
**4.59% to 31.77%**. This is evidence that optimizing the specified reward can
improve compliance while producing degenerate melodies; it is not evidence of
better music.

This validation finding motivated a [separate reward ablation](diversity-protocol.json),
committed before any ID test was opened. The test coordinator was held while the
new policy was trained. Both RL policies start from the **same SFT adapter**;
the second policy is not a continuation of the repetitive RL policy. Configuration
differs only in `reward`, and the first four sampled outputs before any update
were verified identical. Both use 128 groups, four samples and learning rate 5e-5.

The alternative reward preserves compliance priority and differentiates correct
melodies by adjacent repetition and a three-distinct-pitch proxy. This may also
reduce skipped groups and increase the actual update count. Best-of-four selection
uses the same graded reward for every policy during evaluation, so the selector
is not another changing variable. All-candidate metrics remain the primary focus.

Reproduce the complete controlled comparison, training both policies before test:

```bash
HF_HUB_OFFLINE=1 python scripts/run_followup.py \
  --sft-adapter runs/local-study/sft/adapter \
  --diversity-config configs/diversity-study.json --run runs/new-reward-ablation
```

The diversity proxy can still favor an alternating template. It does not address
all cross-prompt variety or tonal-conditioning shortcuts, and it must not be
presented as a validated measure of musical quality.

For context on the algorithm's zero-reward-variance and loss diagnostics, see the
[TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer). This repo
implements its own small GRPO loop and does not claim equivalence to every TRL option.

## Completed outcome

The [final three-policy comparison](../README.md) now includes the previously
reserved ID test. All policies were fixed first, as recorded in the
[frozen-policy audit](frozen-policy-audit.json). Both RL variants improve raw
sample compliance over SFT on this test; the graded reward is more accurate and
much more repetitive. The diversity reward reduces that repetition at a cost in
compliance. No claim of musical quality follows from either metric.
