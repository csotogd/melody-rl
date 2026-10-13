# Experiment evidence

All results here were measured locally on an Apple M4 / 16 GB MacBook Air.
No paid inference or cloud compute was used. Audio is a simple sine rendering of
symbolic note sequences, not learned waveform generation.

## Controlled reward study: completed

The main result is a **held-out compliance improvement with a measurable reward
tradeoff**, on one training seed. [All numeric results](study-summary.json) are
computed from saved predictions; the reports contain all 48 tasks and audio.

### ID test — policies frozen before evaluation

| Policy | Greedy success | All stochastic samples | Sample repetition¹ | Best of 4 |
| --- | ---: | ---: | ---: | ---: |
| SFT | 34/48 | 78/192 (40.6%) | 7.0% | 38/48 |
| SFT + GRPO, constraint reward | 43/48 | 148/192 (77.1%) | 27.1% | 43/48 |
| SFT + GRPO, diversity reward | 39/48 | 130/192 (67.7%) | 7.7% | 45/48 |

### Validation — used for development

| Policy | Greedy success | All stochastic samples | Sample repetition¹ | Best of 4 |
| --- | ---: | ---: | ---: | ---: |
| SFT | 40/48 | 69/192 (35.9%) | 4.6% | 34/48 |
| SFT + GRPO, constraint reward | 44/48 | 158/192 (82.3%) | 31.8% | 45/48 |
| SFT + GRPO, diversity reward | 41/48 | 119/192 (62.0%) | 8.0% | 45/48 |

¹ Repetition is the mean per task of adjacent-repeat fractions among valid
samples. All-candidate success is the primary endpoint; the four candidates for a
task do not count as four independent tasks. Greedy decoding and best-of-four
selection are secondary endpoints. The selector is the **same graded reward** for
all three policies, including the diversity-trained model.

- [Validation report and complete audio](extended-validation.html)
- [ID test report and complete audio](extended-test.html)
- [Diagnosis of the original plateau](diagnosis/README.md)
- [Protocol fixed before extended training](diagnosis/protocol.json)
- [Reward-ablation protocol fixed before ID test](diagnosis/diversity-protocol.json)
- [Frozen adapter hashes and split audit](diagnosis/frozen-policy-audit.json)
- [Extended training audio snapshots](extended-progress/)
- [Diversity training audio snapshots](diversity-progress/)

Constraint RL improves ID-test all-candidate success over SFT by **36.5 percentage points**, with a paired task-bootstrap 95% interval of **[25.0, 47.4] points**.

Diversity RL improves ID-test all-candidate success over SFT by **27.1 percentage points**, with a paired task-bootstrap 95% interval of **[17.7, 36.5] points**.

Intervals are conditional on the trained policies and this small task set; they
do not establish robustness across training seeds. Raw comparison files also
include exact McNemar tests for the secondary selected binary endpoint, where
few discordant tasks can make percentile-bootstrap intervals overconfident.

### What changed, and what remains weak

A larger evaluation confirmed 40/48 for both SFT and the old 32-group RL policy.
That alone did not improve the model. Extending RL to 128 groups and increasing
the learning rate fivefold improved compliance, but cannot isolate the effect of
training duration from learning rate. Graded RL's greedy adjacent repetition is
about 71% on test, versus about 10% for SFT: repeating a safe note is a valid way
to maximize the original reward.

The reward-only ablation starts from the same SFT adapter, uses the same ordered
128 prompts and four samples per group, and changes only the reward configuration.
Graded RL made 87 updates (41 equal-reward groups skipped); diversity RL made 114
updates (14 skipped). Equal sampling budgets therefore do not imply equal update
counts. Training took 750.3 and 855.8 seconds respectively, including loading.
Timings are observations on one local machine, not an isolated hardware benchmark.

The diversity proxy reduces adjacent repetition substantially relative to graded
RL but sacrifices some compliance. Both policies still favor templates. C, D, E,
G and A belong to all supported keys, so passing the key check can exploit shared
notes rather than demonstrate tonal understanding. Neither reward measures
musical quality. The procedural solver remains the stronger choice when the sole
requirement is exact constraint satisfaction.

All models were frozen before the first ID-test evaluation. The diversity reward
was chosen after seeing validation reward exploitation, and before opening test.
No retraining followed the test results. Composition and length tests remain
reserved. More seeds, an extra-SFT compute control and sampling-temperature
controls have not been run. The earlier failed experiments remain below.

## Local development study (historical, 24 tasks)

[Open the comparison and audio report](local-study.html) after cloning the repo.
The configuration is `configs/local-study.json`: seed 42, 348 SFT updates over
174 synthetic training examples, followed by 32 GRPO groups of four samples.
GRPO made 28 optimizer updates; four groups had equal rewards and were skipped.
SFT took 208.7 seconds and GRPO took 288.3 seconds, including model loading.

On the same first 24 validation tasks, both greedy SFT and greedy SFT + GRPO
satisfy all constraints on **21/24** tasks (87.5%). Both have 100% valid format.
There is **no observed compliance improvement from this RL run**. Their three
failures are the same: one wrong note count and two excessive pitch intervals.
Only three of the 24 output strings change, all on already-successful tasks.
Mean adjacent repetition among valid outputs rises from 9.48% to 10.67%.

The initial model scores 0/24 and the procedural control scores 24/24. Selecting
the highest training reward among four stochastic SFT samples scores 18/24, with
4.25% adjacent repetition. It generates 1,216 tokens in 103.1 seconds, versus
298 tokens in 20.9 seconds for greedy SFT. Sampling uses temperature 1; the greedy
output is not included among the four candidates, so best-of-4 need not match
greedy accuracy. All 96 candidates are retained in its predictions file.

This is a single-seed development result, not a claim of generalization or musical
quality. The per-method Wilson 95% interval for 21/24 is approximately 69%–96%;
it does not account for training-seed variability. No hyperparameter sweep was
performed after seeing these results. All predictions and sampled training
rollouts, including failed outputs, are exported alongside the report.

## Initial feasibility pilot

[Open the HTML report](pilot.html) after cloning/downloading this repository.
GitHub displays HTML source rather than hosting a playable page; local browsers
can play the bundled audio. The report's adjacent assets contain raw predictions,
training logs, configurations, MIDI and WAV files.

The four-example pilot verified SFT and GRPO execution and adapter reload. It did
not demonstrate an RL gain. SFT ran 24 updates; RL processed four groups and made
three updates (one equal-reward group was skipped). The initial pilot used the
algebraically equivalent `exp(delta) - delta - 1` KL expression, since changed to
`expm1(delta) - delta` for numerical accuracy near zero.

## Audible progress

The first snapshot compares the **same validation task**:

- [Before SFT — WAV](progress/001-sft/base.wav) · [MIDI](progress/001-sft/base.mid)
- [After 348 SFT updates — WAV](progress/001-sft/sft.wav) · [MIDI](progress/001-sft/sft.mid)
- [Exact prompt, outputs and per-constraint checks](progress/001-sft/outputs.json)

The request is two 3/4 bars in F major, pitches C4–C5, ending F4, with maximum
leap seven semitones. The initial model adds a seventh note. SFT produces six
notes and satisfies all constraints, but includes an adjacent repeated note.
This illustrates why compliance and musical quality must be measured separately.

RL progress snapshots use predetermined training groups 1, 8, 16 and 32. Each
directory preserves **all** sampled outputs, including failures. They are sampled
on training prompts before the corresponding update, and must not be compared
as if they were a fixed held-out benchmark. Only valid scores receive audio files.

- [Group 1](progress/rl-step-001/outputs.json)
- [Group 8](progress/rl-step-008/outputs.json)
- [Group 16](progress/rl-step-016/outputs.json)
- [Group 32](progress/rl-step-032/outputs.json)

```bash
python scripts/export_rollouts.py --rollouts runs/local-study/grpo/rollouts.jsonl --output results/progress
```

The historical pilot used validation only. The later controlled study below has
now evaluated test_id; composition and length tests remain reserved.
