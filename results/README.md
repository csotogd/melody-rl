# Experiment evidence

All results here were measured locally on an Apple M4 / 16 GB MacBook Air.
No paid inference or cloud compute was used. Audio is a simple sine rendering of
symbolic note sequences, not learned waveform generation.

## Local development study

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

The reserved test splits have not been used for model selection or reported scores.
