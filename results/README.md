# Experiment evidence

All results here were measured locally on an Apple M4 / 16 GB MacBook Air.
No paid inference or cloud compute was used. Audio is a simple sine rendering of
symbolic note sequences, not learned waveform generation.

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

```bash
python scripts/export_rollouts.py --rollouts runs/local-study/grpo/rollouts.jsonl --output results/progress
```

The reserved test splits have not been used for model selection or reported scores.
