"""Deliberately small, strict score language: quarter-note pitches, bars separated by |.

This is NOT ABC notation. Restricting durations makes the first local experiment
cheap and isolates compositional instruction following from format complexity.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
import re
import struct
import wave
from pathlib import Path

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
ROOTS = {"C": 0, "F": 5, "G": 7}
MAJOR = {0, 2, 4, 5, 7, 9, 11}
NOTE = re.compile(r"([A-G]#?)([0-8])")


def pitch_name(pitch: int) -> str:
    return f"{NAMES[pitch % 12]}{pitch // 12 - 1}"


def parse_pitch(token: str) -> int:
    match = NOTE.fullmatch(token)
    if not match or match[1] not in NAMES:
        raise ValueError(f"Invalid pitch: {token!r}")
    return (int(match[2]) + 1) * 12 + NAMES.index(match[1])


@dataclass(frozen=True)
class Task:
    key: str
    beats: int
    bars: int
    low: int
    high: int
    final: int
    max_leap: int

    def __post_init__(self):
        if (self.key not in ROOTS or self.beats not in (3, 4)
                or not 1 <= self.bars <= 8 or not 12 <= self.low <= self.final <= self.high <= 119
                or self.max_leap <= 0 or (self.final - ROOTS[self.key]) % 12 not in MAJOR):
            raise ValueError("Invalid musical constraints")

    @property
    def id(self):
        return hashlib.sha256(repr(tuple(asdict(self).values())).encode()).hexdigest()[:16]

    def prompt(self):
        return (
            "Write only a melody, without explanation or markdown. "
            "Use pitch names with octave numbers, e.g. C4 D4 E4 | G4 E4 C4. "
            "Every note is a quarter note. Separate bars with |. No rests. "
            f"Requirements: {self.key} major; {self.beats}/4 meter; exactly {self.bars} bars; "
            f"all pitches between {pitch_name(self.low)} and {pitch_name(self.high)} inclusive; "
            f"last note {pitch_name(self.final)}; no melodic leap larger than {self.max_leap} semitones."
        )


def parse_score(text: str) -> list[list[int]]:
    # Reject extra prose, markdown, empty measures, trailing bars, and oversized outputs.
    if not text.strip() or len(text) > 2048:
        raise ValueError("Empty or oversized score")
    bars = text.strip().split("|")
    if len(bars) > 8:
        raise ValueError("Too many bars")
    result = []
    for bar in bars:
        tokens = bar.split()
        if not 1 <= len(tokens) <= 32:
            raise ValueError("Empty or oversized bar")
        result.append([parse_pitch(token) for token in tokens])
    if sum(map(len, result)) > 64:
        raise ValueError("Too many notes")
    return result


def evaluate_score(text: str, task: Task, *, truncated: bool = False) -> dict:
    names = ("bars", "meter", "key", "range", "final", "leap")
    try:
        if truncated:
            raise ValueError("Generation reached token limit without EOS")
        bars = parse_score(text)
    except ValueError as error:
        return {"valid": False, "all_satisfied": False, "checks": dict.fromkeys(names, False),
                "partial": 0.0, "diversity": 0.0, "repeat_rate": 0.0, "notes": 0,
                "unique_pitches": 0, "error": str(error)}
    pitches = [p for bar in bars for p in bar]
    checks = {
        "bars": len(bars) == task.bars,
        "meter": all(len(bar) == task.beats for bar in bars),
        "key": all((p - ROOTS[task.key]) % 12 in MAJOR for p in pitches),
        "range": all(task.low <= p <= task.high for p in pitches),
        "final": pitches[-1] == task.final,
        "leap": all(abs(a - b) <= task.max_leap for a, b in zip(pitches, pitches[1:])),
    }
    repeats = sum(a == b for a, b in zip(pitches, pitches[1:])) / max(1, len(pitches) - 1)
    return {"valid": True, "all_satisfied": all(checks.values()), "checks": checks,
            "partial": sum(checks.values()) / len(checks),
            "diversity": min(1.0, len(set(pitches)) / 3) * (1 - repeats),
            "repeat_rate": repeats, "notes": len(pitches), "unique_pitches": len(set(pitches)), "error": None}


def reward(metrics: dict, variant: str = "graded") -> float:
    if variant not in ("binary", "graded", "diverse"):
        raise ValueError(f"Unknown reward: {variant}")
    if not metrics["valid"]:
        return 0.0
    if variant == "binary":
        return float(metrics["all_satisfied"])
    base = 0.75 * metrics["partial"] + 0.25 * metrics["all_satisfied"]
    if variant == "diverse":
        # Diversity is a proxy, never labelled musical quality. Apply only on success.
        return 0.9 * base + 0.1 * metrics["all_satisfied"] * metrics["diversity"]
    return base


def _vlq(value):
    result = [value & 127]
    while value >> 7:
        value >>= 7
        result.insert(0, (value & 127) | 128)
    return bytes(result)


def write_midi(text: str, path: Path, bpm: int = 120):
    pitches = [p for bar in parse_score(text) for p in bar]
    tempo = int(60_000_000 / bpm)
    track = b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big") + b"\x00\xc0\x00"
    for pitch in pitches:
        track += b"\x00\x90" + bytes((pitch, 85))
        track += _vlq(480) + b"\x80" + bytes((pitch, 0))
    track += b"\x00\xff\x2f\x00"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480) + b"MTrk" + struct.pack(">I", len(track)) + track)


def write_wav(text: str, path: Path, bpm: int = 120):
    """Simple sine preview; not a learned audio model or a piano synthesizer."""
    import array
    pitches = [p for bar in parse_score(text) for p in bar]
    rate = 22050
    length = int(rate * 60 / bpm)
    samples = array.array("h")
    for pitch in pitches:
        freq = 440 * 2 ** ((pitch - 69) / 12)
        for i in range(length):
            envelope = min(1, i / (rate * 0.01), (length - i) / (rate * 0.04))
            samples.append(int(10000 * envelope * math.sin(2 * math.pi * freq * i / rate)))
    import sys
    if sys.byteorder != "little":
        samples.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
