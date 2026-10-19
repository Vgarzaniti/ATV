"""Camelot Wheel harmonic translation and distance logic."""

from __future__ import annotations

import re

# Pitch Class (0=C, 1=C#, ..., 11=B) and Mode (0=Minor, 1=Major) -> Camelot Key
_PITCH_MODE_TO_CAMELOT: dict[tuple[int, int], str] = {
    # Minor keys (Mode = 0 -> 'A')
    (0, 0): "5A",   # C minor
    (1, 0): "12A",  # C# / Db minor
    (2, 0): "7A",   # D minor
    (3, 0): "2A",   # D# / Eb minor
    (4, 0): "9A",   # E minor
    (5, 0): "4A",   # F minor
    (6, 0): "11A",  # F# / Gb minor
    (7, 0): "6A",   # G minor
    (8, 0): "1A",   # G# / Ab minor
    (9, 0): "8A",   # A minor
    (10, 0): "3A",  # A# / Bb minor
    (11, 0): "10A", # B minor
    # Major keys (Mode = 1 -> 'B')
    (0, 1): "8B",   # C major
    (1, 1): "3B",   # C# / Db major
    (2, 1): "10B",  # D major
    (3, 1): "5B",   # D# / Eb major
    (4, 1): "12B",  # E major
    (5, 1): "7B",   # F major
    (6, 1): "2B",   # F# / Gb major
    (7, 1): "9B",   # G major
    (8, 1): "4B",   # G# / Ab major
    (9, 1): "11B",  # A major
    (10, 1): "6B",  # A# / Bb major
    (11, 1): "1B",  # B major
}

_NOTE_NAME_TO_PITCH_CLASS: dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

_CAMELOT_PATTERN = re.compile(r"^(1[0-2]|[1-9])([AB])$", re.IGNORECASE)


def pitch_class_to_camelot(pitch_class: int, mode: int) -> str:
    """Translate standard Pitch Class (0-11) and Mode (0=Minor, 1=Major) to Camelot notation."""
    if not (0 <= pitch_class <= 11):
        raise ValueError(f"pitch_class must be between 0 and 11, got {pitch_class}")
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 (minor) or 1 (major), got {mode}")
    return _PITCH_MODE_TO_CAMELOT[(pitch_class, mode)]


def key_string_to_camelot(key_str: str, mode_str: str | int | None = None) -> str:
    """Translate musical key strings (e.g., 'Am', 'C# minor', 'F', or already '8A') to Camelot."""
    cleaned = key_str.strip().upper()
    if _CAMELOT_PATTERN.match(cleaned):
        return cleaned

    # Check if mode is embedded in key_str (e.g. "Am", "C#m", "A minor", "C major")
    inferred_mode = 1
    note_part = cleaned

    if cleaned.endswith("MINOR") or cleaned.endswith("MIN"):
        inferred_mode = 0
        note_part = re.sub(r"\s*(MINOR|MIN)$", "", cleaned).strip()
    elif cleaned.endswith("MAJOR") or cleaned.endswith("MAJ"):
        inferred_mode = 1
        note_part = re.sub(r"\s*(MAJOR|MAJ)$", "", cleaned).strip()
    elif len(cleaned) >= 2 and cleaned.endswith("M") and not cleaned.endswith("BM"):
        inferred_mode = 0
        note_part = cleaned[:-1].strip()
    elif cleaned.endswith("BM") and len(cleaned) == 3:
        # e.g., Bbm -> Bb minor
        inferred_mode = 0
        note_part = cleaned[:-1].strip()

    if mode_str is not None:
        if isinstance(mode_str, int):
            inferred_mode = 0 if mode_str == 0 else 1
        elif isinstance(mode_str, str):
            m_clean = mode_str.strip().lower()
            inferred_mode = 0 if m_clean in ("0", "minor", "min", "m", "a") else 1

    pitch_class = _NOTE_NAME_TO_PITCH_CLASS.get(note_part)
    if pitch_class is None:
        raise ValueError(f"Unrecognized musical key string: '{key_str}'")

    return pitch_class_to_camelot(pitch_class, inferred_mode)


def parse_camelot(camelot_key: str) -> tuple[int, str]:
    """Parse a Camelot string like '8A' into (8, 'A')."""
    match = _CAMELOT_PATTERN.match(camelot_key.strip())
    if not match:
        raise ValueError(f"Invalid Camelot key format: '{camelot_key}'. Expected 1A-12B.")
    return int(match.group(1)), match.group(2).upper()


def camelot_distance(key_a: str, key_b: str) -> int:
    """Compute harmonic distance on the Camelot Wheel.

    Returns:
        0: Identical key (e.g., 8A -> 8A).
        1: Adjacent hour on same ring (8A -> 7A/9A) OR relative major/minor (8A -> 8B).
        >1: Minimum steps on the wheel (circular hour distance + ring switch).
    """
    num_a, let_a = parse_camelot(key_a)
    num_b, let_b = parse_camelot(key_b)

    diff = abs(num_a - num_b)
    hour_dist = min(diff, 12 - diff)
    ring_dist = 0 if let_a == let_b else 1

    return hour_dist + ring_dist


def camelot_to_circular_coords(camelot_key: str) -> tuple[float, float, float]:
    """Encode a Camelot key into (sin(theta), cos(theta), ring_binary) in [0, 1] range."""
    import math

    num, letter = parse_camelot(camelot_key)
    angle = 2.0 * math.pi * ((num - 1) / 12.0)
    sin_norm = (math.sin(angle) + 1.0) / 2.0
    cos_norm = (math.cos(angle) + 1.0) / 2.0
    ring_val = 1.0 if letter == "B" else 0.0
    return sin_norm, cos_norm, ring_val
