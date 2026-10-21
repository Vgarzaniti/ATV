"""Core DJ Transition Curves, Crossfade mathematics, and Volume automation."""

from __future__ import annotations

from enum import Enum
import math
from typing import Literal

CurveShape = Literal["LINEAR", "EQUAL_POWER", "EXPONENTIAL", "S_CURVE", "SHARP_CUT"]


def calculate_volume_levels(
    progress_ratio: float,
    curve: CurveShape = "EQUAL_POWER",
) -> tuple[float, float]:
    """Compute (volume_outgoing, volume_incoming) in [0.0, 1.0] along a transition progress ratio t in [0.0, 1.0].

    - LINEAR: vol_out = 1 - t, vol_in = t
    - EQUAL_POWER: cos(t * pi/2), sin(t * pi/2) -> maintains constant acoustic energy without a -3dB volume drop.
    - EXPONENTIAL: smooth logarithmic volume perception for natural human hearing.
    - S_CURVE: smooth start and tail dampening.
    - SHARP_CUT: stays at 1.0 until t >= 0.95 then drops instantly to 0.0 while in enters at 1.0.
    """
    t = max(0.0, min(1.0, float(progress_ratio)))

    if curve == "EQUAL_POWER":
        vol_out = math.cos(t * 0.5 * math.pi)
        vol_in = math.sin(t * 0.5 * math.pi)
    elif curve == "LINEAR":
        vol_out = 1.0 - t
        vol_in = t
    elif curve == "EXPONENTIAL":
        vol_out = math.pow(1.0 - t, 2.0)
        vol_in = math.pow(t, 0.5)
    elif curve == "S_CURVE":
        s = t * t * (3.0 - 2.0 * t)
        vol_out = 1.0 - s
        vol_in = s
    elif curve == "SHARP_CUT":
        vol_out = 0.0 if t >= 0.95 else 1.0
        vol_in = 1.0 if t >= 0.95 else 0.0
    else:
        vol_out = math.cos(t * 0.5 * math.pi)
        vol_in = math.sin(t * 0.5 * math.pi)

    return max(0.0, min(1.0, vol_out)), max(0.0, min(1.0, vol_in))
