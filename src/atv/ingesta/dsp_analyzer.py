"""Lightweight DSP analysis on audio arrays or preview streams."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from atv.models import EndingType


@dataclass(frozen=True)
class DspAnalysisResult:
    first_downbeat_ms: int
    rms_energy: float
    ending_type: EndingType


def analyze_waveform_numpy(
    samples: np.ndarray,
    sample_rate: int = 22050,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> DspAnalysisResult:
    """Compute first_downbeat_ms, normalized RMS energy, and ending_type using pure NumPy DSP.

    If `librosa` is installed, onset strength can optionally refine the first downbeat,
    but this function works deterministically with zero external audio C-dependencies.
    """
    if samples.ndim > 1:
        samples = np.mean(samples, axis=-1)
    samples = np.asarray(samples, dtype=np.float64)

    if samples.size < frame_length:
        return DspAnalysisResult(first_downbeat_ms=0, rms_energy=0.5, ending_type="cold_end")

    # Frame the signal using stride tricks
    num_frames = 1 + (samples.size - frame_length) // hop_length
    shape = (num_frames, frame_length)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)

    rms_per_frame = np.sqrt(np.mean(frames**2, axis=1))
    peak_rms = float(np.max(rms_per_frame)) if rms_per_frame.size > 0 else 0.0

    if peak_rms <= 1e-9:
        return DspAnalysisResult(first_downbeat_ms=0, rms_energy=0.0, ending_type="cold_end")

    norm_rms = rms_per_frame / peak_rms
    global_energy = float(np.clip(np.mean(norm_rms) * 1.35, 0.0, 1.0))

    # Detect first downbeat: first sharp positive energy flux exceeding 35% of peak RMS
    flux = np.diff(norm_rms, prepend=0.0)
    strong_onset_indices = np.where((norm_rms >= 0.35) & (flux >= 0.08))[0]
    if strong_onset_indices.size > 0:
        first_frame = int(strong_onset_indices[0])
    else:
        above_thresh = np.where(norm_rms >= 0.25)[0]
        first_frame = int(above_thresh[0]) if above_thresh.size > 0 else 0

    first_downbeat_ms = int(round((first_frame * hop_length / sample_rate) * 1000.0))

    # Ending type detection: compare tail 15% RMS envelope slope & ratio against body RMS
    tail_start = max(1, int(num_frames * 0.85))
    body_rms = float(np.mean(norm_rms[:tail_start]))
    tail_rms = float(np.mean(norm_rms[tail_start:]))

    # Check monotonic decay across the last 3 segments of the tail
    tail_slice = norm_rms[tail_start:]
    if tail_slice.size >= 6:
        splits = np.array_split(tail_slice, 3)
        means = [float(np.mean(s)) for s in splits]
        is_decaying = means[0] > means[1] > means[2] and (means[2] < body_rms * 0.45)
    else:
        is_decaying = tail_rms < body_rms * 0.45

    ending_type: EndingType = "fade_out" if is_decaying else "cold_end"

    return DspAnalysisResult(
        first_downbeat_ms=first_downbeat_ms,
        rms_energy=round(global_energy, 3),
        ending_type=ending_type,
    )
