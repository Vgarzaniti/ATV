"""Phase 1: Vectorial Pre-Filtering and Macro Similarity Ranking using NumPy."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from atv.harmony.camelot import camelot_distance
from atv.models import SessionState, TrackNode


@dataclass(frozen=True)
class RankedCandidate:
    track: TrackNode
    macro_score: float
    effective_bpm_diff_pct: float
    camelot_dist: int
    jaccard_genres: float
    filter_notes: list[str]


def effective_bpm_difference_pct(
    bpm_a: float,
    bpm_b: float,
    allow_double_half: bool = True,
) -> tuple[float, str]:
    """Compute minimum relative BPM difference, supporting Double/Half Time (2x / 0.5x)."""
    direct_pct = abs(bpm_a - bpm_b) / bpm_a
    if not allow_double_half:
        return direct_pct, "1:1"

    double_pct = abs(bpm_a * 2.0 - bpm_b) / (bpm_a * 2.0)
    half_pct = abs(bpm_a * 0.5 - bpm_b) / (bpm_a * 0.5)

    best = min(
        (direct_pct, "1:1"),
        (double_pct, "2:1 (Double Time)"),
        (half_pct, "1:2 (Half Time)"),
        key=lambda item: item[0],
    )
    return best


def jaccard_similarity(genres_a: list[str], genres_b: list[str]) -> float:
    """Compute Jaccard similarity index between two genre lists."""
    set_a = {g.strip().lower() for g in genres_a if g.strip()}
    set_b = {g.strip().lower() for g in genres_b if g.strip()}
    if not set_a and not set_b:
        return 0.5  # Neutral similarity when both lack genre tags
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def filter_and_rank_candidates(
    current_track: TrackNode,
    candidate_pool: list[TrackNode],
    session_state: SessionState,
) -> list[RankedCandidate]:
    """Execute Phase 1 (Hard Filters + Vectorial Macro Similarity) to return Top-K candidates."""
    survivors: list[tuple[TrackNode, float, str]] = []

    current_author_norm = current_track.autor.strip().lower()
    last_played_author = (
        session_state.played_artistsHistory[-1].strip().lower()
        if session_state.played_artistsHistory
        else current_author_norm
    )

    for cand in candidate_pool:
        # Exclude self and already played tracks
        if cand.track_id == current_track.track_id or cand.track_id in session_state.played_track_ids:
            continue

        # Hard Filter 1: No consecutive same author
        cand_author_norm = cand.autor.strip().lower()
        if cand_author_norm == current_author_norm or cand_author_norm == last_played_author:
            continue

        # Hard Filter 2: Time signature compatibility
        if cand.time_signature != current_track.time_signature:
            continue

        # Hard Filter 3: BPM tolerance (including Double/Half time if enabled)
        bpm_diff_pct, ratio_mode = effective_bpm_difference_pct(
            current_track.bpm,
            cand.bpm,
            allow_double_half=session_state.allow_double_half_time,
        )
        if bpm_diff_pct > session_state.bpm_tolerance_pct:
            continue

        survivors.append((cand, bpm_diff_pct, ratio_mode))

    if not survivors:
        # Fallback for small initial pools: relax author and bpm tolerance so the session never crashes
        for cand in candidate_pool:
            if cand.track_id == current_track.track_id or cand.track_id in session_state.played_track_ids:
                continue
            bpm_diff_pct, ratio_mode = effective_bpm_difference_pct(
                current_track.bpm,
                cand.bpm,
                allow_double_half=True,
            )
            survivors.append((cand, bpm_diff_pct, ratio_mode))

    if not survivors:
        return []

    # Vectorized matrix computation over all survivors
    cand_tracks = [item[0] for item in survivors]
    bpm_diffs = np.array([item[1] for item in survivors], dtype=np.float64)
    ratio_modes = [item[2] for item in survivors]

    # Stack feature_vectors into (N, 7) matrix for fast dot-product cosine similarity
    feature_matrix = np.vstack([t.feature_vector for t in cand_tracks])
    current_vec = current_track.feature_vector.reshape(7, 1)
    cosine_sims = np.clip((feature_matrix @ current_vec).flatten(), 0.0, 1.0)

    # 1. BPM proximity score in [0, 1]
    bpm_scores = np.clip(1.0 - (bpm_diffs / session_state.bpm_tolerance_pct), 0.0, 1.0)

    # 2. Camelot harmonic score: 0 -> 1.0, 1 -> 0.90, >1 -> exponential decay exp(-0.85 * (d - 1))
    camelot_dists = np.array(
        [camelot_distance(current_track.camelot_key, t.camelot_key) for t in cand_tracks],
        dtype=np.int32,
    )
    camelot_scores = np.where(
        camelot_dists == 0,
        1.0,
        np.where(
            camelot_dists == 1,
            0.90,
            np.exp(-0.85 * (camelot_dists - 1.0)),
        ),
    )

    # 3. Genre Jaccard similarity
    jaccard_scores = np.array(
        [jaccard_similarity(current_track.generos, t.generos) for t in cand_tracks],
        dtype=np.float64,
    )

    # 4. Target Energy proximity
    cand_energies = np.array([t.energia_global for t in cand_tracks], dtype=np.float64)
    energy_scores = np.clip(1.0 - np.abs(cand_energies - session_state.target_energy), 0.0, 1.0)

    # 5. Soft Language & Era coherence
    lang_scores = np.array(
        [
            1.0
            if (
                t.idioma.lower() == current_track.idioma.lower()
                or "instrumental" in (t.idioma.lower(), current_track.idioma.lower())
            )
            else 0.65
            for t in cand_tracks
        ],
        dtype=np.float64,
    )
    cand_years = np.array([t.anio for t in cand_tracks], dtype=np.float64)
    year_scores = np.exp(-np.abs(cand_years - current_track.anio) / 15.0)

    # Weighted Macro Combination
    macro_scores = (
        0.25 * camelot_scores
        + 0.22 * bpm_scores
        + 0.20 * energy_scores
        + 0.15 * jaccard_scores
        + 0.08 * cosine_sims
        + 0.05 * lang_scores
        + 0.05 * year_scores
    )

    ranked_indices = np.argsort(-macro_scores)[: session_state.top_k_candidates]

    results: list[RankedCandidate] = []
    for idx in ranked_indices:
        i = int(idx)
        t = cand_tracks[i]
        c_dist = int(camelot_dists[i])
        j_sim = float(jaccard_scores[i])
        m_score = float(macro_scores[i])
        b_diff = float(bpm_diffs[i])

        notes = [
            f"MacroScore={m_score:.3f}",
            f"BPM={t.bpm:.1f} ({ratio_modes[i]}, diff={b_diff * 100:.2f}%)",
            f"Camelot={current_track.camelot_key}->{t.camelot_key} (dist={c_dist})",
            f"GenresJaccard={j_sim:.2f}",
            f"TargetEnergyDiff={abs(t.energia_global - session_state.target_energy):.2f}",
        ]
        results.append(
            RankedCandidate(
                track=t,
                macro_score=m_score,
                effective_bpm_diff_pct=b_diff,
                camelot_dist=c_dist,
                jaccard_genres=j_sim,
                filter_notes=notes,
            )
        )

    return results
