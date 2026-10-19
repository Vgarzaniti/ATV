"""Deterministic synchronized lyrics (.lrc) parser for structural section and vocal inference."""

from __future__ import annotations

import re
from dataclasses import dataclass

from atv.models import SectionType, TrackSection

_LRC_LINE_REGEX = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\](.*)")


@dataclass(frozen=True)
class LrcLine:
    timestamp_ms: int
    text: str


@dataclass(frozen=True)
class LyricStanza:
    start_ms: int
    end_ms: int
    lines: list[LrcLine]
    normalized_fingerprint: str


def _normalize_lyric_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_lrc_lines(lrc_content: str) -> list[LrcLine]:
    """Parse raw .lrc string into chronologically sorted LrcLine entries."""
    entries: list[LrcLine] = []
    for raw_line in lrc_content.splitlines():
        match = _LRC_LINE_REGEX.match(raw_line.strip())
        if not match:
            continue
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frac_str = match.group(3) or "0"
        if len(frac_str) == 1:
            millis = int(frac_str) * 100
        elif len(frac_str) == 2:
            millis = int(frac_str) * 10
        else:
            millis = int(frac_str[:3])

        total_ms = (minutes * 60 + seconds) * 1000 + millis
        lyric_text = match.group(4).strip()
        entries.append(LrcLine(timestamp_ms=total_ms, text=lyric_text))

    entries.sort(key=lambda x: x.timestamp_ms)
    return entries


def infer_sections_from_lrc(
    lrc_content: str,
    duration_ms: int,
    base_energy: float = 0.65,
    gap_threshold_ms: int = 7500,
    default_line_duration_ms: int = 3500,
) -> list[TrackSection]:
    """Deduce TrackSections and vocal presence deterministically from synchronized .lrc lyrics.

    Rules:
    - Silence gaps > gap_threshold_ms split vocal lines into stanzas.
    - Leading silence before first lyric -> 'intro' (tiene_vocal=False).
    - Repeated stanzas (matching normalized line fingerprints) -> 'estribillo' (higher energy).
    - Unique vocal stanzas -> 'verso', or 'puente' if occurring in the final third after a chorus.
    - Long instrumental gaps mid-track -> 'puente' or 'drop' depending on base_energy.
    - Trailing silence after last lyric -> 'outro' (tiene_vocal=False).
    """
    lines = parse_lrc_lines(lrc_content)
    non_empty_lines = [ln for ln in lines if _normalize_lyric_text(ln.text)]

    if not non_empty_lines:
        return build_fallback_instrumental_sections(duration_ms, base_energy)

    # Group into stanzas separated by silence gaps
    raw_groups: list[list[LrcLine]] = []
    current_group: list[LrcLine] = [non_empty_lines[0]]

    for prev_line, curr_line in zip(non_empty_lines, non_empty_lines[1:]):
        gap = curr_line.timestamp_ms - prev_line.timestamp_ms
        if gap >= gap_threshold_ms:
            raw_groups.append(current_group)
            current_group = [curr_line]
        else:
            current_group.append(curr_line)
    raw_groups.append(current_group)

    stanzas: list[LyricStanza] = []
    fingerprint_counts: dict[str, int] = {}

    for idx, grp in enumerate(raw_groups):
        start_ms = grp[0].timestamp_ms
        if idx + 1 < len(raw_groups):
            next_start = raw_groups[idx + 1][0].timestamp_ms
            end_ms = min(grp[-1].timestamp_ms + default_line_duration_ms, next_start)
        else:
            end_ms = min(grp[-1].timestamp_ms + default_line_duration_ms, duration_ms)

        if end_ms <= start_ms:
            end_ms = min(start_ms + 1000, duration_ms)

        # Fingerprint uses the first two normalized lines of the stanza to detect chorus repetition
        norm_lines = [_normalize_lyric_text(x.text) for x in grp if _normalize_lyric_text(x.text)]
        fp = " | ".join(norm_lines[:2]) if norm_lines else f"stanza_{idx}"
        fingerprint_counts[fp] = fingerprint_counts.get(fp, 0) + 1
        stanzas.append(LyricStanza(start_ms=start_ms, end_ms=end_ms, lines=grp, normalized_fingerprint=fp))

    sections: list[TrackSection] = []
    cursor_ms = 0

    # 1. Leading Intro
    if stanzas[0].start_ms >= 2000:
        sections.append(
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=stanzas[0].start_ms,
                energia_local=round(max(0.15, base_energy * 0.55), 3),
                tiene_vocal=False,
                percusion_fuerte=base_energy >= 0.5,
                bajo_predominante=base_energy >= 0.65,
                elemento_melodico=round(min(1.0, base_energy * 0.5), 3),
            )
        )
        cursor_ms = stanzas[0].start_ms

    seen_chorus = False
    for idx, stanza in enumerate(stanzas):
        # Fill any instrumental gap between previous stanza end and current stanza start
        if stanza.start_ms - cursor_ms >= 3000:
            gap_type: SectionType = "drop" if base_energy >= 0.75 and seen_chorus else "puente"
            sections.append(
                TrackSection(
                    tipo=gap_type,
                    start_ms=cursor_ms,
                    end_ms=stanza.start_ms,
                    energia_local=round(min(1.0, base_energy * (1.1 if gap_type == "drop" else 0.7)), 3),
                    tiene_vocal=False,
                    percusion_fuerte=gap_type == "drop",
                    bajo_predominante=gap_type == "drop",
                    elemento_melodico=round(min(1.0, base_energy * 0.8), 3),
                )
            )

        is_repeated = fingerprint_counts.get(stanza.normalized_fingerprint, 0) > 1
        if is_repeated:
            sec_type: SectionType = "estribillo"
            seen_chorus = True
            local_energy = min(1.0, base_energy * 1.15)
            perc = True
            bass = True
            melodic = min(1.0, base_energy * 0.9)
        elif seen_chorus and stanza.start_ms > duration_ms * 0.62 and idx == len(stanzas) - 2:
            sec_type = "puente"
            local_energy = max(0.2, base_energy * 0.75)
            perc = False
            bass = False
            melodic = min(1.0, base_energy * 0.85)
        else:
            sec_type = "verso"
            local_energy = max(0.25, base_energy * 0.85)
            perc = base_energy >= 0.45
            bass = base_energy >= 0.55
            melodic = min(1.0, base_energy * 0.7)

        actual_start = max(cursor_ms if not sections else sections[-1].end_ms if stanza.start_ms - cursor_ms < 3000 else stanza.start_ms, 0)
        if actual_start >= stanza.end_ms:
            actual_start = stanza.start_ms

        sections.append(
            TrackSection(
                tipo=sec_type,
                start_ms=actual_start,
                end_ms=stanza.end_ms,
                energia_local=round(local_energy, 3),
                tiene_vocal=True,
                percusion_fuerte=perc,
                bajo_predominante=bass,
                elemento_melodico=round(melodic, 3),
            )
        )
        cursor_ms = stanza.end_ms

    # Trailing Outro
    if duration_ms - cursor_ms >= 2000:
        sections.append(
            TrackSection(
                tipo="outro",
                start_ms=cursor_ms,
                end_ms=duration_ms,
                energia_local=round(max(0.15, base_energy * 0.5), 3),
                tiene_vocal=False,
                percusion_fuerte=base_energy >= 0.55,
                bajo_predominante=False,
                elemento_melodico=round(max(0.1, base_energy * 0.4), 3),
            )
        )
    elif sections and sections[-1].end_ms < duration_ms:
        last = sections[-1]
        sections[-1] = TrackSection(
            tipo=last.tipo,
            start_ms=last.start_ms,
            end_ms=duration_ms,
            energia_local=last.energia_local,
            tiene_vocal=last.tiene_vocal,
            percusion_fuerte=last.percusion_fuerte,
            bajo_predominante=last.bajo_predominante,
            elemento_melodico=last.elemento_melodico,
        )

    return sections


def build_fallback_instrumental_sections(duration_ms: int, base_energy: float = 0.7) -> list[TrackSection]:
    """Deterministic structural grid when no synchronized lyrics exist (instrumental or missing LRC)."""
    p1 = int(duration_ms * 0.15)
    p2 = int(duration_ms * 0.35)
    p3 = int(duration_ms * 0.45)
    p4 = int(duration_ms * 0.75)
    p5 = int(duration_ms * 0.85)

    return [
        TrackSection(
            tipo="intro",
            start_ms=0,
            end_ms=max(1, p1),
            energia_local=round(max(0.2, base_energy * 0.55), 3),
            tiene_vocal=False,
            percusion_fuerte=True,
            bajo_predominante=False,
            elemento_melodico=0.3,
        ),
        TrackSection(
            tipo="verso",
            start_ms=max(1, p1),
            end_ms=max(2, p2),
            energia_local=round(max(0.3, base_energy * 0.75), 3),
            tiene_vocal=False,
            percusion_fuerte=True,
            bajo_predominante=True,
            elemento_melodico=0.6,
        ),
        TrackSection(
            tipo="build_up",
            start_ms=max(2, p2),
            end_ms=max(3, p3),
            energia_local=round(min(1.0, base_energy * 0.9), 3),
            tiene_vocal=False,
            percusion_fuerte=True,
            bajo_predominante=False,
            elemento_melodico=0.8,
        ),
        TrackSection(
            tipo="drop",
            start_ms=max(3, p3),
            end_ms=max(4, p4),
            energia_local=round(min(1.0, base_energy * 1.15), 3),
            tiene_vocal=False,
            percusion_fuerte=True,
            bajo_predominante=True,
            elemento_melodico=0.9,
        ),
        TrackSection(
            tipo="puente",
            start_ms=max(4, p4),
            end_ms=max(5, p5),
            energia_local=round(max(0.25, base_energy * 0.65), 3),
            tiene_vocal=False,
            percusion_fuerte=False,
            bajo_predominante=False,
            elemento_melodico=0.7,
        ),
        TrackSection(
            tipo="outro",
            start_ms=max(5, p5),
            end_ms=duration_ms,
            energia_local=round(max(0.15, base_energy * 0.45), 3),
            tiene_vocal=False,
            percusion_fuerte=True,
            bajo_predominante=False,
            elemento_melodico=0.25,
        ),
    ]
