"""Phase 2: Structural Directed Weighted Graph Evaluation (Intra-Track Loops & Inter-Track Transitions)."""

from __future__ import annotations

from dataclasses import dataclass

from atv.engine.vector_filter import RankedCandidate, filter_and_rank_candidates
from atv.harmony.camelot import camelot_distance
from atv.models import (
    DJActionPlan,
    DJTimelineEvent,
    SessionState,
    TrackNode,
    TrackSection,
    TransitionPlan,
    TransitionType,
)


def snap_to_beatgrid(
    target_ms: int,
    first_downbeat_ms: int,
    bar_duration_ms: float,
    max_duration_ms: int | None = None,
) -> int:
    """Quantize a timestamp in milliseconds to the nearest musical bar boundary.

    Formula: first_downbeat_ms + round((target_ms - first_downbeat_ms) / bar_duration_ms) * bar_duration_ms
    """
    if bar_duration_ms <= 0:
        return max(0, target_ms)
    if target_ms <= first_downbeat_ms:
        snapped = first_downbeat_ms
    else:
        bars_index = round((target_ms - first_downbeat_ms) / bar_duration_ms)
        snapped = int(round(first_downbeat_ms + bars_index * bar_duration_ms))

    snapped = max(0, snapped)
    if max_duration_ms is not None and snapped > max_duration_ms:
        bars_floor = int((max_duration_ms - first_downbeat_ms) // bar_duration_ms)
        snapped = max(0, int(round(first_downbeat_ms + bars_floor * bar_duration_ms)))
    return snapped


def calculate_quantized_loop(
    track: TrackNode,
    section: TrackSection,
    bars: int,
    latency_ms: int = 250,
    loop_iterations: int = 1,
    anchor_at_end: bool = False,
) -> DJTimelineEvent:
    """Calculate a beatgrid-quantized SEEK_LOOP event with Spotify Connect latency compensation.

    Safety Rule:
    - Loop duration MUST be at least 4 bars (`4 * bar_duration_ms`) to prevent Spotify API
      buffering or rate limiting. Micro-loops (< 4 bars) raise a ValueError.
    - HTTP command is dispatched at `trigger_ms = end_loop_ms - latency_ms` so that playback
      lands on beat 1 of the target bar N seamlessly.
    """
    if bars < 4:
        raise ValueError(
            f"Loop safety violation: requested {bars} bars, but minimum allowed is 4 bars "
            f"({4.0 * track.bar_duration_ms:.1f}ms) to prevent Spotify API buffering/rate-limiting."
        )

    loop_duration_ms = bars * track.bar_duration_ms

    if anchor_at_end:
        snapped_end_ms = snap_to_beatgrid(
            target_ms=section.end_ms,
            first_downbeat_ms=track.first_downbeat_ms,
            bar_duration_ms=track.bar_duration_ms,
            max_duration_ms=track.duration_ms,
        )
        raw_start_ms = int(round(snapped_end_ms - loop_duration_ms))
        snapped_start_ms = snap_to_beatgrid(
            target_ms=max(track.first_downbeat_ms, raw_start_ms),
            first_downbeat_ms=track.first_downbeat_ms,
            bar_duration_ms=track.bar_duration_ms,
            max_duration_ms=track.duration_ms,
        )
        # Ensure at least 4 bars span even if section ends early
        if snapped_end_ms - snapped_start_ms < int(round(4 * track.bar_duration_ms)) - 5:
            snapped_end_ms = int(round(snapped_start_ms + loop_duration_ms))
    else:
        snapped_start_ms = snap_to_beatgrid(
            target_ms=max(track.first_downbeat_ms, section.start_ms),
            first_downbeat_ms=track.first_downbeat_ms,
            bar_duration_ms=track.bar_duration_ms,
            max_duration_ms=track.duration_ms,
        )
        snapped_end_ms = int(round(snapped_start_ms + loop_duration_ms))

    trigger_ms = max(0, snapped_end_ms - latency_ms)
    actual_loop_ms = snapped_end_ms - snapped_start_ms

    return DJTimelineEvent(
        action_type="SEEK_LOOP",
        current_track_uri=track.spotify_uri,
        target_track_uri=track.spotify_uri,
        from_section_type=section.tipo,
        to_section_type=section.tipo,
        quantized_loop_end_ms=snapped_end_ms,
        trigger_at_progress_ms=trigger_ms,
        target_position_ms=snapped_start_ms,
        bars_duration=bars,
        loop_iterations=loop_iterations,
        transition_type=None,
        recommended_fade_duration_ms=0,
        event_score=0.92,
        reasoning_log=[
            f"Intra-Track SEEK_LOOP on '{section.tipo}': {bars} bars ({actual_loop_ms}ms >= 4-bar safety threshold)",
            f"Quantized loop window: {snapped_start_ms}ms <- {snapped_end_ms}ms (bar={track.bar_duration_ms:.1f}ms)",
            f"Spotify Connect Latency Compensation: trigger_ms = {snapped_end_ms}ms - {latency_ms}ms = {trigger_ms}ms "
            f"(iterations={loop_iterations})",
        ],
    )


def evaluate_intra_track_edges(
    track: TrackNode,
    session_state: SessionState,
) -> list[DJTimelineEvent]:
    """Evaluate Intra-Track maneuvers (Intro/Outro Loop Extensions, Double Drop / Re-Chorus, Section Jumps)."""
    intra_events: list[DJTimelineEvent] = []
    min_four_bars_ms = 4.0 * track.bar_duration_ms
    latency_ms = session_state.network_latency_offset_ms

    # 1. Instrumental Intro Extension (Loop de Mezcla)
    # If the track starts with a short instrumental intro (< 4 bars), extend it with a 4-bar SEEK_LOOP
    intro_sections = [s for s in track.secciones if s.tipo == "intro"]
    if intro_sections:
        intro_sec = intro_sections[0]
        if not intro_sec.tiene_vocal and intro_sec.duration_ms < min_four_bars_ms:
            loop_ev = calculate_quantized_loop(
                track=track,
                section=intro_sec,
                bars=4,
                latency_ms=latency_ms,
                loop_iterations=1,
                anchor_at_end=False,
            )
            loop_ev.reasoning_log.insert(
                0,
                f"Rule 1 (Instrumental Intro Extension): Intro duration ({intro_sec.duration_ms}ms) < 4 bars "
                f"({min_four_bars_ms:.0f}ms) and tiene_vocal=False -> Executing 4-bar mix loop",
            )
            intra_events.append(loop_ev)

    # 2. Double Drop / Re-Estribillo (Peak Energy Maintenance)
    # If target_energy >= 0.85 and track has a chorus/drop with energia_local >= 0.85, loop back once
    if session_state.target_energy >= 0.85:
        peak_sections = [
            s for s in track.secciones if s.tipo in ("estribillo", "drop") and s.energia_local >= 0.85
        ]
        if peak_sections:
            # Pick the main peak chorus/drop before the outro
            peak_sec = peak_sections[-1]
            sec_bars = max(4, int(round(peak_sec.duration_ms / track.bar_duration_ms)))
            # Quantize to 4 or 8 bars for clean phrase structure
            loop_bars = 8 if sec_bars >= 8 else 4
            re_chorus_ev = calculate_quantized_loop(
                track=track,
                section=peak_sec,
                bars=loop_bars,
                latency_ms=latency_ms,
                loop_iterations=1,
                anchor_at_end=True,
            )
            re_chorus_ev.reasoning_log.insert(
                0,
                f"Rule 2 (Double Drop / Re-Estribillo): target_energy={session_state.target_energy:.2f}>=0.85 and "
                f"{peak_sec.tipo}.energia_local={peak_sec.energia_local:.2f}>=0.85 -> Repeating {loop_bars} bars (iterations=1)",
            )
            intra_events.append(re_chorus_ev)

    # 3. Instrumental Outro Extension (if short instrumental outro < 4 bars and cold_end)
    outro_sections = [s for s in track.secciones if s.tipo == "outro"]
    if outro_sections and track.ending_type == "cold_end":
        outro_sec = outro_sections[-1]
        if not outro_sec.tiene_vocal and outro_sec.duration_ms < min_four_bars_ms:
            outro_loop_ev = calculate_quantized_loop(
                track=track,
                section=outro_sec,
                bars=4,
                latency_ms=latency_ms,
                loop_iterations=1,
                anchor_at_end=False,
            )
            outro_loop_ev.reasoning_log.insert(
                0,
                f"Rule 1 (Instrumental Outro Extension): Short outro ({outro_sec.duration_ms}ms < 4 bars) "
                f"with tiene_vocal=False -> Executing 4-bar mix loop before next track",
            )
            intra_events.append(outro_loop_ev)

    intra_events.sort(key=lambda ev: ev.trigger_at_progress_ms)
    return intra_events


def evaluate_callback_tease(
    current_track: TrackNode,
    session_state: SessionState,
    exit_progress_ms: int,
) -> tuple[DJTimelineEvent, int] | None:
    """Optionally generate a CALLBACK_TEASE event to a previous track's peak section for 8 or 16 bars."""
    if not session_state.enable_callback_tease or not session_state.session_history:
        return None

    for past_track in reversed(session_state.session_history):
        if past_track.track_id == current_track.track_id:
            continue
        c_dist = camelot_distance(current_track.camelot_key, past_track.camelot_key)
        bpm_diff = abs(current_track.bpm - past_track.bpm) / current_track.bpm
        if c_dist <= 1 and bpm_diff <= session_state.bpm_tolerance_pct:
            peaks = [s for s in past_track.secciones if s.tipo in ("estribillo", "drop")]
            if not peaks:
                continue
            peak_sec = max(peaks, key=lambda s: s.energia_local)
            tease_bars = 8
            tease_start_ms = snap_to_beatgrid(
                peak_sec.start_ms,
                past_track.first_downbeat_ms,
                past_track.bar_duration_ms,
                past_track.duration_ms,
            )
            tease_end_ms = int(round(tease_start_ms + tease_bars * past_track.bar_duration_ms))
            ev = DJTimelineEvent(
                action_type="CALLBACK_TEASE",
                current_track_uri=current_track.spotify_uri,
                target_track_uri=past_track.spotify_uri,
                from_section_type="estribillo",
                to_section_type=peak_sec.tipo,
                quantized_loop_end_ms=tease_end_ms,
                trigger_at_progress_ms=exit_progress_ms,
                target_position_ms=tease_start_ms,
                bars_duration=tease_bars,
                loop_iterations=1,
                transition_type="QUICK_CUT",
                recommended_fade_duration_ms=0,
                event_score=0.89,
                reasoning_log=[
                    f"CALLBACK_TEASE: Returning to history track '{past_track.autor} - {past_track.titulo}' "
                    f"at peak '{peak_sec.tipo}' ({tease_start_ms}ms) for {tease_bars} bars",
                ],
            )
            return ev, tease_end_ms - session_state.network_latency_offset_ms
    return None


@dataclass(frozen=True)
class EvaluatedEdge:
    candidate: TrackNode
    section_out: TrackSection
    section_in: TrackSection
    transition_type: TransitionType
    trigger_at_progress_ms: int
    start_next_at_position_ms: int
    recommended_fade_duration_ms: int
    edge_score: float
    total_score: float
    logs: list[str]


def _get_valid_exit_sections(track: TrackNode) -> list[tuple[int, TrackSection]]:
    """Return valid outgoing sections (outro, estribillo, build_up, puente) in the second half of Track A."""
    valid_types = {"outro", "estribillo", "build_up", "puente", "drop"}
    candidates = [
        (idx, sec)
        for idx, sec in enumerate(track.secciones)
        if sec.tipo in valid_types and sec.end_ms >= track.duration_ms * 0.35
    ]
    if not candidates and track.secciones:
        return [(len(track.secciones) - 1, track.secciones[-1])]
    return candidates


def _get_valid_entry_sections(track: TrackNode) -> list[tuple[int, TrackSection]]:
    """Return valid incoming sections (intro, verso, drop, estribillo) in the first half of Track B."""
    valid_types = {"intro", "verso", "drop", "estribillo"}
    candidates = [
        (idx, sec)
        for idx, sec in enumerate(track.secciones)
        if sec.tipo in valid_types and sec.start_ms <= track.duration_ms * 0.55
    ]
    if not candidates and track.secciones:
        return [(0, track.secciones[0])]
    return candidates


def evaluate_section_edge(
    track_a: TrackNode,
    sec_a_idx: int,
    sec_a: TrackSection,
    ranked_b: RankedCandidate,
    sec_b_idx: int,
    sec_b: TrackSection,
    target_energy: float,
) -> EvaluatedEdge | None:
    """Evaluate a single directed Inter-Track edge (A_out -> B_in) applying deterministic DJ rules."""
    track_b = ranked_b.track
    logs: list[str] = [
        f"Candidate '{track_b.autor} - {track_b.titulo}' ({track_b.spotify_uri})",
        *ranked_b.filter_notes,
        f"Edge evaluation: A[{sec_a.tipo} {sec_a.start_ms}-{sec_a.end_ms}ms] -> B[{sec_b.tipo} {sec_b.start_ms}-{sec_b.end_ms}ms]",
    ]

    score = 0.50
    transition_type: TransitionType = "CROSSFADE"
    fade_ms = 6000
    raw_exit_ms = sec_a.start_ms
    raw_entry_ms = sec_b.start_ms

    chorus_indices = [i for i, s in enumerate(track_a.secciones) if s.tipo == "estribillo"]
    is_second_or_later_chorus = (
        sec_a.tipo == "estribillo"
        and len(chorus_indices) >= 2
        and sec_a_idx >= chorus_indices[1]
    ) or (sec_a.tipo == "estribillo" and len(chorus_indices) == 1)

    # --- RULE 3: Hook Strategy by ending_type and Sections ---
    if sec_a.tipo == "build_up" and sec_b.tipo in ("drop", "estribillo"):
        bpm_abs_diff = abs(track_a.bpm - track_b.bpm)
        if bpm_abs_diff <= 2.0 and ranked_b.camelot_dist <= 1:
            transition_type = "DROP_SWAP"
            fade_ms = 0
            raw_exit_ms = sec_a.end_ms
            raw_entry_ms = sec_b.start_ms
            score += 0.34
            logs.append(
                f"Strategy=DROP_SWAP: build_up->{sec_b.tipo} valid (|diff_BPM|={bpm_abs_diff:.2f}<=2.0, CamelotDist={ranked_b.camelot_dist})"
            )
        else:
            return None

    elif sec_a.tipo == "outro" and sec_b.tipo == "intro":
        if track_a.ending_type == "cold_end" and not sec_a.tiene_vocal and not sec_b.tiene_vocal:
            transition_type = "CROSSFADE"
            two_bars_ms = int(round(2.0 * track_a.bar_duration_ms))
            fade_ms = max(6000, min(8000, two_bars_ms))
            raw_exit_ms = max(sec_a.start_ms, sec_a.end_ms - fade_ms)
            raw_entry_ms = max(track_b.first_downbeat_ms, sec_b.start_ms)
            score += 0.28
            logs.append(f"Strategy=CLASSIC_MIX (outro->intro): low vocal density, fade={fade_ms}ms")
        elif track_a.ending_type == "cold_end":
            transition_type = "COLD_END_START"
            fade_ms = 0
            raw_exit_ms = sec_a.end_ms
            raw_entry_ms = max(track_b.first_downbeat_ms, sec_b.start_ms)
            score += 0.20
            logs.append("Strategy=COLD_END_START: cold ending at outro boundary")
        else:
            transition_type = "CROSSFADE"
            fade_ms = 6000
            raw_exit_ms = sec_a.start_ms
            raw_entry_ms = max(track_b.first_downbeat_ms, sec_b.start_ms)
            score -= 0.22
            logs.append("Penalty: Track A has fade_out; waiting until outro loses energy (-0.22)")

    elif sec_a.tipo == "estribillo" and sec_b.tipo in ("verso", "intro"):
        is_short_intro = sec_b.tipo == "intro" and sec_b.duration_ms <= 16000
        wants_high_energy = target_energy >= 0.70 or track_a.ending_type == "fade_out"

        if wants_high_energy and is_second_or_later_chorus and (sec_b.tipo == "verso" or is_short_intro):
            transition_type = "QUICK_CUT"
            fade_ms = 0
            raw_exit_ms = sec_a.end_ms
            raw_entry_ms = (
                track_b.first_downbeat_ms
                if sec_b.tipo == "intro" and sec_b_idx == 0
                else sec_b.start_ms
            )
            score += 0.32
            logs.append(
                f"Strategy=POWER_MIX (estribillo->{sec_b.tipo}): bypassed fade_out/kept high energy at chorus end"
            )
        else:
            transition_type = "QUICK_CUT"
            fade_ms = 0
            raw_exit_ms = sec_a.end_ms
            raw_entry_ms = max(track_b.first_downbeat_ms, sec_b.start_ms)
            score += 0.12
            logs.append(f"Strategy=CHORUS_CUT (estribillo->{sec_b.tipo})")
    else:
        if track_a.ending_type == "cold_end" and sec_a.tipo == "outro":
            transition_type = "COLD_END_START"
            fade_ms = 0
            raw_exit_ms = sec_a.end_ms
        elif track_a.ending_type == "fade_out" and sec_a.tipo == "outro":
            transition_type = "CROSSFADE"
            fade_ms = 6000
            raw_exit_ms = sec_a.start_ms
            score -= 0.22
            logs.append("Penalty: Track A has fade_out; waiting until outro loses energy (-0.22)")
        else:
            transition_type = "CROSSFADE"
            fade_ms = 6000
            raw_exit_ms = max(sec_a.start_ms, sec_a.end_ms - fade_ms)
        raw_entry_ms = max(track_b.first_downbeat_ms, sec_b.start_ms)
        logs.append(f"Strategy=STANDARD ({sec_a.tipo}->{sec_b.tipo})")

    # --- RULE 1: Anti-Vocal Clash ---
    if sec_a.tiene_vocal and sec_b.tiene_vocal:
        if transition_type == "CROSSFADE":
            transition_type = "QUICK_CUT"
            fade_ms = 0
            raw_exit_ms = sec_a.end_ms
            score -= 0.18
            logs.append(
                "Anti-Vocal Clash: Both A_out and B_in have vocals -> FORCED QUICK_CUT at section end & penalized (-0.18)"
            )
        else:
            score -= 0.10
            logs.append("Anti-Vocal Clash: Vocal-to-vocal boundary -> applied caution penalty (-0.10)")
    elif not sec_a.tiene_vocal and not sec_b.tiene_vocal:
        score += 0.08
        logs.append("Vocal Clearance: Clean instrumental bridge (+0.08)")

    # --- RULE 2: Timbral & Spectral Bridge ---
    if sec_a.percusion_fuerte and sec_a.elemento_melodico <= 0.45 and (
        sec_b.elemento_melodico >= 0.55 or sec_b.bajo_predominante
    ):
        score += 0.14
        logs.append(
            "Spectral Bridge: Percussive A_out complements melodic/bass entry in B_in (+0.14)"
        )
    elif sec_a.bajo_predominante and sec_b.bajo_predominante and transition_type == "CROSSFADE":
        score -= 0.12
        logs.append("Spectral Clash: Dual sub-bass overlap during CROSSFADE (-0.12)")
    elif sec_a.percusion_fuerte == sec_b.percusion_fuerte:
        score += 0.05
        logs.append("Spectral Continuity: Matched percussion drive (+0.05)")

    if track_a.groove_type == track_b.groove_type:
        score += 0.06
        logs.append(f"Groove Lock: Maintained '{track_a.groove_type}' pattern (+0.06)")

    local_energy_delta = abs(sec_b.energia_local - target_energy)
    energy_step_bonus = max(0.0, 0.10 * (1.0 - local_energy_delta))
    score += energy_step_bonus

    # --- RULE 4: Beatgrid Snapping ---
    snapped_exit_ms = snap_to_beatgrid(
        target_ms=raw_exit_ms,
        first_downbeat_ms=track_a.first_downbeat_ms,
        bar_duration_ms=track_a.bar_duration_ms,
        max_duration_ms=track_a.duration_ms,
    )
    snapped_entry_ms = snap_to_beatgrid(
        target_ms=raw_entry_ms,
        first_downbeat_ms=track_b.first_downbeat_ms,
        bar_duration_ms=track_b.bar_duration_ms,
        max_duration_ms=track_b.duration_ms,
    )
    logs.append(
        f"Beatgrid Snapping: Exit {raw_exit_ms}ms->{snapped_exit_ms}ms (bar={track_a.bar_duration_ms:.1f}ms), "
        f"Entry {raw_entry_ms}ms->{snapped_entry_ms}ms (bar={track_b.bar_duration_ms:.1f}ms)"
    )

    edge_score = float(max(0.0, min(1.0, score)))
    total_score = round(0.45 * ranked_b.macro_score + 0.55 * edge_score, 4)
    logs.append(
        f"FinalScore={total_score:.4f} (Macro={ranked_b.macro_score:.3f} * 0.45 + Edge={edge_score:.3f} * 0.55)"
    )

    return EvaluatedEdge(
        candidate=track_b,
        section_out=sec_a,
        section_in=sec_b,
        transition_type=transition_type,
        trigger_at_progress_ms=snapped_exit_ms,
        start_next_at_position_ms=snapped_entry_ms,
        recommended_fade_duration_ms=fade_ms,
        edge_score=edge_score,
        total_score=total_score,
        logs=logs,
    )


class HybridDJDecisionEngine:
    """Deterministic Autonomous DJ Decision Engine (Vector Filter + Intra/Inter-Track Graph)."""

    def plan_next_transition(
        self,
        current_track: TrackNode,
        candidate_pool: list[TrackNode],
        session_state: SessionState,
    ) -> TransitionPlan:
        """Select the optimal next track and exact structural cue points (Inter-Track contract)."""
        top_k = filter_and_rank_candidates(
            current_track=current_track,
            candidate_pool=candidate_pool,
            session_state=session_state,
        )
        if not top_k:
            raise ValueError(
                "No compatible candidate tracks survived Phase 1 hard filters "
                "(check author repetition, time_signature, or BPM tolerance)."
            )

        exit_sections = _get_valid_exit_sections(current_track)
        if not exit_sections:
            raise ValueError(f"Current track '{current_track.track_id}' has no valid sections.")

        best_edge: EvaluatedEdge | None = None

        for ranked_cand in top_k:
            entry_sections = _get_valid_entry_sections(ranked_cand.track)
            for sec_a_idx, sec_a in exit_sections:
                for sec_b_idx, sec_b in entry_sections:
                    evaluated = evaluate_section_edge(
                        track_a=current_track,
                        sec_a_idx=sec_a_idx,
                        sec_a=sec_a,
                        ranked_b=ranked_cand,
                        sec_b_idx=sec_b_idx,
                        sec_b=sec_b,
                        target_energy=session_state.target_energy,
                    )
                    if evaluated is None:
                        continue
                    if best_edge is None or evaluated.total_score > best_edge.total_score:
                        best_edge = evaluated

        if best_edge is None:
            raise ValueError("No valid structural transition edges could be formed in Phase 2.")

        return TransitionPlan(
            next_track_uri=best_edge.candidate.spotify_uri,
            transition_type=best_edge.transition_type,
            from_section_type=best_edge.section_out.tipo,
            to_section_type=best_edge.section_in.tipo,
            trigger_at_progress_ms=best_edge.trigger_at_progress_ms,
            start_next_at_position_ms=best_edge.start_next_at_position_ms,
            recommended_fade_duration_ms=best_edge.recommended_fade_duration_ms,
            transition_score=best_edge.total_score,
            reasoning_log=best_edge.logs,
        )

    def plan_dj_action_timeline(
        self,
        current_track: TrackNode,
        candidate_pool: list[TrackNode],
        session_state: SessionState,
    ) -> DJActionPlan:
        """Build a complete chronological DJActionPlan combining Intra-Track loops and Inter-Track transitions."""
        inter_plan = self.plan_next_transition(
            current_track=current_track,
            candidate_pool=candidate_pool,
            session_state=session_state,
        )

        # Evaluate Intra-Track Loops (Intro Mix Loop, Double Drop / Re-Chorus, Outro Loop)
        intra_events = evaluate_intra_track_edges(
            track=current_track,
            session_state=session_state,
        )

        # Filter intra-track events so none occur after the chosen inter-track exit point
        valid_intra = [
            ev
            for ev in intra_events
            if (ev.quantized_loop_end_ms or ev.trigger_at_progress_ms) <= inter_plan.trigger_at_progress_ms
        ]

        timeline: list[DJTimelineEvent] = list(valid_intra)

        # Optional Callback Tease before transitioning to the new track
        tease_res = evaluate_callback_tease(
            current_track=current_track,
            session_state=session_state,
            exit_progress_ms=inter_plan.trigger_at_progress_ms,
        )
        final_trigger_ms = inter_plan.trigger_at_progress_ms
        final_source_uri = current_track.spotify_uri

        if tease_res is not None:
            tease_ev, next_trigger_ms = tease_res
            timeline.append(tease_ev)
            final_trigger_ms = next_trigger_ms
            final_source_uri = tease_ev.target_track_uri

        # Terminal TRACK_TRANSITION event
        transition_event = DJTimelineEvent(
            action_type="TRACK_TRANSITION",
            current_track_uri=final_source_uri,
            target_track_uri=inter_plan.next_track_uri,
            from_section_type=inter_plan.from_section_type,
            to_section_type=inter_plan.to_section_type,
            quantized_loop_end_ms=None,
            trigger_at_progress_ms=final_trigger_ms,
            target_position_ms=inter_plan.start_next_at_position_ms,
            bars_duration=0,
            loop_iterations=0,
            transition_type=inter_plan.transition_type,
            recommended_fade_duration_ms=inter_plan.recommended_fade_duration_ms,
            event_score=inter_plan.transition_score,
            reasoning_log=inter_plan.reasoning_log,
        )
        timeline.append(transition_event)

        combined_logs: list[str] = []
        for idx, ev in enumerate(timeline, start=1):
            combined_logs.append(
                f"[Step {idx}: {ev.action_type}] trigger={ev.trigger_at_progress_ms}ms -> target={ev.target_position_ms}ms "
                f"({ev.from_section_type}->{ev.to_section_type})"
            )
            combined_logs.extend(f"  {line}" for line in ev.reasoning_log)

        return DJActionPlan(
            source_track_uri=current_track.spotify_uri,
            next_track_uri=inter_plan.next_track_uri,
            network_latency_offset_ms=session_state.network_latency_offset_ms,
            timeline=timeline,
            final_transition=inter_plan,
            overall_score=inter_plan.transition_score,
            reasoning_log=combined_logs,
        )


def plan_next_transition(
    current_track: TrackNode,
    candidate_pool: list[TrackNode],
    session_state: SessionState,
) -> TransitionPlan:
    """Functional entrypoint for Inter-Track TransitionPlan."""
    return HybridDJDecisionEngine().plan_next_transition(
        current_track=current_track,
        candidate_pool=candidate_pool,
        session_state=session_state,
    )


def plan_dj_action_timeline(
    current_track: TrackNode,
    candidate_pool: list[TrackNode],
    session_state: SessionState,
) -> DJActionPlan:
    """Functional entrypoint for complete Intra+Inter-Track DJActionPlan."""
    return HybridDJDecisionEngine().plan_dj_action_timeline(
        current_track=current_track,
        candidate_pool=candidate_pool,
        session_state=session_state,
    )
