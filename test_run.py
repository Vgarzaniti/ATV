"""Simulation script (test_run.py) demonstrating Intra-Track Loops, Latency Compensation, and Inter-Track Transitions."""

from __future__ import annotations

import json

from atv.engine.decision_engine import HybridDJDecisionEngine
from atv.models import SessionState, TrackNode, TrackSection


def build_simulation_scenario() -> tuple[TrackNode, list[TrackNode], SessionState]:
    """Build a scenario where the DJ:
    1. Loops a short (< 4 bars) instrumental intro for 4 bars (`SEEK_LOOP`).
    2. Lets the track run through the verse and repeats the peak chorus once (`SEEK_LOOP` with 250ms latency offset).
    3. Transitions cleanly into the next compatible track (`TRACK_TRANSITION`).
    """
    # At 125 BPM (4/4), 1 bar = (60000 / 125) * 4 = 1920.0 ms.
    # 4 bars = 7680.0 ms.
    current_track = TrackNode(
        track_id="trk_bicep_solaris",
        spotify_uri="spotify:track:trk_bicep_solaris",
        isrc="GBCFB2400101",
        titulo="Solaris Horizon (Original Mix)",
        autor="Bicep",
        generos=["melodic techno", "electronic", "house"],
        anio=2024,
        idioma="en",
        duration_ms=192000,
        bpm=125.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=0,
        camelot_key="8A",
        energia_global=0.86,
        valencia_mood=0.68,
        ending_type="cold_end",
        secciones=[
            # Short instrumental intro (2 bars = 3840ms < 4 bars) -> Triggers Rule 1: 4-bar SEEK_LOOP
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=3840,
                energia_local=0.55,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.35,
            ),
            TrackSection(
                tipo="verso",
                start_ms=3840,
                end_ms=49920,
                energia_local=0.72,
                tiene_vocal=True,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.60,
            ),
            TrackSection(
                tipo="estribillo",
                start_ms=49920,
                end_ms=80640,
                energia_local=0.88,
                tiene_vocal=True,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.85,
            ),
            TrackSection(
                tipo="verso",
                start_ms=80640,
                end_ms=111360,
                energia_local=0.74,
                tiene_vocal=True,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.65,
            ),
            # Second Chorus with high energy (0.92 >= 0.85) -> Triggers Rule 2: Re-Chorus SEEK_LOOP
            TrackSection(
                tipo="estribillo",
                start_ms=111360,
                end_ms=142080,  # 16 bars duration (30,720 ms)
                energia_local=0.92,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.40,
            ),
            # Instrumental Outro
            TrackSection(
                tipo="outro",
                start_ms=142080,
                end_ms=192000,
                energia_local=0.50,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.30,
            ),
        ],
    )

    candidate_next = TrackNode(
        track_id="trk_anyma_consciousness",
        spotify_uri="spotify:track:trk_anyma_consciousness",
        isrc="USUM72400999",
        titulo="Consciousness (Club Mix)",
        autor="Anyma",
        generos=["melodic techno", "electronic"],
        anio=2024,
        idioma="en",
        duration_ms=215000,
        bpm=126.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=0,
        camelot_key="8B",  # Relative major (Camelot dist = 1)
        energia_global=0.90,
        valencia_mood=0.72,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=15238,
                energia_local=0.78,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.70,
            ),
            TrackSection(
                tipo="verso",
                start_ms=15238,
                end_ms=60952,
                energia_local=0.88,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.85,
            ),
        ],
    )

    session = SessionState(
        target_energy=0.90,
        bpm_tolerance_pct=0.06,
        allow_double_half_time=True,
        top_k_candidates=15,
        network_latency_offset_ms=250,
    )

    return current_track, [candidate_next], session


def main() -> None:
    current_track, pool, session = build_simulation_scenario()
    engine = HybridDJDecisionEngine()
    action_plan = engine.plan_dj_action_timeline(
        current_track=current_track,
        candidate_pool=pool,
        session_state=session,
    )

    print("=" * 78)
    print("[ATV DETERMINISTIC DJ ENGINE - TIMELINE SIMULATION (test_run.py)]")
    print("=" * 78)
    print(f"Current Track   : {current_track.autor} - {current_track.titulo} ({current_track.spotify_uri})")
    print(f"Rhythmic Grid   : {current_track.bpm} BPM | {current_track.time_signature}/4 | Bar={current_track.bar_duration_ms:.1f}ms")
    print(f"Target Energy   : {session.target_energy} | Spotify Latency Offset: {session.network_latency_offset_ms}ms")
    print("-" * 78)
    for idx, ev in enumerate(action_plan.timeline, start=1):
        print(
            f"EVENT #{idx}: [{ev.action_type}] "
            f"{ev.from_section_type.upper()} -> {ev.to_section_type.upper()}"
        )
        print(
            f"  Trigger HTTP at : {ev.trigger_at_progress_ms} ms "
            + (f"(Quantized Bar End: {ev.quantized_loop_end_ms} ms - {session.network_latency_offset_ms} ms)" if ev.quantized_loop_end_ms else "")
        )
        print(f"  Seek / Start at : {ev.target_position_ms} ms on {ev.target_track_uri}")
        if ev.bars_duration:
            print(f"  Loop Geometry   : {ev.bars_duration} bars | Iterations={ev.loop_iterations}")
        if ev.transition_type:
            print(f"  Transition Mode : {ev.transition_type} (Fade={ev.recommended_fade_duration_ms} ms)")
        for note in ev.reasoning_log:
            print(f"    * {note}")
        print("-" * 78)

    print("\nFULL DJActionPlan JSON PAYLOAD:")
    print(json.dumps(action_plan.model_dump(), indent=2, ensure_ascii=True))
    print("=" * 78)


if __name__ == "__main__":
    main()
