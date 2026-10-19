"""Simulation script (test_run.py) running 3 consecutive stochastic iterations over the same playlist and initial track."""

from __future__ import annotations

import numpy as np

from atv.engine.decision_engine import HybridDJDecisionEngine
from atv.models import CandidatePlaybook, PlaybookArchetype, SessionState, TrackNode, TrackSection


def build_playlist_and_initial_track() -> tuple[TrackNode, list[TrackNode], list[TrackNode]]:
    """Return (current_track, candidate_pool, session_history)."""
    # Previously played track in session_history (enables Archetype 5: HISTORY_CALLBACK)
    history_track = TrackNode(
        track_id="trk_hist_camelphat",
        spotify_uri="spotify:track:trk_hist_camelphat",
        isrc="GBARL2300555",
        titulo="Cola (Club Mix)",
        autor="CamelPhat",
        generos=["melodic techno", "house"],
        anio=2023,
        idioma="en",
        duration_ms=200000,
        bpm=124.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=0,
        camelot_key="8A",
        energia_global=0.88,
        valencia_mood=0.75,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="estribillo",
                start_ms=61935,
                end_ms=92903,
                energia_local=0.94,
                tiene_vocal=True,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.85,
            )
        ],
    )

    # Current Track A: 125 BPM (bar = 1920.0 ms)
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
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=3840,  # 2 bars (< 4 bars)
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
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.42,
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
            TrackSection(
                tipo="estribillo",
                start_ms=111360,
                end_ms=142080,
                energia_local=0.92,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.40,
            ),
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

    cand_1 = TrackNode(
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
        camelot_key="8B",
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

    cand_2 = TrackNode(
        track_id="trk_artbat_horizon",
        spotify_uri="spotify:track:trk_artbat_horizon",
        isrc="DEUM72400222",
        titulo="Horizon",
        autor="ARTBAT",
        generos=["melodic techno", "electronic", "house"],
        anio=2024,
        idioma="en",
        duration_ms=230000,
        bpm=125.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=0,
        camelot_key="8A",
        energia_global=0.89,
        valencia_mood=0.70,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=15360,
                energia_local=0.75,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.65,
            ),
            TrackSection(
                tipo="estribillo",
                start_ms=30720,
                end_ms=76800,
                energia_local=0.91,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.88,
            ),
        ],
    )

    cand_3 = TrackNode(
        track_id="trk_taleofus_nova",
        spotify_uri="spotify:track:trk_taleofus_nova",
        isrc="ITUM72400333",
        titulo="Nova",
        autor="Tale Of Us",
        generos=["melodic techno", "electronic"],
        anio=2024,
        idioma="en",
        duration_ms=205000,
        bpm=124.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=0,
        camelot_key="7A",
        energia_global=0.85,
        valencia_mood=0.66,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=15484,
                energia_local=0.70,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.60,
            ),
            TrackSection(
                tipo="verso",
                start_ms=15484,
                end_ms=54193,
                energia_local=0.82,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.78,
            ),
        ],
    )

    return current_track, [cand_1, cand_2, cand_3], [history_track]


def format_playbook_row(pb: CandidatePlaybook, is_winner: bool, cooldown_active: bool) -> str:
    plan = pb.action_plan
    tr = plan.final_transition
    steps_summary = " -> ".join(f"{e.action_type}({e.from_section_type}->{e.to_section_type})" for e in plan.timeline)
    marker = ">> WINNER <<" if is_winner else (" [COOLDOWN -30%]" if cooldown_active else "")
    return (
        f"| {pb.archetype:<18} | {pb.raw_score:>6.2f} | {pb.selection_probability * 100:>6.2f}% | "
        f"{tr.next_track_uri.split(':')[-1]:<23} | {steps_summary:<50} | {marker}"
    )


def main() -> None:
    current_track, pool, history = build_playlist_and_initial_track()
    engine = HybridDJDecisionEngine()

    # Fixed RNG seed for reproducible 3-run demonstration showing 3 distinct archetypes
    rng = np.random.default_rng(11)
    last_archetype: PlaybookArchetype | None = None

    print("=" * 135)
    print("[ATV STOCHASTIC MULTI-OPTION DJ ENGINE - 3 CONSECUTIVE RUNS ON SAME TRACK & PLAYLIST]")
    print("=" * 135)
    print(f"Initial Track : {current_track.autor} - {current_track.titulo} ({current_track.bpm} BPM, Key={current_track.camelot_key})")
    print(f"Candidate Pool: {[f'{c.autor} - {c.titulo}' for c in pool]}")
    print(f"Temperature   : 0.40 | Min Viable Score: 65.0 | Maneuver Cooldown Factor: -30% on consecutive repeat")
    print("=" * 135)

    for iteration in range(1, 4):
        session = SessionState(
            target_energy=0.88,
            session_history=history,
            network_latency_offset_ms=250,
            creativity_temperature=0.40,
            last_maneuver_archetype=last_archetype,
            min_viable_score=65.0,
        )

        output = engine.plan_stochastic_playbooks(
            current_track=current_track,
            candidate_pool=pool,
            session_state=session,
            rng=rng,
        )

        all_pbs = [output.selected_playbook, *output.alternative_playbooks]
        all_pbs.sort(key=lambda x: x.raw_score, reverse=True)

        print(f"\n>>> ITERATION #{iteration} (Previous Maneuver Cooldown: {last_archetype or 'None'})")
        print("-" * 135)
        print(
            f"| {'ARCHETYPE':<18} | {'SCORE':>6} | {'PROB %':>7} | {'TARGET CANDIDATE':<23} | {'TIMELINE MANEUVER SEQUENCE':<50} | STATUS"
        )
        print("-" * 135)
        for pb in all_pbs:
            is_win = pb.archetype == output.selected_playbook.archetype
            is_cd = last_archetype is not None and pb.archetype == last_archetype
            print(format_playbook_row(pb, is_winner=is_win, cooldown_active=is_cd))
        print("-" * 135)

        win = output.selected_playbook
        print(
            f"Selected Playbook : [{win.archetype}] (Prob={win.selection_probability * 100:.2f}%, Score={win.raw_score:.2f})"
        )
        for idx, ev in enumerate(win.action_plan.timeline, start=1):
            print(
                f"  Step {idx}: {ev.action_type:<16} | trigger={ev.trigger_at_progress_ms:>6}ms -> target={ev.target_position_ms:>6}ms "
                f"on {ev.target_track_uri} ({ev.from_section_type}->{ev.to_section_type})"
            )

        # Feed chosen archetype into next iteration's maneuver_cooldown
        last_archetype = win.archetype

    print("\n" + "=" * 135)


if __name__ == "__main__":
    main()
