"""Comprehensive test suite for ATV Deterministic DJ Decision Engine."""

from __future__ import annotations

import numpy as np
import pytest

from atv.engine import (
    HybridDJDecisionEngine,
    calculate_quantized_loop,
    evaluate_intra_track_edges,
    plan_dj_action_timeline,
    plan_next_transition,
)
from atv.engine.vector_filter import effective_bpm_difference_pct, filter_and_rank_candidates
from atv.harmony.camelot import camelot_distance, key_string_to_camelot, pitch_class_to_camelot
from atv.ingesta.dsp_analyzer import analyze_waveform_numpy
from atv.ingesta.lrc_parser import infer_sections_from_lrc
from atv.ingesta.pipeline import OsintAcousticFeatures, SpotifyBaseMetadata, TrackHydrationPipeline
from atv.models import SessionState, TrackNode, TrackSection


def test_camelot_translation_and_distance() -> None:
    assert pitch_class_to_camelot(9, 0) == "8A"
    assert pitch_class_to_camelot(0, 1) == "8B"
    assert key_string_to_camelot("Am") == "8A"
    assert key_string_to_camelot("C major") == "8B"
    assert key_string_to_camelot("12A") == "12A"

    assert camelot_distance("8A", "8A") == 0
    assert camelot_distance("8A", "8B") == 1
    assert camelot_distance("8A", "7A") == 1
    assert camelot_distance("12A", "1A") == 1
    assert camelot_distance("8A", "2B") == 7


def test_track_node_bar_duration_and_normalized_vector() -> None:
    node = TrackNode(
        track_id="t1",
        spotify_uri="spotify:track:t1",
        isrc="US1234567890",
        titulo="Test Track",
        autor="DJ Test",
        generos=["tech house", "house"],
        anio=2024,
        idioma="en",
        duration_ms=210000,
        bpm=120.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=500,
        camelot_key="8A",
        energia_global=0.8,
        valencia_mood=0.7,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=30000,
                energia_local=0.5,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.3,
            )
        ],
    )
    assert node.bar_duration_ms == pytest.approx(2000.0)
    assert node.feature_vector.shape == (7,)
    assert np.linalg.norm(node.feature_vector) == pytest.approx(1.0)


def test_lrc_parser_section_and_vocal_deduction() -> None:
    sample_lrc = """
    [00:16.00] Walking through the neon lights
    [00:20.00] Shadows dancing in the street
    [00:32.00] Feel the rhythm in your soul
    [00:36.00] Let the bass take full control
    [00:50.00] Moving faster through the night
    [00:54.00] Everything is out of sight
    [01:06.00] Feel the rhythm in your soul
    [01:10.00] Let the bass take full control
    """
    sections = infer_sections_from_lrc(sample_lrc, duration_ms=95000, base_energy=0.75)
    types = [s.tipo for s in sections]

    assert types[0] == "intro"
    assert sections[0].tiene_vocal is False
    assert "verso" in types
    assert types.count("estribillo") == 2
    assert types[-1] == "outro"
    assert sections[-1].tiene_vocal is False


def test_dsp_waveform_fade_out_vs_cold_end() -> None:
    sr = 22050
    t = np.linspace(0, 10, sr * 10)
    carrier = np.sin(2 * np.pi * 110 * t)

    cold_res = analyze_waveform_numpy(carrier, sample_rate=sr)
    assert cold_res.ending_type == "cold_end"

    envelope = np.ones_like(t)
    tail_idx = int(t.size * 0.8)
    envelope[tail_idx:] = np.linspace(1.0, 0.02, t.size - tail_idx) ** 2
    fade_res = analyze_waveform_numpy(carrier * envelope, sample_rate=sr)
    assert fade_res.ending_type == "fade_out"


def test_phase1_hard_filters_and_double_half_time() -> None:
    base_sec = [
        TrackSection(
            tipo="intro",
            start_ms=0,
            end_ms=16000,
            energia_local=0.5,
            tiene_vocal=False,
            percusion_fuerte=True,
            bajo_predominante=False,
            elemento_melodico=0.4,
        ),
        TrackSection(
            tipo="outro",
            start_ms=160000,
            end_ms=180000,
            energia_local=0.4,
            tiene_vocal=False,
            percusion_fuerte=True,
            bajo_predominante=False,
            elemento_melodico=0.3,
        ),
    ]

    current = TrackNode(
        track_id="curr",
        spotify_uri="spotify:track:curr",
        isrc="US0001",
        titulo="Current",
        autor="Artist A",
        generos=["dnb", "electronic"],
        anio=2023,
        duration_ms=180000,
        bpm=70.0,
        time_signature=4,
        camelot_key="8A",
        energia_global=0.75,
        valencia_mood=0.6,
        secciones=base_sec,
    )

    same_author = current.model_copy(update={"track_id": "c1", "spotify_uri": "spotify:track:c1"})
    wrong_meter = current.model_copy(
        update={"track_id": "c2", "spotify_uri": "spotify:track:c2", "autor": "Artist B", "time_signature": 3}
    )
    double_time_match = current.model_copy(
        update={"track_id": "c3", "spotify_uri": "spotify:track:c3", "autor": "Artist C", "bpm": 140.0}
    )

    state = SessionState(target_energy=0.8, bpm_tolerance_pct=0.06, allow_double_half_time=True)
    ranked = filter_and_rank_candidates(current, [same_author, wrong_meter, double_time_match], state)

    assert len(ranked) == 1
    assert ranked[0].track.track_id == "c3"
    diff_pct, mode = effective_bpm_difference_pct(70.0, 140.0, allow_double_half=True)
    assert diff_pct == pytest.approx(0.0)
    assert "Double Time" in mode


def test_anti_vocal_clash_forbids_crossfade() -> None:
    track_a = TrackNode(
        track_id="a",
        spotify_uri="spotify:track:a",
        isrc="US1",
        titulo="Song A",
        autor="Author A",
        generos=["pop"],
        anio=2022,
        duration_ms=180000,
        bpm=120.0,
        camelot_key="8A",
        energia_global=0.7,
        valencia_mood=0.7,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="outro",
                start_ms=150000,
                end_ms=180000,
                energia_local=0.6,
                tiene_vocal=True,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.5,
            )
        ],
    )
    track_b = TrackNode(
        track_id="b",
        spotify_uri="spotify:track:b",
        isrc="US2",
        titulo="Song B",
        autor="Author B",
        generos=["pop"],
        anio=2022,
        duration_ms=180000,
        bpm=120.0,
        camelot_key="8A",
        energia_global=0.7,
        valencia_mood=0.7,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=20000,
                energia_local=0.6,
                tiene_vocal=True,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.5,
            )
        ],
    )

    plan = plan_next_transition(track_a, [track_b], SessionState(target_energy=0.7))
    assert plan.transition_type != "CROSSFADE"
    assert plan.recommended_fade_duration_ms == 0


def test_drop_swap_and_beatgrid_snapping() -> None:
    track_a = TrackNode(
        track_id="a",
        spotify_uri="spotify:track:a",
        isrc="US10",
        titulo="Build Track",
        autor="Producer A",
        generos=["tech house"],
        anio=2024,
        duration_ms=200000,
        bpm=128.0,
        first_downbeat_ms=250,
        camelot_key="8A",
        energia_global=0.85,
        valencia_mood=0.7,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="build_up",
                start_ms=90000,
                end_ms=105200,
                energia_local=0.88,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.35,
            )
        ],
    )

    track_b = TrackNode(
        track_id="b",
        spotify_uri="spotify:track:b",
        isrc="US20",
        titulo="Drop Track",
        autor="Producer B",
        generos=["tech house"],
        anio=2024,
        duration_ms=210000,
        bpm=129.0,
        first_downbeat_ms=100,
        camelot_key="8B",
        energia_global=0.92,
        valencia_mood=0.8,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="drop",
                start_ms=44800,
                end_ms=90000,
                energia_local=0.95,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.8,
            )
        ],
    )

    plan = plan_next_transition(track_a, [track_b], SessionState(target_energy=0.92))
    assert plan.transition_type == "DROP_SWAP"
    assert plan.from_section_type == "build_up"
    assert plan.to_section_type == "drop"
    assert plan.recommended_fade_duration_ms == 0

    bars_a = (plan.trigger_at_progress_ms - track_a.first_downbeat_ms) / track_a.bar_duration_ms
    assert abs(bars_a - round(bars_a)) < 1e-3


def test_quantized_loop_safety_and_latency_offset() -> None:
    # 120 BPM, 4/4 -> bar_duration_ms = 2000ms. 4 bars = 8000ms.
    track = TrackNode(
        track_id="loop_trk",
        spotify_uri="spotify:track:loop_trk",
        isrc="US99",
        titulo="Loop Test",
        autor="Loop Artist",
        generos=["house"],
        anio=2024,
        duration_ms=180000,
        bpm=120.0,
        first_downbeat_ms=0,
        camelot_key="8A",
        energia_global=0.85,
        valencia_mood=0.7,
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=4000,  # Only 2 bars (< 4 bars)
                energia_local=0.6,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=False,
                elemento_melodico=0.3,
            ),
            TrackSection(
                tipo="estribillo",
                start_ms=60000,
                end_ms=92000,  # 16 bars
                energia_local=0.90,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.5,
            ),
        ],
    )

    # 1. Prohibits micro-loops < 4 bars
    with pytest.raises(ValueError, match="minimum allowed is 4 bars"):
        calculate_quantized_loop(track, track.secciones[0], bars=2, latency_ms=250)

    # 2. Valid 4-bar loop with 250ms network latency compensation
    ev = calculate_quantized_loop(track, track.secciones[0], bars=4, latency_ms=250)
    assert ev.action_type == "SEEK_LOOP"
    assert ev.quantized_loop_end_ms == 8000
    assert ev.trigger_at_progress_ms == 8000 - 250  # 7750ms
    assert ev.target_position_ms == 0

    # 3. Full timeline includes Intro 4-bar loop + Re-Chorus loop + Track Transition
    cand = track.model_copy(
        update={
            "track_id": "next_trk",
            "spotify_uri": "spotify:track:next_trk",
            "autor": "Other Artist",
            "camelot_key": "8B",
        }
    )
    state = SessionState(target_energy=0.90, network_latency_offset_ms=250)
    action_plan = plan_dj_action_timeline(track, [cand], state)

    action_types = [e.action_type for e in action_plan.timeline]
    assert action_types == ["SEEK_LOOP", "SEEK_LOOP", "TRACK_TRANSITION"]
    assert action_plan.timeline[0].from_section_type == "intro"
    assert action_plan.timeline[1].from_section_type == "estribillo"
    assert action_plan.timeline[1].trigger_at_progress_ms == 92000 - 250


def test_hydration_pipeline_end_to_end() -> None:
    spotify_meta = SpotifyBaseMetadata.from_spotify_dict(
        {
            "id": "sp123",
            "uri": "spotify:track:sp123",
            "name": "Midnight Protocol",
            "artists": [{"name": "Synth Architect"}],
            "duration_ms": 120000,
            "explicit": False,
            "album": {"release_date": "2025-05-10"},
            "external_ids": {"isrc": "USAT22500001"},
        }
    )
    osint = OsintAcousticFeatures(
        bpm=125.0,
        pitch_class=9,
        mode=0,
        time_signature=4,
        genres=["melodic techno", "progressive house"],
        energy=0.82,
        valence=0.65,
        language="en",
    )
    lrc = "[00:15.00] System online\n[00:35.00] Enter the grid\n[00:55.00] Enter the grid\n"

    pipeline = TrackHydrationPipeline()
    node = pipeline.hydrate_track(spotify_meta=spotify_meta, osint_features=osint, lrc_text=lrc)

    assert node.track_id == "sp123"
    assert node.camelot_key == "8A"
    assert node.anio == 2025
    assert len(node.secciones) >= 3


def test_stochastic_playbooks_and_maneuver_cooldown() -> None:
    from test_run import build_playlist_and_initial_track
    from atv.engine.decision_engine import StochasticSelector, plan_stochastic_playbooks

    current_track, pool, history = build_playlist_and_initial_track()
    state = SessionState(
        target_energy=0.88,
        session_history=history,
        creativity_temperature=0.40,
        session_seed=11,
        min_viable_score=65.0,
    )

    out_1 = plan_stochastic_playbooks(current_track, pool, state)
    all_pbs = [out_1.selected_playbook, *out_1.alternative_playbooks]

    # Must generate at least 4 distinct tactical playbooks above min_viable_score
    assert len(all_pbs) >= 4
    assert all(pb.raw_score >= 65.0 for pb in all_pbs)
    assert sum(pb.selection_probability for pb in all_pbs) == pytest.approx(1.0, abs=1e-2)

    # Verify 30% maneuver cooldown reduction when repeating the same archetype
    winner_arch = out_1.selected_playbook.archetype
    prob_without_cooldown = out_1.selected_playbook.selection_probability

    cooled_pbs = StochasticSelector.compute_probabilities(
        playbooks=all_pbs,
        temperature=0.40,
        last_archetype=winner_arch,
    )
    prob_with_cooldown = next(pb.selection_probability for pb in cooled_pbs if pb.archetype == winner_arch)
    assert prob_with_cooldown < prob_without_cooldown


def test_spotify_pkce_and_playlist_parser(tmp_path) -> None:
    from atv.config import SpotifyAppConfig
    from atv.spotify import extract_playlist_id, generate_pkce_pair

    verifier, challenge = generate_pkce_pair()
    assert len(verifier) >= 43
    assert len(challenge) >= 40
    assert "=" not in challenge

    assert extract_playlist_id("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc") == "37i9dQZF1DXcBWIGoYBM5M"
    assert extract_playlist_id("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M") == "37i9dQZF1DXcBWIGoYBM5M"

    env_file = tmp_path / ".env"
    env_file.write_text(
        "SPOTIFY_CLIENT_ID=test_client_123\nDJ_TARGET_ENERGY=0.91\nDJ_CREATIVITY_TEMPERATURE=0.55\n",
        encoding="utf-8",
    )
    cfg = SpotifyAppConfig.from_env(env_file)
    assert cfg.client_id == "test_client_123"
    assert cfg.target_energy == pytest.approx(0.91)
    assert cfg.creativity_temperature == pytest.approx(0.55)


