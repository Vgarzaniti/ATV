"""Interactive simulation script to test the ATV DJ Decision Engine in the terminal."""

from __future__ import annotations

import json
import numpy as np

from atv.engine.decision_engine import HybridDJDecisionEngine
from atv.ingesta.pipeline import OsintAcousticFeatures, SpotifyBaseMetadata, TrackHydrationPipeline
from atv.models import SessionState, TrackNode, TrackSection


def build_demo_pool() -> tuple[TrackNode, list[TrackNode]]:
    pipeline = TrackHydrationPipeline()

    # Track A (Currently Playing): Melodic Techno / House with a fade_out ending and synced lyrics
    meta_a = SpotifyBaseMetadata(
        track_id="trk_current_01",
        uri="spotify:track:trk_current_01",
        name="Solaris Horizon",
        artists=["Bicep"],
        duration_ms=210000,
        explicit=False,
        release_date="2024-03-15",
        isrc="GBCFB2400101",
    )
    osint_a = OsintAcousticFeatures(
        bpm=126.0,
        pitch_class=9,  # A minor -> 8A
        mode=0,
        time_signature=4,
        genres=["melodic techno", "electronic", "house"],
        energy=0.78,
        valence=0.65,
        language="en",
    )
    lrc_a = """
    [00:16.00] Holding on to the frequency
    [00:22.00] Drifting past the atmosphere
    [00:45.00] We are rising into the light
    [00:52.00] Shining through the endless night
    [01:15.00] Signals echo in the dark
    [01:22.00] Waiting for another spark
    [01:45.00] We are rising into the light
    [01:52.00] Shining through the endless night
    """
    # Simulate a fade-out waveform for Track A so the engine prefers a Power Mix or clean cut
    sr = 22050
    t = np.linspace(0, 10, sr * 10)
    env = np.ones_like(t)
    tail = int(t.size * 0.8)
    env[tail:] = np.linspace(1.0, 0.01, t.size - tail) ** 2
    waveform_a = np.sin(2 * np.pi * 110 * t) * env

    current_track = pipeline.hydrate_track(
        spotify_meta=meta_a,
        osint_features=osint_a,
        lrc_text=lrc_a,
        waveform_samples=waveform_a,
        sample_rate=sr,
    )

    # Candidate 1: Ideal harmonic & tempo match (127 BPM, 8B relative major, strong drop & intro)
    cand_1 = TrackNode(
        track_id="cand_01_anyma",
        spotify_uri="spotify:track:cand_01_anyma",
        isrc="USUM72400999",
        titulo="Consciousness (Club Mix)",
        autor="Anyma",
        generos=["melodic techno", "progressive house"],
        anio=2024,
        idioma="en",
        duration_ms=225000,
        bpm=127.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=350,
        camelot_key="8B",  # Distance 1 from 8A
        energia_global=0.88,
        valencia_mood=0.70,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=15000,
                energia_local=0.72,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.65,
            ),
            TrackSection(
                tipo="verso",
                start_ms=15000,
                end_ms=45000,
                energia_local=0.84,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.80,
            ),
            TrackSection(
                tipo="drop",
                start_ms=45000,
                end_ms=110000,
                energia_local=0.95,
                tiene_vocal=False,
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.90,
            ),
        ],
    )

    # Candidate 2: Same artist as current ("Bicep") -> Should be discarded by Hard Filter 1
    cand_2_same_artist = cand_1.model_copy(
        update={
            "track_id": "cand_02_bicep",
            "spotify_uri": "spotify:track:cand_02_bicep",
            "titulo": "Glue",
            "autor": "Bicep",
        }
    )

    # Candidate 3: Incompatible BPM (95 BPM Reggaeton) -> Discarded by Hard Filter 3
    cand_3_bad_bpm = cand_1.model_copy(
        update={
            "track_id": "cand_03_feid",
            "spotify_uri": "spotify:track:cand_03_feid",
            "titulo": "Classy 101",
            "autor": "Feid",
            "bpm": 95.0,
            "groove_type": "dembow",
            "generos": ["reggaeton"],
        }
    )

    # Candidate 4: Compatible BPM (125 BPM), Distant Camelot Key (2B, dist=7) & Vocal Clash on Intro
    cand_4_clash = TrackNode(
        track_id="cand_04_clash",
        spotify_uri="spotify:track:cand_04_clash",
        isrc="GBUM72300111",
        titulo="Vocal Overload",
        autor="Disclosure",
        generos=["house", "uk garage"],
        anio=2023,
        idioma="en",
        duration_ms=195000,
        bpm=125.0,
        time_signature=4,
        groove_type="4x4",
        first_downbeat_ms=200,
        camelot_key="2B",
        energia_global=0.80,
        valencia_mood=0.75,
        ending_type="cold_end",
        secciones=[
            TrackSection(
                tipo="intro",
                start_ms=0,
                end_ms=20000,
                energia_local=0.75,
                tiene_vocal=True,  # Heavy vocals right from the start
                percusion_fuerte=True,
                bajo_predominante=True,
                elemento_melodico=0.85,
            )
        ],
    )

    return current_track, [cand_1, cand_2_same_artist, cand_3_bad_bpm, cand_4_clash]


def main() -> None:
    current_track, pool = build_demo_pool()
    session = SessionState(
        target_energy=0.88,
        bpm_tolerance_pct=0.06,
        allow_double_half_time=True,
        top_k_candidates=15,
    )

    engine = HybridDJDecisionEngine()
    plan = engine.plan_next_transition(
        current_track=current_track,
        candidate_pool=pool,
        session_state=session,
    )

    print("=" * 72)
    print("[ATV DETERMINISTIC DJ ENGINE - SIMULATION OUTPUT]")
    print("=" * 72)
    print(f"Current Track : {current_track.autor} - {current_track.titulo}")
    print(
        f"Properties    : {current_track.bpm} BPM | Key={current_track.camelot_key} | "
        f"Ending={current_track.ending_type} | Bar={current_track.bar_duration_ms:.1f}ms"
    )
    print("-" * 72)
    print("TRANSITION PLAN (JSON):")
    print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=True))
    print("=" * 72)


if __name__ == "__main__":
    main()
