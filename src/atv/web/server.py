"""Web Backend & Interactive DJ Deck server (FastAPI + WebSockets + Spotify Connect)."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from atv.config import SpotifyAppConfig
from atv.engine.decision_engine import HybridDJDecisionEngine
from atv.ingesta.pipeline import DeezerAcousticBrainzClient, SpotifyBaseMetadata, TrackHydrationPipeline
from atv.models import (
    CandidatePlaybook,
    DJActionPlan,
    DJTimelineEvent,
    DecisionEngineOutput,
    PlaybookArchetype,
    SessionState,
    TrackNode,
)
from atv.spotify.auth import SpotifyPKCEAuthenticator
from atv.spotify.player import PlaybackStatus, SpotifyConnectClient

app = FastAPI(title="ATV DJ Deck & Mix Preview Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App State
config = SpotifyAppConfig.from_env(".env")
auth = SpotifyPKCEAuthenticator(config)
spotify_client = SpotifyConnectClient(auth)
hydrator = TrackHydrationPipeline(
    acoustic_provider=DeezerAcousticBrainzClient(),
    lrc_provider=DeezerAcousticBrainzClient(),
)
engine = HybridDJDecisionEngine()


class DJSessionManager:
    """Coordinates real-time playback, background candidate pool hydration, and playbook override."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.is_running = False
        self.current_track: TrackNode | None = None
        self.raw_pool: list[SpotifyBaseMetadata] = []
        self.candidate_nodes: list[TrackNode] = []
        self.candidate_by_uri: dict[str, TrackNode] = {}
        self.played_ids: set[str] = set()
        self.played_artists: list[str] = []
        self.session_history: list[TrackNode] = []
        self.last_archetype: PlaybookArchetype | None = None
        self.forced_playbook: CandidatePlaybook | None = None
        self.current_decision: DecisionEngineOutput | None = None
        self.playback_status: PlaybackStatus | None = None
        self.active_device_id: str | None = None
        self.timeline_pending: list[DJTimelineEvent] = []
        self.timeline_done: list[dict[str, Any]] = []
        self.worker_thread: threading.Thread | None = None
        self.runner_thread: threading.Thread | None = None
        self.stop_signal = threading.Event()
        self.active_playlist_name = "Tus Me Gusta"

    def set_pool(self, raw_items: list[SpotifyBaseMetadata], name: str = "Tus Me Gusta") -> None:
        with self.lock:
            self.stop_signal.set()
            self.is_running = False
            self.raw_pool = list(raw_items)
            self.active_playlist_name = name
            self.candidate_nodes.clear()
            self.candidate_by_uri.clear()
            self.current_decision = None
            self.forced_playbook = None
            self.timeline_pending.clear()
            self.timeline_done.clear()

    def select_playbook_override(self, archetype: str) -> bool:
        with self.lock:
            if not self.current_decision:
                return False
            all_pbs = [self.current_decision.selected_playbook, *self.current_decision.alternative_playbooks]
            match = next((pb for pb in all_pbs if pb.archetype == archetype), None)
            if match:
                self.forced_playbook = match
                self.timeline_pending = list(match.action_plan.timeline)
                return True
        return False

    def select_candidate_override(self, track_uri: str) -> bool:
        """User manually chooses which track MUST play next; engine generates a playbook for it."""
        with self.lock:
            target_node = self.candidate_by_uri.get(track_uri)
            if not target_node or not self.current_track:
                return False

            session_state = SessionState(
                target_energy=config.target_energy,
                played_track_ids=self.played_ids,
                played_artistsHistory=self.played_artists,
                session_history=self.session_history,
                network_latency_offset_ms=config.network_latency_offset_ms,
                creativity_temperature=config.creativity_temperature,
                last_maneuver_archetype=self.last_archetype,
            )
            # Re-generate decision prioritizing this specific candidate
            sub_decision = engine.plan_stochastic_playbooks(
                current_track=self.current_track,
                candidate_pool=[target_node],
                session_state=session_state,
            )
            self.current_decision = sub_decision
            self.forced_playbook = sub_decision.selected_playbook
            self.timeline_pending = list(self.forced_playbook.action_plan.timeline)
            return True

    def start_dj(self) -> None:
        if self.is_running:
            return
        self.stop_signal.clear()
        self.is_running = True
        self.runner_thread = threading.Thread(target=self._run_session, daemon=True)
        self.runner_thread.start()

    def stop_dj(self) -> None:
        self.stop_signal.set()
        self.is_running = False

    def _run_session(self) -> None:
        if not self.raw_pool:
            return

        # 1. Detect device & current track
        status = spotify_client.get_playback_status()
        device_id = status.device_id if status else None
        if not device_id:
            devs = spotify_client.get_available_devices()
            if devs:
                device_id = devs[0].get("id")
        self.active_device_id = device_id

        # 2. Immediate hydration of initial track
        active_raw = self.raw_pool[0]
        if status:
            match = next((r for r in self.raw_pool if r.uri == status.track_uri), None)
            if match:
                active_raw = match

        init_node = hydrator.hydrate_track(active_raw)
        with self.lock:
            self.current_track = init_node
            self.candidate_by_uri[init_node.spotify_uri] = init_node

        if status and status.track_uri != init_node.spotify_uri and device_id:
            spotify_client.start_or_transfer_playback(
                track_uri=init_node.spotify_uri,
                position_ms=init_node.first_downbeat_ms,
                device_id=device_id,
            )

        # 3. Background worker for progressive candidate pool hydration
        remaining = [r for r in self.raw_pool if r.uri != init_node.spotify_uri]

        def _bg_worker() -> None:
            for raw in remaining:
                if self.stop_signal.is_set():
                    break
                try:
                    c_node = hydrator.hydrate_track(raw)
                    with self.lock:
                        self.candidate_nodes.append(c_node)
                        self.candidate_by_uri[c_node.spotify_uri] = c_node
                except Exception:
                    pass

        self.worker_thread = threading.Thread(target=_bg_worker, daemon=True)
        self.worker_thread.start()

        # 4. Main Transitions Loop
        while not self.stop_signal.is_set():
            # Wait for at least 1 candidate to be ready
            while not self.stop_signal.is_set():
                with self.lock:
                    avail = [c for c in self.candidate_nodes if c.track_id != self.current_track.track_id and c.track_id not in self.played_ids]
                if avail:
                    break
                time.sleep(0.5)

            if self.stop_signal.is_set():
                break

            session_state = SessionState(
                target_energy=config.target_energy,
                played_track_ids=self.played_ids,
                played_artistsHistory=self.played_artists,
                session_history=self.session_history,
                network_latency_offset_ms=config.network_latency_offset_ms,
                creativity_temperature=config.creativity_temperature,
                last_maneuver_archetype=self.last_archetype,
            )

            with self.lock:
                if not self.forced_playbook:
                    dec = engine.plan_stochastic_playbooks(
                        current_track=self.current_track,
                        candidate_pool=avail,
                        session_state=session_state,
                    )
                    self.current_decision = dec
                    winner = dec.selected_playbook
                else:
                    winner = self.forced_playbook
                    self.forced_playbook = None

                self.timeline_pending = list(winner.action_plan.timeline)
                self.last_archetype = winner.archetype

            # Execute events
            loop_counts: dict[int, int] = {}
            while self.timeline_pending and not self.stop_signal.is_set():
                pb_stat = spotify_client.get_playback_status()
                with self.lock:
                    self.playback_status = pb_stat

                if not pb_stat:
                    time.sleep(0.4)
                    continue

                ev = self.timeline_pending[0]
                if pb_stat.progress_ms >= ev.trigger_at_progress_ms:
                    # Dispatch to Spotify
                    if ev.action_type in ("SEEK_LOOP", "SECTION_JUMP") and ev.current_track_uri == ev.target_track_uri:
                        spotify_client.seek_position(ev.target_position_ms, device_id=self.active_device_id)
                    else:
                        spotify_client.start_or_transfer_playback(
                            track_uri=ev.target_track_uri,
                            position_ms=ev.target_position_ms,
                            device_id=self.active_device_id,
                        )

                    ev_id = id(ev)
                    times = loop_counts.get(ev_id, 0) + 1
                    loop_counts[ev_id] = times

                    with self.lock:
                        self.timeline_done.append({
                            "action_type": ev.action_type,
                            "trigger_ms": ev.trigger_at_progress_ms,
                            "target_ms": ev.target_position_ms,
                            "timestamp": time.time(),
                        })

                    if ev.action_type == "SEEK_LOOP" and times < ev.loop_iterations:
                        time.sleep(0.6)
                    else:
                        with self.lock:
                            if self.timeline_pending:
                                self.timeline_pending.pop(0)
                        if ev.action_type == "TRACK_TRANSITION":
                            break
                        time.sleep(0.6)
                else:
                    time.sleep(0.4)

            # Advance to next song
            with self.lock:
                self.played_ids.add(self.current_track.track_id)
                self.played_artists.append(self.current_track.autor)
                self.session_history.append(self.current_track)
                next_node = self.candidate_by_uri.get(winner.action_plan.next_track_uri)
                if next_node:
                    self.current_track = next_node
                else:
                    break


session_mgr = DJSessionManager()


# --- API Endpoints ---

@app.get("/api/playlists")
def list_playlists() -> list[dict[str, Any]]:
    """Return Spotify playlists + default Liked Songs option."""
    playlists = [
        {
            "id": "liked",
            "name": "❤️ Canciones que te gustan (Liked Songs)",
            "uri": "spotify:liked",
            "tracks_count": "Biblioteca",
            "image": None,
        }
    ]
    try:
        user_lists = spotify_client.get_user_playlists(limit=50)
        playlists.extend(user_lists)
    except Exception:
        pass
    return playlists


@app.post("/api/session/load")
def load_playlist(payload: dict[str, str]) -> dict[str, Any]:
    """Load tracks from a playlist or liked songs and prime the DJ session."""
    target = payload.get("playlist_id", "liked").strip()
    name = payload.get("name", "Playlist").strip()

    if target == "liked" or target == "spotify:liked":
        raws = spotify_client.fetch_saved_tracks(limit=30)
        name = "Canciones que te gustan"
    else:
        raws = spotify_client.fetch_playlist_tracks(target, limit=30)

    if not raws:
        raise HTTPException(status_code=400, detail="No se encontraron canciones en esta playlist.")

    session_mgr.set_pool(raws, name=name)
    session_mgr.start_dj()
    return {"status": "ok", "tracks_loaded": len(raws), "playlist_name": name}


@app.post("/api/session/control")
def control_session(payload: dict[str, str]) -> dict[str, Any]:
    cmd = payload.get("action", "").lower()
    if cmd == "start":
        session_mgr.start_dj()
    elif cmd == "stop":
        session_mgr.stop_dj()
    elif cmd == "override_archetype":
        arch = payload.get("archetype", "")
        session_mgr.select_playbook_override(arch)
    elif cmd == "override_track":
        uri = payload.get("track_uri", "")
        session_mgr.select_candidate_override(uri)
    return {"status": "ok", "is_running": session_mgr.is_running}


@app.get("/api/session/state")
def get_session_state() -> dict[str, Any]:
    """Live state for frontend UI: current track, next playbooks, candidate pool, and preview sync audio."""
    with session_mgr.lock:
        curr = session_mgr.current_track
        dec = session_mgr.current_decision

        curr_data = None
        if curr:
            curr_data = {
                "track_id": curr.track_id,
                "spotify_uri": curr.spotify_uri,
                "titulo": curr.titulo,
                "autor": curr.autor,
                "bpm": curr.bpm,
                "camelot_key": curr.camelot_key,
                "duration_ms": curr.duration_ms,
                "preview_url": curr.preview_url,
                "secciones": [
                    {
                        "tipo": s.tipo,
                        "start_ms": s.start_ms,
                        "end_ms": s.end_ms,
                        "tiene_vocal": s.tiene_vocal,
                        "energia_local": s.energia_local,
                    }
                    for s in curr.secciones
                ],
            }

        playbooks_data = []
        if dec:
            all_pbs = [dec.selected_playbook, *dec.alternative_playbooks]
            for pb in all_pbs:
                next_cand = session_mgr.candidate_by_uri.get(pb.action_plan.next_track_uri)
                playbooks_data.append({
                    "archetype": pb.archetype,
                    "raw_score": pb.raw_score,
                    "selection_probability": pb.selection_probability,
                    "is_selected": pb.archetype == dec.selected_playbook.archetype,
                    "next_track": {
                        "uri": pb.action_plan.next_track_uri,
                        "titulo": next_cand.titulo if next_cand else "Next Song",
                        "autor": next_cand.autor if next_cand else "Artist",
                        "bpm": next_cand.bpm if next_cand else 120.0,
                        "camelot_key": next_cand.camelot_key if next_cand else "8A",
                        "preview_url": next_cand.preview_url if next_cand else None,
                    },
                    "timeline": [
                        {
                            "action_type": ev.action_type,
                            "trigger_ms": ev.trigger_at_progress_ms,
                            "target_ms": ev.target_position_ms,
                            "from_section": ev.from_section_type,
                            "to_section": ev.to_section_type,
                            "bars": ev.bars_duration,
                            "iterations": ev.loop_iterations,
                        }
                        for ev in pb.action_plan.timeline
                    ],
                })

        candidates_pool_data = [
            {
                "uri": c.spotify_uri,
                "titulo": c.titulo,
                "autor": c.autor,
                "bpm": c.bpm,
                "camelot_key": c.camelot_key,
                "preview_url": c.preview_url,
                "is_played": c.track_id in session_mgr.played_ids,
            }
            for c in session_mgr.candidate_nodes
        ]

        # Live Spotify progress
        progress_ms = 0
        if session_mgr.playback_status:
            progress_ms = session_mgr.playback_status.progress_ms

        next_trigger_ms = 0
        if session_mgr.timeline_pending:
            next_trigger_ms = session_mgr.timeline_pending[0].trigger_at_progress_ms

        return {
            "is_running": session_mgr.is_running,
            "active_playlist": session_mgr.active_playlist_name,
            "current_track": curr_data,
            "progress_ms": progress_ms,
            "next_trigger_ms": next_trigger_ms,
            "playbooks": playbooks_data,
            "candidate_pool": candidates_pool_data,
            "timeline_pending": [
                {
                    "action_type": ev.action_type,
                    "trigger_ms": ev.trigger_at_progress_ms,
                    "target_ms": ev.target_position_ms,
                    "from_section": ev.from_section_type,
                    "to_section": ev.to_section_type,
                }
                for ev in session_mgr.timeline_pending
            ],
        }


# --- Static Front-end UI (Desktop Web App) ---

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if html_path.is_file():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ATV UI is loading...</h1>")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
