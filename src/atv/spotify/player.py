"""Spotify Connect Web API client and Live Autonomous DJ Executor."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any
import httpx

from atv.config import SpotifyAppConfig
from atv.engine.decision_engine import HybridDJDecisionEngine
from atv.ingesta.pipeline import DeezerAcousticBrainzClient, SpotifyBaseMetadata, TrackHydrationPipeline
from atv.models import DJActionPlan, DJTimelineEvent, PlaybookArchetype, SessionState, TrackNode
from atv.spotify.auth import SpotifyPKCEAuthenticator

SPOTIFY_API_BASE = "https://api.spotify.com/v1"


def extract_playlist_id(playlist_input: str) -> str:
    """Extract the bare Spotify playlist ID from a URL, URI, or raw ID string."""
    cleaned = playlist_input.strip()
    if cleaned.startswith("spotify:playlist:"):
        return cleaned.split(":")[-1]
    match = re.search(r"playlist/([a-zA-Z0-9]+)", cleaned)
    if match:
        return match.group(1)
    return cleaned


@dataclass(frozen=True)
class PlaybackStatus:
    is_playing: bool
    progress_ms: int
    track_id: str
    track_uri: str
    track_name: str
    artist_name: str
    device_id: str | None
    volume_percent: int


class SpotifyConnectClient:
    """HTTP wrapper for Spotify Web API `/v1/me/player` and `/v1/playlists`."""

    def __init__(self, authenticator: SpotifyPKCEAuthenticator, timeout_s: float = 8.0) -> None:
        self._auth = authenticator
        self._timeout = timeout_s

    def _headers(self) -> dict[str, str]:
        token = self._auth.get_valid_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_current_user_profile(self) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{SPOTIFY_API_BASE}/me", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    def get_playback_status(self) -> PlaybackStatus | None:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{SPOTIFY_API_BASE}/me/player", headers=self._headers())
            if resp.status_code == 204 or not resp.content:
                return None
            resp.raise_for_status()
            data = resp.json()

        item = data.get("item")
        if not item:
            return None
        device = data.get("device") or {}
        artists = [a.get("name", "") for a in item.get("artists", []) if a.get("name")]
        return PlaybackStatus(
            is_playing=bool(data.get("is_playing", False)),
            progress_ms=int(data.get("progress_ms") or 0),
            track_id=str(item.get("id", "")),
            track_uri=str(item.get("uri", "")),
            track_name=str(item.get("name", "")),
            artist_name=artists[0] if artists else "Unknown",
            device_id=device.get("id"),
            volume_percent=int(device.get("volume_percent") or 100),
        )

    def fetch_saved_tracks(self, limit: int = 50) -> list[SpotifyBaseMetadata]:
        """Fetch saved/liked tracks (canciones que te gustan) from Spotify library (/v1/me/tracks)."""
        results: list[SpotifyBaseMetadata] = []
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                f"{SPOTIFY_API_BASE}/me/tracks",
                headers=self._headers(),
                params={"limit": min(limit, 50)},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            for entry in items:
                track_obj = entry.get("track")
                if not track_obj or not track_obj.get("id"):
                    continue
                results.append(SpotifyBaseMetadata.from_spotify_dict(track_obj))
        return results

    def fetch_playlist_tracks(self, playlist_input: str, limit: int = 50) -> list[SpotifyBaseMetadata]:
        """Fetch track metadata items (including ISRC) from a Spotify playlist."""
        pid = extract_playlist_id(playlist_input)
        results: list[SpotifyBaseMetadata] = []
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                f"{SPOTIFY_API_BASE}/playlists/{pid}/tracks",
                headers=self._headers(),
                params={"limit": min(limit, 100)},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            for entry in items:
                track_obj = entry.get("track")
                if not track_obj or not track_obj.get("id"):
                    continue
                results.append(SpotifyBaseMetadata.from_spotify_dict(track_obj))
        return results

    def start_or_transfer_playback(self, track_uri: str, position_ms: int = 0, device_id: str | None = None) -> None:
        """Trigger `PUT /v1/me/player/play` at an exact millisecond cue point."""
        params: dict[str, Any] = {}
        if device_id:
            params["device_id"] = device_id
        body = {
            "uris": [track_uri],
            "position_ms": max(0, int(position_ms)),
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.put(
                f"{SPOTIFY_API_BASE}/me/player/play",
                headers=self._headers(),
                params=params,
                json=body,
            )
            resp.raise_for_status()

    def seek_position(self, position_ms: int, device_id: str | None = None) -> None:
        """Trigger `PUT /v1/me/player/seek` for intra-track loops and section jumps."""
        params: dict[str, Any] = {"position_ms": max(0, int(position_ms))}
        if device_id:
            params["device_id"] = device_id
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.put(
                f"{SPOTIFY_API_BASE}/me/player/seek",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()

    def set_volume(self, volume_percent: int, device_id: str | None = None) -> None:
        params: dict[str, Any] = {"volume_percent": max(0, min(100, int(volume_percent)))}
        if device_id:
            params["device_id"] = device_id
        with httpx.Client(timeout=self._timeout) as client:
            client.put(
                f"{SPOTIFY_API_BASE}/me/player/volume",
                headers=self._headers(),
                params=params,
            )


class LiveDJSessionRunner:
    """Real-time orchestrator polling Spotify Connect and firing DJActionPlan events on the beatgrid."""

    def __init__(
        self,
        spotify_client: SpotifyConnectClient,
        config: SpotifyAppConfig,
    ) -> None:
        self._spotify = spotify_client
        self._config = config
        osint_client = DeezerAcousticBrainzClient()
        self._hydrator = TrackHydrationPipeline(acoustic_provider=osint_client, lrc_provider=osint_client)
        self._engine = HybridDJDecisionEngine()

    def hydrate_tracks(self, raw_tracks: list[SpotifyBaseMetadata]) -> list[TrackNode]:
        hydrated: list[TrackNode] = []
        print(f"Hydrating {len(raw_tracks)} tracks via OSINT (Deezer BPM/Preview + LRCLIB Synced Lyrics)...")
        for idx, meta in enumerate(raw_tracks, start=1):
            node = self._hydrator.hydrate_track(meta)
            hydrated.append(node)
            print(
                f"  [{idx:02d}/{len(raw_tracks):02d}] {node.autor} - {node.titulo} | "
                f"{node.bpm:.1f} BPM | Key={node.camelot_key} | Sections={len(node.secciones)}"
            )
        return hydrated

    def hydrate_playlist_pool(self, playlist_input: str, max_tracks: int = 25) -> list[TrackNode]:
        raw_tracks = self._spotify.fetch_playlist_tracks(playlist_input, limit=max_tracks)
        return self.hydrate_tracks(raw_tracks)

    def hydrate_liked_tracks_pool(self, max_tracks: int = 25) -> list[TrackNode]:
        raw_tracks = self._spotify.fetch_saved_tracks(limit=max_tracks)
        return self.hydrate_tracks(raw_tracks)

    def execute_timeline_event(self, ev: DJTimelineEvent, device_id: str | None = None) -> None:
        """Dispatch the corresponding Spotify Connect API call for a DJTimelineEvent."""
        if ev.action_type in ("SEEK_LOOP", "SECTION_JUMP") and ev.current_track_uri == ev.target_track_uri:
            print(
                f"\n[LIVE ACTION -> {ev.action_type}] Seeking back to {ev.target_position_ms}ms "
                f"({ev.from_section_type} -> {ev.to_section_type}, {ev.bars_duration} bars)"
            )
            self._spotify.seek_position(ev.target_position_ms, device_id=device_id)
        else:
            print(
                f"\n[LIVE ACTION -> {ev.action_type}] Switching to {ev.target_track_uri} "
                f"at position_ms={ev.target_position_ms}ms (Mode={ev.transition_type})"
            )
            self._spotify.start_or_transfer_playback(
                track_uri=ev.target_track_uri,
                position_ms=ev.target_position_ms,
                device_id=device_id,
            )

    def run_live_loop(self, track_pool: list[TrackNode], max_transitions: int = 10) -> None:
        """Monitor active Spotify playback and execute stochastic DJ playbooks in real time."""
        if len(track_pool) < 2:
            raise ValueError("At least 2 tracks are required in the pool to run a live DJ session.")

        pool_by_uri = {t.spotify_uri: t for t in track_pool}
        played_ids: set[str] = set()
        played_artists: list[str] = []
        session_history: list[TrackNode] = []
        last_archetype: PlaybookArchetype | None = None

        status = self._spotify.get_playback_status()
        if status is None:
            raise RuntimeError(
                "No active Spotify playback device detected. Open Spotify on your phone, desktop, or web player, "
                "press play on any song so Spotify Connect registers the active device, and run again."
            )

        device_id = status.device_id
        current_track = pool_by_uri.get(status.track_uri, track_pool[0])
        if status.track_uri != current_track.spotify_uri:
            print(f"Starting initial track from pool: {current_track.autor} - {current_track.titulo}")
            self._spotify.start_or_transfer_playback(
                track_uri=current_track.spotify_uri,
                position_ms=current_track.first_downbeat_ms,
                device_id=device_id,
            )
            time.sleep(1.0)

        for step in range(1, max_transitions + 1):
            session_state = SessionState(
                target_energy=self._config.target_energy,
                played_track_ids=played_ids,
                played_artistsHistory=played_artists,
                session_history=session_history,
                network_latency_offset_ms=self._config.network_latency_offset_ms,
                creativity_temperature=self._config.creativity_temperature,
                last_maneuver_archetype=last_archetype,
            )

            decision = self._engine.plan_stochastic_playbooks(
                current_track=current_track,
                candidate_pool=track_pool,
                session_state=session_state,
            )
            winner = decision.selected_playbook
            plan: DJActionPlan = winner.action_plan
            last_archetype = winner.archetype

            print("\n" + "=" * 88)
            print(f"[LIVE DJ DECK - TRACK #{step}] Now Playing: {current_track.autor} - {current_track.titulo}")
            print(
                f"Selected Playbook: [{winner.archetype}] (Prob={winner.selection_probability * 100:.1f}%, "
                f"Score={winner.raw_score:.1f}) -> Next: {plan.next_track_uri}"
            )
            for idx, ev in enumerate(plan.timeline, start=1):
                print(
                    f"  Queue #{idx}: {ev.action_type} at {ev.trigger_at_progress_ms}ms -> "
                    f"target {ev.target_position_ms}ms ({ev.from_section_type}->{ev.to_section_type})"
                )
            print("=" * 88)

            # Execute timeline events chronologically as Spotify progress_ms reaches each trigger_at_progress_ms
            pending_events = list(plan.timeline)
            loop_counts: dict[int, int] = {}

            while pending_events:
                playback = self._spotify.get_playback_status()
                if playback is None:
                    time.sleep(self._config.poll_interval_sec)
                    continue

                next_ev = pending_events[0]
                if playback.progress_ms >= next_ev.trigger_at_progress_ms:
                    self.execute_timeline_event(next_ev, device_id=device_id)
                    ev_id = id(next_ev)
                    executed_times = loop_counts.get(ev_id, 0) + 1
                    loop_counts[ev_id] = executed_times

                    if next_ev.action_type == "SEEK_LOOP" and executed_times < next_ev.loop_iterations:
                        # Keep event at front of queue for remaining loop_iterations
                        time.sleep(0.6)
                    else:
                        pending_events.pop(0)
                        if next_ev.action_type == "TRACK_TRANSITION":
                            break
                        time.sleep(0.6)
                else:
                    time.sleep(self._config.poll_interval_sec)

            played_ids.add(current_track.track_id)
            played_artists.append(current_track.autor)
            session_history.append(current_track)

            next_node = pool_by_uri.get(plan.next_track_uri)
            if next_node is None:
                break
            current_track = next_node
