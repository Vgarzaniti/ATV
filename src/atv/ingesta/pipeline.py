"""OSINT & Spotify Ingestion Pipeline to hydrate TrackNode instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import httpx
import numpy as np

from atv.harmony.camelot import key_string_to_camelot, pitch_class_to_camelot
from atv.ingesta.dsp_analyzer import DspAnalysisResult, analyze_waveform_numpy
from atv.ingesta.lrc_parser import build_fallback_instrumental_sections, infer_sections_from_lrc
from atv.models import EndingType, GrooveType, TrackNode


@dataclass(frozen=True)
class SpotifyBaseMetadata:
    """Base metadata extracted from Spotify Web API (/v1/tracks/{id})."""

    track_id: str
    uri: str
    name: str
    artists: list[str]
    duration_ms: int
    explicit: bool
    release_date: str
    isrc: str

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else "Unknown Artist"

    @property
    def release_year(self) -> int:
        if len(self.release_date) >= 4 and self.release_date[:4].isdigit():
            return int(self.release_date[:4])
        return 2020

    @classmethod
    def from_spotify_dict(cls, payload: dict[str, Any]) -> SpotifyBaseMetadata:
        artists = [a.get("name", "") for a in payload.get("artists", []) if a.get("name")]
        album = payload.get("album", {})
        ext_ids = payload.get("external_ids", {})
        return cls(
            track_id=str(payload["id"]),
            uri=str(payload["uri"]),
            name=str(payload["name"]),
            artists=artists,
            duration_ms=int(payload["duration_ms"]),
            explicit=bool(payload.get("explicit", False)),
            release_date=str(album.get("release_date", payload.get("release_date", "2020"))),
            isrc=str(ext_ids.get("isrc", payload.get("isrc", ""))),
        )


@dataclass(frozen=True)
class OsintAcousticFeatures:
    """Acoustic metadata gathered from Deezer / AcousticBrainz / GetSongBPM."""

    bpm: float = 120.0
    pitch_class: int | None = None
    mode: int | None = None
    camelot_key: str | None = None
    time_signature: int = 4
    genres: list[str] = field(default_factory=list)
    preview_mp3_url: str | None = None
    energy: float = 0.70
    valence: float = 0.60
    language: str = "en"
    groove_type: GrooveType = "4x4"


class OsintAcousticProvider(Protocol):
    def fetch_by_isrc_or_query(self, isrc: str, artist: str, title: str) -> OsintAcousticFeatures:
        ...


class LrcProvider(Protocol):
    def fetch_synced_lyrics(self, artist: str, title: str, duration_ms: int) -> str | None:
        ...


class DeezerAcousticBrainzClient:
    """HTTP OSINT client querying Deezer API (by ISRC) and LRCLIB API."""

    def __init__(self, timeout_s: float = 8.0) -> None:
        self._timeout = timeout_s

    def fetch_by_isrc_or_query(self, isrc: str, artist: str, title: str) -> OsintAcousticFeatures:
        bpm = 120.0
        preview_url: str | None = None
        genres: list[str] = []

        with httpx.Client(timeout=self._timeout) as client:
            if isrc:
                resp = client.get(f"https://api.deezer.com/track/isrc:{isrc}")
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" not in data:
                        if data.get("bpm") and float(data["bpm"]) > 0:
                            bpm = float(data["bpm"])
                        preview_url = data.get("preview") or None

            if bpm == 120.0 and not preview_url:
                q = f'artist:"{artist}" track:"{title}"'
                resp = client.get("https://api.deezer.com/search", params={"q": q, "limit": 1})
                if resp.status_code == 200:
                    items = resp.json().get("data", [])
                    if items:
                        preview_url = items[0].get("preview") or None

        return OsintAcousticFeatures(
            bpm=bpm,
            pitch_class=9,  # Default A minor (8A) if external key provider unavailable
            mode=0,
            time_signature=4,
            genres=genres,
            preview_mp3_url=preview_url,
        )

    def fetch_synced_lyrics(self, artist: str, title: str, duration_ms: int) -> str | None:
        duration_sec = round(duration_ms / 1000.0)
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                "https://lrclib.net/api/get",
                params={
                    "artist_name": artist,
                    "track_name": title,
                    "duration": duration_sec,
                },
            )
            if resp.status_code == 200:
                payload = resp.json()
                synced = payload.get("syncedLyrics")
                if isinstance(synced, str) and synced.strip():
                    return synced
        return None


def infer_groove_from_genres_and_bpm(genres: list[str], bpm: float) -> GrooveType:
    """Deterministically infer GrooveType from genre tags and tempo."""
    g_text = " ".join(g.lower() for g in genres)
    if any(k in g_text for k in ("reggaeton", "urbano", "dembow", "latin", "perreo")):
        return "dembow"
    if any(k in g_text for k in ("drum and bass", "dnb", "jungle", "breakbeat", "uk garage")):
        return "breakbeat"
    if any(k in g_text for k in ("funk", "salsa", "jazz", "afrobeat", "cumbia")):
        return "sincopado"
    if any(k in g_text for k in ("acoustic", "folk", "ambient", "classical", "ballad")):
        return "acustico_libre"
    if 88.0 <= bpm <= 102.0 and "pop" not in g_text:
        return "dembow"
    return "4x4"


class TrackHydrationPipeline:
    """Hydrates a full TrackNode from Spotify metadata, OSINT providers, LRC lyrics, and optional DSP."""

    def __init__(
        self,
        acoustic_provider: OsintAcousticProvider | None = None,
        lrc_provider: LrcProvider | None = None,
    ) -> None:
        self._acoustic_provider = acoustic_provider
        self._lrc_provider = lrc_provider

    def hydrate_track(
        self,
        spotify_meta: SpotifyBaseMetadata,
        osint_features: OsintAcousticFeatures | None = None,
        lrc_text: str | None = None,
        waveform_samples: np.ndarray | None = None,
        sample_rate: int = 22050,
    ) -> TrackNode:
        """Assemble and validate a complete TrackNode."""
        if osint_features is None:
            if self._acoustic_provider is not None:
                osint_features = self._acoustic_provider.fetch_by_isrc_or_query(
                    spotify_meta.isrc,
                    spotify_meta.primary_artist,
                    spotify_meta.name,
                )
            else:
                osint_features = OsintAcousticFeatures()

        if lrc_text is None and self._lrc_provider is not None:
            lrc_text = self._lrc_provider.fetch_synced_lyrics(
                spotify_meta.primary_artist,
                spotify_meta.name,
                spotify_meta.duration_ms,
            )

        # Resolve Camelot Key
        if osint_features.camelot_key:
            camelot = key_string_to_camelot(osint_features.camelot_key)
        elif osint_features.pitch_class is not None and osint_features.mode is not None:
            camelot = pitch_class_to_camelot(osint_features.pitch_class, osint_features.mode)
        else:
            camelot = "8A"

        # DSP Analysis (if waveform provided)
        dsp_res: DspAnalysisResult | None = None
        if waveform_samples is not None:
            dsp_res = analyze_waveform_numpy(waveform_samples, sample_rate=sample_rate)

        global_energy = dsp_res.rms_energy if dsp_res is not None else osint_features.energy
        first_downbeat_ms = dsp_res.first_downbeat_ms if dsp_res is not None else 0
        ending_type: EndingType = dsp_res.ending_type if dsp_res is not None else "cold_end"

        # Structural Section Inference via LRC
        if lrc_text and lrc_text.strip():
            sections = infer_sections_from_lrc(
                lrc_content=lrc_text,
                duration_ms=spotify_meta.duration_ms,
                base_energy=global_energy,
            )
            idioma = osint_features.language
        else:
            sections = build_fallback_instrumental_sections(
                duration_ms=spotify_meta.duration_ms,
                base_energy=global_energy,
            )
            idioma = osint_features.language or "instrumental"

        groove = (
            osint_features.groove_type
            if osint_features.groove_type != "4x4"
            else infer_groove_from_genres_and_bpm(osint_features.genres, osint_features.bpm)
        )

        return TrackNode(
            track_id=spotify_meta.track_id,
            spotify_uri=spotify_meta.uri,
            isrc=spotify_meta.isrc,
            titulo=spotify_meta.name,
            autor=spotify_meta.primary_artist,
            generos=list(osint_features.genres),
            anio=spotify_meta.release_year,
            idioma=idioma,
            duration_ms=spotify_meta.duration_ms,
            bpm=osint_features.bpm,
            time_signature=osint_features.time_signature,
            groove_type=groove,
            first_downbeat_ms=first_downbeat_ms,
            camelot_key=camelot,
            energia_global=global_energy,
            valencia_mood=osint_features.valence,
            ending_type=ending_type,
            secciones=sections,
        )
