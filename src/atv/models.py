"""Strict data models for TrackNode, TrackSection, SessionState, TransitionPlan, and DJActionPlan."""

from __future__ import annotations

from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from atv.harmony.camelot import camelot_to_circular_coords, parse_camelot

GrooveType = Literal["4x4", "dembow", "sincopado", "breakbeat", "acustico_libre"]
EndingType = Literal["cold_end", "fade_out"]
SectionType = Literal["intro", "verso", "estribillo", "build_up", "drop", "puente", "outro"]
TransitionType = Literal["CROSSFADE", "QUICK_CUT", "COLD_END_START", "DROP_SWAP"]
DJActionType = Literal["SEEK_LOOP", "SECTION_JUMP", "TRACK_TRANSITION", "CALLBACK_TEASE"]


class TrackSection(BaseModel):
    """Sub-node representing a temporal section (Cue Point) of a track."""

    model_config = ConfigDict(extra="forbid")

    tipo: SectionType
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    energia_local: float = Field(ge=0.0, le=1.0)

    # Spectral Roles / Instrument Families
    tiene_vocal: bool
    percusion_fuerte: bool
    bajo_predominante: bool
    elemento_melodico: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_interval(self) -> TrackSection:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"end_ms ({self.end_ms}) must be greater than start_ms ({self.start_ms})")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class TrackNode(BaseModel):
    """Global song node combining Spotify metadata, OSINT enrichment, and DSP structures."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Identity & Context
    track_id: str
    spotify_uri: str
    isrc: str
    titulo: str
    autor: str
    generos: list[str] = Field(default_factory=list)
    anio: int = Field(ge=1900, le=2100)
    idioma: str = "en"
    duration_ms: int = Field(gt=0)

    # Rhythmic Physics
    bpm: float = Field(gt=0.0)
    time_signature: int = Field(default=4, ge=1, le=12)
    groove_type: GrooveType = "4x4"
    first_downbeat_ms: int = Field(default=0, ge=0)

    # Global Harmony & Vibe
    camelot_key: str
    energia_global: float = Field(ge=0.0, le=1.0)
    valencia_mood: float = Field(ge=0.0, le=1.0)

    # Morphology
    ending_type: EndingType = "cold_end"
    secciones: list[TrackSection] = Field(default_factory=list)
    feature_vector: np.ndarray = Field(default_factory=lambda: np.zeros(7, dtype=np.float64))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bar_duration_ms(self) -> float:
        """Duration of a single musical bar in milliseconds: (60000 / bpm) * time_signature."""
        return (60000.0 / self.bpm) * self.time_signature

    @field_validator("camelot_key")
    @classmethod
    def _validate_camelot(cls, v: str) -> str:
        num, letter = parse_camelot(v)
        return f"{num}{letter}"

    @model_validator(mode="after")
    def _compute_normalized_feature_vector(self) -> TrackNode:
        """Build or normalize the 7D macro feature vector automatically if not explicitly provided."""
        if self.feature_vector is None or self.feature_vector.size == 0 or np.all(self.feature_vector == 0):
            self.feature_vector = self.build_feature_vector()
        else:
            arr = np.asarray(self.feature_vector, dtype=np.float64)
            norm = np.linalg.norm(arr)
            self.feature_vector = arr / norm if norm > 1e-12 else arr
        return self

    def build_feature_vector(self) -> np.ndarray:
        """Construct the L2-normalized numerical feature vector for fast similarity lookup."""
        bpm_norm = float(np.clip((self.bpm - 60.0) / 140.0, 0.0, 1.0))
        c_sin, c_cos, c_ring = camelot_to_circular_coords(self.camelot_key)
        year_norm = float(np.clip((self.anio - 1950.0) / 80.0, 0.0, 1.0))

        raw = np.array(
            [
                bpm_norm,
                c_sin,
                c_cos,
                c_ring,
                self.energia_global,
                self.valencia_mood,
                year_norm,
            ],
            dtype=np.float64,
        )
        norm = np.linalg.norm(raw)
        return raw / norm if norm > 1e-12 else raw


class SessionState(BaseModel):
    """Runtime context of the active DJ session, including Spotify Connect latency compensation."""

    model_config = ConfigDict(extra="forbid")

    target_energy: float = Field(ge=0.0, le=1.0)
    played_track_ids: set[str] = Field(default_factory=set)
    played_artistsHistory: list[str] = Field(default_factory=list)
    session_history: list[TrackNode] = Field(default_factory=list)
    bpm_tolerance_pct: float = Field(default=0.06, gt=0.0, le=0.50)
    allow_double_half_time: bool = True
    top_k_candidates: int = Field(default=15, ge=1)
    network_latency_offset_ms: int = Field(default=250, ge=0, le=2000)
    enable_callback_tease: bool = False


class TransitionPlan(BaseModel):
    """Deterministic output contract for an inter-track transition."""

    model_config = ConfigDict(extra="forbid")

    next_track_uri: str
    transition_type: TransitionType
    from_section_type: str
    to_section_type: str
    trigger_at_progress_ms: int      # Exact ms in Track A to trigger transition
    start_next_at_position_ms: int   # Exact ms in Track B to start playback (position_ms in Spotify API)
    recommended_fade_duration_ms: int # 0 if QUICK_CUT
    transition_score: float          # Final decision score
    reasoning_log: list[str]         # Deterministic audit trail explaining why this edge won


class DJTimelineEvent(BaseModel):
    """Single deterministic event in the DJ execution timeline (Intra-Track or Inter-Track)."""

    model_config = ConfigDict(extra="forbid")

    action_type: DJActionType
    current_track_uri: str
    target_track_uri: str
    from_section_type: str
    to_section_type: str
    quantized_loop_end_ms: int | None = None  # Musical bar boundary M before latency offset
    trigger_at_progress_ms: int               # HTTP dispatch timestamp (end_loop_ms - network_latency_offset_ms for loops)
    target_position_ms: int                   # Destination ms (start_loop_ms or position_ms in target track)
    bars_duration: int = Field(default=0, ge=0)
    loop_iterations: int = Field(default=0, ge=0)
    transition_type: TransitionType | None = None
    recommended_fade_duration_ms: int = Field(default=0, ge=0)
    event_score: float = Field(default=1.0)
    reasoning_log: list[str] = Field(default_factory=list)


class DJActionPlan(BaseModel):
    """Chronological sequence of DJ commands (Intra-Track Loops, Section Jumps, Callbacks, and Track Transitions)."""

    model_config = ConfigDict(extra="forbid")

    source_track_uri: str
    next_track_uri: str
    network_latency_offset_ms: int = 250
    timeline: list[DJTimelineEvent] = Field(default_factory=list)
    final_transition: TransitionPlan
    overall_score: float
    reasoning_log: list[str] = Field(default_factory=list)
