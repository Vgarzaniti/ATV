"""ATV Deterministic Autonomous DJ Decision Engine."""

from atv.engine.decision_engine import (
    HybridDJDecisionEngine,
    StochasticSelector,
    calculate_quantized_loop,
    evaluate_intra_track_edges,
    plan_dj_action_timeline,
    plan_next_transition,
    plan_stochastic_playbooks,
    snap_to_beatgrid,
)
from atv.harmony.camelot import camelot_distance, key_string_to_camelot, pitch_class_to_camelot
from atv.ingesta.lrc_parser import infer_sections_from_lrc
from atv.ingesta.pipeline import OsintAcousticFeatures, SpotifyBaseMetadata, TrackHydrationPipeline
from atv.models import (
    CandidatePlaybook,
    DJActionPlan,
    DJTimelineEvent,
    DecisionEngineOutput,
    PlaybookArchetype,
    SessionState,
    TrackNode,
    TrackSection,
    TransitionPlan,
)

__all__ = [
    "CandidatePlaybook",
    "DJActionPlan",
    "DJTimelineEvent",
    "DecisionEngineOutput",
    "HybridDJDecisionEngine",
    "OsintAcousticFeatures",
    "PlaybookArchetype",
    "SessionState",
    "SpotifyBaseMetadata",
    "StochasticSelector",
    "TrackHydrationPipeline",
    "TrackNode",
    "TrackSection",
    "TransitionPlan",
    "calculate_quantized_loop",
    "camelot_distance",
    "evaluate_intra_track_edges",
    "infer_sections_from_lrc",
    "key_string_to_camelot",
    "pitch_class_to_camelot",
    "plan_dj_action_timeline",
    "plan_next_transition",
    "plan_stochastic_playbooks",
    "snap_to_beatgrid",
]
