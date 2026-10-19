# ATV — Deterministic Autonomous DJ Decision Engine

Purely mathematical and deterministic DJ decision engine (Zero LLMs) designed for Spotify Premium playback orchestration. Combines **NumPy Vectorial Pre-Filtering** with a **Directed Weighted Structural Graph** to compute optimal track transitions, exact bar-quantized cue points, and anti-vocal-clash rules.

## Architecture

1. **Data Model (`src/atv/models.py`)**:
   - `TrackNode`: Global track representation with strict Pydantic validation, automatic `bar_duration_ms` calculation (`(60000 / bpm) * time_signature`), and L2-normalized 7D `feature_vector`.
   - `TrackSection`: Structural sub-node (`intro`, `verso`, `estribillo`, `build_up`, `drop`, `puente`, `outro`) with spectral/instrument role layers (`tiene_vocal`, `percusion_fuerte`, `bajo_predominante`, `elemento_melodico`).
   - `TransitionPlan`: Execution contract for the Spotify player.
2. **Harmonic Engine (`src/atv/harmony/camelot.py`)**:
   - Pitch Class + Mode and key string translation to the Camelot Wheel (`1A`–`12B`) and circular distance calculation.
3. **OSINT & DSP Ingestion Pipeline (`src/atv/ingesta/`)**:
   - `lrc_parser.py`: Deterministic `.lrc` synchronized lyrics parser that detects silence gaps and stanza repetition to classify structural sections and vocal presence.
   - `dsp_analyzer.py`: Pure NumPy RMS envelope & onset flux analyzer for `first_downbeat_ms`, `energia_global`, and `ending_type` (`cold_end` vs `fade_out`).
   - `pipeline.py`: Hydrates `TrackNode` instances from Spotify metadata + Deezer/LRCLIB OSINT + DSP analysis.
4. **Hybrid Decision Engine (`src/atv/engine/`)**:
   - **Phase 1 (`vector_filter.py`)**: Hard filters (consecutive author exclusion, `time_signature` match, configurable BPM tolerance + Double/Half time) and vectorized macro ranking (Camelot exponential decay, BPM proximity, Jaccard genre index, target energy, language, and era).
   - **Phase 2 (`decision_engine.py`)**: Evaluates the Cartesian product of valid exit/entry sections ($A_{\text{out}} \times B_{\text{in}}$) enforcing Anti-Vocal Clash, Spectral Continuity, Classic/Power/Drop-Swap strategies, and Beatgrid Snapping.

## Running Tests

```bash
python -m pytest -v
```
