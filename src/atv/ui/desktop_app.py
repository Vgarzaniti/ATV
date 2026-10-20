"""Native Desktop Application for ATV Autonomous DJ Studio (Tkinter DJ Deck GUI)."""

from __future__ import annotations

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import io
import math
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import httpx
import pygame

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

# Color Palette (Dark DJ Console)
BG_MAIN = "#0d0e12"
BG_CARD = "#151821"
BG_CARD_ACCENT = "#1b1f2b"
TEXT_MAIN = "#f3f4f6"
TEXT_MUTED = "#94a3b8"
ACCENT_GREEN = "#10b981"
ACCENT_CYAN = "#38bdf8"
ACCENT_ORANGE = "#f59e0b"
ACCENT_RED = "#f43f5e"
BORDER_COLOR = "#232838"


class DualAudioPreviewEngine:
    """Native audio player that streams Deezer 30s previews and mixes them with a crossfader."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.set_num_channels(8)
        self.channel_a = pygame.mixer.Channel(0)
        self.channel_b = pygame.mixer.Channel(1)
        self.sound_a: pygame.mixer.Sound | None = None
        self.sound_b: pygame.mixer.Sound | None = None
        self.is_playing = False
        self.crossfader_pos = 0.5  # 0.0 (Deck A) to 1.0 (Deck B)

    def _fetch_sound(self, url: str) -> pygame.mixer.Sound | None:
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return pygame.mixer.Sound(io.BytesIO(resp.content))
        except Exception:
            pass
        return None

    def start_dual_preview(self, url_a: str | None, url_b: str | None) -> None:
        def _loader() -> None:
            with self.lock:
                self.channel_a.stop()
                self.channel_b.stop()
                self.sound_a = self._fetch_sound(url_a) if url_a else None
                self.sound_b = self._fetch_sound(url_b) if url_b else None

                if self.sound_a:
                    self.channel_a.play(self.sound_a, loops=-1)
                if self.sound_b:
                    self.channel_b.play(self.sound_b, loops=-1)

                self.apply_crossfader(self.crossfader_pos)
                self.is_playing = True

        threading.Thread(target=_loader, daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            self.channel_a.stop()
            self.channel_b.stop()
            self.is_playing = False

    def apply_crossfader(self, value: float) -> None:
        """Constant-power crossfade curve between Track A and Track B."""
        self.crossfader_pos = max(0.0, min(1.0, value))
        vol_a = math.cos(self.crossfader_pos * 0.5 * math.pi)
        vol_b = math.cos((1.0 - self.crossfader_pos) * 0.5 * math.pi)
        self.channel_a.set_volume(vol_a)
        self.channel_b.set_volume(vol_b)


class DesktopDJApp(tk.Tk):
    """Main Desktop Window simulating a dual DJ Deck with Spotify live controls."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ATV — Autonomous DJ Studio")
        self.geometry("1180x760")
        self.minsize(980, 680)
        self.configure(bg=BG_MAIN)

        # Force window to top and visible
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)
        self.focus_force()

        # Core Backend State
        self.config = SpotifyAppConfig.from_env(".env")
        self.auth = SpotifyPKCEAuthenticator(self.config)
        self.spotify = SpotifyConnectClient(self.auth)
        self.hydrator = TrackHydrationPipeline(
            acoustic_provider=DeezerAcousticBrainzClient(),
            lrc_provider=DeezerAcousticBrainzClient(),
        )
        self.engine = HybridDJDecisionEngine()
        self.preview_engine = DualAudioPreviewEngine()

        # Session variables
        self.raw_pool: list[SpotifyBaseMetadata] = []
        self.candidate_pool: list[TrackNode] = []
        self.candidate_by_uri: dict[str, TrackNode] = {}
        self.current_track: TrackNode | None = None
        self.current_decision: DecisionEngineOutput | None = None
        self.forced_playbook: CandidatePlaybook | None = None
        self.timeline_pending: list[DJTimelineEvent] = []
        self.played_ids: set[str] = set()
        self.played_artists: list[str] = []
        self.session_history: list[TrackNode] = []
        self.last_archetype: PlaybookArchetype | None = None
        self.active_device_id: str | None = None
        self.is_dj_running = False
        self.stop_signal = threading.Event()
        self.pool_lock = threading.Lock()

        self._build_ui()
        self._load_playlists_async()

        # Cached status updated in background to avoid blocking the Tkinter event loop
        self.cached_playback_status: PlaybackStatus | None = None
        self._start_status_poller()

        self.after(200, self._periodic_ui_update)

    def _build_ui(self) -> None:
        # 1. Top Navigation Bar
        top_bar = tk.Frame(self, bg=BG_CARD, height=60, padx=16, pady=12, highlightthickness=1, highlightbackground=BORDER_COLOR)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        logo_lbl = tk.Label(top_bar, text="🎧 ATV DECK", font=("Segoe UI", 14, "bold"), fg=TEXT_MAIN, bg=BG_CARD)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 8))

        badge_lbl = tk.Label(top_bar, text="DESKTOP STUDIO", font=("Segoe UI", 8, "bold"), fg="#000", bg=ACCENT_GREEN, padx=6, pady=2)
        badge_lbl.pack(side=tk.LEFT)

        self.btn_toggle_dj = tk.Button(
            top_bar, text="▶ INICIAR DJ", font=("Segoe UI", 10, "bold"),
            bg=ACCENT_GREEN, fg="#000", activebackground="#059669", relief=tk.FLAT,
            padx=14, pady=4, cursor="hand2", command=self._toggle_dj_session
        )
        self.btn_toggle_dj.pack(side=tk.RIGHT, padx=6)

        self.btn_load_pl = tk.Button(
            top_bar, text="Cargar Playlist", font=("Segoe UI", 9),
            bg=BG_CARD_ACCENT, fg=TEXT_MAIN, relief=tk.FLAT, padx=12, pady=4,
            cursor="hand2", command=self._on_load_playlist
        )
        self.btn_load_pl.pack(side=tk.RIGHT, padx=6)

        self.pl_combo = ttk.Combobox(top_bar, width=38, state="readonly")
        self.pl_combo.pack(side=tk.RIGHT, padx=6)
        self.pl_combo.set("❤️ Canciones que te gustan (Liked Songs)")

        # 2. Main Content Layout (Left: Pool, Center: Deck & Options, Right: Timeline)
        content = tk.Frame(self, bg=BG_MAIN, padx=12, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=5)
        content.columnconfigure(2, weight=3)
        content.rowconfigure(0, weight=1)

        # === LEFT PANEL: Playlist & Candidate Pool ===
        p_left = tk.Frame(content, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=10, pady=10)
        p_left.grid(row=0, column=0, sticky="nsew", padx=6)

        lbl_cand_title = tk.Label(p_left, text="Bolsa de Canciones", font=("Segoe UI", 11, "bold"), fg=TEXT_MAIN, bg=BG_CARD)
        lbl_cand_title.pack(anchor="w")

        self.lbl_pool_status = tk.Label(p_left, text="0 temas cargados", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_pool_status.pack(anchor="w", pady=(0, 8))

        scroll_cand = tk.Scrollbar(p_left)
        scroll_cand.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox_candidates = tk.Listbox(
            p_left, bg=BG_MAIN, fg=TEXT_MAIN, font=("Segoe UI", 9),
            selectbackground=ACCENT_CYAN, selectforeground="#000",
            relief=tk.FLAT, highlightthickness=0, yscrollcommand=scroll_cand.set
        )
        self.listbox_candidates.pack(fill=tk.BOTH, expand=True)
        scroll_cand.config(command=self.listbox_candidates.yview)

        btn_force_cand = tk.Button(
            p_left, text="Forzar Siguiente Canción", font=("Segoe UI", 9, "bold"),
            bg=BG_CARD_ACCENT, fg=ACCENT_CYAN, relief=tk.FLAT, pady=6, cursor="hand2",
            command=self._on_force_candidate_track
        )
        btn_force_cand.pack(fill=tk.X, pady=(8, 0))

        # === CENTER PANEL: Deck A (Current), Sintonizador Dual, Deck B (Playbooks) ===
        p_center = tk.Frame(content, bg=BG_MAIN)
        p_center.grid(row=0, column=1, sticky="nsew", padx=6)

        # Deck A: Active Playing Track
        deck_card = tk.Frame(p_center, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=16, pady=14)
        deck_card.pack(fill=tk.X, pady=(0, 10))

        self.lbl_deck_tag = tk.Label(deck_card, text="● BANDEJA EN VIVO (SPOTIFY)", font=("Segoe UI", 8, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        self.lbl_deck_tag.pack(anchor="w")

        self.lbl_track_title = tk.Label(deck_card, text="Esperando reproducción...", font=("Segoe UI", 15, "bold"), fg=TEXT_MAIN, bg=BG_CARD)
        self.lbl_track_title.pack(anchor="w", pady=(2, 0))

        self.lbl_track_artist = tk.Label(deck_card, text="—", font=("Segoe UI", 11), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_track_artist.pack(anchor="w", pady=(0, 10))

        deck_meta_box = tk.Frame(deck_card, bg=BG_CARD)
        deck_meta_box.pack(anchor="w", pady=(0, 10))

        self.lbl_track_bpm = tk.Label(deck_meta_box, text="— BPM", font=("Consolas", 9, "bold"), fg=ACCENT_ORANGE, bg=BG_CARD_ACCENT, padx=8, pady=3)
        self.lbl_track_bpm.pack(side=tk.LEFT, padx=(0, 6))

        self.lbl_track_key = tk.Label(deck_meta_box, text="— KEY", font=("Consolas", 9, "bold"), fg=ACCENT_CYAN, bg=BG_CARD_ACCENT, padx=8, pady=3)
        self.lbl_track_key.pack(side=tk.LEFT, padx=(0, 6))

        self.lbl_track_secs = tk.Label(deck_meta_box, text="— Secciones", font=("Segoe UI", 9), fg=TEXT_MAIN, bg=BG_CARD_ACCENT, padx=8, pady=3)
        self.lbl_track_secs.pack(side=tk.LEFT)

        # Progress Bar
        self.progress_canvas = tk.Canvas(deck_card, height=12, bg="#202431", highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X, pady=(4, 4))

        time_box = tk.Frame(deck_card, bg=BG_CARD)
        time_box.pack(fill=tk.X)
        self.lbl_curr_time = tk.Label(time_box, text="00:00", font=("Consolas", 9), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_curr_time.pack(side=tk.LEFT)

        self.lbl_trigger_time = tk.Label(time_box, text="Enganche: --:--", font=("Consolas", 9, "bold"), fg=ACCENT_RED, bg=BG_CARD)
        self.lbl_trigger_time.pack(side=tk.LEFT, expand=True)

        self.lbl_total_time = tk.Label(time_box, text="00:00", font=("Consolas", 9), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_total_time.pack(side=tk.RIGHT)

        # Dual Sintonizador Preview (Crossfader)
        sintonizador_box = tk.Frame(p_center, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=14, pady=12)
        sintonizador_box.pack(fill=tk.X, pady=(0, 10))

        sinto_top = tk.Frame(sintonizador_box, bg=BG_CARD)
        sinto_top.pack(fill=tk.X, pady=(0, 6))
        tk.Label(sinto_top, text="Sintonizador de Previo (Audio Preview A ⟷ B)", font=("Segoe UI", 9, "bold"), fg=TEXT_MAIN, bg=BG_CARD).pack(side=tk.LEFT)
        self.lbl_sinto_status = tk.Label(sinto_top, text="Inactivo", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_sinto_status.pack(side=tk.RIGHT)

        sinto_ctrls = tk.Frame(sintonizador_box, bg=BG_CARD)
        sinto_ctrls.pack(fill=tk.X, pady=(0, 6))

        self.btn_preview_both = tk.Button(
            sinto_ctrls, text="▶ Escuchar Sintonización de Ambas Canciones", font=("Segoe UI", 9, "bold"),
            bg=BG_CARD_ACCENT, fg=ACCENT_CYAN, relief=tk.FLAT, padx=10, pady=4, cursor="hand2",
            command=self._on_play_dual_preview
        )
        self.btn_preview_both.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_stop_prev = tk.Button(
            sinto_ctrls, text="⏹ Detener", font=("Segoe UI", 9),
            bg=BG_CARD_ACCENT, fg=TEXT_MAIN, relief=tk.FLAT, padx=10, pady=4, cursor="hand2",
            command=self.preview_engine.stop
        )
        btn_stop_prev.pack(side=tk.RIGHT)

        self.crossfader_scale = tk.Scale(
            sintonizador_box, from_=0, to=100, orient=tk.HORIZONTAL,
            showvalue=0, bg=BG_CARD, fg=TEXT_MAIN, troughcolor="#202431",
            activebackground=ACCENT_GREEN, highlightthickness=0,
            command=self._on_crossfader_move
        )
        self.crossfader_scale.set(50)
        self.crossfader_scale.pack(fill=tk.X)

        cross_labels = tk.Frame(sintonizador_box, bg=BG_CARD)
        cross_labels.pack(fill=tk.X, pady=(2, 0))
        tk.Label(cross_labels, text="Tema Actual (A)", font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT)
        tk.Label(cross_labels, text="Mezcla 50/50", font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, expand=True)
        tk.Label(cross_labels, text="Tema Siguiente (B)", font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.RIGHT)

        # Playbooks Options List
        p_playbooks = tk.Frame(p_center, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=12, pady=10)
        p_playbooks.pack(fill=tk.BOTH, expand=True)

        tk.Label(p_playbooks, text="Maniobras Generadas (Toca una y haz clic en 'Elegir Esta')", font=("Segoe UI", 10, "bold"), fg=TEXT_MAIN, bg=BG_CARD).pack(anchor="w", pady=(0, 6))

        scroll_pb = tk.Scrollbar(p_playbooks)
        scroll_pb.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox_playbooks = tk.Listbox(
            p_playbooks, bg=BG_MAIN, fg=TEXT_MAIN, font=("Segoe UI", 9),
            selectbackground=ACCENT_GREEN, selectforeground="#000",
            relief=tk.FLAT, highlightthickness=0, yscrollcommand=scroll_pb.set
        )
        self.listbox_playbooks.pack(fill=tk.BOTH, expand=True)
        scroll_pb.config(command=self.listbox_playbooks.yview)

        btn_select_pb = tk.Button(
            p_playbooks, text="Elegir Esta Maniobra Táctica", font=("Segoe UI", 9, "bold"),
            bg=ACCENT_GREEN, fg="#000", relief=tk.FLAT, pady=6, cursor="hand2",
            command=self._on_override_playbook
        )
        btn_select_pb.pack(fill=tk.X, pady=(6, 0))

        # === RIGHT PANEL: Real-time Timeline ===
        p_right = tk.Frame(content, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=10, pady=10)
        p_right.grid(row=0, column=2, sticky="nsew", padx=6)

        tk.Label(p_right, text="Timeline de Eventos", font=("Segoe UI", 11, "bold"), fg=TEXT_MAIN, bg=BG_CARD).pack(anchor="w")
        tk.Label(p_right, text="Acciones en tiempo real", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0, 8))

        scroll_tl = tk.Scrollbar(p_right)
        scroll_tl.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox_timeline = tk.Listbox(
            p_right, bg=BG_MAIN, fg=TEXT_MAIN, font=("Consolas", 9),
            selectbackground=BG_CARD_ACCENT, selectforeground=TEXT_MAIN,
            relief=tk.FLAT, highlightthickness=0, yscrollcommand=scroll_tl.set
        )
        self.listbox_timeline.pack(fill=tk.BOTH, expand=True)
        scroll_tl.config(command=self.listbox_timeline.yview)

    # --- Actions & Networking ---

    def _load_playlists_async(self) -> None:
        def _loader() -> None:
            try:
                pls = self.spotify.get_user_playlists(limit=50)
                names = ["❤️ Canciones que te gustan (Liked Songs)"]
                mapping = {"❤️ Canciones que te gustan (Liked Songs)": "liked"}
                for p in pls:
                    title = f"{p['name']} ({p['tracks_count']})"
                    names.append(title)
                    mapping[title] = p["id"]

                def _apply() -> None:
                    self.pl_mapping = mapping
                    self.pl_combo["values"] = names
                    if names:
                        self.pl_combo.current(0)
                self.after(0, _apply)
            except Exception:
                pass
        threading.Thread(target=_loader, daemon=True).start()

    def _on_load_playlist(self) -> None:
        sel_text = self.pl_combo.get()
        target_id = getattr(self, "pl_mapping", {}).get(sel_text, "liked")

        self.lbl_pool_status.config(text="Cargando temas desde Spotify...")
        self.stop_signal.set()
        self.is_dj_running = False
        self.btn_toggle_dj.config(text="▶ INICIAR DJ", bg=ACCENT_GREEN)

        def _fetcher() -> None:
            if target_id == "liked":
                raws = self.spotify.fetch_saved_tracks(limit=30)
            else:
                raws = self.spotify.fetch_playlist_tracks(target_id, limit=30)

            def _done() -> None:
                self.raw_pool = raws
                self.lbl_pool_status.config(text=f"{len(raws)} temas encontrados. Listo para iniciar.")
                self.listbox_candidates.delete(0, tk.END)
                for r in raws:
                    self.listbox_candidates.insert(tk.END, f"{r.primary_artist} - {r.name}")
            self.after(0, _done)

        threading.Thread(target=_fetcher, daemon=True).start()

    def _toggle_dj_session(self) -> None:
        if self.is_dj_running:
            self.stop_signal.set()
            self.is_dj_running = False
            self.btn_toggle_dj.config(text="▶ INICIAR DJ", bg=ACCENT_GREEN)
            self.lbl_deck_tag.config(text="PAUSADO", fg=TEXT_MUTED)
        else:
            if not self.raw_pool:
                # Default to liked songs if none loaded
                self._on_load_playlist()
                self.after(1200, self._toggle_dj_session)
                return

            self.stop_signal.clear()
            self.is_dj_running = True
            self.btn_toggle_dj.config(text="⏸ PAUSAR DJ", bg=ACCENT_ORANGE)
            self.lbl_deck_tag.config(text="● EN VIVO (SPOTIFY)", fg=ACCENT_GREEN)
            threading.Thread(target=self._run_live_session_thread, daemon=True).start()

    def _run_live_session_thread(self) -> None:
        # 1. Device check
        status = self.spotify.get_playback_status()
        device_id = status.device_id if status else None
        if not device_id:
            devs = self.spotify.get_available_devices()
            if devs:
                device_id = devs[0].get("id")
        self.active_device_id = device_id

        # 2. Initial track immediate hydration
        active_raw = self.raw_pool[0]
        if status:
            match = next((r for r in self.raw_pool if r.uri == status.track_uri), None)
            if match:
                active_raw = match

        init_node = self.hydrator.hydrate_track(active_raw)
        with self.pool_lock:
            self.current_track = init_node
            self.candidate_by_uri[init_node.spotify_uri] = init_node

        if status and status.track_uri != init_node.spotify_uri and device_id:
            self.spotify.start_or_transfer_playback(
                track_uri=init_node.spotify_uri,
                position_ms=init_node.first_downbeat_ms,
                device_id=device_id,
            )

        # 3. Background candidate pool worker
        remaining = [r for r in self.raw_pool if r.uri != init_node.spotify_uri]

        def _bg_worker() -> None:
            for raw in remaining:
                if self.stop_signal.is_set():
                    break
                try:
                    c_node = self.hydrator.hydrate_track(raw)
                    with self.pool_lock:
                        self.candidate_pool.append(c_node)
                        self.candidate_by_uri[c_node.spotify_uri] = c_node
                except Exception:
                    pass

        threading.Thread(target=_bg_worker, daemon=True).start()

        # 4. DJ Loop
        while not self.stop_signal.is_set():
            # Wait for at least 1 candidate
            while not self.stop_signal.is_set():
                with self.pool_lock:
                    avail = [c for c in self.candidate_pool if c.track_id != self.current_track.track_id and c.track_id not in self.played_ids]
                if avail:
                    break
                time.sleep(0.5)

            if self.stop_signal.is_set():
                break

            session_state = SessionState(
                target_energy=self.config.target_energy,
                played_track_ids=self.played_ids,
                played_artistsHistory=self.played_artists,
                session_history=self.session_history,
                network_latency_offset_ms=self.config.network_latency_offset_ms,
                creativity_temperature=self.config.creativity_temperature,
                last_maneuver_archetype=self.last_archetype,
            )

            with self.pool_lock:
                if not self.forced_playbook:
                    decision = self.engine.plan_stochastic_playbooks(
                        current_track=self.current_track,
                        candidate_pool=avail,
                        session_state=session_state,
                    )
                    self.current_decision = decision
                    winner = decision.selected_playbook
                else:
                    winner = self.forced_playbook
                    self.forced_playbook = None

                self.timeline_pending = list(winner.action_plan.timeline)
                self.last_archetype = winner.archetype

            # Execute events
            loop_counts: dict[int, int] = {}
            while self.timeline_pending and not self.stop_signal.is_set():
                pb_stat = self.spotify.get_playback_status()
                if not pb_stat:
                    time.sleep(0.4)
                    continue

                ev = self.timeline_pending[0]
                if pb_stat.progress_ms >= ev.trigger_at_progress_ms:
                    if ev.action_type in ("SEEK_LOOP", "SECTION_JUMP") and ev.current_track_uri == ev.target_track_uri:
                        self.spotify.seek_position(ev.target_position_ms, device_id=self.active_device_id)
                    else:
                        self.spotify.start_or_transfer_playback(
                            track_uri=ev.target_track_uri,
                            position_ms=ev.target_position_ms,
                            device_id=self.active_device_id,
                        )

                    ev_id = id(ev)
                    times = loop_counts.get(ev_id, 0) + 1
                    loop_counts[ev_id] = times

                    if ev.action_type == "SEEK_LOOP" and times < ev.loop_iterations:
                        time.sleep(0.6)
                    else:
                        with self.pool_lock:
                            if self.timeline_pending:
                                self.timeline_pending.pop(0)
                        if ev.action_type == "TRACK_TRANSITION":
                            break
                        time.sleep(0.6)
                else:
                    time.sleep(0.4)

            # Finish track
            with self.pool_lock:
                self.played_ids.add(self.current_track.track_id)
                self.played_artists.append(self.current_track.autor)
                self.session_history.append(self.current_track)
                next_node = self.candidate_by_uri.get(winner.action_plan.next_track_uri)
                if next_node:
                    self.current_track = next_node
                else:
                    break

    # --- UI Periodic Refresh ---

    def _start_status_poller(self) -> None:
        def _poller() -> None:
            while True:
                try:
                    st = self.spotify.get_playback_status()
                    self.cached_playback_status = st
                except Exception:
                    pass
                time.sleep(0.5)

        threading.Thread(target=_poller, daemon=True).start()

    def _periodic_ui_update(self) -> None:
        try:
            # Update Current Track & Progress
            if self.current_track:
                self.lbl_track_title.config(text=self.current_track.titulo)
                self.lbl_track_artist.config(text=self.current_track.autor)
                self.lbl_track_bpm.config(text=f"{self.current_track.bpm:.1f} BPM")
                self.lbl_track_key.config(text=f"{self.current_track.camelot_key} KEY")
                self.lbl_track_secs.config(text=f"{len(self.current_track.secciones)} Secciones")

                dur_s = self.current_track.duration_ms / 1000.0
                m_tot, s_tot = divmod(int(dur_s), 60)
                self.lbl_total_time.config(text=f"{m_tot:02d}:{s_tot:02d}")

                pb = self.cached_playback_status
                prog_ms = pb.progress_ms if pb else 0
                prog_s = prog_ms / 1000.0
                m_cur, s_cur = divmod(int(prog_s), 60)
                self.lbl_curr_time.config(text=f"{m_cur:02d}:{s_cur:02d}")

                # Draw progress bar
                self.progress_canvas.delete("all")
                w = self.progress_canvas.winfo_width()
                if w > 10 and self.current_track.duration_ms > 0:
                    fill_x = (prog_ms / self.current_track.duration_ms) * w
                    self.progress_canvas.create_rectangle(0, 0, fill_x, 12, fill=ACCENT_GREEN, width=0)

                    # Trigger marker
                    if self.timeline_pending:
                        trig_ms = self.timeline_pending[0].trigger_at_progress_ms
                        trig_x = (trig_ms / self.current_track.duration_ms) * w
                        self.progress_canvas.create_line(trig_x, 0, trig_x, 12, fill=ACCENT_RED, width=3)
                        m_trig, s_trig = divmod(int(trig_ms / 1000.0), 60)
                        self.lbl_trigger_time.config(text=f"Enganche: {m_trig:02d}:{s_trig:02d}")

            # Update Candidates listbox if changed
            with self.pool_lock:
                cand_count = len(self.candidate_pool)
            if cand_count != self.listbox_candidates.size():
                self.listbox_candidates.delete(0, tk.END)
                with self.pool_lock:
                    for c in self.candidate_pool:
                        self.listbox_candidates.insert(tk.END, f"{c.autor} - {c.titulo}  ({c.bpm:.1f} BPM, {c.camelot_key})")
                self.lbl_pool_status.config(text=f"{cand_count} candidatos hidratados")

            # Update Playbooks listbox
            if self.current_decision:
                all_pbs = [self.current_decision.selected_playbook, *self.current_decision.alternative_playbooks]
                items = []
                for p in all_pbs:
                    nxt = self.candidate_by_uri.get(p.action_plan.next_track_uri)
                    nxt_label = f"{nxt.autor} - {nxt.titulo}" if nxt else "Siguiente Pista"
                    mark = "▶ [ELEGIDO] " if p.archetype == self.current_decision.selected_playbook.archetype else "  "
                    items.append(f"{mark}{p.archetype:<18} ({p.selection_probability * 100:>4.1f}%) ➔ {nxt_label}")

                current_items = list(self.listbox_playbooks.get(0, tk.END))
                if items != current_items:
                    self.listbox_playbooks.delete(0, tk.END)
                    for it in items:
                        self.listbox_playbooks.insert(tk.END, it)

            # Update Timeline
            tl_items = []
            for idx, ev in enumerate(self.timeline_pending, start=1):
                m_t, s_t = divmod(int(ev.trigger_at_progress_ms / 1000.0), 60)
                tl_items.append(f"#{idx} {ev.action_type} @ {m_t:02d}:{s_t:02d} ({ev.from_section_type}➔{ev.to_section_type})")
            if tl_items != list(self.listbox_timeline.get(0, tk.END)):
                self.listbox_timeline.delete(0, tk.END)
                for t_str in tl_items:
                    self.listbox_timeline.insert(tk.END, t_str)

        except Exception:
            pass

        self.after(500, self._periodic_ui_update)

    # --- Sintonizador Dual Audio Previews ---

    def _on_play_dual_preview(self) -> None:
        if not self.current_track or not self.current_decision:
            messagebox.showinfo("Previo", "Inicia el DJ para tener las pistas sintonizadas.")
            return

        winner = self.current_decision.selected_playbook
        nxt_node = self.candidate_by_uri.get(winner.action_plan.next_track_uri)

        url_a = self.current_track.preview_url
        url_b = nxt_node.preview_url if nxt_node else None

        if not url_a and not url_b:
            messagebox.showwarning("Preview", "No hay clips de vista previa disponibles para estas pistas.")
            return

        self.lbl_sinto_status.config(text="Sintonizando pistas A y B...", fg=ACCENT_CYAN)
        self.preview_engine.start_dual_preview(url_a, url_b)

    def _on_crossfader_move(self, val_str: str) -> None:
        val = float(val_str) / 100.0
        self.preview_engine.apply_crossfader(val)

    def _on_override_playbook(self) -> None:
        sel = self.listbox_playbooks.curselection()
        if not sel or not self.current_decision:
            return
        idx = sel[0]
        all_pbs = [self.current_decision.selected_playbook, *self.current_decision.alternative_playbooks]
        chosen = all_pbs[idx]
        with self.pool_lock:
            self.forced_playbook = chosen
            self.timeline_pending = list(chosen.action_plan.timeline)
            self.current_decision = DecisionEngineOutput(
                selected_playbook=chosen,
                alternative_playbooks=[p for p in all_pbs if p.archetype != chosen.archetype],
            )

    def _on_force_candidate_track(self) -> None:
        sel = self.listbox_candidates.curselection()
        if not sel:
            return
        idx = sel[0]
        with self.pool_lock:
            if idx < len(self.candidate_pool):
                target_node = self.candidate_pool[idx]
            else:
                return

        if not self.current_track:
            return

        session_state = SessionState(
            target_energy=self.config.target_energy,
            played_track_ids=self.played_ids,
            played_artistsHistory=self.played_artists,
            session_history=self.session_history,
            network_latency_offset_ms=self.config.network_latency_offset_ms,
            creativity_temperature=self.config.creativity_temperature,
            last_maneuver_archetype=self.last_archetype,
        )

        with self.pool_lock:
            sub_decision = self.engine.plan_stochastic_playbooks(
                current_track=self.current_track,
                candidate_pool=[target_node],
                session_state=session_state,
            )
            self.current_decision = sub_decision
            self.forced_playbook = sub_decision.selected_playbook
            self.timeline_pending = list(self.forced_playbook.action_plan.timeline)


def run_desktop_app() -> None:
    app = DesktopDJApp()
    app.mainloop()


if __name__ == "__main__":
    run_desktop_app()
