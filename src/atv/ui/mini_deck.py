"""Minimalist Mini-Deck Widget for ATV Autonomous DJ (Spotify Overlay / Compact Bar)."""

from __future__ import annotations

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import io
import math
import threading
import time
import tkinter as tk
from tkinter import ttk
import httpx
import pygame

from atv.config import SpotifyAppConfig
from atv.engine import MultiDeckMixer, calculate_volume_levels
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

# Palette Minimalist Dark Console
BG_DARK = "#121318"
BG_CARD = "#1a1c24"
BG_ITEM = "#222530"
BG_HOVER = "#2d3142"
TEXT_WHITE = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_GREEN = "#10b981"
ACCENT_CYAN = "#38bdf8"
ACCENT_ORANGE = "#f59e0b"
ACCENT_RED = "#f43f5e"
BORDER = "#2b2f3e"


class MiniPreviewEngine:
    """Lightweight crossfader audio preview engine."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.set_num_channels(4)
        self.channel_a = pygame.mixer.Channel(0)
        self.channel_b = pygame.mixer.Channel(1)
        self.sound_a: pygame.mixer.Sound | None = None
        self.sound_b: pygame.mixer.Sound | None = None
        self.is_playing = False
        self.pos = 0.5

    def _fetch(self, url: str) -> pygame.mixer.Sound | None:
        try:
            with httpx.Client(timeout=6.0) as client:
                res = client.get(url)
                if res.status_code == 200:
                    return pygame.mixer.Sound(io.BytesIO(res.content))
        except Exception:
            pass
        return None

    def play_blend(self, url_a: str | None, url_b: str | None) -> None:
        def _run() -> None:
            with self.lock:
                self.channel_a.stop()
                self.channel_b.stop()
                self.sound_a = self._fetch(url_a) if url_a else None
                self.sound_b = self._fetch(url_b) if url_b else None
                if self.sound_a:
                    self.channel_a.play(self.sound_a, loops=-1)
                if self.sound_b:
                    self.channel_b.play(self.sound_b, loops=-1)
                self.set_crossfader(self.pos)
                self.is_playing = True
        threading.Thread(target=_run, daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            self.channel_a.stop()
            self.channel_b.stop()
            self.is_playing = False

    def set_crossfader(self, val: float) -> None:
        self.pos = max(0.0, min(1.0, val))
        vol_a = math.cos(self.pos * 0.5 * math.pi)
        vol_b = math.cos((1.0 - self.pos) * 0.5 * math.pi)
        self.channel_a.set_volume(vol_a)
        self.channel_b.set_volume(vol_b)


class MiniDJDeckApp(tk.Tk):
    """Compact, minimalist rectangular mini-player window that sits cleanly over or beside Spotify."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ATV Mini Deck")
        # Sleek compact rectangle (e.g. 720 x 290 px)
        self.geometry("740x310")
        self.minsize(660, 280)
        self.configure(bg=BG_DARK)

        # Always-on-top option toggle
        self.attributes("-topmost", True)
        self.config = SpotifyAppConfig.from_env(".env")
        self.auth = SpotifyPKCEAuthenticator(self.config)
        self.spotify = SpotifyConnectClient(self.auth)
        self.hydrator = TrackHydrationPipeline(
            acoustic_provider=DeezerAcousticBrainzClient(),
            lrc_provider=DeezerAcousticBrainzClient(),
        )
        self.engine = HybridDJDecisionEngine()
        self.multi_mixer = MultiDeckMixer(max_decks=4)
        self.previewer = MiniPreviewEngine()

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
        self.cached_status: PlaybackStatus | None = None

        self._build_mini_ui()
        self._start_background_status_poller()
        self._load_initial_library_async()

        self.after(300, self._tick_ui)

    def _build_mini_ui(self) -> None:
        # TOP BAR: Minimal title, playlist status & Start/Pause
        top_bar = tk.Frame(self, bg=BG_DARK, height=36, padx=12, pady=6)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        logo = tk.Label(top_bar, text="🎧 ATV MINI DECK", font=("Segoe UI", 9, "bold"), fg=ACCENT_GREEN, bg=BG_DARK)
        logo.pack(side=tk.LEFT)

        self.lbl_status_mini = tk.Label(top_bar, text="Conectando a Spotify...", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_DARK)
        self.lbl_status_mini.pack(side=tk.LEFT, padx=12)

        self.btn_run = tk.Button(
            top_bar, text="▶ INICIAR DJ", font=("Segoe UI", 8, "bold"),
            bg=ACCENT_GREEN, fg="#000", activebackground="#059669", relief=tk.FLAT,
            padx=10, pady=2, cursor="hand2", command=self._toggle_dj
        )
        self.btn_run.pack(side=tk.RIGHT)

        # MAIN SPLIT: Left = Active Deck & Preview / Right = Scrollable Up Next & Playbooks
        body = tk.Frame(self, bg=BG_DARK, padx=8, pady=4)
        body.pack(fill=tk.BOTH, expand=True)

        body.columnconfigure(0, weight=6)
        body.columnconfigure(1, weight=5)
        body.rowconfigure(0, weight=1)

        # --- LEFT CARD: CURRENT DECK & COMPACT CROSSFADER ---
        deck_card = tk.Frame(body, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER, padx=12, pady=10)
        deck_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        tk.Label(deck_card, text="SONANDO AHORA", font=("Segoe UI", 7, "bold"), fg=ACCENT_CYAN, bg=BG_CARD).pack(anchor="w")

        self.lbl_title = tk.Label(deck_card, text="Cargando música...", font=("Segoe UI", 12, "bold"), fg=TEXT_WHITE, bg=BG_CARD, anchor="w")
        self.lbl_title.pack(fill=tk.X, pady=(2, 0))

        self.lbl_artist = tk.Label(deck_card, text="—", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, anchor="w")
        self.lbl_artist.pack(fill=tk.X, pady=(0, 6))

        # Badges row (BPM, Key, Next Hook)
        meta_row = tk.Frame(deck_card, bg=BG_CARD)
        meta_row.pack(anchor="w", pady=(0, 6))

        self.badge_bpm = tk.Label(meta_row, text="-- BPM", font=("Consolas", 8, "bold"), fg=ACCENT_ORANGE, bg=BG_ITEM, padx=6, pady=2)
        self.badge_bpm.pack(side=tk.LEFT, padx=(0, 4))

        self.badge_key = tk.Label(meta_row, text="-- KEY", font=("Consolas", 8, "bold"), fg=ACCENT_CYAN, bg=BG_ITEM, padx=6, pady=2)
        self.badge_key.pack(side=tk.LEFT, padx=(0, 4))

        self.badge_hook = tk.Label(meta_row, text="Corte: --:--", font=("Consolas", 8, "bold"), fg=ACCENT_RED, bg=BG_ITEM, padx=6, pady=2)
        self.badge_hook.pack(side=tk.LEFT)

        # Slim Progress Bar
        self.progress_canvas = tk.Canvas(deck_card, height=6, bg="#262938", highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X, pady=(4, 2))

        time_row = tk.Frame(deck_card, bg=BG_CARD)
        time_row.pack(fill=tk.X, pady=(0, 8))
        self.lbl_time_cur = tk.Label(time_row, text="00:00", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_time_cur.pack(side=tk.LEFT)
        self.lbl_time_tot = tk.Label(time_row, text="00:00", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_time_tot.pack(side=tk.RIGHT)

        # Compact Preview Sintonizador Bar
        sinto_box = tk.Frame(deck_card, bg=BG_ITEM, highlightthickness=1, highlightbackground=BORDER, padx=8, pady=6)
        sinto_box.pack(fill=tk.X, side=tk.BOTTOM)

        sinto_head = tk.Frame(sinto_box, bg=BG_ITEM)
        sinto_head.pack(fill=tk.X, pady=(0, 4))

        self.btn_preview = tk.Button(
            sinto_head, text="▶ Sintonizar A ⟷ B", font=("Segoe UI", 7, "bold"),
            bg=BG_CARD, fg=ACCENT_CYAN, relief=tk.FLAT, padx=6, pady=2, cursor="hand2",
            command=self._toggle_preview_audio
        )
        self.btn_preview.pack(side=tk.LEFT)

        self.lbl_preview_info = tk.Label(sinto_head, text="50/50 Preview", font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_ITEM)
        self.lbl_preview_info.pack(side=tk.RIGHT)

        self.cross_slider = tk.Scale(
            sinto_box, from_=0, to=100, orient=tk.HORIZONTAL,
            showvalue=0, bg=BG_ITEM, troughcolor="#121318",
            activebackground=ACCENT_GREEN, highlightthickness=0,
            command=self._on_slider_change
        )
        self.cross_slider.set(50)
        self.cross_slider.pack(fill=tk.X)

        # --- RIGHT CARD: SCROLLABLE CANDIDATES & DJ PLAYBOOK OVERRIDE ---
        right_card = tk.Frame(body, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER, padx=10, pady=8)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        right_head = tk.Frame(right_card, bg=BG_CARD)
        right_head.pack(fill=tk.X, pady=(0, 4))
        tk.Label(right_head, text="POSIBLES ENGANCHES (CLICK PARA FORZAR)", font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT)

        # Notebook tabs (Candidates vs Tactical Playbooks)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=BG_CARD, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_ITEM, foreground=TEXT_MUTED, padding=[8, 2], font=("Segoe UI", 7, "bold"))
        style.map("TNotebook.Tab", background=[("selected", BG_HOVER)], foreground=[("selected", ACCENT_GREEN)])

        self.tabs = ttk.Notebook(right_card)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Candidates Pool
        tab_cand = tk.Frame(self.tabs, bg=BG_CARD)
        self.tabs.add(tab_cand, text="Canciones")

        scroll_c = tk.Scrollbar(tab_cand)
        scroll_c.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_cand = tk.Listbox(
            tab_cand, bg=BG_DARK, fg=TEXT_WHITE, font=("Segoe UI", 8),
            selectbackground=ACCENT_GREEN, selectforeground="#000",
            relief=tk.FLAT, highlightthickness=0, yscrollcommand=scroll_c.set
        )
        self.list_cand.pack(fill=tk.BOTH, expand=True)
        scroll_c.config(command=self.list_cand.yview)
        self.list_cand.bind("<Double-Button-1>", lambda e: self._on_double_click_candidate())

        # Tab 2: Tactical Playbooks
        tab_pb = tk.Frame(self.tabs, bg=BG_CARD)
        self.tabs.add(tab_pb, text="Maniobras DJ")

        scroll_p = tk.Scrollbar(tab_pb)
        scroll_p.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_pb = tk.Listbox(
            tab_pb, bg=BG_DARK, fg=TEXT_WHITE, font=("Segoe UI", 8),
            selectbackground=ACCENT_CYAN, selectforeground="#000",
            relief=tk.FLAT, highlightthickness=0, yscrollcommand=scroll_p.set
        )
        self.list_pb.pack(fill=tk.BOTH, expand=True)
        scroll_p.config(command=self.list_pb.yview)
        self.list_pb.bind("<Double-Button-1>", lambda e: self._on_double_click_playbook())

    # --- Background Polling & Stream ---

    def _start_background_status_poller(self) -> None:
        def _poll() -> None:
            while True:
                try:
                    st = self.spotify.get_playback_status()
                    self.cached_status = st
                except Exception:
                    pass
                time.sleep(0.4)
        threading.Thread(target=_poll, daemon=True).start()

    def _load_initial_library_async(self) -> None:
        def _fetch() -> None:
            try:
                raws = self.spotify.fetch_saved_tracks(limit=35)
                self.raw_pool = raws
                def _update() -> None:
                    self.lbl_status_mini.config(text=f"{len(raws)} temas en biblioteca. Listo.")
                    for r in raws:
                        self.list_cand.insert(tk.END, f"{r.primary_artist} - {r.name}")
                self.after(0, _update)
            except Exception:
                pass
        threading.Thread(target=_fetch, daemon=True).start()

    def _toggle_dj(self) -> None:
        if self.is_dj_running:
            self.stop_signal.set()
            self.is_dj_running = False
            self.btn_run.config(text="▶ INICIAR DJ", bg=ACCENT_GREEN)
            self.lbl_status_mini.config(text="DJ Pausado")
        else:
            if not self.raw_pool:
                self._load_initial_library_async()
                self.after(1500, self._toggle_dj)
                return
            self.stop_signal.clear()
            self.is_dj_running = True
            self.btn_run.config(text="⏸ PAUSAR", bg=ACCENT_ORANGE)
            self.lbl_status_mini.config(text="DJ Activo en vivo", fg=ACCENT_GREEN)
            threading.Thread(target=self._run_engine_session, daemon=True).start()

    def _run_engine_session(self) -> None:
        status = self.cached_status or self.spotify.get_playback_status()
        dev_id = status.device_id if status else None
        if not dev_id:
            devs = self.spotify.get_available_devices()
            if devs:
                dev_id = devs[0].get("id")
        self.active_device_id = dev_id

        # Initial track
        active_raw = self.raw_pool[0]
        if status:
            match = next((r for r in self.raw_pool if r.uri == status.track_uri), None)
            if match:
                active_raw = match

        init_node = self.hydrator.hydrate_track(active_raw)
        with self.pool_lock:
            self.current_track = init_node
            self.candidate_by_uri[init_node.spotify_uri] = init_node

        if status and status.track_uri != init_node.spotify_uri and dev_id:
            self.spotify.start_or_transfer_playback(
                track_uri=init_node.spotify_uri,
                position_ms=init_node.first_downbeat_ms,
                device_id=dev_id,
            )

        # Background hydration worker
        remaining = [r for r in self.raw_pool if r.uri != init_node.spotify_uri]

        def _bg_hydrator() -> None:
            for raw in remaining:
                if self.stop_signal.is_set():
                    break
                try:
                    node = self.hydrator.hydrate_track(raw)
                    with self.pool_lock:
                        self.candidate_pool.append(node)
                        self.candidate_by_uri[node.spotify_uri] = node
                except Exception:
                    pass

        threading.Thread(target=_bg_hydrator, daemon=True).start()

        # Decision loop
        while not self.stop_signal.is_set():
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
                    dec = self.engine.plan_stochastic_playbooks(
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
                pb = self.cached_status
                if not pb:
                    time.sleep(0.4)
                    continue

                ev = self.timeline_pending[0]
                if pb.progress_ms >= ev.trigger_at_progress_ms:
                    fade_ms = ev.recommended_fade_duration_ms or 5000
                    nxt_node = self.candidate_by_uri.get(ev.target_track_uri)

                    # Trigger local multi-track crossfade if previews are available
                    if nxt_node and nxt_node.preview_url and self.multi_mixer:
                        self.multi_mixer.transition_to_track(
                            next_track_uri=ev.target_track_uri,
                            next_preview_url=nxt_node.preview_url,
                            duration_ms=fade_ms,
                            curve=getattr(ev, "fade_curve", "EQUAL_POWER"),
                        )

                    # Execute Spotify Connect Action
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

            with self.pool_lock:
                self.played_ids.add(self.current_track.track_id)
                self.played_artists.append(self.current_track.autor)
                self.session_history.append(self.current_track)
                nxt = self.candidate_by_uri.get(winner.action_plan.next_track_uri)
                if nxt:
                    self.current_track = nxt
                else:
                    break

    # --- UI Periodic Ticker ---

    def _tick_ui(self) -> None:
        try:
            if self.current_track:
                self.lbl_title.config(text=self.current_track.titulo)
                self.lbl_artist.config(text=self.current_track.autor)
                self.badge_bpm.config(text=f"{self.current_track.bpm:.1f} BPM")
                self.badge_key.config(text=f"{self.current_track.camelot_key}")

                dur_s = self.current_track.duration_ms / 1000.0
                m_tot, s_tot = divmod(int(dur_s), 60)
                self.lbl_time_tot.config(text=f"{m_tot:02d}:{s_tot:02d}")

                prog_ms = self.cached_status.progress_ms if self.cached_status else 0
                prog_s = prog_ms / 1000.0
                m_cur, s_cur = divmod(int(prog_s), 60)
                self.lbl_time_cur.config(text=f"{m_cur:02d}:{s_cur:02d}")

                # Progress line
                self.progress_canvas.delete("all")
                w = self.progress_canvas.winfo_width()
                if w > 10 and self.current_track.duration_ms > 0:
                    fill_x = (prog_ms / self.current_track.duration_ms) * w
                    self.progress_canvas.create_rectangle(0, 0, fill_x, 6, fill=ACCENT_GREEN, width=0)

                    if self.timeline_pending:
                        trig_ms = self.timeline_pending[0].trigger_at_progress_ms
                        trig_x = (trig_ms / self.current_track.duration_ms) * w
                        self.progress_canvas.create_line(trig_x, 0, trig_x, 6, fill=ACCENT_RED, width=2)
                        m_trig, s_trig = divmod(int(trig_ms / 1000.0), 60)
                        self.badge_hook.config(text=f"Enganche: {m_trig:02d}:{s_trig:02d}")

            # Update Candidate Listbox
            with self.pool_lock:
                cand_len = len(self.candidate_pool)
            if cand_len != self.list_cand.size():
                self.list_cand.delete(0, tk.END)
                with self.pool_lock:
                    for c in self.candidate_pool:
                        self.list_cand.insert(tk.END, f"{c.autor} - {c.titulo} ({c.bpm:.0f} BPM, {c.camelot_key})")

            # Update Playbooks Listbox
            if self.current_decision:
                all_pbs = [self.current_decision.selected_playbook, *self.current_decision.alternative_playbooks]
                lines = []
                for p in all_pbs:
                    nxt = self.candidate_by_uri.get(p.action_plan.next_track_uri)
                    nxt_name = f"{nxt.autor} - {nxt.titulo}" if nxt else "Siguiente"
                    mark = "▶ " if p.archetype == self.current_decision.selected_playbook.archetype else "  "
                    lines.append(f"{mark}{p.archetype:<17} ({p.selection_probability*100:>4.1f}%) ➔ {nxt_name}")
                if lines != list(self.list_pb.get(0, tk.END)):
                    self.list_pb.delete(0, tk.END)
                    for l in lines:
                        self.list_pb.insert(tk.END, l)

        except Exception:
            pass

        self.after(300, self._tick_ui)

    # --- Double click actions ---

    def _on_double_click_candidate(self) -> None:
        sel = self.list_cand.curselection()
        if not sel:
            return
        idx = sel[0]
        with self.pool_lock:
            if idx < len(self.candidate_pool) and self.current_track:
                target = self.candidate_pool[idx]
                session_state = SessionState(
                    target_energy=self.config.target_energy,
                    played_track_ids=self.played_ids,
                    played_artistsHistory=self.played_artists,
                    session_history=self.session_history,
                    network_latency_offset_ms=self.config.network_latency_offset_ms,
                    creativity_temperature=self.config.creativity_temperature,
                    last_maneuver_archetype=self.last_archetype,
                )
                sub_dec = self.engine.plan_stochastic_playbooks(
                    current_track=self.current_track,
                    candidate_pool=[target],
                    session_state=session_state,
                )
                self.current_decision = sub_dec
                self.forced_playbook = sub_dec.selected_playbook
                self.timeline_pending = list(self.forced_playbook.action_plan.timeline)

    def _on_double_click_playbook(self) -> None:
        sel = self.list_pb.curselection()
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

    def _toggle_preview_audio(self) -> None:
        if self.previewer.is_playing:
            self.previewer.stop()
            self.btn_preview.config(text="▶ Sintonizar A ⟷ B", fg=ACCENT_CYAN)
            self.lbl_preview_info.config(text="Detenido")
        else:
            if not self.current_track or not self.current_decision:
                return
            nxt_uri = self.current_decision.selected_playbook.action_plan.next_track_uri
            nxt = self.candidate_by_uri.get(nxt_uri)
            url_a = self.current_track.preview_url
            url_b = nxt.preview_url if nxt else None
            self.previewer.play_blend(url_a, url_b)
            self.btn_preview.config(text="⏹ Detener Previo", fg=ACCENT_RED)
            self.lbl_preview_info.config(text="Sintonizando...")

    def _on_slider_change(self, val_str: str) -> None:
        val = float(val_str) / 100.0
        self.previewer.set_crossfader(val)


def run_mini_app() -> None:
    app = MiniDJDeckApp()
    app.mainloop()


if __name__ == "__main__":
    run_mini_app()
