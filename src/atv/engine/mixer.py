"""Multi-Track Acoustic Mixing Engine (Runs 2 to N channels simultaneously with volume automation curves)."""

from __future__ import annotations

import io
import math
import threading
import time
from typing import Any, Callable
import httpx
import pygame

from atv.engine.curves import CurveShape, calculate_volume_levels


class AudioTrackDeck:
    """Represents a single independent virtual DJ deck playing a loaded audio stream."""

    def __init__(self, channel_id: int) -> None:
        self.channel_id = channel_id
        self.channel = pygame.mixer.Channel(channel_id)
        self.sound: pygame.mixer.Sound | None = None
        self.track_uri: str = ""
        self.current_volume: float = 0.0
        self.is_active: bool = False

    def load_and_play(self, sound: pygame.mixer.Sound, uri: str, initial_volume: float = 0.0) -> None:
        self.channel.stop()
        self.sound = sound
        self.track_uri = uri
        self.current_volume = max(0.0, min(1.0, initial_volume))
        self.channel.set_volume(self.current_volume)
        self.channel.play(sound, loops=-1)
        self.is_active = True

    def set_volume(self, volume: float) -> None:
        self.current_volume = max(0.0, min(1.0, float(volume)))
        self.channel.set_volume(self.current_volume)

    def stop(self) -> None:
        self.channel.stop()
        self.sound = None
        self.track_uri = ""
        self.current_volume = 0.0
        self.is_active = False


class MultiDeckMixer:
    """N-Deck Virtual DJ Mixer that can run multiple tracks in parallel, fading volumes dynamically."""

    def __init__(self, max_decks: int = 4) -> None:
        self.lock = threading.Lock()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.set_num_channels(max(8, max_decks * 2))
        self.decks: list[AudioTrackDeck] = [AudioTrackDeck(i) for i in range(max_decks)]
        self.active_deck_idx: int = 0
        self.is_transitioning: bool = False
        self._fade_thread: threading.Thread | None = None
        self._stop_fade_signal = threading.Event()

    def _fetch_sound(self, preview_url: str) -> pygame.mixer.Sound | None:
        try:
            with httpx.Client(timeout=6.0) as client:
                res = client.get(preview_url)
                if res.status_code == 200:
                    return pygame.mixer.Sound(io.BytesIO(res.content))
        except Exception:
            pass
        return None

    def start_single_track(self, track_uri: str, preview_url: str) -> None:
        """Start playing a track at 100% volume on the main active deck."""
        with self.lock:
            self._stop_fade_signal.set()
            for d in self.decks:
                d.stop()

        snd = self._fetch_sound(preview_url) if preview_url else None
        if not snd:
            return

        with self.lock:
            self.active_deck_idx = 0
            self.decks[0].load_and_play(snd, uri=track_uri, initial_volume=1.0)

    def transition_to_track(
        self,
        next_track_uri: str,
        next_preview_url: str,
        duration_ms: int = 6000,
        curve: CurveShape = "EQUAL_POWER",
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Fade from the currently active deck into an incoming deck over duration_ms without cutting audio."""
        snd = self._fetch_sound(next_preview_url) if next_preview_url else None
        if not snd:
            return

        with self.lock:
            self._stop_fade_signal.set()
            time.sleep(0.05)
            self._stop_fade_signal.clear()

            out_deck = self.decks[self.active_deck_idx]
            in_deck_idx = (self.active_deck_idx + 1) % len(self.decks)
            in_deck = self.decks[in_deck_idx]

            # Start incoming track in silence (0.0 volume)
            in_deck.load_and_play(snd, uri=next_track_uri, initial_volume=0.0)
            self.is_transitioning = True

        def _fade_worker() -> None:
            steps = 40
            step_duration_s = (max(500, duration_ms) / 1000.0) / steps
            for i in range(1, steps + 1):
                if self._stop_fade_signal.is_set():
                    return
                t = i / steps
                vol_out, vol_in = calculate_volume_levels(t, curve=curve)
                with self.lock:
                    out_deck.set_volume(vol_out)
                    in_deck.set_volume(vol_in)
                time.sleep(step_duration_s)

            # Transition completed
            with self.lock:
                out_deck.stop()
                in_deck.set_volume(1.0)
                self.active_deck_idx = in_deck_idx
                self.is_transitioning = False

            if on_complete:
                on_complete()

        self._fade_thread = threading.Thread(target=_fade_worker, daemon=True)
        self._fade_thread.start()

    def stop_all(self) -> None:
        with self.lock:
            self._stop_fade_signal.set()
            for d in self.decks:
                d.stop()
            self.is_transitioning = False
