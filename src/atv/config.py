"""Configuration loader for `.env` files and environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def load_dotenv_file(dotenv_path: str | Path = ".env") -> dict[str, str]:
    """Parse a `.env` file and populate `os.environ` without external dependencies."""
    path = Path(dotenv_path)
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        k = key.strip()
        v = val.strip().strip('"').strip("'")
        if k:
            loaded[k] = v
            if not os.environ.get(k):
                os.environ[k] = v
    return loaded


@dataclass(frozen=True)
class SpotifyAppConfig:
    """Runtime configuration for Spotify OAuth2 PKCE and DJ Engine parameters."""

    client_id: str
    client_secret: str
    redirect_uri: str = "http://127.0.0.1:8888/callback"
    default_playlist: str = ""
    target_energy: float = 0.85
    creativity_temperature: float = 0.40
    network_latency_offset_ms: int = 250
    poll_interval_sec: float = 0.40
    token_cache_path: Path = Path(".spotify_token_cache.json")

    @classmethod
    def from_env(cls, dotenv_path: str | Path = ".env") -> SpotifyAppConfig:
        file_vars = load_dotenv_file(dotenv_path)

        def _get(key: str, default: str = "") -> str:
            if key in file_vars and file_vars[key]:
                return file_vars[key]
            return os.environ.get(key, default).strip()

        return cls(
            client_id=_get("SPOTIFY_CLIENT_ID"),
            client_secret=_get("SPOTIFY_CLIENT_SECRET"),
            redirect_uri=_get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            default_playlist=_get("SPOTIFY_DEFAULT_PLAYLIST"),
            target_energy=float(_get("DJ_TARGET_ENERGY", "0.85") or 0.85),
            creativity_temperature=float(_get("DJ_CREATIVITY_TEMPERATURE", "0.40") or 0.40),
            network_latency_offset_ms=int(_get("DJ_NETWORK_LATENCY_OFFSET_MS", "250") or 250),
            poll_interval_sec=float(_get("DJ_POLL_INTERVAL_SEC", "0.40") or 0.40),
        )
