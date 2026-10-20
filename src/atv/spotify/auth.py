"""Spotify OAuth2 PKCE authentication flow allowing any Spotify Premium user to log in via browser."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import time
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser
import httpx

from atv.config import SpotifyAppConfig

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "playlist-read-private "
    "playlist-read-collaborative "
    "user-library-read"
)


@dataclass
class SpotifyTokenBundle:
    access_token: str
    refresh_token: str
    expires_at: float
    scope: str

    @property
    def is_expired(self) -> bool:
        return time.time() >= (self.expires_at - 60.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SpotifyTokenBundle:
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data.get("refresh_token", "")),
            expires_at=float(data["expires_at"]),
            scope=str(data.get("scope", SPOTIFY_SCOPES)),
        )


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a high-entropy PKCE (code_verifier, code_challenge) pair."""
    verifier = secrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


class SpotifyPKCEAuthenticator:
    """Manages OAuth2 PKCE browser login, token caching, and automatic token refresh."""

    def __init__(self, config: SpotifyAppConfig) -> None:
        if not config.client_id or config.client_id == "your_spotify_client_id_here":
            raise ValueError(
                "Missing SPOTIFY_CLIENT_ID in .env. Please create a Spotify App at "
                "https://developer.spotify.com/dashboard, add http://127.0.0.1:8888/callback "
                "to Redirect URIs, and set SPOTIFY_CLIENT_ID in your .env file."
            )
        self._config = config
        self._token_bundle: SpotifyTokenBundle | None = None

    def get_valid_access_token(self) -> str:
        """Return a valid access token, loading from cache, refreshing, or launching browser login."""
        if self._token_bundle is None:
            self._token_bundle = self._load_cached_token()

        if self._token_bundle is not None:
            if not self._token_bundle.is_expired:
                return self._token_bundle.access_token
            if self._token_bundle.refresh_token:
                try:
                    self._token_bundle = self._refresh_access_token(self._token_bundle.refresh_token)
                    self._save_cached_token(self._token_bundle)
                    return self._token_bundle.access_token
                except Exception:
                    self._token_bundle = None

        self._token_bundle = self._run_interactive_browser_login()
        self._save_cached_token(self._token_bundle)
        return self._token_bundle.access_token

    def _load_cached_token(self) -> SpotifyTokenBundle | None:
        path: Path = self._config.token_cache_path
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return SpotifyTokenBundle.from_dict(raw)
        except Exception:
            return None

    def _save_cached_token(self, bundle: SpotifyTokenBundle) -> None:
        path: Path = self._config.token_cache_path
        path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")

    def _refresh_access_token(self, refresh_token: str) -> SpotifyTokenBundle:
        payload: dict[str, str] = {
            "client_id": self._config.client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        if self._config.client_secret:
            payload["client_secret"] = self._config.client_secret

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(SPOTIFY_TOKEN_URL, data=payload)
            resp.raise_for_status()
            data = resp.json()

        return SpotifyTokenBundle(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_at=time.time() + float(data.get("expires_in", 3600)),
            scope=data.get("scope", SPOTIFY_SCOPES),
        )

    def _run_interactive_browser_login(self) -> SpotifyTokenBundle:
        verifier, challenge = generate_pkce_pair()
        state = secrets.token_urlsafe(16)
        parsed_redirect = urlparse(self._config.redirect_uri)
        host = parsed_redirect.hostname or "127.0.0.1"
        port = parsed_redirect.port or 8888

        captured: dict[str, str] = {}

        class _CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                qs = parse_qs(urlparse(self.path).query)
                if "code" in qs:
                    captured["code"] = qs["code"][0]
                if "state" in qs:
                    captured["state"] = qs["state"][0]
                if "error" in qs:
                    captured["error"] = qs["error"][0]

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = (
                    "<html><body style='font-family:sans-serif;background:#121212;color:#1DB954;"
                    "display:flex;align-items:center;justify-content:center;height:100vh;'>"
                    "<div><h1>ATV Autonomous DJ Connected</h1>"
                    "<p style='color:#fff;'>Authentication complete. You can close this window and return to the terminal.</p>"
                    "</div></body></html>"
                )
                self.wfile.write(html.encode("utf-8"))

            def log_message(self, format: str, *args: object) -> None:
                return

        params = {
            "client_id": self._config.client_id,
            "response_type": "code",
            "redirect_uri": self._config.redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "state": state,
            "scope": SPOTIFY_SCOPES,
        }
        auth_url = f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"

        print("=" * 78)
        print("[SPOTIFY OAUTH2 PKCE LOGIN]")
        print("Opening your browser to authorize Spotify Premium playback control...")
        print(f"If the browser does not open automatically, visit:\n{auth_url}")
        print("=" * 78)
        webbrowser.open(auth_url)

        with HTTPServer((host, port), _CallbackHandler) as server:
            server.handle_request()

        if "error" in captured:
            raise RuntimeError(f"Spotify authorization failed: {captured['error']}")
        if captured.get("state") != state or "code" not in captured:
            raise RuntimeError("Spotify OAuth callback state mismatch or missing authorization code.")

        token_payload: dict[str, str] = {
            "client_id": self._config.client_id,
            "grant_type": "authorization_code",
            "code": captured["code"],
            "redirect_uri": self._config.redirect_uri,
            "code_verifier": verifier,
        }
        if self._config.client_secret:
            token_payload["client_secret"] = self._config.client_secret

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(SPOTIFY_TOKEN_URL, data=token_payload)
            resp.raise_for_status()
            data = resp.json()

        return SpotifyTokenBundle(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            expires_at=time.time() + float(data.get("expires_in", 3600)),
            scope=data.get("scope", SPOTIFY_SCOPES),
        )
