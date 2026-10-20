"""Live CLI entrypoint to log into Spotify Premium (OAuth2 PKCE) and run the Autonomous DJ Engine."""

from __future__ import annotations

import argparse
import sys

from atv.config import SpotifyAppConfig
from atv.engine.decision_engine import HybridDJDecisionEngine
from atv.models import SessionState
from atv.spotify.auth import SpotifyPKCEAuthenticator
from atv.spotify.player import LiveDJSessionRunner, SpotifyConnectClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ATV Autonomous DJ — Spotify Premium OAuth2 PKCE & Live Session Runner"
    )
    parser.add_argument(
        "--liked",
        action="store_true",
        help="Use your Spotify 'Tus me gusta' / Liked Songs library as the track pool",
    )
    parser.add_argument(
        "--playlist",
        type=str,
        default="",
        help="Spotify Playlist URL, URI, or ID (falls back to SPOTIFY_DEFAULT_PLAYLIST in .env)",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=20,
        help="Maximum number of tracks to hydrate from the playlist (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Authenticate, hydrate the real Spotify playlist, and print the generated Playbooks without altering live playback",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SpotifyAppConfig.from_env(".env")

    try:
        auth = SpotifyPKCEAuthenticator(config)
        token = auth.get_valid_access_token()
        assert token
    except ValueError as exc:
        print(f"[CONFIGURATION ERROR] {exc}")
        return 1
    except Exception as exc:
        print(f"[AUTHENTICATION ERROR] {exc}")
        return 1

    client = SpotifyConnectClient(auth)
    profile = client.get_current_user_profile()
    display_name = profile.get("display_name") or profile.get("id") or "Spotify User"
    product = profile.get("product", "unknown")
    print(f"Authenticated as: {display_name} (Account tier: {product})")

    runner = LiveDJSessionRunner(spotify_client=client, config=config)

    # 1. Fetch lightweight metadata instantly from Spotify (takes < 0.5s)
    if args.liked:
        print("Fetching your Spotify Liked Songs ('Tus me gusta')...")
        raw_tracks = client.fetch_saved_tracks(limit=args.max_tracks)
    else:
        playlist_target = args.playlist.strip() or config.default_playlist
        if not playlist_target:
            choice = input("Enter Spotify Playlist URL/URI (or press Enter for 'Tus me gusta'): ").strip()
            if not choice:
                args.liked = True
                print("Fetching your Spotify Liked Songs ('Tus me gusta')...")
                raw_tracks = client.fetch_saved_tracks(limit=args.max_tracks)
            else:
                playlist_target = choice

        if not args.liked:
            if not playlist_target:
                print("No playlist or source selected. Exiting.")
                return 1
            raw_tracks = client.fetch_playlist_tracks(playlist_target, limit=args.max_tracks)

    if len(raw_tracks) < 2:
        print("Need at least 2 tracks in library/playlist.")
        return 1

    # In dry-run mode, hydrate a few synchronously to inspect
    if args.dry_run:
        pool = runner.hydrate_tracks(raw_tracks[:5])
        engine = HybridDJDecisionEngine()
        session = SessionState(
            target_energy=config.target_energy,
            network_latency_offset_ms=config.network_latency_offset_ms,
            creativity_temperature=config.creativity_temperature,
        )
        out = engine.plan_stochastic_playbooks(pool[0], pool[1:], session)
        all_pbs = [out.selected_playbook, *out.alternative_playbooks]
        print("\n" + "=" * 95)
        print(f"[DRY-RUN PLAYBOOKS FOR INITIAL TRACK: {pool[0].autor} - {pool[0].titulo}]")
        print("=" * 95)
        for pb in all_pbs:
            mark = ">> SELECTED <<" if pb.archetype == out.selected_playbook.archetype else ""
            print(
                f"  * {pb.archetype:<18} | Score={pb.raw_score:>6.2f} | Prob={pb.selection_probability * 100:>5.1f}% | "
                f"Next={pb.action_plan.next_track_uri} {mark}"
            )
        print("=" * 95)
        return 0

    # Live Mode: Instant start! Runs playback immediately and hydrates candidate pool in background
    runner.run_live_loop(raw_tracks, max_transitions=15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
