import os
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

class SpotifyClient:
    def __init__(self):
        cred = SpotifyClientCredentials(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
        )
        self.sp = Spotify(auth_manager=cred)

    def get_track_metadata(self, track_id: str) -> dict | None:
        return self.sp.track(track_id)

    def get_preview_url(self, track_id: str) -> str | None:
        track = self.sp.track(track_id)
        if track is not None:
            return track.get("preview_url")
        return "not found"

    def get_audio_features(self, track_id: str) -> dict | None:
        features_list = self.sp.audio_features([track_id])
        if features_list and features_list[0] is not None:
            return features_list[0]
        return {"error": "not found"}

    def get_audio_analysis(self, track_id: str) -> dict | None:
        analysis = self.sp.audio_analysis(track_id)
        if analysis is not None:
            return analysis
        return {"error": "not found"}