import requests
import tempfile
from pydub import AudioSegment

def download_preview_audio(preview_url: str) -> str:
    """
    Downloads Spotify preview (MP3, up to 30s) and converts to WAV.
    Returns path to WAV file.
    """
    # Download MP3
    res = requests.get(preview_url, stream=True)
    res.raise_for_status()
    mp3_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    for chunk in res.iter_content(chunk_size=8192):
        mp3_tmp.write(chunk)
    mp3_tmp.close()

    # Convert to WAV
    wav_path = mp3_tmp.name.replace(".mp3", ".wav")
    audio = AudioSegment.from_file(mp3_tmp.name, format="mp3")
    audio.export(wav_path, format="wav")
    return wav_path