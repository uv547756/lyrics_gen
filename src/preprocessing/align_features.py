import numpy as np

def align_spotify_features(analysis, hop_length, sr=16000):
    beats = [b['start'] for b in analysis['beats']]
    frame_times = np.arange(
        0, analysis['track']['duration'], hop_length/sr
    )
    beat_flags = np.isin(
        np.round(frame_times,3), np.round(beats,3)
    ).astype(int)
    return beat_flags  # shape (T,)