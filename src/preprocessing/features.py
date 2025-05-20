import librosa

def extract_log_mel(path, sr=16000, n_mels=80, hop_length=160, win_length=400):
    y, _ = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(path= y, sr=_, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
    return librosa.power_to_db(mel)