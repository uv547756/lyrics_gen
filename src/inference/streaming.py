import torch
from models.lyrics_model import LyricsGenModel
from preprocessing.download_audio import download_preview_audio
from preprocessing.separate import separate_vocals
from preprocessing.features import extract_log_mel
from preprocessing.dataset import LyricsDataset

# Constants and setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate dataset for token mappings
dataset = LyricsDataset(data_dir='data/processed', split='train')
START_TOKEN = dataset.start_token
END_TOKEN = dataset.end_token

# Load model checkpoint
model = LyricsGenModel(vocab_size=10000, feat_dim=4).to(device)
model.load_state_dict(torch.load('checkpoints/best.pt', map_location=device))
model.eval()


def chunked_inference(preview_url: str, max_tokens: int = 200) -> str:
    # Download and convert preview
    wav = download_preview_audio(preview_url)
    # Separate vocals
    vocals = separate_vocals(wav)
    # Extract mel-spectrogram
    mel = extract_log_mel(vocals)              # shape (T_src, 80)
    # Create dummy features (e.g., zeros) or load aligned features
    features = torch.zeros(1, mel.shape[0], 4)  # shape (1, T_src, feat_dim)

    # Prepare initial token sequence
    tokens = [START_TOKEN]
    for _ in range(max_tokens):
        tgt_input = torch.tensor([tokens], dtype=torch.long).to(device)  # (1, len)
        mel_tensor = torch.tensor([mel], dtype=torch.float).to(device)   # (1, T_src, 80)
        features_tensor = features.to(device)

        # Forward pass
        logits, _ = model(mel_tensor, features_tensor, tgt_input)
        next_token = logits.argmax(-1)[0, -1].item()
        tokens.append(next_token)

        if next_token == END_TOKEN:
            break

    # Convert token list to text (implement decode in dataset)
    return dataset.decode(tokens)