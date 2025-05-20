import os
import torch
from torch.utils.data import Dataset
from preprocessing.features import extract_log_mel
from preprocessing.align_features import align_spotify_features

class LyricsDataset(Dataset):
    def __init__(self, data_dir, split='train', max_len=500, vocab_path='data/vocab.json'):
        """
        data_dir/
          train/
            track1.wav
            track1.json   # contains 'audio_analysis', 'tokens'
            ...
          val/
        vocab_path: JSON file mapping token IDs to strings ("id2token").
        """
        self.data_dir = os.path.join(data_dir, split)
        self.files = [f.split('.wav')[0] for f in os.listdir(self.data_dir) if f.endswith('.wav')]
        self.max_len = max_len
        # Load vocabulary mapping
        import json
        vocab = json.load(open(vocab_path, 'r'))
        self.id2token = {int(k):v for k,v in vocab.get('id2token', {}).items()}
        self.pad_token = int(vocab.get('pad_token', 0))
        self.start_token = int(vocab.get('start_token', 1))
        self.end_token = int(vocab.get('end_token', 2))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        wav_path = os.path.join(self.data_dir, f"{name}.wav")
        meta_path = os.path.join(self.data_dir, f"{name}.json")

        # Load audio features
        mel = extract_log_mel(wav_path)            # (T_src, 80)
        # Load precomputed Spotify analysis
        import json
        analysis = json.load(open(meta_path, 'r'))['audio_analysis']
        beat_flags = align_spotify_features(analysis, hop_length=160)

        # Load token sequence
        tokens = json.load(open(meta_path, 'r'))['tokens']  # list of ints
        # Prepare input/output
        tgt_input = [self.start_token] + tokens
        tgt_output = tokens + [self.end_token]

        return (
            torch.tensor(mel, dtype=torch.float),              # mel
            torch.tensor(beat_flags, dtype=torch.float).unsqueeze(-1),     # features
            torch.tensor(tgt_input, dtype=torch.long),          # for decoder input
            torch.tensor(tgt_output, dtype=torch.long),         # for loss target
            torch.tensor(beat_flags, dtype=torch.long)          # alignment labels
        )

    def collate_fn(self, batch):
        # pad sequences to max length
        mel_batch, feat_batch, inp_batch, out_batch, align_batch = zip(*batch)
        mel_batch = list(mel_batch)
        feat_batch = list(feat_batch)
        inp_batch = list(inp_batch)
        out_batch = list(out_batch)
        align_batch = list(align_batch)
        # Implementation of padding omitted for brevity
        # Return padded tensors
        return (
            torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(feat_batch, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(inp_batch, batch_first=True, padding_value=self.pad_token),
            torch.nn.utils.rnn.pad_sequence(out_batch, batch_first=True, padding_value=self.pad_token),
            torch.nn.utils.rnn.pad_sequence(align_batch, batch_first=True, padding_value=0)
        )
    
    def decode(self, token_ids):
        """
        Convert a list of token IDs back to a lyric string.
        """
        tokens = [self.id2token.get(i, '') for i in token_ids]
        # Remove start/end and pads
        tokens = [tok for tok in tokens if tok not in ('<pad>', '<start>', '<end>', '')]
        return ' '.join(tokens)