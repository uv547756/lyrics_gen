import torch
import torch.nn as nn

# 4.1 Feed-Forward Module

def FeedForward(dim, hidden, dropout):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, dim),
        nn.Dropout(dropout)
    )

# 4.2 Conformer Block

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, conv_kernel=31, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, dim * ff_mult, dropout)
        self.mhsa = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel//2, groups=dim),
            nn.GLU(dim),
            nn.Conv1d(dim//2, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ff2 = FeedForward(dim, dim * ff_mult, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Macaron-style FF
        x = x + 0.5 * self.ff1(x)
        # Multi-head Self-Attention
        x2 = self.norm(x).transpose(0, 1)
        x = x + self.mhsa(x2, x2, x2)[0].transpose(0, 1)
        # Convolutional Module
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        # Final FF
        x = x + 0.5 * self.ff2(x)
        return x

# 4.3 Feature Fusion Module

class FeatureFusion(nn.Module):
    def __init__(self, feat_dim, model_dim):
        super().__init__()
        self.proj = nn.Linear(feat_dim, model_dim)

    def forward(self, enc_out, feat):
        # enc_out: (B, T, D), feat: (B, T, F)
        return enc_out + self.proj(feat)

# 4.4 Lyrics Generation Model

dim, heads = 512, 8  # model dimension and transformer heads

class LyricsGenModel(nn.Module):
    def __init__(self, vocab_size, feat_dim, max_len=1000):
        super().__init__()
        # Encoder: stacked Conformer blocks
        self.encoder = nn.Sequential(
            *[ConformerBlock(dim, heads) for _ in range(12)]
        )
        # Fuse audio features (e.g., tempo, key, beat flags)
        self.fusion = FeatureFusion(feat_dim, dim)
        # Alignment head for beat/bar boundary classification
        self.align_head = nn.Linear(dim, 2)
        # Token embedding and positional encoding for decoder
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        # Transformer decoder
        self.decoder = nn.Transformer(
            d_model=dim,
            nhead=heads,
            num_encoder_layers=0,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        # Output projection to vocabulary logits
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, mel, features, tgt_tokens):
        # mel: (B, T_src, 80), features: (B, T_src, F)
        enc = self.encoder(mel)                 # (B, T_src, D)
        fused = self.fusion(enc, features)     # (B, T_src, D)
        align_logits = self.align_head(fused)  # (B, T_src, 2)

        # Prepare decoder inputs
        # Memory for decoder (source) expects shape (B, T_src, D)
        memory = fused
        # Embed target tokens and add positional encoding
        tgt_emb = self.token_embed(tgt_tokens)  # (B, T_tgt, D)
        seq_len = tgt_emb.size(1)
        tgt_emb = tgt_emb + self.pos_embed[:, :seq_len, :]

        # Run decoder: returns (B, T_tgt, D)
        dec_out = self.decoder(src=memory, tgt=tgt_emb)

        # Project to vocabulary
        logits = self.out_proj(dec_out)         # (B, T_tgt, vocab_size)
        return logits, align_logits
