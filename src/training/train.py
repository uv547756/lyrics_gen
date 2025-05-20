import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_scheduler
from models.lyrics_model import LyricsGenModel
from preprocessing.dataset import LyricsDataset

# Hyperparameters and paths
vocab_size = 10000
feat_dim = 4
epochs = 10
batch_size = 16
learning_rate = 1e-4
warmup_steps = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare datasets and loaders
train_dataset = LyricsDataset(data_dir='data/processed', split='train')
val_dataset   = LyricsDataset(data_dir='data/processed', split='val')
train_loader  = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=val_dataset.collate_fn
)

# Model, optimizer, scheduler, and loss
model = LyricsGenModel(vocab_size=vocab_size, feat_dim=feat_dim).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = epochs * len(train_loader)
scheduler = get_scheduler(
    name='inverse_sqrt',
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_token)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for mel, features, tgt_input, tgt_output, align_labels in train_loader:
        mel = mel.to(device)
        features = features.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        align_labels = align_labels.to(device)

        optimizer.zero_grad()
        logits, align_logits = model(mel, features, tgt_input)

        # Reshape for loss
        logits = logits.view(-1, vocab_size)
        tgt_output = tgt_output.view(-1)
        align_logits = align_logits.view(-1, 2)
        align_labels = align_labels.view(-1)

        # Compute losses
        loss_text = criterion(logits, tgt_output)
        loss_align = criterion(align_logits, align_labels)
        loss = loss_text + 0.5 * loss_align

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    # (Optional) Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mel, features, tgt_input, tgt_output, align_labels in val_loader:
            mel = mel.to(device)
            features = features.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            align_labels = align_labels.to(device)
            logits, align_logits = model(mel, features, tgt_input)
            logits = logits.view(-1, vocab_size)
            tgt_output = tgt_output.view(-1)
            align_logits = align_logits.view(-1, 2)
            align_labels = align_labels.view(-1)
            loss_text = criterion(logits, tgt_output)
            loss_align = criterion(align_logits, align_labels)
            val_loss += (loss_text + 0.5 * loss_align).item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch}/{epochs} - Val Loss: {avg_val_loss:.4f}")