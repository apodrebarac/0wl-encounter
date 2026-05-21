"""
2_TRAIN.py
──────────
Run this second. It will take time. That is correct.

This script trains the degrading model on your corpus.
It saves a checkpoint every 10 epochs so you can use
intermediate states — an earlier checkpoint is a more
degraded model (less learned, more broken).

The training process is slow by design. The model is small,
running on CPU is expected. Let it run. Check back.
The waiting is part of the work.

HOW TO RUN:
  python 2_train.py

  To resume from a checkpoint:
  python 2_train.py --resume checkpoints/epoch_050.pt

REQUIREMENTS:
  pip install torch

WHAT TO EXPECT:
  The loss (printed during training) will decrease over time.
  Lower loss = the model has learned more patterns from your text.
  You do not need to train until the loss stops decreasing.
  A partially trained model has its own quality.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import argparse
import json

# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS — these are the parameters you can tune
#  Read the comments. Each one does something specific.
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_SIZE = 256
# The model's memory capacity. This is how many numbers the model uses
# to hold its understanding of what it has read so far.
# Smaller = less can be held = more forgetting = more degraded output.
# Larger = more faithful to patterns = less degraded.
#
# Try: 128  (more degraded, trains faster)
#      256  (default — the balance point)
#      512  (more faithful, trains slower)

NUM_LAYERS = 2
# How many LSTM layers are stacked. Each layer processes the output of the one before.
# More layers = the model can learn longer-range patterns.
# But for a corpus this size, 2 is correct.
#
# Try: 1  (shallower memory, more immediate, faster)
#      2  (default)
#      3  (deeper — unlikely to help much here)

DROPOUT = 0.3
# The rate at which connections are randomly severed during training.
# At each training step, 30% of the network's connections are temporarily cut.
# The model learns to work with loss already built in.
# This IS the degradation mechanism, architecturally.
#
# Try: 0.1  (less degradation — more coherent output)
#      0.3  (default)
#      0.5  (heavy degradation — broken, strange output)
#      0.0  (no degradation — not recommended for this project)

SEQUENCE_LENGTH = 100
# How many characters the model reads at once before making a prediction.
# This is the model's window of attention.
# Shorter = more fragmented, less context, more surprising leaps.
# Longer = more coherent sequences, more recognizable patterns.
#
# Try: 50   (very fragmented)
#      100  (default)
#      150  (longer memory per step)

BATCH_SIZE = 32
# How many sequences are trained simultaneously.
# This is a technical parameter — higher = faster training, more memory used.
# 32 is safe for most machines.

LEARNING_RATE = 0.002
# How large a step the model takes when adjusting its weights.
# Too high: the model overshoots and learns nothing.
# Too low: the model learns very slowly.
# 0.002 is a reliable starting point for this corpus size.

NUM_EPOCHS = 150
# How many complete passes through the training data.
# More epochs = more learned = less degraded output.
# A model at epoch 30 will feel different from a model at epoch 150.
# Both are valid. Earlier = more broken. Later = more formed.
# Save checkpoints (default: every 10 epochs) to preserve earlier states.

CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 10   # Save a checkpoint every N epochs

CORPUS_PATH = "corpus.txt"

# ─────────────────────────────────────────────────────────────────────────────
#  THE MODEL ARCHITECTURE
#  Read this to understand what is being built.
# ─────────────────────────────────────────────────────────────────────────────

class DegradingModel(nn.Module):
    """
    A character-level Long Short-Term Memory (LSTM) network.

    WHY CHARACTER-LEVEL:
    The model reads and generates one character at a time. This preserves
    the grain of your writing — the idiosyncratic spacing, the broken grammar,
    the abrupt stops. A word-level model would smooth those edges. This one keeps them.
    Your corpus is already degraded language (RNN outputs). This model learns
    the texture of degradation itself, not the words.

    WHY LSTM:
    The LSTM has four gates: input, forget, cell, output.
    The FORGET GATE is the critical one here. At each character, the model
    decides what to discard from memory. It learns which patterns are worth
    holding and which can be released.
    This is not a metaphor. It is the literal mechanism.

    WHY TWO LAYERS:
    The first layer learns low-level patterns (letter sequences, common endings).
    The second layer learns higher-level patterns (line shapes, rhythmic structures).
    Together they produce output that feels like language without quite being it.

    WHY DROPOUT:
    Randomly zeroing connections during training means the model must learn
    redundant representations — it cannot rely on any single pathway.
    The result is a model that generates with uncertainty baked in.
    """

    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding: turns each character (an integer) into a vector
        # the LSTM can process. Think of it as giving each character a position
        # in a space the model can navigate.
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # The LSTM: the memory architecture.
        # batch_first=True means input shape is (batch, sequence, features)
        # dropout is applied between layers (not on the final layer)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Additional dropout after the LSTM — applied during generation too
        # when model.train() is active. This adds variability at generation time
        # if you choose to enable it.
        self.dropout = nn.Dropout(dropout)

        # Decoder: maps from the LSTM's hidden state back to a probability
        # distribution over all characters. This is what generates the next character.
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        One forward pass: given a sequence of characters, predict the next character.

        x      : (batch_size, sequence_length) — integer indices of characters
        hidden : the LSTM's memory state from the previous step
                 if None, starts with zeros (a blank memory)
        """
        embedded = self.embedding(x)                    # (batch, seq, hidden)
        output, hidden = self.lstm(embedded, hidden)    # output: (batch, seq, hidden)
        output = self.dropout(output)
        logits = self.decoder(output)                   # (batch, seq, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """
        Returns a fresh hidden state — zeros, no memory.
        This is the model's state at the start of a new generation.
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(path):
    """Load the text corpus and build character vocabulary."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, path)

    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Every unique character in the corpus gets an integer index.
    # The model works with these integers, not the characters directly.
    chars = sorted(set(text))
    vocab_size = len(chars)

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    print(f"Corpus length: {len(text):,} characters")
    print(f"Unique characters (vocabulary size): {vocab_size}")

    return text, chars, char_to_idx, idx_to_char, vocab_size


def make_batches(text, char_to_idx, seq_length, batch_size):
    """
    Convert text into training batches.

    The corpus is split into sequences of length seq_length.
    For each sequence, the target is the same sequence shifted by one character:
    the model learns to predict the next character given the current one.

    This is the fundamental learning task: given what has come before, what comes next?
    For a corpus of degraded language, the answer is always almost-right.
    """
    encoded = np.array([char_to_idx[c] for c in text], dtype=np.int64)

    # Trim to be divisible by (batch_size * seq_length)
    n = (len(encoded) - 1) // (batch_size * seq_length) * (batch_size * seq_length)
    inputs = encoded[:n].reshape(batch_size, -1)
    targets = encoded[1:n+1].reshape(batch_size, -1)

    num_batches = inputs.shape[1] // seq_length

    for i in range(num_batches):
        x = inputs[:, i * seq_length:(i + 1) * seq_length]
        y = targets[:, i * seq_length:(i + 1) * seq_length]
        yield torch.tensor(x), torch.tensor(y)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(resume_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load corpus
    text, chars, char_to_idx, idx_to_char, vocab_size = load_corpus(CORPUS_PATH)

    # Save vocabulary so the generation script can use it
    vocab_path = os.path.join(script_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "chars": chars,
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
            "vocab_size": vocab_size,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT
        }, f, ensure_ascii=False, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining device: {device}")
    if device.type == "cpu":
        print("Running on CPU. Training will be slow. This is expected.")
        print("You can let it run in the background.\n")

    # Build model
    model = DegradingModel(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0

    # Resume from checkpoint if requested
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"\nSettings:")
    print(f"  Hidden size:     {HIDDEN_SIZE}  (memory capacity)")
    print(f"  Layers:          {NUM_LAYERS}")
    print(f"  Dropout:         {DROPOUT}  (degradation rate)")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Epochs:          {NUM_EPOCHS}")
    print(f"\nTraining...\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0
        start_time = time.time()

        hidden = model.init_hidden(BATCH_SIZE, device)

        for x_batch, y_batch in make_batches(text, char_to_idx, SEQUENCE_LENGTH, BATCH_SIZE):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Detach hidden state to prevent backprop through full history
            hidden = tuple(h.detach() for h in hidden)

            optimizer.zero_grad()
            logits, hidden = model(x_batch, hidden)

            # logits: (batch, seq, vocab) → reshape for loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                y_batch.reshape(-1)
            )

            loss.backward()

            # Gradient clipping prevents the model from making catastrophically
            # large updates. Keeps training stable.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1:>4}/{NUM_EPOCHS}  |  loss: {avg_loss:.4f}  |  {elapsed:.1f}s")

        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"epoch_{epoch + 1:04d}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            }, checkpoint_path)
            print(f"           → checkpoint saved: epoch_{epoch + 1:04d}.pt")

    print(f"\nTraining complete.")
    print(f"Checkpoints saved in: {checkpoint_dir}/")
    print(f"Run 3_generate.py to generate text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_path=args.resume)
