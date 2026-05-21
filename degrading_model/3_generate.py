"""
3_GENERATE.py
─────────────
Run this after training. Run it many times.
Each run produces a different output even from the same seed.

This script feeds a seed text to the trained model and lets it
continue — generating characters one at a time, each prediction
shaping the next.

The seed text is the door. What comes through it is not yours
and not the model's. It is between.

HOW TO RUN:
  python 3_generate.py

  To use a specific checkpoint (e.g. an earlier, more degraded model):
  python 3_generate.py --checkpoint checkpoints/epoch_0030.pt

  To save output to a file:
  python 3_generate.py --save output.txt

REQUIREMENTS:
  Run 1_prepare_corpus.py and 2_train.py first.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse

# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS — the primary expressive controls
# ─────────────────────────────────────────────────────────────────────────────

TEMPERATURE = 0.8
# The grief parameter. Controls how much the model diverges from what it knows.
#
# 0.3   faithful, eerie, close to your text — recognisable patterns repeat
# 0.5   slightly loosened — familiar rhythms, unexpected words
# 0.8   the balance point — almost-meaning, the sweet spot
# 1.0   standard sampling — no compression or dilation
# 1.2   dissociation begins — language untethers from sense
# 1.5   unmoored — fragments, the machine reaching past its knowledge
# 2.0   near-collapse — barely language, mostly texture
#
# There is no wrong temperature. Each one produces a different kind of text.
# Low temperature is grief that is still legible.
# High temperature is grief that has lost its grammar.

SEED_TEXT = "Dear baka,"
# What you feed the model to start generating.
# The model reads this and then continues from where it leaves off.
#
# Suggestions:
#   "Dear baka,"                          — opens like the letter
#   "0wl"                                 — the name, as a door
#   "I cannot bring you"                  — from the corpus itself
#   "The machine remembers"               — describing what is happening
#   "Please find me"                      — from Bridge Place
#   ""                                    — empty string: the model begins from silence

GENERATE_LENGTH = 800
# How many characters to generate after the seed.
# 500  — roughly a short poem
# 800  — a longer poem or short prose piece
# 1500 — an extended essay-poem
# The model does not know it has a length limit. It simply continues.

CHECKPOINT_PATH = None
# Leave as None to use the most recent checkpoint automatically.
# Or specify: "checkpoints/epoch_0050.pt" for a specific training state.
# Earlier checkpoints = less trained = more degraded output.
# This is a valid artistic choice.

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL (copy from 2_train.py — must match exactly)
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn as nn

class DegradingModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.decoder(output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_checkpoint(checkpoint_dir):
    """Find the most recently saved checkpoint."""
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("epoch_") and f.endswith(".pt")
    ]
    if not checkpoints:
        return None
    checkpoints.sort()
    return os.path.join(checkpoint_dir, checkpoints[-1])


def temperature_sample(logits, temperature):
    """
    Sample a character from the model's probability distribution,
    scaled by temperature.

    HOW IT WORKS:
    The model produces a score (logit) for every character in the vocabulary.
    We convert these to probabilities, then sample from them.

    Temperature divides the logits before converting to probabilities.
    - Low temperature (< 1.0): the distribution sharpens — high-probability
      characters become even more likely. The model plays it safe.
    - High temperature (> 1.0): the distribution flattens — unlikely characters
      become more possible. The model takes risks.

    At temperature=1.0, this is standard sampling with no modification.
    """
    logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)
    # Sample one character index from this probability distribution
    sampled = torch.multinomial(probabilities, num_samples=1)
    return sampled.item()


def generate(model, seed_text, char_to_idx, idx_to_char, vocab_size,
             length, temperature, device):
    """
    Generate text character by character.

    The model reads the seed_text first (the 'priming' phase),
    building up a hidden state from those characters.
    Then it begins generating: each new character is fed back in
    as the input for the next step.

    The hidden state carries memory forward across the entire generation.
    Because of dropout and the model's limited hidden_size, this memory
    degrades as the generation grows longer. Earlier parts of the generated
    text influence later parts, but less and less precisely over time.
    """
    model.eval()

    # Handle characters in seed that the model has never seen
    # (characters not in the training vocabulary)
    cleaned_seed = ""
    skipped = []
    for c in seed_text:
        if c in char_to_idx:
            cleaned_seed += c
        else:
            skipped.append(c)

    if skipped:
        print(f"Note: seed contained {len(skipped)} character(s) not in vocabulary, removed.")

    if not cleaned_seed:
        # If seed is empty or all characters were unknown, start from a random character
        cleaned_seed = idx_to_char[np.random.randint(vocab_size)]

    hidden = model.init_hidden(1, device)
    generated = cleaned_seed

    with torch.no_grad():
        # Prime the model: read through the seed to build hidden state
        for char in cleaned_seed[:-1]:
            idx = torch.tensor([[char_to_idx[char]]], dtype=torch.long).to(device)
            _, hidden = model(idx, hidden)

        # Generate: start from the last character of the seed
        current_char = cleaned_seed[-1]

        for _ in range(length):
            idx = torch.tensor([[char_to_idx[current_char]]], dtype=torch.long).to(device)
            logits, hidden = model(idx, hidden)

            # logits shape: (1, 1, vocab_size) → squeeze to (vocab_size,)
            next_char_idx = temperature_sample(logits.squeeze(), temperature)
            next_char = idx_to_char[str(next_char_idx)]

            generated += next_char
            current_char = next_char

    return generated


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--seed", type=str, default=SEED_TEXT)
    parser.add_argument("--length", type=int, default=GENERATE_LENGTH)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    # Load vocabulary
    vocab_path = os.path.join(script_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print("ERROR: vocab.json not found. Run 2_train.py first.")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    chars = vocab["chars"]
    char_to_idx = vocab["char_to_idx"]
    idx_to_char = vocab["idx_to_char"]
    vocab_size = vocab["vocab_size"]
    hidden_size = vocab["hidden_size"]
    num_layers = vocab["num_layers"]
    dropout = vocab["dropout"]

    # Find checkpoint
    checkpoint_dir = os.path.join(script_dir, CHECKPOINT_PATH or "checkpoints")
    if args.checkpoint:
        checkpoint_path = os.path.join(script_dir, args.checkpoint)
    else:
        checkpoint_path = get_latest_checkpoint(os.path.join(script_dir, "checkpoints"))

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("ERROR: No checkpoint found. Run 2_train.py first.")
        return

    print(f"Loading: {os.path.basename(checkpoint_path)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DegradingModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    epoch = checkpoint.get("epoch", "?")
    loss = checkpoint.get("loss", "?")

    print(f"Model state: epoch {epoch}, training loss {loss:.4f}" if isinstance(loss, float) else f"Model state: epoch {epoch}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: '{args.seed}'")
    print(f"Length: {args.length} characters")
    print(f"\n{'─' * 60}\n")

    # Generate
    output = generate(
        model=model,
        seed_text=args.seed,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        vocab_size=vocab_size,
        length=args.length,
        temperature=args.temperature,
        device=device
    )

    print(output)
    print(f"\n{'─' * 60}")

    # Save if requested
    if args.save:
        save_path = os.path.join(script_dir, args.save)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
