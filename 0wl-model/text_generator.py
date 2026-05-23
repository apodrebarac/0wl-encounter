"""
RNN and LSTM Text Generator
============================
Compatible with Mac (CPU/MPS) and Windows (CPU).
Trained on plain UTF-8 text files (e.g. English translations of Dostoyevsky).

Commands:
  train    -- train a new model from scratch
  generate -- generate text from a saved checkpoint
  finetune -- continue training an existing checkpoint on new text
"""

import argparse
import os
import sys
import time
import pickle

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  Loaded {len(text):,} characters from '{path}'")
    return text


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return chars, char2idx, idx2char


def encode(text, char2idx):
    return [char2idx[c] for c in text if c in char2idx]


def make_sequences(encoded, seq_len):
    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_len):
        inputs.append(encoded[i: i + seq_len])
        targets.append(encoded[i + 1: i + seq_len + 1])
    return inputs, targets


def batch_iter(inputs, targets, batch_size, device):
    n = len(inputs)
    idx = list(range(n))
    np.random.shuffle(idx)
    for start in range(0, n - batch_size + 1, batch_size):
        batch_idx = idx[start: start + batch_size]
        x = torch.tensor([inputs[i] for i in batch_idx], dtype=torch.long).to(device)
        y = torch.tensor([targets[i] for i in batch_idx], dtype=torch.long).to(device)
        yield x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, model_type, dropout=0.3):
        super().__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if model_type == "lstm":
            self.rnn = nn.LSTM(
                hidden_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
        else:  # vanilla rnn
            self.rnn = nn.RNN(
                hidden_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                nonlinearity="tanh"
            )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        if self.model_type == "lstm":
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return (h, c)
        return h

    def detach_hidden(self, hidden):
        if self.model_type == "lstm":
            return (hidden[0].detach(), hidden[1].detach())
        return hidden.detach()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, char2idx, idx2char, args_dict):
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "model_type": args_dict["model_type"],
        "hidden_size": args_dict["hidden_size"],
        "num_layers": args_dict["num_layers"],
        "vocab_size": len(char2idx),
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TextModel(
        vocab_size=ckpt["vocab_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        model_type=ckpt["model_type"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["char2idx"], ckpt["idx2char"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device(args.cpu)
    print(f"\n[train] Device: {device}")

    text = load_text(args.input_text)
    chars, char2idx, idx2char = build_vocab(text)
    vocab_size = len(chars)
    print(f"  Vocabulary: {vocab_size} unique characters")

    encoded = encode(text, char2idx)
    inputs, targets = make_sequences(encoded, args.seq_len)
    print(f"  Sequences: {len(inputs):,}")

    model = TextModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        model_type=args.model_type,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model_type.upper()} | hidden={args.hidden_size} | layers={args.num_layers} | params={total_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        t0 = time.time()

        for x, y in batch_iter(inputs, targets, args.batch_size, device):
            hidden = model.init_hidden(x.size(0), device)
            optimizer.zero_grad()
            logits, hidden = model(x, hidden)
            # logits: (batch, seq_len, vocab_size)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")

        # Save checkpoint each epoch so you never lose progress
        save_checkpoint(args.checkpoint, model, char2idx, idx2char, vars(args))

    print(f"\n[train] Done. Final checkpoint: {args.checkpoint}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(args):
    device = get_device(args.cpu)
    print(f"\n[generate] Device: {device}")

    model, char2idx, idx2char = load_checkpoint(args.checkpoint, device)
    model.eval()

    # Filter prompt to known characters only
    prompt = args.prompt
    filtered = [c for c in prompt if c in char2idx]
    if len(filtered) < len(prompt):
        skipped = set(prompt) - set(char2idx.keys())
        print(f"  Warning: skipped characters not in vocabulary: {skipped}")
    if not filtered:
        print("  Error: prompt contains no characters from training vocabulary.")
        sys.exit(1)

    # Encode prompt
    encoded = [char2idx[c] for c in filtered]
    x = torch.tensor([encoded], dtype=torch.long).to(device)

    hidden = model.init_hidden(1, device)

    # Prime hidden state on prompt
    with torch.no_grad():
        _, hidden = model(x, hidden)

    generated = list(filtered)
    current_char = torch.tensor([[encoded[-1]]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(args.length):
            logits, hidden = model(current_char, hidden)
            logits = logits[0, -1, :] / args.temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(idx2char[next_idx])
            current_char = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    output = "".join(generated)
    print(f"\n{'─'*60}")
    print(output)
    print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(args):
    device = get_device(args.cpu)
    print(f"\n[finetune] Device: {device}")

    model, char2idx, idx2char = load_checkpoint(args.checkpoint, device)
    print(f"  Base checkpoint loaded: {args.checkpoint}")
    print(f"  Base vocabulary size: {len(char2idx)}")

    text = load_text(args.input_text)

    # Report any new characters that will be skipped
    new_chars = set(text) - set(char2idx.keys())
    if new_chars:
        print(f"  Warning: {len(new_chars)} new character(s) not in base vocabulary will be skipped.")
        print(f"  Skipped: {sorted(new_chars)}")

    encoded = encode(text, char2idx)  # encode() already filters unknown chars
    if len(encoded) < args.seq_len + 1:
        print("  Error: not enough text after filtering. Use a larger file or retrain from scratch.")
        sys.exit(1)

    inputs, targets = make_sequences(encoded, args.seq_len)
    print(f"  Sequences for fine-tuning: {len(inputs):,}\n")

    vocab_size = len(char2idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        t0 = time.time()

        for x, y in batch_iter(inputs, targets, args.batch_size, device):
            hidden = model.init_hidden(x.size(0), device)
            optimizer.zero_grad()
            logits, hidden = model(x, hidden)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")

    save_checkpoint(
        args.output_checkpoint,
        model,
        char2idx,
        idx2char,
        {
            "model_type": model.model_type,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
        },
    )
    print(f"\n[finetune] Done. Saved → {args.output_checkpoint}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RNN / LSTM Text Generator — Mac & Windows compatible"
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a new model from scratch")
    p_train.add_argument("--input-text",   required=True,  help="Path to training text file (UTF-8)")
    p_train.add_argument("--model-type",   default="lstm", choices=["lstm", "rnn"])
    p_train.add_argument("--epochs",       type=int,   default=20)
    p_train.add_argument("--seq-len",      type=int,   default=80)
    p_train.add_argument("--batch-size",   type=int,   default=64)
    p_train.add_argument("--hidden-size",  type=int,   default=256)
    p_train.add_argument("--num-layers",   type=int,   default=2)
    p_train.add_argument("--lr",           type=float, default=0.002)
    p_train.add_argument("--checkpoint",   required=True,  help="Where to save the model (.pt)")
    p_train.add_argument("--cpu",          action="store_true", help="Force CPU (skip MPS/CUDA)")

    # ── generate ───────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate text from a checkpoint")
    p_gen.add_argument("--checkpoint",  required=True)
    p_gen.add_argument("--prompt",      default="The",  help="Seed text to start generation")
    p_gen.add_argument("--length",      type=int,   default=300, help="Number of characters to generate")
    p_gen.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.5=focused, 1.2=wild)")
    p_gen.add_argument("--cpu",         action="store_true")

    # ── finetune ───────────────────────────────────────────────────────────
    p_ft = sub.add_parser("finetune", help="Fine-tune an existing checkpoint on new text")
    p_ft.add_argument("--checkpoint",        required=True, help="Base checkpoint to start from")
    p_ft.add_argument("--input-text",        required=True, help="New text to fine-tune on")
    p_ft.add_argument("--output-checkpoint", required=True, help="Where to save the adapted model")
    p_ft.add_argument("--epochs",            type=int,   default=3)
    p_ft.add_argument("--batch-size",        type=int,   default=64)
    p_ft.add_argument("--seq-len",           type=int,   default=80)
    p_ft.add_argument("--lr",               type=float, default=0.001)
    p_ft.add_argument("--cpu",              action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "finetune":
        finetune(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
