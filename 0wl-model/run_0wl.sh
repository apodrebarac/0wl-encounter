#!/bin/bash
# ============================================================
# 0wl — full training workflow
# Run this from inside the 0wl-model/ folder:
#   cd 0wl-model
#   bash run_0wl.sh
#
# Before running, make sure your corpus files are in place:
#   ../corpus.txt          — books 0 and 1 (already exists)
#   ../0wl-corpus.txt      — Dear F0x / Dear 0wl letters (already exists)
#   grandma.txt            — grandmother's writings (add this when ready)
#   peerly_poems.txt       — Peerly × Decameron poems (add this when ready)
# ============================================================

set -e  # stop if any command fails

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         0wl — training pipeline      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── STEP 1: install dependencies ─────────────────────────────
echo "→ Installing dependencies..."
pip install -r requirements.txt -q
echo "  Done."
echo ""

# ── STEP 2: train base model on books 0+1 ────────────────────
echo "→ Step 1/4 — Training base model on books 0 and 1..."
echo "  (This is the longest step — go make tea.)"
echo ""
python3 text_generator.py train \
  --input-text ../corpus.txt \
  --checkpoint checkpoints/0wl_base.pt \
  --epochs 50 \
  --hidden-size 256 \
  --num-layers 2 \
  --lr 0.002

echo ""
echo "  ✓ Base model saved → checkpoints/0wl_base.pt"
echo ""

# ── STEP 3: fine-tune on grandmother's writings ───────────────
if [ -f "grandma.txt" ]; then
  echo "→ Step 2/4 — Fine-tuning on grandmother's writings..."
  python3 text_generator.py finetune \
    --checkpoint checkpoints/0wl_base.pt \
    --input-text grandma.txt \
    --output-checkpoint checkpoints/0wl_grandma.pt \
    --epochs 10 \
    --lr 0.001
  echo "  ✓ Saved → checkpoints/0wl_grandma.pt"
  PREV_CHECKPOINT="checkpoints/0wl_grandma.pt"
else
  echo "→ Step 2/4 — Skipping grandmother's writings (grandma.txt not found yet)"
  echo "  Add grandma.txt to this folder when ready and re-run."
  PREV_CHECKPOINT="checkpoints/0wl_base.pt"
fi
echo ""

# ── STEP 4: fine-tune on letters (Dear F0x / Dear 0wl) ───────
echo "→ Step 3/4 — Fine-tuning on Dear F0x / Dear 0wl letters..."
python3 text_generator.py finetune \
  --checkpoint $PREV_CHECKPOINT \
  --input-text ../0wl-corpus.txt \
  --output-checkpoint checkpoints/0wl_letters.pt \
  --epochs 10 \
  --lr 0.001
echo "  ✓ Saved → checkpoints/0wl_letters.pt"
echo ""

# ── STEP 5: fine-tune on Peerly poems ────────────────────────
if [ -f "peerly_poems.txt" ]; then
  echo "→ Step 4/4 — Fine-tuning on Peerly poems..."
  python3 text_generator.py finetune \
    --checkpoint checkpoints/0wl_letters.pt \
    --input-text peerly_poems.txt \
    --output-checkpoint checkpoints/0wl_final.pt \
    --epochs 15 \
    --lr 0.001
  echo "  ✓ Final model saved → checkpoints/0wl_final.pt"
  FINAL="checkpoints/0wl_final.pt"
else
  echo "→ Step 4/4 — Skipping Peerly poems (peerly_poems.txt not found yet)"
  echo "  Add peerly_poems.txt to this folder when ready and re-run from Step 4."
  cp checkpoints/0wl_letters.pt checkpoints/0wl_final.pt
  FINAL="checkpoints/0wl_final.pt"
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         Training complete            ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Final checkpoint: $FINAL"
echo ""
echo "To generate poetry:"
echo ""
echo "  python3 text_generator.py generate \\"
echo "    --checkpoint $FINAL \\"
echo "    --prompt \"Dear baka,\" \\"
echo "    --temperature 0.8 \\"
echo "    --length 800"
echo ""
