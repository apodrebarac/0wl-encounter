"""
1_PREPARE_CORPUS.py
───────────────────
Run this first. Once.

This script extracts all text from your two poetry books (book-0.pdf and book-1.pdf)
and saves it as a single plain text file: corpus.txt

That file becomes the training data for the degrading model.

WHAT GETS INCLUDED:
  - All poems, titles, whitespace, line breaks
  - The pages of numbers (the model weights you printed inside the books)
    These number sequences will occasionally surface in the generated output,
    which reads as the machine's numerical interior breaking through language.
    This is intentional. They stay in.

HOW TO RUN:
  python 1_prepare_corpus.py

REQUIREMENTS:
  pip install pymupdf
"""

import fitz   # PyMuPDF — reads PDFs
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS — change these if your files are elsewhere
# ─────────────────────────────────────────────────────────────────────────────

PDF_PATHS = [
    "../book-0.pdf",   # Poems from F0X  — dedicated "Read them to trees"
    "../book-1.pdf",   # Poems from am1  — dedicated "Read them to a mirror"
]

OUTPUT_PATH = "corpus.txt"

# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF, page by page.

    PyMuPDF reads the text in the order it appears on the page.
    For visually arranged poems (words scattered across the page, rotated text,
    different font sizes) the extraction will be imperfect — words may appear
    in unexpected order, or merge strangely.

    This imperfection is not a problem to fix.
    The RNN trains on characters, not on meaning. The slightly scrambled
    extraction will become part of the texture the model learns from.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")   # plain text extraction
        if text.strip():               # skip truly empty pages (but keep sparse ones)
            pages_text.append(text)

    doc.close()
    return "\n".join(pages_text)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_text_parts = []

    print("Extracting corpus from poetry books...\n")

    for pdf_path in PDF_PATHS:
        full_path = os.path.join(script_dir, pdf_path)

        if not os.path.exists(full_path):
            print(f"ERROR: Cannot find {full_path}")
            print("Check that PDF_PATHS in this script points to the right location.")
            sys.exit(1)

        print(f"Reading: {os.path.basename(full_path)}")
        text = extract_text_from_pdf(full_path)
        all_text_parts.append(text)
        print(f"  → {len(text):,} characters extracted")

    # Join both books with a separator the model will learn to cross
    corpus = "\n\n---\n\n".join(all_text_parts)

    output_path = os.path.join(script_dir, OUTPUT_PATH)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\n───────────────────────────────────────────────")
    print(f"Corpus saved to: {output_path}")
    print(f"Total characters: {len(corpus):,}")
    print(f"Unique characters: {len(set(corpus))}")
    print(f"\nFirst 300 characters of corpus:")
    print("─" * 40)
    print(corpus[:300])
    print("─" * 40)
    print("\nCorpus ready. Run 2_train.py next.")


if __name__ == "__main__":
    main()
