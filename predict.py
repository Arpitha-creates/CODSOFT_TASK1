#!/usr/bin/env python3
"""
CODSOFT Machine Learning Internship -- Task 1
Predict the genre of a movie from its plot summary.

Usage
-----
    python predict.py --text "A young wizard discovers he has magical powers ..."
    python predict.py --file plots.txt        # one plot per line
    python predict.py --text "..." --model models/best_model.joblib
"""

from __future__ import annotations

import argparse
import re
import sys

import joblib

# Reuse the exact same cleaner used at training time so features match.
from train import clean_text


def predict(model, text: str) -> str:
    return model.predict([clean_text(text)])[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Movie genre predictor")
    ap.add_argument("--text", help="a single plot summary")
    ap.add_argument("--file", help="a text file with one plot per line")
    ap.add_argument("--model", default="models/best_model.joblib")
    args = ap.parse_args()

    if not args.text and not args.file:
        ap.error("provide either --text or --file")

    bundle = joblib.load(args.model)
    model = bundle["pipeline"]
    print(f"[info] using model: {bundle.get('model_name', 'unknown')}")
    print(f"[info] known genres: {', '.join(bundle.get('classes', []))}\n")

    if args.text:
        print(f"Plot  : {args.text[:120]}{'...' if len(args.text) > 120 else ''}")
        print(f"Genre : {predict(model, args.text)}")

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                print(f"{i:>3}. [{predict(model, line)}] {line[:90]}")


if __name__ == "__main__":
    sys.exit(main())
