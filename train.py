#!/usr/bin/env python3
"""
CODSOFT Machine Learning Internship
Task 1 -- Movie Genre Classification
=====================================

Trains and evaluates several classic ML classifiers that predict a movie's
genre from its plot summary / description text.

Models compared (all on TF-IDF features):
    * Multinomial Naive Bayes
    * Logistic Regression
    * Linear Support Vector Machine (LinearSVC)

Usage
-----
    python train.py --data sample_data.csv
    python train.py --data /path/to/train_data.txt      # Kaggle IMDb format
    python train.py --data movies.csv --text-col Plot --label-col Genre

Outputs (into --outdir, default ./models)
-----------------------------------------
    best_model.joblib       best-performing trained pipeline (vectoriser + model)
    metrics.json            accuracy / macro-F1 for every model
    model_comparison.png    bar chart comparing the models
    confusion_matrix.png    confusion matrix of the best model
"""

from __future__ import annotations

import argparse
import json
import os
import re

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe: never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data loading & cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Lower-case, strip punctuation/digits and collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str, text_col: str | None = None,
                 label_col: str | None = None) -> tuple[pd.Series, pd.Series]:
    """Load a dataset and return (texts, labels).

    Supports:
      * the Kaggle IMDb file ``train_data.txt`` with ``ID ::: TITLE ::: GENRE ::: DESCRIPTION``
      * any CSV/TSV that has a text column and a genre column
    """
    if path.endswith(".txt") and ":::" in open(path, encoding="utf-8",
                                               errors="ignore").read(2000):
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split(" ::: ")
                if len(parts) == 4:
                    rows.append({"title": parts[1].strip(),
                                 "genre": parts[2].strip(),
                                 "description": parts[3].strip()})
        df = pd.DataFrame(rows)
        text_col, label_col = "description", "genre"
    else:
        sep = "\t" if path.endswith((".tsv", ".tab")) else ","
        df = pd.read_csv(path, sep=sep)

    df = df.dropna().reset_index(drop=True)
    cols = list(df.columns)

    if text_col is None or label_col is None:
        # Heuristic auto-detection: label = low-cardinality column,
        # text = the column with the longest average string length.
        # Works across pandas versions (object dtype in <3.0, 'str' in 3.x).
        str_cols = [c for c in cols
                    if pd.api.types.is_string_dtype(df[c])
                    and not pd.api.types.is_numeric_dtype(df[c])]
        label_col = label_col or min(
            str_cols, key=lambda c: df[c].nunique())
        remaining = [c for c in str_cols if c != label_col]
        text_col = text_col or max(
            remaining, key=lambda c: df[c].astype(str).str.len().mean())

    print(f"[data] text column = '{text_col}', label column = '{label_col}'")
    texts = df[text_col].map(clean_text)
    labels = df[label_col].astype(str).str.strip()
    return texts, labels


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(max_features: int) -> dict[str, Pipeline]:
    """Return a dict of {name: sklearn Pipeline} sharing the same TF-IDF step."""
    def tfidf():
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )

    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", tfidf()),
            ("clf", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=300, C=5,
                                       random_state=RANDOM_STATE)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LinearSVC(random_state=RANDOM_STATE)),
        ]),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_model_comparison(results: dict, outpath: str) -> None:
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["macro_f1"] for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4C72B0")
    ax.bar(x + w / 2, f1s, w, label="Macro F1", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Movie Genre Classification - model comparison")
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8),
                                    max(5, len(labels) * 0.7)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    ax.set_title("Confusion matrix - best model")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_metrics(outdir: str) -> dict:
    path = os.path.join(outdir, "metrics.json")
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        return data.get("results", data)  # tolerate both file shapes
    return {}


def _eval_split(args):
    """Return (X_test, y_test) for evaluation."""
    if args.test_data:
        return load_dataset(args.test_data)
    texts, labels = load_dataset(args.data, args.text_col, args.label_col)
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=args.test_size,
        random_state=RANDOM_STATE, stratify=labels)
    return X_test, y_test


def finalize(args) -> None:
    """Pick the best saved model, write best_model.joblib, metrics and plots."""
    results = _load_metrics(args.outdir)
    if not results:
        raise SystemExit("no metrics.json found - run training first")

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    bundle = joblib.load(os.path.join(args.outdir,
                                      f"model_{_slug(best_name)}.joblib"))
    best_model = bundle["pipeline"]
    print(f"[best] {best_name} (macro_f1={results[best_name]['macro_f1']})")

    X_test, y_test = _eval_split(args)
    preds = best_model.predict(X_test)
    report = classification_report(y_test, preds, zero_division=0)
    print("\n" + report)

    joblib.dump({"pipeline": best_model, "model_name": best_name,
                 "classes": list(best_model.classes_)},
                os.path.join(args.outdir, "best_model.joblib"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump({"results": results, "best_model": best_name,
                   "classification_report": report}, fh, indent=2)

    plot_model_comparison(results, os.path.join(args.outdir,
                                                "model_comparison.png"))
    labels_sorted = sorted(best_model.classes_)
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    plot_confusion_matrix(cm, labels_sorted,
                          os.path.join(args.outdir, "confusion_matrix.png"))
    print(f"\n[done] best model + plots written to '{args.outdir}/'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Movie genre classifier trainer")
    ap.add_argument("--data", default="sample_data.csv",
                    help="path to the dataset (CSV/TSV or Kaggle IMDb txt)")
    ap.add_argument("--test-data", default=None,
                    help="optional labelled test file, e.g. the Kaggle "
                         "'test_data_solution.txt' (evaluates on the official split)")
    ap.add_argument("--text-col", default=None, help="name of the text column")
    ap.add_argument("--label-col", default=None, help="name of the genre column")
    ap.add_argument("--outdir", default="models", help="output directory")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--only", default=None,
                    help="train a single model by name (useful on big datasets)")
    ap.add_argument("--finalize", action="store_true",
                    help="assemble best_model.joblib + plots from saved metrics")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.finalize:
        finalize(args)
        return

    texts, labels = load_dataset(args.data, args.text_col, args.label_col)
    print(f"[data] {len(texts)} rows, {labels.nunique()} genres")

    if args.test_data:
        # Evaluate on the official Kaggle test split (train on everything).
        X_train, y_train = texts, labels
        X_test, y_test = load_dataset(args.test_data)
        print(f"[eval] official test set: {len(X_test)} rows")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=args.test_size,
            random_state=RANDOM_STATE, stratify=labels)

    models = build_models(args.max_features)
    if args.only:
        models = {k: v for k, v in models.items() if k == args.only}
        if not models:
            ap.error(f"unknown model '{args.only}'")

    results = _load_metrics(args.outdir)
    for name, pipe in models.items():
        print(f"[train] {name} ...")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        joblib.dump({"pipeline": pipe, "model_name": name,
                     "classes": list(pipe.classes_)},
                    os.path.join(args.outdir, f"model_{_slug(name)}.joblib"))
        print(f"        accuracy={acc:.4f}  macro_f1={f1:.4f}")

    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[done] trained {len(models)} model(s); metrics -> metrics.json")


if __name__ == "__main__":
    main()
