# CODSOFT_TASK1 — Movie Genre Classification

A machine-learning project that predicts a movie's **genre** from its plot
summary / description text. Built for the **CodSoft Machine Learning
Internship (Task 1)**.

## Overview

The pipeline converts each plot summary into **TF-IDF** features (unigrams +
bigrams, English stop-words removed) and compares three classic classifiers:

| Model | Notes |
|---|---|
| Multinomial Naive Bayes | fast, strong text baseline |
| Logistic Regression | linear, calibrated probabilities |
| Linear SVM | often the best for high-dimensional text |

The best model (by macro-F1) is saved and can be reused to classify new plots.

## Project structure

```
CODSOFT_TASK1/
├── train.py           # train + evaluate + save the best model
├── predict.py         # predict the genre of new plot summaries
├── requirements.txt
├── sample_data.csv    # small demo dataset (runs out of the box)
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
# with the bundled sample data
python train.py --data sample_data.csv

# with the real Kaggle IMDb dataset (Genre Classification Dataset IMDb)
python train.py --data /path/to/train_data.txt

# with your own CSV (columns auto-detected, or specify them)
python train.py --data movies.csv --text-col Plot --label-col Genre
```

Artefacts are written to `models/`:

* `best_model.joblib` — the trained pipeline (vectoriser + classifier)
* `metrics.json` — accuracy and macro-F1 per model
* `model_comparison.png` — bar chart comparing the models
* `confusion_matrix.png` — confusion matrix of the best model

## Predict

```bash
python predict.py --text "A young wizard discovers he has magical powers and must fight an evil sorcerer"
python predict.py --file plots.txt
```

## Results

Evaluated on the official Kaggle test split (54,200 movies, 27 genres),
training on the 54,214-row `train_data.txt`:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Multinomial Naive Bayes | 0.512 | 0.119 |
| Linear SVM | 0.581 | 0.351 |
| **Logistic Regression** (best) | **0.595** | **0.357** |

Genre is a genuinely hard, noisy target (plots overlap heavily across genres),
so ~60% accuracy with a linear TF-IDF model is a solid baseline. The class
distribution is very imbalanced, which is why macro-F1 is much lower than
accuracy. Natural next steps: `LinearSVC`/`SGDClassifier` with class weights,
`GridSearchCV` over `C`, or a transformer (DistilBERT) for a large gain.

## Large datasets (the official IMDb files)

Training all three models on 54k rows can be slow, so `train.py` can run one
model at a time and then assemble the results:

```bash
D="/path/to/Genre Classification Dataset"
python train.py --data "$D/train_data.txt" --test-data "$D/test_data_solution.txt" --only "Multinomial Naive Bayes" --outdir models
python train.py --data "$D/train_data.txt" --test-data "$D/test_data_solution.txt" --only "Linear SVM"              --outdir models
python train.py --data "$D/train_data.txt" --test-data "$D/test_data_solution.txt" --only "Logistic Regression"    --outdir models
python train.py --data "$D/train_data.txt" --test-data "$D/test_data_solution.txt" --outdir models --finalize
```

`--finalize` picks the best model, writes `best_model.joblib` and the two plots.

## Dataset

* **Kaggle:** "Genre Classification Dataset IMDb" — file `train_data.txt`
  with the format `ID ::: TITLE ::: GENRE ::: DESCRIPTION`.
* Any CSV with a plot/description column and a genre column also works.

## Notes

* Text is lower-cased and stripped of punctuation/digits before vectorising.
* `train_test_split` is stratified so every genre appears in train and test.
* Everything runs offline — no dataset downloads inside the code.
