# Breast Cancer Classification with Logistic Regression

A beginner-friendly, self-contained machine learning project that trains a
**Logistic Regression** model to classify breast tumors as **malignant**
(cancerous) or **benign** (non-cancerous) using the
[Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
(bundled with scikit-learn).

Everything lives in a single Jupyter notebook, written for people who are
completely new to machine learning — every block is commented, every section
is explained, and every evaluation metric is taught from first principles.

## Features

- **Step-by-step notebook** with clear Markdown headings and concise comments
- Covers the full workflow: data loading → exploration → train/test split →
  training → predictions → evaluation
- **Evaluation metrics taught from scratch**: accuracy, precision, recall,
  F1 score, confusion matrix, and classification report — each verified with a
  manual calculation from the confusion matrix
- **Minimal, clean visualizations**:
  - Class distribution bar chart
  - Confusion matrix heatmap
- **Deep-dive learning resource**: [`Explainer.md`](./Explainer.md) teaches
  every concept in the project with formulas, tables, and real-world analogies
- Reproducible results via a fixed `random_state`

## Repository Structure

```
breast-cancer-classifier/
├── breast_cancer_metrics.ipynb   # The main notebook (run this)
├── Explainer.md                  # In-depth explanations of every concept
├── README.md                     # This file
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Locked dependency versions
└── .python-version               # Python version pin (3.12)
```

## Prerequisites

- [Python 3.12](https://www.python.org/downloads/) (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) — a fast Python package and environment manager

## Local Setup (using `uv`)

Clone the repository and create the environment:

```bash
git clone <your-repo-url>
cd breast-cancer-classifier

# Create a virtual environment and install all dependencies
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates a `.venv`, and
installs everything you need:

- `jupyterlab`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`

## Running the Notebook

Start Jupyter Lab:

```bash
uv run jupyter lab
```

Open `breast_cancer_metrics.ipynb`, then run all cells
(`Run ▸ Run All Cells`) or step through them one at a time with
`Shift + Enter`.

## Sample Output

Running the notebook produces a `LogisticRegression` model with the following
evaluation on the held-out test set:

```
Accuracy: 92.98%
Precision (sklearn): 90.62%
Recall (sklearn): 96.67%
F1 (sklearn): 93.55%

Confusion matrix:
[[48  6]
 [ 2 58]]

Classification report:
              precision    recall  f1-score   support

           0       0.96      0.89      0.92        54
           1       0.91      0.97      0.94        60

    accuracy                           0.93       114
   macro avg       0.93      0.93      0.93       114
weighted avg       0.93      0.93      0.93       114
```

The notebook also renders a **class distribution bar chart** and a
**confusion matrix heatmap** as you execute it.

## Learning Outcomes

After working through this project you will be able to:

- Load and inspect a dataset with scikit-learn and `pandas`
- Explain the difference between **features (X)** and **labels (y)**
- Understand why we split data into **training** and **testing** sets
- Train and evaluate a **Logistic Regression** classifier
- Distinguish `fit()` (learning) from `predict()` (inference)
- Read a **confusion matrix** and explain TN, FP, FN, and TP
- Compute and interpret **accuracy**, **precision**, **recall**, and **F1**
- Read a **classification report** and know when to trust each metric
- Apply the same workflow to any new tabular classification problem

> New to machine learning? Read [`Explainer.md`](./Explainer.md) first for a
> gentle, analogy-driven introduction to every concept used in this project.

## License

This project is for educational purposes and uses the openly available
scikit-learn Wisconsin Breast Cancer dataset.
