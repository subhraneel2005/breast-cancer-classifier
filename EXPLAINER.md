# Explainer: Breast Cancer Classification with Logistic Regression

This guide explains **every concept** used in
[`breast_cancer_metrics.ipynb`](./breast_cancer_metrics.ipynb) in depth. It is
written for someone completely new to machine learning, so we start from the
very basics and build up step by step. Each section ends with an **intuitive
analogy** to help the idea stick.

---

## Table of Contents

1. [The Dataset](#1-the-dataset)
2. [Features (X) and Labels (y)](#2-features-x-and-labels-y)
3. [The Bunch Object](#3-the-bunch-object)
4. [Pandas DataFrame](#4-pandas-dataframe)
5. [Train/Test Split](#5-train-test-split)
6. [Logistic Regression](#6-logistic-regression)
7. [fit() vs predict()](#7-fit-vs-predict)
8. [The Confusion Matrix (TN, FP, FN, TP)](#8-the-confusion-matrix-tn-fp-fn-tp)
9. [Accuracy](#9-accuracy)
10. [Precision](#10-precision)
11. [Recall](#11-recall)
12. [F1 Score](#12-f1-score)
13. [The Classification Report](#13-the-classification-report)
14. [Why do all these metrics exist?](#14-why-do-all-these-metrics-exist)
15. [How to read the results of this project](#15-how-to-read-the-results-of-this-project)

---

## 1. The Dataset

The project uses the **Wisconsin Breast Cancer dataset**. It contains data for
**569 patients**. For each patient, a doctor took a sample of a suspicious
breast mass using a procedure called a *fine needle aspirate (FNA)*, and a
computer measured 30 different properties of the cells in that sample.

Each patient gets two pieces of information:

1. **30 measurements** of their cells (the *features*), such as the average
   radius, texture, smoothness, and symmetry of the cells.
2. **A diagnosis label** telling us whether the tumor was:
   - `0` → **malignant** (cancerous), or
   - `1` → **benign** (non-cancerous)

The dataset is a classic, real-world dataset used for teaching, and it is
bundled directly inside scikit-learn — no file download needed.

> **Analogy:** Think of a box of fruit. Each apple is described by its weight,
> color, and size (the features), and carries a label "apple" or "pear". A
> model's job is to look at the features and guess the label. Here, each
> patient is described by cell measurements (the features) and carries a label
> "malignant" or "benign".

---

## 2. Features (X) and Labels (y)

In supervised machine learning, we always split our data into two parts:

| Variable | Name | What it is |
|----------|------|------------|
| `X` | **Features** (inputs) | The information we use to make a prediction |
| `y` | **Labels / target** (outputs) | The answer we want to predict |

In this project:

- `X` is a table with **569 rows** (one per patient) and **30 columns**
  (one per measurement).
- `y` is a list of **569 values**, each either `0` (malignant) or `1`
  (benign).

The model's goal: learn a rule that turns `X` into a good guess for `y`.

> **Why lowercase `y` but uppercase `X`?** This is a convention inherited from
> mathematics and linear algebra. `X` is conventionally written as a matrix
> (a 2D table) while `y` is a vector (a 1D list), so `X` gets the uppercase
> letter.

> **Analogy:** Filling in a survey about yourself. The questions you answer
> (age, height, habits) are the features `X`. The thing being studied
> (e.g., whether you exercise daily) is the label `y`. We want to predict `y`
> from `X`.

---

## 3. The Bunch Object

When you call `load_breast_cancer()`, scikit-learn returns a **Bunch object**.

A `Bunch` is a **dictionary-like container** that lets you access its contents
with dot notation. A normal dictionary gives you values via keys:

```python
student = {"name": "Alice", "age": 21}
student["name"]   # "Alice"
```

A `Bunch` does the same, but with dots:

```python
student.name      # "Alice"
```

Our dataset's `Bunch` contains:

| Attribute | Contains |
|-----------|----------|
| `data` | The features as a 2D array (569 × 30) |
| `target` | The labels as a 1D array (569 values: 0 or 1) |
| `feature_names` | The names of the 30 features |
| `target_names` | The names of the classes: `["malignant", "benign"]` |
| `DESCR` | A full written description of the dataset |

> **Analogy:** A Bunch is like an envelope that holds several papers — the
> features, the labels, and the names of everything. You can grab any paper by
> name.

---

## 4. Pandas DataFrame

A **pandas DataFrame** is Python's spreadsheet. It is a 2D table of rows and
columns, where every column has a name.

Why use a DataFrame instead of the raw numpy array?

- **Named columns** — instead of asking "what is column 17?", you ask for
  `"mean radius"`.
- **Easy exploration** — methods like `.head()`, `.shape`, `.value_counts()`
  make inspection quick.
- **Convenient columns** — you can add a new column with one line, as the
  notebook does with `df["target"] = data.target`.

In the notebook we build the DataFrame like this:

```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```

This says: "Make a table from the features array, and name the columns using
the feature names."

| Feature name | Meaning |
|--------------|---------|
| `mean radius` | Average distance from center to points on the cell perimeter |
| `mean texture` | Variation in cell surface gray-level values |
| `mean smoothness` | Local variation in cell radius lengths |
| `worst area` | The largest (worst) area among the cells |
| … | 26 more measurements |

> **Analogy:** A DataFrame is a well-labeled filing cabinet. Each drawer has a
> label (column name), and each drawer holds one item per patient (row).

---

## 5. Train/Test Split

If a student practices on exam questions and then takes **the same questions**
as their final exam, a perfect score proves nothing. The same logic applies to
models: **a model must be evaluated on data it has never seen.**

`train_test_split` shuffles the data and cuts it into two pieces:

- **Training set (80%)** — used to teach the model the patterns.
- **Test set (20%)** — held out and only used at the very end, to measure how
  well the model generalizes to new patients.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69
)
```

In our project this gives **455 training samples** and **114 test samples**.

The `random_state` parameter is a **seed** for the random number generator.
Setting it to a fixed number means the split is *reproducible* — you (and
anyone else) get the exact same split every time. Without it, every run would
produce a different split and different results.

> **Analogy:** Cooking shows. The chef practices the dish all week (training),
> then cooks it once on camera for the judges (testing). The judges' score only
> counts if they never saw the dish before. A chef trained on the judging dish
> would score perfectly but couldn't cook anything new.

---

## 6. Logistic Regression

Despite its name, **Logistic Regression is a classification** algorithm — it
predicts **categories** (like "malignant" vs "benign"), not continuous numbers.

### How it works (in plain terms)

1. The model assigns a **weight** to every feature. A large positive weight
   means that feature pushes the prediction toward class 1; a large negative
   weight pushes it toward class 0.
2. For a patient, it computes a weighted sum of all their features.
3. It runs that sum through the **sigmoid function**, which squashes any number
   into a probability between 0 and 1:

   - close to **0** → predicts class **0** (malignant)
   - close to **1** → predicts class **1** (benign)
   - exactly **0.5** → the decision boundary (tie)

The training process (`fit`) finds the weights that best separate the two
classes in the training data.

In the notebook:

```python
model = LogisticRegression(max_iter=10000)
```

`max_iter` is the maximum number of optimization steps allowed for the
algorithm to find good weights. For this dataset with its default solver, we
raise it to `10000` so the optimizer is guaranteed to converge instead of
raising a warning.

> **Analogy:** A teacher draws a line between two groups of students: those
> who passed and those who failed, based on study hours and attendance. A new
> student's position relative to the line decides their predicted outcome.
> Logistic regression is the math that finds the *best* line.

---

## 7. fit() vs predict()

Scikit-learn models have a simple, consistent interface with two central
methods:

### `fit(X_train, y_train)` — the learning step

The model **studies** the training data and learns the weights (the "rule").

- Inputs: the features `X_train` and the true labels `y_train`.
- Output: nothing useful to us — the model simply *remembers* what it learned.

### `predict(X_test)` — the prediction step

The model **applies** the learned rule to new data.

- Input: the features `X_test` of samples it has never seen.
- Output: `y_pred` — its predicted labels (0 or 1) for those samples.

```python
model.fit(X_train, y_train)   # learn the rule
y_pred = model.predict(X_test) # apply the rule
```

> **Key idea:** You must **always `fit` before `predict`**. Using an unfitted
> model to predict is like asking someone to grade a test they never studied
> for.

> **Analogy:** `fit` is studying for a driver's license test; `predict` is
> taking the actual test. The questions (inputs) feed in, and the answers
> (outputs) come out — but only after studying.

---

## 8. The Confusion Matrix (TN, FP, FN, TP)

The confusion matrix is a **2×2 table** that compares the model's predictions
against reality for every test sample.

Each cell counts one of four possible outcomes:

|  | Predicted 0 (malignant) | Predicted 1 (benign) |
|---|---|---|
| **Actually 0 (malignant)** | **TN** — True Negative | **FP** — False Positive |
| **Actually 1 (benign)** | **FN** — False Negative | **TP** — True Positive |

Where:

- **True** / **False** → whether the prediction matched reality.
- **Positive** / **Negative** → which class was predicted. Here the *positive*
  class is `1` (benign), and the *negative* class is `0` (malignant).

| Term | Meaning | Everyday phrasing |
|------|---------|-------------------|
| **TP** (True Positive) | Actually benign, predicted benign | Correct "yes" |
| **TN** (True Negative) | Actually malignant, predicted malignant | Correct "no" |
| **FP** (False Positive) | Actually malignant, predicted benign | False "yes" |
| **FN** (False Negative) | Actually benign, predicted malignant | Missed "yes" |

The notebook's model produced this confusion matrix:

```python
[[48  6]
 [ 2 58]]
```

| Cell | Value | Meaning |
|------|-------|---------|
| TN = 48 | 48 patients actually malignant and predicted malignant | ✅ correct |
| FP = 6 | 6 patients actually malignant but predicted benign | ❌ dangerous |
| FN = 2 | 2 patients actually benign but predicted malignant | ❌ false alarm |
| TP = 58 | 58 patients actually benign and predicted benign | ✅ correct |

> **Note on class meaning:** The raw confusion matrix does not know anything
> about "malignant is scarier than benign". It just counts. In this notebook
> the *positive* class happens to be `1` (benign). In a real medical pipeline
> you would carefully choose which class you treat as positive so the metrics
> match your priorities.

> **Analogy:** A fire alarm. Each room is either actually on fire or not, and
> the alarm either goes off or not:
> - TP: fire + alarm goes off ✅
> - TN: no fire + silence ✅
> - FP: no fire + alarm goes off (false alarm) 😅
> - FN: fire + silence (disaster) 😱

---

## 9. Accuracy

**Accuracy** is the simplest metric: the fraction of all predictions that were
correct.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

For our model:

```
(58 + 48) / 114 = 106 / 114 = 0.9298 → 92.98%
```

### When accuracy is a good metric

- Classes are **balanced** (similar number of each).
- All mistakes are **equally costly**.

### When accuracy misleads you

Consider a rare disease affecting **1%** of people. A model that always
predicts "healthy" gets **99% accuracy** — yet it detects *no* sick patients
at all. Accuracy is high, but the model is useless for screening.

> **Analogy:** A student who guesses "B" for every multiple-choice question.
> If 90% of answers are B, their accuracy is 90% — but they learned nothing.

---

## 10. Precision

**Precision** asks: *of everything the model called positive, how much was
actually positive?*

```
Precision = TP / (TP + FP)
```

For our model:

```
58 / (58 + 6) = 58 / 64 = 0.90625 → 90.62%
```

In this project, precision answers: **when the model says a tumor is benign,
how often is it right?**

Low precision means **many false positives** — the model cries wolf too often.

### When precision matters most

When a **false positive is expensive or harmful**, you want high precision.
Examples:

- **Spam filters** — marking a good email as spam (FP) loses your customer's
  trust, so precision matters.
- **Product recommendations** — recommending a product the user hates (FP) is
  annoying, so precision matters.

> **Analogy:** A screening test that flags people as "guilty". Precision is the
> fraction of flagged people who actually committed the crime. If you flag 100
> people but only 50 are guilty, precision is 50% — lots of innocent people
> were wrongly accused (false positives).

---

## 11. Recall

**Recall** (also called **sensitivity** or **true positive rate**) asks: *of
everything that was actually positive, how much did the model catch?*

```
Recall = TP / (TP + FN)
```

For our model:

```
58 / (58 + 2) = 58 / 60 = 0.9667 → 96.67%
```

In this project, recall answers: **of all truly benign tumors, how many did
the model find?**

Low recall means **many false negatives** — the model is missing positive
cases.

### When recall matters most

When a **false negative is expensive or harmful**, you want high recall.
Examples:

- **Medical screening** — missing a real cancer (FN) can be life-threatening,
  so you want to catch every case, even at the cost of some false alarms.
- **Fraud detection** — letting a fraudulent transaction through (FN) costs
  real money.

> **Analogy:** A net fishing for invasive fish. Recall is the fraction of
> invasive fish that the net actually catches. If 100 invasive fish are in the
> pond and the net catches 90, recall is 90%. The 10 that escape are false
> negatives — and in conservation, each escapee is a real problem.

---

## 12. F1 Score

**F1 score** is the **harmonic mean** of precision and recall. It produces a
single number that balances both, which is useful because there is usually a
trade-off: you can often improve one by sacrificing the other.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

For our model:

```
2 × (0.90625 × 0.9667) / (0.90625 + 0.9667) = 0.9355 → 93.55%
```

The **harmonic mean** is stricter than the arithmetic mean (the usual
average). If precision is 1.0 but recall is 0.0, the arithmetic mean is 0.5 —
but the F1 score is **0**, because a model that misses *every* positive case
is not half-good, it is broken.

### When to use F1

Use F1 when you care about **both** false positives and false negatives and
want one number to compare models.

> **Analogy:** Precision and recall are like a delivery service that must be
> both fast and careful. If it's either incredibly fast but loses packages, or
> perfectly careful but impossibly slow, the overall service is bad. F1
> punishes being terrible at either one.

---

## 13. The Classification Report

`classification_report` prints one row per class, containing that class's
precision, recall, F1, and support:

```
              precision    recall  f1-score   support

           0       0.96      0.89      0.92        54
           1       0.91      0.97      0.94        60

    accuracy                           0.93       114
   macro avg       0.93      0.93      0.93       114
weighted avg       0.93      0.93      0.93       114
```

Reading each column:

- **precision** — when the model predicted this class, how often was it right?
- **recall** — of all true members of this class, how many were found?
- **f1-score** — harmonic mean of the two.
- **support** — the number of true samples of this class in the test set
  (54 malignant, 60 benign).

The bottom three rows summarize the whole table:

- **accuracy** — overall fraction correct (0.93).
- **macro avg** — the *plain average* across classes; it treats both classes
  equally regardless of size.
- **weighted avg** — the average *weighted by support*; a class with more
  samples has more influence on the average.

> **Analogy:** A report card. Precision, recall, and F1 are grades in
> different subjects. `macro avg` is the report card average where every
> subject counts equally, while `weighted avg` gives more weight to the
> subjects you take more classes in.

---

## 14. Why do all these metrics exist?

Because **different mistakes have different costs**, and one number cannot
capture everything.

| Metric | Question it answers | Best for |
|--------|---------------------|----------|
| **Accuracy** | Overall, how often are we right? | Balanced classes, equal costs |
| **Precision** | Are our "yes" predictions trustworthy? | When false positives are costly |
| **Recall** | Are we catching all the real cases? | When false negatives are costly |
| **F1** | How balanced are precision and recall? | A single summary that penalizes extreme imbalance |
| **Confusion Matrix** | Exactly where are we right and wrong? | Diagnosing the specific kinds of errors |
| **Classification Report** | Per-class metrics at a glance | Understanding behavior on each class |

In healthcare, a **False Negative** (missing cancer) is usually far worse than
a **False Positive** (unnecessary worry), so a screening model would prioritize
**recall**. In spam filtering, a **False Positive** (losing a real email) is
worse, so **precision** takes priority. There is no single "best" metric — you
choose the one that matches the real-world consequences.

---

## 15. How to read the results of this project

Our Logistic Regression model scored:

| Metric | Value |
|--------|-------|
| Accuracy | 92.98% |
| Precision | 90.62% |
| Recall | 96.67% |
| F1 Score | 93.55% |

From the confusion matrix:

- **106 of 114** test patients were classified correctly.
- **6 False Positives** — patients with malignant tumors predicted as benign.
  In a real system these would be the most worrying.
- **2 False Negatives** — patients with benign tumors predicted as malignant.
  These would cause unnecessary procedures but not miss a disease.

The model is accurate overall, with notably **high recall**, which is a good
quality for a screening tool: it rarely misses a positive case. The slightly
lower precision means it occasionally raises false alarms — a trade-off that a
medical team could tune by changing the decision threshold.

**Suggested next steps:**

1. Scale the features with `sklearn.preprocessing.StandardScaler` and see if
   performance improves.
2. Plot an **ROC curve** and compute **AUC** to see how the model behaves
   across all decision thresholds.
3. Try other classifiers (`RandomForestClassifier`, `SVC`, `KNeighborsClassifier`)
   and compare their metrics with a single table.
4. Use `GridSearchCV` to tune hyperparameters.
