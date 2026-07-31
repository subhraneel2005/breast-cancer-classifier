# Explainer: Breast Cancer Classification with Logistic Regression

This guide explains **every concept** used in
[`breast_cancer_metrics.ipynb`](./breast_cancer_metrics.ipynb).

The **main focus** of this guide — like the notebook — is **evaluation
metrics**: the confusion matrix, accuracy, precision, recall, F1 score, and
the classification report. For each metric we show exactly how it is computed
**in the notebook**, line by line, using the **real numbers from the model's
confusion matrix**, and what the result tells us about the model.

Part 1 covers the background you need before the metrics make sense. Part 2 —
the heart of the document — works through every metric in depth. Each section
includes an **intuitive analogy** to help the idea stick.

---

## Table of Contents

### Part 1 — Background

1. [The Dataset](#1-the-dataset)
2. [Features (X) and Labels (y)](#2-features-x-and-labels-y)
3. [The Bunch Object](#3-the-bunch-object)
4. [Pandas DataFrame](#4-pandas-dataframe)
5. [Train/Test Split](#5-train-test-split)
6. [Logistic Regression](#6-logistic-regression)
7. [fit() vs predict()](#7-fit-vs-predict)

### Part 2 — Evaluation Metrics (the main event)

8. [Where all the metrics come from](#8-where-all-the-metrics-come-from)
9. [The Confusion Matrix (TN, FP, FN, TP)](#9-the-confusion-matrix-tn-fp-fn-tp)
10. [Accuracy](#10-accuracy)
11. [Precision](#11-precision)
12. [Recall](#12-recall)
13. [F1 Score](#13-f1-score)
14. [The Classification Report](#14-the-classification-report)
15. [Comparing the metrics on the notebook's results](#15-comparing-the-metrics-on-the-notebooks-results)
16. [When to use which metric](#16-when-to-use-which-metric)

### Part 3 — Putting it together

17. [Reading the notebook results end to end](#17-reading-the-notebook-results-end-to-end)
18. [Next steps](#18-next-steps)

---

# Part 1 — Background

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

> **Analogy:** A box of fruit. Each apple is described by its weight, color,
> and size (the features) and carries a label "apple" or "pear" (the target).
> A model's job is to look at the features and guess the label. Here, each
> patient is described by cell measurements and carries a label "malignant"
> or "benign".

---

## 2. Features (X) and Labels (y)

In supervised machine learning we always split our data into two parts:

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
> mathematics. `X` is conventionally a matrix (a 2D table) while `y` is a
> vector (a 1D list), so `X` gets the uppercase letter.

In the notebook we separate them like this:

```python
# Features: everything except the target column
X = df.drop("target", axis=1)

# Labels: only the target column
y = df["target"]
```

> **Analogy:** Filling in a survey about yourself. The questions you answer
> (age, height, habits) are the features `X`. The thing being studied is the
> label `y`. We want to predict `y` from `X`.

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
> features, the labels, and the names of everything. You can grab any paper
> by name.

---

## 4. Pandas DataFrame

A **pandas DataFrame** is Python's spreadsheet — a 2D table of rows and
columns where every column has a name.

Why use a DataFrame instead of the raw numpy array?

- **Named columns** — instead of asking "what is column 17?", you ask for
  `"mean radius"`.
- **Easy exploration** — methods like `.head()`, `.shape`, and
  `.value_counts()` make inspection quick.
- **Convenient columns** — you can add a new column with one line, as the
  notebook does with `df["target"] = data.target`.

In the notebook we build the DataFrame like this:

```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```

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
Setting it to a fixed number makes the split *reproducible* — you (and anyone
else) get the exact same split every time. Without it, every run would
produce a different split and different results.

> **Analogy:** Cooking shows. The chef practices the dish all week (training),
> then cooks it once on camera for the judges (testing). The judges' score only
> counts if they never saw the dish before.

---

## 6. Logistic Regression

Despite its name, **Logistic Regression is a classification** algorithm — it
predicts **categories** (like "malignant" vs "benign"), not continuous
numbers.

### How it works (in plain terms)

1. The model assigns a **weight** to every feature. A large positive weight
   means that feature pushes the prediction toward class 1; a large negative
   weight pushes it toward class 0.
2. For a patient, it computes a weighted sum of all their features.
3. It runs that sum through the **sigmoid function**, which squashes any
   number into a probability between 0 and 1:
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
algorithm to find good weights. We raise it to `10000` so the optimizer is
guaranteed to converge instead of raising a warning.

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

### `predict(X_test)` — the prediction step

The model **applies** the learned rule to new data and returns `y_pred` — its
predicted labels (0 or 1) for those samples.

```python
model.fit(X_train, y_train)    # learn the rule
y_pred = model.predict(X_test) # apply the rule
```

> **Key idea:** You must **always `fit` before `predict`**. Using an unfitted
> model to predict is like asking someone to grade a test they never studied
> for.

> **Analogy:** `fit` is studying for a driver's license test; `predict` is
> taking the actual test.

---

# Part 2 — Evaluation Metrics (the main event)

This is the part that really matters in the notebook. Everything from here on
uses the model's predictions `y_pred` and the true labels `y_test` produced
by the cells you just saw.

---

## 8. Where all the metrics come from

Every metric in this notebook is computed from **one small table** — the
confusion matrix. It contains only four numbers, and **accuracy, precision,
recall, and F1 are all just different arithmetic done on those same four
numbers.**

The notebook produces this confusion matrix on the 114 test patients:

```python
[[48  6]
 [ 2 58]]
```

One value per cell:

| | Predicted 0 | Predicted 1 |
|---|-------------|-------------|
| **Actual 0** | 48 | 6 |
| **Actual 1** | 2 | 58 |

From these four numbers we will derive, step by step:

| Metric | Notebook result |
|--------|-----------------|
| Accuracy | **92.98%** |
| Precision | **90.62%** |
| Recall | **96.67%** |
| F1 Score | **93.55%** |

Before the arithmetic, we need to give the four numbers their standard names.

---

## 9. The Confusion Matrix (TN, FP, FN, TP)

The confusion matrix is a **2×2 table** that compares the model's predictions
against reality for every test sample. Each cell counts one of four possible
outcomes:

| | Predicted 0 (malignant) | Predicted 1 (benign) |
|---|---|---|
| **Actually 0 (malignant)** | **TN** — True Negative | **FP** — False Positive |
| **Actually 1 (benign)** | **FN** — False Negative | **TP** — True Positive |

Where:

- **True** / **False** → whether the prediction matched reality.
- **Positive** / **Negative** → which class was predicted. Here the *positive*
  class is `1` (benign), and the *negative* class is `0` (malignant).

Filling in the notebook's numbers:

| Term | Value | Meaning | Everyday phrasing |
|------|-------|---------|-------------------|
| **TP** (True Positive) | 58 | Actually benign, predicted benign | Correct "yes" |
| **TN** (True Negative) | 48 | Actually malignant, predicted malignant | Correct "no" |
| **FP** (False Positive) | 6 | Actually malignant, predicted benign | False "yes" |
| **FN** (False Negative) | 2 | Actually benign, predicted malignant | Missed "yes" |

> **Note on class meaning:** The confusion matrix does not know that
> "malignant is scarier than benign" — it just counts. In this notebook the
> *positive* class happens to be `1` (benign). In a real medical pipeline you
> would carefully choose which class is positive so the metrics match your
> priorities.

> **Analogy:** A fire alarm. Each room is either actually on fire or not, and
> the alarm either goes off or not:
> - TP: fire + alarm goes off ✅
> - TN: no fire + silence ✅
> - FP: no fire + alarm goes off (false alarm) 😅
> - FN: fire + silence (disaster) 😱

---

## 10. Accuracy

**Accuracy** is the simplest metric: the fraction of all predictions that were
correct.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### How it's computed in the notebook

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

scikit-learn's `accuracy_score` compares `y_pred` against `y_test` element by
element and counts the matches. Under the hood it does exactly the formula
above. Let's verify with the confusion matrix:

- Correct predictions: **TP + TN = 58 + 48 = 106**
- All predictions: **TP + TN + FP + FN = 58 + 48 + 6 + 2 = 114**

```
Accuracy = 106 / 114 = 0.9298 → 92.98%
```

Out of every 100 test patients, the model classified roughly 93 correctly.

### When accuracy is a good metric

- Classes are **balanced** (similar number of each).
- All mistakes are **equally costly**.

### When accuracy misleads you

Consider a rare disease affecting **1%** of people. A model that always
predicts "healthy" gets **99% accuracy** — yet it detects *no* sick patients
at all. Accuracy is high, but the model is useless for screening. This is
exactly why the notebook goes on to look at more careful metrics.

> **Analogy:** A student who guesses "B" for every multiple-choice question.
> If 90% of the answers are B, their accuracy is 90% — but they learned
> nothing.

---

## 11. Precision

**Precision** asks: *of everything the model called positive, how much was
actually positive?*

```
Precision = TP / (TP + FP)
```

### How it's computed in the notebook

```python
precision = precision_score(y_test, y_pred)
print(f"Precision (sklearn): {precision * 100:.2f}%")

# Manual calculation: TP / (TP + FP)
manual_precision = cm[1][1] / (cm[0][1] + cm[1][1])
print(f"Precision (manual):  {manual_precision * 100:.3f}%")
```

Let's unpack the two calculations.

**Why `cm[1][1]`, `cm[0][1]`, `cm[1][1]`?** The confusion matrix is a 2D list
where `cm[row][column]`:

| Index | Meaning |
|-------|---------|
| `cm[0][0]` | TN = 48 |
| `cm[0][1]` | FP = 6 |
| `cm[1][0]` | FN = 2 |
| `cm[1][1]` | TP = 58 |

So `manual_precision` is `58 / (6 + 58)`:

```
Precision = TP / (TP + FP) = 58 / (58 + 6) = 58 / 64 = 0.90625 → 90.62%
```

Both lines agree: sklearn reports **90.62%** and the manual formula gives
**90.625%** — the same number, just formatted differently.

### What this result means

In this project, the positive class is `1` (benign), so precision answers:
**when the model says a tumor is benign, how often is it right?**

Of the **64** patients the model called benign, **58** actually were benign
and **6** were malignant (false positives). So 90.62% of its "benign" calls
were trustworthy, and ~9% were dangerous false alarms.

### When precision matters most

When a **false positive is expensive or harmful**, you want high precision:

- **Spam filters** — marking a real email as spam (FP) loses user trust.
- **Product recommendations** — recommending something the user hates (FP).

> **Analogy:** A screening test that flags people as "guilty". Precision is the
> fraction of flagged people who actually committed the crime. If you flag 100
> people but only 50 are guilty, precision is 50% — lots of innocent people
> were wrongly accused.

---

## 12. Recall

**Recall** (also called **sensitivity** or **true positive rate**) asks: *of
everything that was actually positive, how much did the model catch?*

```
Recall = TP / (TP + FN)
```

### How it's computed in the notebook

```python
recall = recall_score(y_test, y_pred)
print(f"Recall (sklearn): {recall * 100:.2f}%")

# Manual calculation: TP / (TP + FN)
manual_recall = cm[1][1] / (cm[1][0] + cm[1][1])
print(f"Recall (manual):  {manual_recall * 100:.3f}%")
```

Note the tiny difference from precision: the denominator uses `cm[1][0]`
(the **2 false negatives**) instead of `cm[0][1]` (the 6 false positives).

```
Recall = TP / (TP + FN) = 58 / (58 + 2) = 58 / 60 = 0.9667 → 96.67%
```

sklearn reports **96.67%** and the manual formula gives **96.667%** — again,
the same number.

### What this result means

In this project, recall answers: **of all truly benign tumors, how many did
the model find?**

Of the **60** patients who actually had benign tumors, the model caught
**58** and missed **2**. It only failed to recognize 2 benign cases, so its
recall is a very high 96.67%.

### When recall matters most

When a **false negative is expensive or harmful**, you want high recall:

- **Medical screening** — missing a real cancer (FN) can be life-threatening,
  so you want to catch every case even at the cost of false alarms.
- **Fraud detection** — letting a fraudulent transaction through (FN) costs
  real money.

> **Analogy:** A net fishing for invasive fish. Recall is the fraction of
> invasive fish that the net actually catches. If 100 invasive fish are in the
> pond and the net catches 90, recall is 90%. The 10 that escape are false
> negatives — and each one is a real problem.

---

## 13. F1 Score

**F1 score** is the **harmonic mean** of precision and recall. It produces a
single number that balances both, which is useful because there is usually a
trade-off: you can often improve one by sacrificing the other.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### How it's computed in the notebook

```python
f1 = f1_score(y_test, y_pred)
print(f"F1 (sklearn): {f1 * 100:.2f}%")

# Manual calculation: harmonic mean of precision and recall
manual_f1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 (manual):  {manual_f1 * 100:.2f}%")
```

Notice this manual calculation **reuses the variables `precision` and `recall`
computed earlier in the notebook** — the notebook builds each metric on top of
the ones before it. Plugging in our values:

```
F1 = 2 × (0.90625 × 0.96667) / (0.90625 + 0.96667)
   = 2 × 0.87604 / 1.87292
   = 0.9355 → 93.55%
```

sklearn and the manual formula agree at **93.55%**.

### Why the *harmonic* mean?

The harmonic mean is stricter than the usual arithmetic mean. If precision is
1.0 but recall is 0.0, the arithmetic mean is 0.5 — but the F1 score is **0**,
because a model that catches *none* of the positive cases is not half-good, it
is broken. F1 punishes extreme imbalance between precision and recall.

### When to use F1

Use F1 when you care about **both** false positives and false negatives and
want a single number to compare models.

> **Analogy:** Precision and recall are like a delivery service that must be
> both fast and careful. If it's incredibly fast but loses packages, or
> perfectly careful but impossibly slow, the overall service is bad. F1 punishes
> being terrible at either one.

---

## 14. The Classification Report

The notebook's final evaluation step prints a full per-class report:

```python
print(classification_report(y_test, y_pred))
```

Output:

```
              precision    recall  f1-score   support

           0       0.96      0.89      0.92        54
           1       0.91      0.97      0.94        60

    accuracy                           0.93       114
   macro avg       0.93      0.93      0.93       114
weighted avg       0.93      0.93      0.93       114
```

### How the per-class numbers are computed

The classification report runs the **exact same formulas** as the previous
sections, but treats **each class as the "positive" class in turn**. This is
the key insight that ties the whole notebook together.

**Class 1 (benign)** — the same numbers we already computed:

```
precision = TP / (TP + FP) = 58 / (58 + 6)  = 0.91
recall    = TP / (TP + FN) = 58 / (58 + 2)  = 0.97
```

**Class 0 (malignant)** — now class 0 is "positive", so its own cell
`cm[0][0] = 48` plays the role of TP:

```
precision = 48 / (48 + 2) = 0.96   (its FP are the 2 predicted malignant)
recall    = 48 / (48 + 6) = 0.89   (its FN are the 6 missed malignant)
```

| Column | Meaning |
|--------|---------|
| **precision** | When the model predicted this class, how often was it right? |
| **recall** | Of all true members of this class, how many were found? |
| **f1-score** | Harmonic mean of the two |
| **support** | Number of true samples of this class in the test set (54, 60) |

### The summary rows

- **accuracy (0.93)** — overall fraction correct: `106 / 114`.
- **macro avg (0.93)** — the *plain average* across classes; both classes
  count equally regardless of size.
- **weighted avg (0.93)** — the average *weighted by support*; a class with
  more samples has more influence.

> **Analogy:** A report card. Precision, recall, and F1 are grades in
> different subjects. `macro avg` is the report-card average where every
> subject counts equally, while `weighted avg` gives more weight to subjects
> you take more classes in.

---

## 15. Comparing the metrics on the notebook's results

Now we step back and look at the four metrics side by side:

| Metric | Formula | Notebook value | Numbers used |
|--------|---------|----------------|--------------|
| Accuracy | (TP+TN) / all | **92.98%** | (58+48)/114 = 106/114 |
| Precision | TP / (TP+FP) | **90.62%** | 58/64 |
| Recall | TP / (TP+FN) | **96.67%** | 58/60 |
| F1 | 2·P·R/(P+R) | **93.55%** | from P and R above |

Notice how **one confusion matrix produces four different stories**:

- **Accuracy (92.98%)** is the overall picture: the model is right about 93
  times out of 100.
- **Precision (90.62%)** is dragged down by the **6 false positives**: 6 of
  the 64 patients predicted benign were actually malignant.
- **Recall (96.67%)** is high because there are only **2 false negatives**:
  the model rarely misses a positive case.
- **F1 (93.55%)** sits between them, balancing the two.

The pattern is informative: recall > precision. This means the model errs more
often by raising false alarms (FP) than by missing cases (FN). For a screening
tool this is the *safer* direction of error — it would rather you get an
unnecessary follow-up than miss a tumor.

> **Takeaway:** Accuracy alone hides this. Only by computing the other metrics
> — and reading the confusion matrix — do we learn *how* the model makes
> mistakes, not just *how many*.

---

## 16. When to use which metric

Different mistakes have different costs, and no single metric captures
everything. Use this as a quick decision guide:

| Metric | Question it answers | Best for |
|--------|---------------------|----------|
| **Accuracy** | Overall, how often are we right? | Balanced classes, equal costs |
| **Precision** | Are our "yes" predictions trustworthy? | When false positives are costly |
| **Recall** | Are we catching all the real cases? | When false negatives are costly |
| **F1** | How balanced are precision and recall? | A single summary that penalizes imbalance |
| **Confusion Matrix** | Exactly where are we right and wrong? | Diagnosing specific kinds of errors |
| **Classification Report** | Per-class metrics at a glance | Understanding behavior on each class |

Real-world examples of the trade-off:

- **Medical screening (like this project):** a **False Negative** (missing
  cancer) is far worse than a False Positive (unnecessary worry), so
  **recall** is prioritized.
- **Spam filtering:** a **False Positive** (losing a real email) is worse, so
  **precision** is prioritized.
- **Credit scoring:** both matter — approving a bad loan (FP) and rejecting a
  good customer (FN) both cost money, so **F1** is often the headline number.

---

# Part 3 — Putting it together

## 17. Reading the notebook results end to end

Walking through the notebook's output in one flow:

1. **569 patients** loaded into a DataFrame.
2. Split into **455 training** and **114 test** patients.
3. `LogisticRegression` trained on the training set.
4. Predictions made on the unseen test set.
5. The confusion matrix on those 114 patients:

```
[[48  6]
 [ 2 58]]
```

6. Which gives the four metrics:

| Metric | Value | What it tells us |
|--------|-------|------------------|
| Accuracy | 92.98% | ~93 of 100 patients classified correctly |
| Precision | 90.62% | When it predicts benign, it's right 90.6% of the time |
| Recall | 96.67% | It catches 96.7% of all truly benign cases |
| F1 Score | 93.55% | A balanced summary of the two |

7. The classification report confirms the same numbers per class and shows
   that class 0 (malignant) has precision **0.96** but recall **0.89** — of
   the 54 malignant patients, it flagged 6 as benign (the FP we saw earlier).

The model is accurate overall with notably **high recall**, a good quality for
a screening tool. The lower precision means it occasionally raises false
alarms — a trade-off a medical team could tune by adjusting the decision
threshold.

## 18. Next steps

1. Scale the features with `StandardScaler` and see if performance changes.
2. Plot an **ROC curve** and compute **AUC** to see how the model behaves at
   every decision threshold.
3. Try other classifiers (`RandomForestClassifier`, `SVC`,
   `KNeighborsClassifier`) and compare their metrics side by side.
4. Use `GridSearchCV` to tune hyperparameters.
