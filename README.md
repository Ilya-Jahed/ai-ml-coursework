# Decision Tree from Scratch — EuroRail Passenger Satisfaction

A from-scratch (NumPy/Pandas only, no scikit-learn model) implementation of a Decision Tree classifier, trained and evaluated on a ~130k-row European rail passenger satisfaction survey. Built as a workshop assignment for the *Artificial Intelligence & Intelligent Systems* course at Iran University of Science and Technology (IUST).

## Overview

`DT_Library.py` implements a full CART-style decision tree from the ground up:

- **Splitting criteria:** Information Gain (entropy) and Gini impurity, selectable via a `mode` parameter
- **Mixed feature support:** categorical splits (multi-way, one branch per value) and numeric splits (binary, via candidate-midpoint thresholds)
- **Stopping rules:** max depth, minimum samples per split, and a pruning-threshold on gain/gini improvement
- **Post-pruning:** validation-driven reduced-error pruning that collapses subtrees when doing so doesn't hurt validation accuracy
- **Hyperparameter search:** grid search over depth, split criterion, min samples, and pruning threshold, with per-combination checkpointing so runs can resume
- **Visualization:** Graphviz export of trained trees to PDF, with optional depth-limiting and node-collapsing for readability

The included notebook (`DT_NoteBook (1).ipynb`) walks through the implementation part by part — Node/Tree classes, entropy & information gain, Gini split, tree construction, prediction, hyperparameter tuning, data cleaning, training/testing, post-pruning, and visualization.

## Project structure

```
.
├── DT_Library.py                  # Node & DecisionTree classes (core implementation)
├── backup.py                      # Earlier/backup version of the library
├── DT_NoteBook (1).ipynb          # Walkthrough notebook: build, train, tune, prune, visualize
├── DT_Document.pdf                # Original assignment brief (Farsi)
├── WorkReport.pdf                 # Write-up / analysis report
├── EuroRail_Survey.csv            # Raw dataset (~130k rows)
├── EuroRail_Survey_cleaned.csv    # Cleaned dataset used for training
├── checkpoints/                   # Saved models per hyperparameter combo (joblib .pkl)
└── *.pdf                          # Exported tree visualizations (full / pruned / depth-limited)
```

## Dataset

`EuroRail_Survey.csv` is a rail passenger satisfaction survey with columns such as `Passenger Type`, `Age`, `Type of Trip`, `Ticket Class`, `Trip Distance`, various service ratings (seat comfort, wifi, entertainment, staff service, cleanliness, etc.), delay times, and the target `satisfaction` (satisfied / dissatisfied). `EuroRail_Survey_cleaned.csv` is the preprocessed version (missing values handled, duplicates removed) used for model training.

## Installation

```bash
pip install pandas numpy scikit-learn joblib graphviz
```

- `scikit-learn` is only used for `train_test_split` / `accuracy_score` utilities in the notebook — the tree itself has no sklearn dependency.
- `graphviz` (the Python package) also requires the [Graphviz system binary](https://graphviz.org/download/) to render PDFs.

## Usage

```python
from DT_Library import DecisionTree
import pandas as pd

df = pd.read_csv("EuroRail_Survey_cleaned.csv")
X, y = df.drop(columns=["satisfaction"]), df["satisfaction"]

tree = DecisionTree(mode="gain", max_Depth=12, min_Samples=50, pruning_threshold=0.01)
tree.fit(X, y)

predictions = tree.predict(X)
```

**Hyperparameters:**

| Parameter | Description |
|---|---|
| `mode` | `"gain"` (information gain / entropy) or `"gini"` (Gini split) |
| `max_Depth` | Maximum tree depth |
| `min_Samples` | Minimum samples required to split a node |
| `pruning_threshold` | Minimum gain (or maximum gini) required to keep splitting a node |

## Results

Grid search over `max_depth ∈ {10, 12, 14}`, `mode ∈ {gain, gini}`, `min_Samples ∈ {50, 60, 70}`, and `pruning_threshold ∈ {0.005, 0.01, 0.02}`:

- **Best combo:** `max_depth=12, mode=gain, min_Samples=50, pruning_threshold=0.01`
- **Best validation accuracy:** 91.58%
- **Test accuracy of the best model:** 91.18%

Training the tree directly on the cleaned data (full depth):

- **Train accuracy:** 92.74%
- **Test accuracy:** 91.61%

After post-pruning (reduced-error pruning against a validation set):

- **Train accuracy:** 92.57%
- **Test accuracy:** 91.67%

Information-gain splits consistently outperformed Gini in this grid search (Gini runs plateaued around 74–75% validation accuracy at these settings).

## Visualization

Trained trees are exported to PDF via Graphviz (`euro_rail_tree_full.pdf`, `euro_rail_tree_depth4.pdf`, `euro_rail_tree_collapsed.pdf`, and the post-pruned equivalents), with options to cap the displayed depth or collapse subtrees below a size threshold for readability.

## Notes

- `checkpoints/` contains one saved model (`joblib`) per grid-search combination and is fairly large; consider excluding it from version control (`.gitignore`) if you fork this for further work.
- `backup.py` is an earlier draft of `DT_Library.py`, kept for reference.

## Acknowledgments

Built for the Decision Tree workshop, Artificial Intelligence & Intelligent Systems course, IUST — Fall 1404 (2025), under Dr. Arash Abdi.
