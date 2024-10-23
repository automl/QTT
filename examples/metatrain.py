"""Metatrain


"""
from qtt.predictors import PerfPredictor, CostPredictor
import pandas as pd


"""
The `fit`-method of the predictors takes tabular data as input. If the data is stored in
a CSV file, the expected format of the CSV is shown below:

## Configurations

Hyperparammeter configurations of previous evaluations. Do not apply any preprocessing
to the data. Use native data types as much as possible.

|           | model         | opt   | lr     | sched   | batch_size |
|-----------|---------------|-------|--------|---------|------------|
| 1         | xcit_abc      | adam  | 0.001  | cosine  | 64         |
| 2         | beit_def      | sgd   | 0.0005 | step    | 128        |
| 3         | mobilevit_xyz | adamw | 0.01   | plateau | 32         |
| ... |

## Meta-Features

Meta-features are optional. Meta-features refer to features that describe or summarize
other features in a dataset. They are higher-level characteristics or properties of the
dataset that can provide insight into its structure or complexity.

|           | num-features | num-classes |
|-----------|--------------|-------------|
| 1         | 128          | 42          |
| 2         | 256          | 123         |
| 3         | 384          | 1000        |

## Learning Curves

Learning curves show the performance of a model over time or over iterations as it
learns from training data. For the vision classification task, the learning curves
are the validation accuracy on the validation set.

|           | 1     | 2          | 3        | 4         | 5       | ...     |
|-----------|-------|------------|----------|-----------|---------|---------|
| 1         | 0.11  | 0.12       | 0.13     | 0.14      | 0.15    | ...     |
| 2         | 0.21  | 0.22       | 0.23     | 0.24      | 0.25    | ...     |
| 3         | 0.31  | 0.32       | 0.33     | 0.34      | 0.35    | ...     |

## Cost

The cost of running a pipeline (per fidelity). This refers to the total runtime required
to complete the pipeline. This includes both the training and evaluation phases. We use
the total runtime as the cost measure for each pipeline execution.

|           | cost  |
|-----------|-------|
| 1         | 12.3  |
| 2         | 45.6  |
| 3         | 78.9  |

Ensure that the CSV files follow this structure for proper processing.
"""
config = pd.read_csv("config.csv", index_col=0)  # pipeline configurations
meta = pd.read_csv("meta.csv", index_col=0)  # if meta-features are available
curve = pd.read_csv("curve.csv", index_col=0)  # learning curves
cost = pd.read_csv("cost.csv", index_col=0)  # runtime costs

X = pd.concat([config, meta], axis=1)
curve = curve.values  # predictors expect curves as numpy arrays
cost = cost.values  # predictors expect costs as numpy arrays

perf_predictor = PerfPredictor().fit(X, curve)
cost_predictor = CostPredictor().fit(X, cost)
