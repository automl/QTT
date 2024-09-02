from qtt.predictors import PerfPredictor, CostPredictor
import pandas as pd

config = pd.read_csv("config.csv", index_col=0)  # pipeline configurations
meta = pd.read_csv("meta.csv", index_col=0)  # if meta-features are available
curve = pd.read_csv("curve.csv", index_col=0)  # learning curves
cost = pd.read_csv("cost.csv", index_col=0)  # runtime costs

X = pd.concat([config, meta], axis=1)
curve = curve.values  # predictors expect curves as numpy arrays
cost = cost.values  # predictors expect costs as numpy arrays

perf_predictor = PerfPredictor().fit(X, curve)
cost_predictor = CostPredictor().fit(X, cost)
