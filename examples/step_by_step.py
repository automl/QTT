from qtt import QuickOptimizer, QuickTuner
from qtt.predictors import PerfPredictor, CostPredictor
from qtt.finetune.cv.classification import finetune_script, extract_task_info_metafeat
import pandas as pd
from ConfigSpace import ConfigurationSpace

config = pd.read_csv("config.csv", index_col=0)  # pipeline configurations
meta = pd.read_csv("meta.csv", index_col=0)  # if meta-features are available
curve = pd.read_csv("curve.csv", index_col=0)  # learning curves
cost = pd.read_csv("cost.csv", index_col=0)  # runtime costs

X = pd.concat([config, meta], axis=1)
curve = curve.values  # predictors expect curves as numpy arrays
cost = cost.values  # predictors expect costs as numpy arrays

perf_predictor = PerfPredictor().fit(X, curve)
cost_predictor = CostPredictor().fit(X, cost)

# Define/Load the search space
cs = ConfigurationSpace()  # ConfigurationSpace.from_json("cs.json")

# Define the optimizer
optimizer = QuickOptimizer(
    cs=cs,
    max_fidelity=50,
    perf_predictor=perf_predictor,
    cost_predictor=cost_predictor,
)

task_info, metafeat = extract_task_info_metafeat("path/to/dataset")


optimizer.setup(
    512,
    metafeat=metafeat,
)
# Define the tuner
tuner = QuickTuner(
    optimizer=optimizer,
    f=finetune_script,
)
tuner.run(task_info=task_info, fevals=100, time_budget=3600)
