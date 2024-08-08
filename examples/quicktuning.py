from qtt import QuickTuner, QuickOptimizer
from qtt.finetune.cv.classification import finetune_script, extract_task_info_metafeat

# load a pretrained optimizer
optimizer = QuickOptimizer.load("path/to/optimizer")

# setup the optimizer
n = 100  # number of configs to sample
# we need to extract some information about the dataset
task_info, metafeat = extract_task_info_metafeat("path/to/dataset")
optimizer.setup(n, metafeat)

# setup and run the tuner
qt = QuickTuner(optimizer, finetune_script)  # pass the optimizer and objective function
qt.run(task_info=task_info, time_budget=3600)  # run for 1 hour
