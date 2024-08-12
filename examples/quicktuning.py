from qtt import QuickTuner, get_pretrained_optimizer
from qtt.finetune.cv.classification import finetune_script, extract_task_info_metafeat

# specify the path to the dataset here
# it will extract the meta-features and create two dictionaries for the optimizer and tuner/finetune_script
task_info, metafeat = extract_task_info_metafeat("path/to/dataset")

# load a pretrained optimizer
optimizer = get_pretrained_optimizer("mtlbm/micro")
# setup the optimizer
n = 100  # number of configs to sample
optimizer.setup(n, metafeat)

# setup and run the tuner
qt = QuickTuner(optimizer, finetune_script)  # pass the optimizer and objective function
qt.run(task_info=task_info, time_budget=3600)  # run for 1 hour
