from qtt.factory import get_optimizer
from qtt.tuners import QuickTuner
from qtt.finetune.cv.classification import finetune_script

optimizer = get_optimizer("mtlbm/micro")
qt = QuickTuner(optimizer, finetune_script)
task_info = {"data_path": "path/to/data"}
qt.run(task_info=task_info, time_budget=3600)
