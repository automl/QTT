from qtt.factory import get_opt
from qtt.tuners import QuickTuner
from qtt.finetune.cv.image_classification import finetune_script

opt, cm = get_opt("mtlbm/micro")
qt = QuickTuner(opt, cm, finetune_script)
qt.fit(data_path="path/to/dataset", time_limit=3600)
