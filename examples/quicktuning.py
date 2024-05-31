from qtt.utils.setup import get_opt_from_pretrained
from qtt.tuners import QuickTuner
from qtt.finetune.cv.image_classification import finetune_function

opt = get_opt_from_pretrained("mtlbm/mini")
qt = QuickTuner(opt, finetune_function)
qt.fit(data_path="path/to/dataset", time_limit=3600)
