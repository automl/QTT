from qtt.utils.setup import get_opt_from_pretrained
from qtt.tuners import QuickTuner
from qtt.finetune.finetune_wrapper import eval_finetune_conf

opt = get_opt_from_pretrained("mtlbm/mini")
qt = QuickTuner(opt, eval_finetune_conf)
qt.fit(data_path="path/to/dataset", time_limit=3600)
