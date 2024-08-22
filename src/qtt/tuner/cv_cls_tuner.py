from pathlib import Path

from ..finetune.cv.classification import extract_task_info_metafeat
from ..finetune.cv.classification.finetune_wrapper import finetune_script
from ..optimizers.quick import QuickOptimizer
from .quicktuner import QuickTuner


class ImageClassificationTuner(QuickTuner):
    def __init__(
        self,
        data_path: str,
        path: str | None = None,
        verbosity: int = 2,
    ):
        o_path = Path(__file__).parent.parent / "pretrained" / "mtlbm" / "full"
        o = QuickOptimizer.load(o_path)

        task_info, metafeat = extract_task_info_metafeat(data_path)
        o.setup(512, metafeat=metafeat)

        self.task_info = task_info

        super().__init__(o, finetune_script, path=path, verbosity=verbosity)

    def run(self, fevals: int | None = None, time_budget: float | None = None):
        return super().run(
            task_info=self.task_info, fevals=fevals, time_budget=time_budget
        )
