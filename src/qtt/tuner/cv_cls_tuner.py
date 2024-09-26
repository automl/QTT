from ..finetune.cv.classification import extract_task_info_metafeat
from ..finetune.cv.classification.finetune_wrapper import finetune_script
from ..optimizers.quick import QuickOptimizer
from ..pretrained import load_pretrained_optimizer
from .quicktuner import QuickTuner

import numpy as np


class QuickCVCLSTuner(QuickTuner):
    """QuickTuner for image classification tasks.

    Args:
        data_path (str): Path to the dataset.
        path (str, optional): Path to save the optimizer. Defaults to None.
        verbosity (int, optional): Verbosity level. Defaults to 2.
    """

    def __init__(
        self,
        data_path: str,
        n: int = 512,
        path: str | None = None,
        verbosity: int = 2,
    ):
        quick_opt: QuickOptimizer = load_pretrained_optimizer("mtlbm/full")

        task_info, metafeat = extract_task_info_metafeat(data_path)
        quick_opt.setup(n, metafeat=metafeat)

        self.task_info = task_info

        super().__init__(quick_opt, finetune_script, path=path, verbosity=verbosity)

    def run(self, fevals: int | None = None, time_budget: float | None = None) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """

        Args:
            fevals (int, optional): Number of function evaluations to run. Defaults to None.
            time_budget (float, optional): Time budget in seconds. Defaults to None.

        Returns:
            - np.ndarray: Trajectory of the incumbent scores.
            - np.ndarray: Runtime of the incumbent evaluations.
            - np.ndarray: History of all evaluations.
        """
        return super().run(
            task_info=self.task_info, fevals=fevals, time_budget=time_budget
        )
