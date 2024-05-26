import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from qtt.configuration.manager import ConfigManager
from qtt.utils.log_utils import set_logger_verbosity
from qtt.utils.qt_utils import QTaskStatus, QTunerResult

logger = logging.getLogger("QuickOptimizer")


class RandomOptimizer:
    def __init__(
        self,
        config_manager: ConfigManager,
        num_configs: int,
        max_budget: int = 50,
        fantasize_steps: int = 1,
        verbosity: int = 2,
        seed: Optional[int] = None,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.cm = config_manager

        self.sample_configs = self.cm.sample_configuration(num_configs)

        self.num_configs = num_configs
        self.max_budget = max_budget
        self.fantasize_steps = fantasize_steps

        self.incumbent = -1
        self.incumbent_score = float("-inf")
        self.info_dict = dict()
        self.finished_configs = set()
        self.results: dict[int, List[int]] = dict()
        self.scores: dict[int, List[float]] = dict()
        self.costs: dict[int, List[float]] = dict()

    def set_metafeatures(self, *args, **kwargs):
        pass

    def suggest(self) -> Tuple[int, int]:
        """
        Suggest the next hyperparameter configuration to evaluate.

        Returns
        -------
        next_config_index: int
            The index of the best hyperparameter configuration.
        budget: int
            The budget of the hyperparameter configuration.
        """
        # check if we still have random configurations to evaluate
        if len(self.finished_configs) == len(self.sample_configs):
            return -1, -1

        # get the index of the next configuration to evaluate
        while True:
            index = random.choice(range(self.num_configs))
            if index not in self.finished_configs:
                break
        
        # get the budget of the next configuration to evaluate    
        if index in self.results:
            budget = max(self.results[index]) + self.fantasize_steps
        else:
            budget = self.fantasize_steps

        return index, budget

    def observe(
        self,
        index: int,
        budget: int,
        result: QTunerResult,
    ):
        """
        Observe the learning curve of a hyperparameter configuration.

        Args
        ----
        index: int
            The index of the hyperparameter configuration.
        budget: int
            The budget of the hyperparameter configuration.
        result:
            The performance of the hyperparameter configuration.

        Returns
        -------
        overhead_time: float
            The overhead time of the iteration.
        """
        # if y is an undefined value, append 0 as the overhead since we finish here.
        score = result.score
        if result.status == QTaskStatus.ERROR:
            self.finished_configs.add(index)
            return 0

        # if score >= (100 - threshold)
        # maybe accept config as finished before reaching max performance and do not evaluate further
        if score >= 100 or budget >= self.max_budget:
            self.finished_configs.add(index)

        if index in self.results:
            self.results[index].append(budget)
            self.scores[index].append(score)
        else:
            self.results[index] = [budget]
            self.scores[index] = [score]

        if self.incumbent_score < score:
            self.incumbent = index
            self.incumbent_score = score
            self.no_improvement_patience = 0
        else:
            self.no_improvement_patience += 1

        return 1
