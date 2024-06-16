import logging
import random
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch

from qtt.utils.log_utils import set_logger_verbosity

logger = logging.getLogger("RandomOptimizer")


class RandomOptimizer:
    configs: np.ndarray
    n_iter_no_change: int | None = None
    tol: float = 0.0

    def __init__(
        self,
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

        self.max_budget = max_budget
        self.fantasize_steps = fantasize_steps

        self.incumbent = -1
        self.incumbent_score = 0.0
        self.info_dict = dict()
        self.finished_configs = set()

        self.results: dict[int, List[int]] = defaultdict(list)
        self.scores: dict[int, List[float]] = defaultdict(list)
        self.costs: dict[int, List[float]] = defaultdict(list)

    def set_configs(self, configs):  # np.ndarray | pd.DataFrame
        self.configs = configs
        if self.n_iter_no_change is not None:
            self.score_history = {
                i: np.zeros(self.n_iter_no_change) for i in range(len(configs))
            }

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
        if len(self.finished_configs) == len(self.configs):
            return -1, -1

        # get the index of the next configuration to evaluate
        while True:
            index = random.choice(range(len(self.configs)))
            if index not in self.finished_configs:
                break

        # get the budget of the next configuration to evaluate
        if index in self.results:
            budget = max(self.results[index]) + self.fantasize_steps
        else:
            budget = self.fantasize_steps

        return index, budget

    def observe(self, results: dict | list[dict]):
        if isinstance(results, dict):
            results = [results]

        for result in results:
            score = result["score"]
            budget = result["budget"]
            config_id = result["config_id"]
            status = result["status"]
            cost = result["cost"]

            if not status:
                self.finished_configs.add(config_id)

            else:
                if config_id in self.results:
                    self.results[config_id].append(budget)
                    self.scores[config_id].append(score)
                    self.costs[config_id].append(cost)
                else:
                    self.results[config_id] = [budget]
                    self.scores[config_id] = [score]
                    self.costs[config_id] = [cost]

            if score >= 1.0 or budget >= self.max_budget:
                self.finished_configs.add(config_id)

            if self.incumbent_score < score:
                self.incumbent = config_id
                self.incumbent_score = score

            if self.n_iter_no_change is not None:
                n_last_iter = self.score_history[config_id]
                if not np.any(score + self.tol > n_last_iter):
                    self.finished_configs.add(config_id)
                self.score_history[config_id] = np.roll(
                    self.score_history[config_id], 1
                )
                self.score_history[config_id][0] = score
