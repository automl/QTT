import logging
import random
from typing import Optional
import numpy as np
import torch

from ..configuration import ConfigManager
from ..utils import set_logger_verbosity
from .optimizer import BaseOptimizer

logger = logging.getLogger("RandomOptimizer")


class RandomOptimizer(BaseOptimizer):
    def __init__(
        self,
        cm: ConfigManager,
        num_configs: int = 256,
        fantasize_steps: int = 1,
        total_budget: int = 50,
        tol: float = 0.0,
        n_iter_no_change: Optional[int] = None,
        verbosity: int = 2,
        seed: Optional[int] = None,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.cm = cm
        self.num_configs = num_configs
        self.fantasize_steps = fantasize_steps
        self.total_budget = total_budget

        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        if n_iter_no_change is not None:
            self.score_history = np.zeros(
                (num_configs, n_iter_no_change), dtype=np.float64
            )

        self.candidates = self.cm.sample_configuration(num_configs)

        self.finished_configs = set()
        self.evaluated_configs = set()

        self.budgets = np.zeros(num_configs, dtype=np.int64)
        self.scores = np.zeros((num_configs, total_budget), dtype=np.float64)

    def ask(self):
        indexes = set(range(self.num_configs))
        remaining = indexes - self.finished_configs
        index = random.choice(list(remaining))

        budget = self.budgets[index] + self.fantasize_steps
        budget = min(budget, self.total_budget)

        out = {
            "budget": budget,
            "config": self.candidates[index],
            "config_id": index,
        }
        return out

    def tell(self, results: dict | list[dict]):
        if isinstance(results, dict):
            results = [results]

        for result in results:
            score = result["score"]
            budget = result["budget"]
            config_id = result["config_id"]
            status = result["status"]

            if not status:
                self.finished_configs.add(config_id)

            else:
                self.evaluated_configs.add(config_id)
                self.scores[config_id][budget] = score
                self.budgets[config_id] = budget

            if score >= 1.0 or budget >= self.total_budget:
                self.finished_configs.add(config_id)

            if self.inc_score < score:
                self.inc_config = config_id
                self.inc_score = score

            if self.n_iter_no_change is not None:
                n_last_iter = self.score_history[config_id]
                if not np.any(score + self.tol > n_last_iter):
                    self.finished_configs.add(config_id)
                self.score_history[config_id] = np.roll(
                    self.score_history[config_id], 1
                )
                self.score_history[config_id][0] = score

    def finished(self):
        return len(self.finished_configs) == self.num_configs
