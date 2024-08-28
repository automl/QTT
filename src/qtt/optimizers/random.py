import logging
import random

import numpy as np
from ConfigSpace import ConfigurationSpace

from ..utils import fix_random_seeds, set_logger_verbosity
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class RandomOptimizer(Optimizer):
    """Random search optimizer.

    Args:
        cs (ConfigurationSpace): Configuration space object.
        max_fidelity (int): Maximum fidelity level.
        n (int): Number of configurations to sample.
        patience (int, optional): Determines if early stopping should be applied for a
            single configuration. If the score does not improve for `patience` steps,
            the configuration is stopped. Defaults to None.
        tol (float, optional): Tolerance for early stopping. Training stops if the score
            does not improve by at least `tol` for `patience` iterations (if set). Values
            must be in the range `[0.0, inf)`. Defaults to 0.0.
        score_thresh (float, optional): Score threshold for early stopping. Defaults to 0.0.
        path (str, optional): Path to save the optimizer. Defaults to None.
        seed (int, optional): Random seed. Defaults to None.
        verbosity (int, optional): Verbosity level. Defaults to 2.
    """
    def __init__(
        self,
        cs: ConfigurationSpace,
        max_fidelity: int,
        n: int,
        *,
        patience: int | None = None,
        tol: float = 0.0,
        score_thresh: float = 0.0,
        #
        path: str | None = None,
        seed: int | None = None,
        verbosity: int = 2,
    ):
        super().__init__(path=path)
        set_logger_verbosity(verbosity, logger)
        self.verbosity = verbosity

        if seed is not None:
            fix_random_seeds(seed)
        self.seed = seed

        self.cs = cs
        self.max_fidelity = max_fidelity
        self.candidates = cs.sample_configuration(n)
        self.N = n
        self.patience = patience
        self.tol = tol
        self.scr_thr = score_thresh

        self.reset()

    def reset(self):
        # trackers
        self.iteration = 0
        self.ask_count = 0
        self.tell_count = 0
        self.init_count = 0
        self.eval_count = 0
        self.evaled = set()
        self.stoped = set()
        self.failed = set()
        self.history = []

        self.fidelities: np.ndarray = np.zeros(self.N, dtype=int)
        self.curves: np.ndarray = np.zeros((self.N, self.max_fidelity), dtype=float)
        self.costs: np.ndarray = np.zeros(self.N, dtype=float)

        if self.patience is not None:
            self._score_history = np.zeros((self.N, self.patience), dtype=float)

    def ask(self):
        left = set(range(self.N)) - self.failed - self.stoped
        index = random.choice(list(left))

        fidelity = self.fidelities[index] + 1

        return {
            "config_id": index,
            "config": self.candidates[index],
            "fidelity": fidelity,
        }

    def tell(self, reports: dict | list):
        if isinstance(reports, dict):
            reports = [reports]
        for report in reports:
            self._tell(report)

    def _tell(self, report: dict):
        self.tell_count += 1

        index = report["config_id"]
        fidelity = report["fidelity"]
        cost = report["cost"]
        score = report["score"]
        status = report["status"]

        if not status:
            self.failed.add(index)
            return
        
        # update trackers
        self.curves[index, fidelity - 1] = score
        self.fidelities[index] = fidelity
        self.costs[index] = cost
        self.history.append(report)
        self.evaled.add(index)
        self.eval_count += 1

        if score >= 1.0 - self.scr_thr or fidelity == self.max_fidelity:
            self.stoped.add(index)

        if self.patience is not None:
            if not np.any(self._score_history[index] < (score - self.tol)):
                self.stoped.add(index)
            self._score_history[index][fidelity % self.patience] = score
