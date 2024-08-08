import logging
import pickle
import random
from pathlib import Path

import numpy as np
from ConfigSpace import ConfigurationSpace

from qtt.utils import fix_random_seeds, set_logger_verbosity

from .optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class RandomOptimizer(BaseOptimizer):
    def __init__(
        self,
        cs: ConfigurationSpace,
        n: int,
        n_iter_no_change: int | None = None,
        tol: float = 0.0,
        score_thresh: float = 0.0,
        seed: int | None = None,
        verbosity: int = 2,
    ):
        set_logger_verbosity(verbosity, logger)
        self.verbosity = verbosity

        if seed is not None:
            fix_random_seeds(seed)
        self.seed = seed

        self.cs = cs
        self.max_fidelity = int(cs["max_fidelity"].default_value)
        self.candidates = cs.sample_configuration(n)
        self.N = n

        self.fidelities: np.ndarray = np.zeros(n, dtype=int)
        self.curves: np.ndarray = np.zeros((n, self.max_fidelity), dtype=float)
        self.costs: np.ndarray = np.zeros(n, dtype=float)

        if n_iter_no_change is not None:
            self._score_history = np.zeros((n, n_iter_no_change), dtype=float)

        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.scr_thr = score_thresh

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

    def ask(self):
        left = set(range(self.N)) - self.evaled - self.failed - self.stoped
        index = random.choice(list(left))

        fidelity = self.fidelities[index] + 1

        return {
            "config_id": index,
            "config": self.candidates[index],
            "fidelity": fidelity,
        }

    def tell(self, result: dict | list):
        if isinstance(result, dict):
            result = [result]
        for res in result:
            self._tell(res)

    def _tell(self, result: dict):
        self.tell_count += 1

        index = result["config_id"]
        fidelity = result["fidelity"]
        cost = result["cost"]
        score = result["score"]
        status = result["status"]

        if not status:
            self.failed.add(index)
            return

        if score >= 1.0 - self.scr_thr or fidelity == self.max_fidelity:
            self.stoped.add(index)

        # update trackers
        self.curves[index, fidelity - 1] = score
        self.fidelities[index] = fidelity
        self.costs[index] = cost
        self.history.append(result)
        self.evaled.add(index)
        self.eval_count += 1

        if self.n_iter_no_change is not None:
            if not np.any(self._score_history[index] < (score - self.tol)):
                self.stoped.add(index)
            self._score_history[index][fidelity % self.n_iter_no_change] = score

    def save(self, path: str | Path = ""):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        ckp = dict(vars(self))

        self.cs.to_yaml(path / "cs.yaml")
        ckp.pop("cs")

        # torch.save(ckp, path / "checkpoint.pth")
        with open(path / "checkpoint.pkl", "wb") as f:
            pickle.dump(ckp, f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        assert path.exists(), f"Path {path} does not exist"

        # load configspace(s)
        cs = ConfigurationSpace.from_yaml(path / "cs.yaml")

        # create instance
        opt = cls(cs, 0)

        # load checkpoint
        with open(path / "checkpoint.pkl", "rb") as f:
            checkpoint: dict = pickle.load(f)
        # update instance attributes
        for key, value in vars(opt).items():
            if key == "cs":
                continue
            if key in checkpoint:
                setattr(opt, key, checkpoint.get(key, value))
            else:
                logger.info(f"'{key}' not in checkpoint")
        return opt
