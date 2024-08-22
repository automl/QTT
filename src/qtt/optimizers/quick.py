import logging
import pickle
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from scipy.stats import norm

from ..utils import fix_random_seeds, set_logger_verbosity
from .optimizer import BaseOptimizer
from ..predictors import CostPredictor, DyHPO

logger = logging.getLogger(__name__)


class QuickOptimizer(BaseOptimizer):
    """QuickOptimizer is a class that implements a cost-aware Bayesian optimization
    approach with an ask and tell interface. It uses a DyHPO predictor for performance
    and a simple MLP for cost prediction.

    Args:
        cs: ConfigurationSpace
            The configuration space to optimize over.
        perf_predictor: DyHPO, optional
            The performance predictor to use. If None, a new DyHPO predictor is created.
        cost_predictor: CostPredictor, optional
            The cost predictor to use. If None, a new CostPredictor is created.
        cost_aware: bool, optional
            Whether to use the cost predictor in the acquisition function.
        acq_fn: str, optional
            The acquisition function to use. One of ["ei", "ucb", "thompson", "exploit"].
        explore_factor: float, optional
            The exploration factor to use in the acquisition function.
        patience: int, optional
            ``patience`` is used to decide if early stopping should be applied. If the
            score does not improve for ``patience`` steps, the configuration is stopped.
        tol: float, optional
            Tolerance for the early stopping. When the score is not improving
            by at least tol for ``patience`` iterations (if set to a number),
            the training stops.
            Values must be in the range `[0.0, inf)`.
        score_thresh: float, optional
            A threshold for the score. If the score is above 1 - score_thresh, the
            configuration is stopped.
        init_random_search_steps: int, optional
            The number of random configurations to evaluate (with fidelity 1) before
            starting the optimization.
        num_init_steps: int, optional
            The number of initialization steps before the predictors are refitted.
        path: str, optional
            The path to save the optimizer state.
        seed: int, optional
            The seed to use for reproducibility.
        verbosity: int, optional
            The verbosity level to use for logging.

    """
    def __init__(
        self,
        cs: ConfigurationSpace,
        perf_predictor: DyHPO | None = None,
        cost_predictor: CostPredictor | None = None,
        *,
        cost_aware: bool = False,
        acq_fn: Literal["ei", "ucb", "thompson", "exploit"] = "ei",
        explore_factor: float = 0.0,
        patience: int | None = None,
        tol: float = 1e-4,
        score_thresh: float = 0.0,
        init_random_search_steps: int = 3,
        num_init_steps: int = 32,
        #
        path: str | Path = "",
        seed: int | None = None,
        verbosity: int = 2,
    ):
        set_logger_verbosity(verbosity, logger)
        self.verbosity = verbosity

        if seed is not None:
            fix_random_seeds(seed)
        self.seed = seed

        path = Path(path)

        # configuration space
        self.cs = cs
        self.max_fidelity = int(cs["max_fidelity"].default_value)

        # optimizer related parameters
        self.acq_fn = acq_fn
        self.explore_factor = explore_factor
        self.cost_aware = cost_aware
        self.patience = patience
        self.tol = tol
        self.scr_thr = score_thresh
        self.num_init_steps = num_init_steps

        # predictors
        self.perf_predictor = perf_predictor
        if self.perf_predictor is None:
            self.perf_predictor = DyHPO()
        if cost_predictor is not None and not cost_aware:
            logger.warning(
                "cost_predictor given but cost_aware is False. Setting cost_aware to True."
            )
            cost_aware = True
        self.cost_predictor = cost_predictor
        if self.cost_aware and self.cost_predictor is None:
            self.cost_predictor = CostPredictor()
        # trackers
        self.init_random_search_steps = init_random_search_steps
        self.iteration = 0
        self.ask_count = 0
        self.tell_count = 0
        self.init_count = 0
        self.eval_count = 0
        self.configs: list[Configuration] = []
        self.pipelines: pd.DataFrame
        self.evaled = set()
        self.stoped = set()
        self.failed = set()
        self.history = []

        self._setup_run = False

    def setup(
        self,
        n: int,
        metafeat: Mapping[str, int | float] | None = None,
    ):
        """
        Setup the optimizer with the number of configurations to evaluate and optional
        metafeatures of the dataset.

        Args:
            n: int
                The number of configurations to evaluate.
            metafeat: Mapping[str, int | float], optional
                The metafeatures of the dataset.
        """
        self.N = n
        self.fidelities: np.ndarray = np.zeros(n, dtype=int)
        self.curves: np.ndarray = np.full((n, self.max_fidelity), np.nan, dtype=float)
        self.costs: np.ndarray = np.full(n, np.nan, dtype=float)
        if self.patience is not None:
            self._score_history = np.zeros((n, self.patience), dtype=float)

        if self.seed is not None:
            self.cs.seed(self.seed)
        self.configs = self.cs.sample_configuration(n)
        cfg_dict = [dict(c) for c in self.configs]
        self.pipelines = pd.DataFrame(cfg_dict)

        self.metafeat = metafeat
        if self.metafeat is not None:
            self.metafeat = pd.DataFrame([metafeat] * n)
        self.pipelines = pd.concat([self.pipelines, self.metafeat], axis=1)

        self._setup_run = True

    def _predict(self):
        """Predict the performance and cost of the configurations."""
        pipeline, curve = self.pipelines, self.curves

        pred = self.perf_predictor.predict(pipeline, curve)  # type: ignore
        pred_mean, pred_std = pred

        cost = self.costs
        if self.cost_aware:
            pred_cost = self.cost_predictor.predict(pipeline)  # type: ignore
            pred_cost = pred_cost.squeeze()
            mask = np.isnan(cost)
            cost[mask] = pred_cost[mask]

        return pred_mean, pred_std, cost

    def _calc_acq_val(self, mean, std, y_max):
        """Calculate the acquisition value.
        
        Args:
            mean: np.ndarray
                The mean of the predictions.
            std: np.ndarray
                The standard deviation of the predictions.
            y_max: np.ndarray
                The maximum score per fidelity.

        Returns:
            The acquisition values.
        """
        fn = self.acq_fn
        xi = self.explore_factor
        match fn:
            # Expected Improvement
            case "ei":
                mask = std == 0
                std = std + mask * 1.0
                z = (mean - y_max - xi) / std
                acq_value = (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
                acq_value[mask] = 0.0
            # Upper Confidence Bound
            case "ucb":
                acq_value = mean + xi * std
            # Thompson Sampling
            case "thompson":
                acq_value = np.random.normal(mean, std)
            # Exploitation
            case "exploit":
                acq_value = mean
            case _:
                raise ValueError
        return acq_value

    def _optimize_acq_fn(self, mean, std, cost) -> list[int]:
        """Optimize the acquisition function.
        
        Args:
            mean: np.ndarray
                The mean of the predictions.
            std: np.ndarray
                The standard deviation of the predictions.
            cost: np.ndarray
                The cost of the pipeline.
        
        Returns:
            A sorted list of indices of the pipeline.
        """
        # max score per fidelity
        y_max = self.curves.max(axis=0)
        y_max[y_max == 0] = y_max.max()

        # get the ymax for the next fidelity of the pipelines
        next_fidelitys = np.minimum(self.fidelities + 1, self.max_fidelity)
        y_max = y_max[next_fidelitys - 1]

        acq_values = self._calc_acq_val(mean, std, y_max)
        if self.cost_aware:
            cost += 1  # avoid division by zero
            acq_values /= cost

        return np.argsort(acq_values).tolist()

    def _ask(self):
        pred_mean, pred_std, cost = self._predict()
        ranks = self._optimize_acq_fn(pred_mean, pred_std, cost)
        ranks = [r for r in ranks if r not in self.stoped | self.failed]
        idx = ranks[-1]
        logger.info(f"predicted score: {pred_mean[idx]:.4f}")
        return ranks[-1]

    def ask(self) -> dict:
        """Ask the optimizer for a configuration to evaluate.

        Returns:
            A dictionary with the configuration to evaluate.
        """
        if not self._setup_run:
            raise RuntimeError("Call setup() before ask()")

        self.ask_count += 1
        if len(self.evaled) < self.init_random_search_steps:
            left = set(range(self.N)) - self.evaled - self.failed - self.stoped
            index = left.pop()
            fidelity = 1
        else:
            index = self._ask()
            fidelity = self.fidelities[index] + 1

        return {
            "config_id": index,
            "config": self.configs[index],
            "fidelity": fidelity,
        }

    def tell(self, result: dict | list):
        """Tell the optimizer the result for an asked trial.

        Args:
            result: dict | list[dict]
                The result(s) for a trial.
        """
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

        if self.patience is not None:
            if not np.any(self._score_history[index] < (score - self.tol)):
                self.stoped.add(index)
            self._score_history[index][fidelity % self.patience] = score

    def ante(self):
        """Pre processing before each iteration.
        
        Refit the predictors after the initialization steps.
        """
        if self.eval_count >= self.num_init_steps:
            self.refit()

    def post(self):
        """Post processing after each iteration.
        
        Used by tuners.
        """
        pass

    def refit(self):
        """Refit the predictors with observed data."""
        pipeline, curve = self.pipelines, self.curves
        self.perf_predictor.refit(pipeline, curve)  # type: ignore

    def fit(self, X, curve, cost):
        """
        Fit the predictors with the given training data.
        """
        self.perf_predictor.fit(X, curve)  # type: ignore
        if self.cost_predictor is not None:
            self.cost_predictor.fit(X, cost)

    def save(self, path: str | Path = ""):
        """Save the current state of the optimizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "quick.pkl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "QuickOptimizer":
        path = Path(path)
        assert path.exists(), f"Path {path} does not exist"

        with open(path / "quick.pkl", "rb") as f:
            opt = pickle.load(f)

        return opt
