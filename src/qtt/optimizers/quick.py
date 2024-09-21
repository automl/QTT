import logging
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from scipy.stats import norm

from ..predictors import CostPredictor, PerfPredictor
from ..utils import fix_random_seeds, set_logger_verbosity
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class QuickOptimizer(Optimizer):
    """QuickOptimizer implements a cost-aware Bayesian optimization approach with an ask
    and tell interface. It uses a DyHPO predictor for performance and a simple MLP for
    cost.

    Args:
        cs (ConfigurationSpace): The configuration space to optimize over.
        max_fidelity (int): The maximum fidelity to optimize. Fidelity is a measure of
            a resource used by a configuration, such as the number of epochs.
        perf_predictor (PerfPredictor, optional): The performance predictor to use. If
            None, a new predictor is created.
        cost_predictor (CostPredictor, optional): The cost predictor to use. If None,
            a new CostPredictor is created if `cost_aware` is True.
        cost_aware (bool, optional): Whether to use the cost predictor. Defaults to False.
        cost_factor (float, optional): A factor to control the scaling of cost values.
            Values must be in the range `[0.0, inf)`. A cost factor smaller than 1
            compresses the cost values closer together (with 0 equalizing them), while
            values larger than 1 expand them. Defaults to 1.0.
        acq_fn (str, optional): The acquisition function to use. One of ["ei", "ucb",
            "thompson", "exploit"]. Defaults to "ei".
        explore_factor (float, optional): The exploration factor in the acquisition
            function. Defaults to 1.0.
        patience (int, optional): Determines if early stopping should be applied for a
            single configuration. If the score does not improve for `patience` steps,
            the configuration is stopped. Defaults to None.
        tol (float, optional): Tolerance for early stopping. Training stops if the score
            does not improve by at least `tol` for `patience` iterations (if set). Values
            must be in the range `[0.0, inf)`. Defaults to 0.0.
        score_thresh (float, optional): Threshold for early stopping. If the score is
            above `1 - score_thresh`, the configuration is stopped. Defaults to 0.0.
        init_random_search_steps (int, optional): Number of configurations to evaluate
            randomly at the beginning of the optimization (with fidelity 1) before using
            predictors/acquisition function. Defaults to 10.
        refit_init_steps (int, optional): Number of steps (successful evaluations) before
            refitting the predictors. Defaults to 0.
        refit (bool, optional): Whether to refit the predictors with observed data.
            Defaults to False.
        refit_interval (int, optional): Interval for refitting the predictors. Defaults
            to 1.
        path (str, optional): Path to save the optimizer state. Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to None.
        verbosity (int, optional): Verbosity level for logging. Defaults to 2.
    """

    def __init__(
        self,
        cs: ConfigurationSpace,
        max_fidelity: int,
        perf_predictor: PerfPredictor | None = None,
        cost_predictor: CostPredictor | None = None,
        *,
        cost_aware: bool = False,
        cost_factor: float = 1.0,
        acq_fn: Literal["ei", "ucb", "thompson", "exploit"] = "ei",
        explore_factor: float = 0.0,
        patience: int | None = None,
        tol: float = 1e-4,
        score_thresh: float = 0.0,
        init_random_search_steps: int = 3,
        refit_init_steps: int = 0,
        refit: bool = False,
        refit_interval: int = 1,
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

        # configuration space
        self.cs = cs
        self.max_fidelity = max_fidelity

        # optimizer related parameters
        self.acq_fn = acq_fn
        self.explore_factor = explore_factor
        self.cost_aware = cost_aware
        self.cost_factor = cost_factor
        self.patience = patience
        self.tol = tol
        self.scr_thr = score_thresh
        self.refit_init_steps = refit_init_steps
        self.refit = refit
        self.refit_interval = refit_interval

        # predictors
        self.perf_predictor = perf_predictor
        if self.perf_predictor is None:
            self.perf_predictor = PerfPredictor(path=path)
        self.cost_predictor = cost_predictor
        if self.cost_aware and self.cost_predictor is None:
            self.cost_predictor = CostPredictor(path=path)

        # trackers
        self.init_random_search_steps = init_random_search_steps
        self.ask_count = 0
        self.tell_count = 0
        self.init_count = 0
        self.eval_count = 0
        self.configs: list[dict] = []
        self.evaled = set()
        self.stoped = set()
        self.failed = set()
        self.history = []

        # placeholders
        self.pipelines: pd.DataFrame
        self.curves: np.ndarray
        self.fidelities: np.ndarray
        self.costs: np.ndarray
        self.score_history: np.ndarray | None = None

        # flags
        self.ready = False
        self.finished = False

    def setup(
        self,
        n: int,
        metafeat: Mapping[str, int | float] | None = None,
    ) -> None:
        """Setup the optimizer for optimization.

        Create the configurations to evaluate. The configurations are sampled from the
        configuration space. Optionally, metafeatures of the dataset can be provided.

        Args:
            n (int): The number of configurations to create.
            metafeat (Mapping[str, int | float], optional): The metafeatures of the dataset.
        """
        self.N = n
        self.fidelities: np.ndarray = np.zeros(self.N, dtype=int)
        self.curves: np.ndarray = np.full(
            (self.N, self.max_fidelity), np.nan, dtype=float
        )
        self.costs = np.ones(self.N, dtype=float)
        if self.patience is not None:
            self.score_history = np.zeros((n, self.patience), dtype=float)

        if self.seed is not None:
            self.cs.seed(self.seed)
        configs = self.cs.sample_configuration(n)
        self.configs = [dict(c) for c in configs]
        self.pipelines = pd.DataFrame(self.configs)

        self.metafeat = metafeat
        if self.metafeat is not None:
            self.metafeat = pd.DataFrame([metafeat] * self.N)
        self.pipelines = pd.concat([self.pipelines, self.metafeat], axis=1)

        self.ready = True

    def setup_pandas(
        self,
        df: pd.DataFrame,
        metafeat: Mapping[str, int | float] | None = None,
    ):
        """Setup the optimizer for optimization.

        Use an existing DataFrame to create the configurations to evaluate. Optionally,
        metafeatures of the dataset can be provided.

        Args:
            df (pd.DataFrame): The DataFrame with the configurations to evaluate.
            metafeat (Mapping[str, int | float], optional): The metafeatures of the dataset.
        """
        self.pipelines = df
        self.N = len(df)
        self.fidelities: np.ndarray = np.zeros(self.N, dtype=int)
        self.curves: np.ndarray = np.full(
            (self.N, self.max_fidelity), np.nan, dtype=float
        )
        self.costs = np.ones(self.N, dtype=float)
        if self.patience is not None:
            self.score_history = np.zeros((self.N, self.patience), dtype=float)

        self.metafeat = metafeat
        if self.metafeat is not None:
            self.metafeat = pd.DataFrame([metafeat] * self.N)
        self.pipelines = pd.concat([self.pipelines, self.metafeat], axis=1)
        self.configs = self.pipelines.to_dict(orient="records")

        self.ready = True

    def _predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the performance and cost of the configurations.
        
        Returns:
            The mean and standard deviation of the performance of the pipelines and their costs.
        """
        pipeline, curve = self.pipelines, self.curves

        pred = self.perf_predictor.predict(pipeline, curve)  # type: ignore
        pred_mean, pred_std = pred

        costs = self.costs
        if self.cost_aware:
            costs = self.cost_predictor.predict(pipeline)  # type: ignore
            min_clip = 1e-6
            if self.cost_predictor.meta_data_mean is not None:  # type: ignore
                min_clip = self.cost_predictor.meta_data_mean  # type: ignore
            
            costs = np.clip(costs, min_clip, None)  # avoid division by zero
            costs /= costs.max()  # normalize
            costs = np.power(costs, self.cost_factor)  # rescale

        return pred_mean, pred_std, costs

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
                acq_value = (mean - y_max) * norm.cdf(z) + std * norm.pdf(z)
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
        # maximum score per fidelity
        curves = np.nan_to_num(self.curves)
        y_max = curves.max(axis=0)
        y_max = np.maximum.accumulate(y_max)

        # get the ymax for the next fidelity of the pipelines
        next_fidelitys = np.minimum(self.fidelities + 1, self.max_fidelity)
        y_max_next = y_max[next_fidelitys - 1]

        acq_values = self._calc_acq_val(mean, std, y_max_next)
        if self.cost_aware:
            # acq_values /= max(cost, meta.avg)  TODO: QuickFix
            acq_values /= cost 

        return np.argsort(acq_values).tolist()

    def _ask(self):
        pred_mean, pred_std, cost = self._predict()
        ranks = self._optimize_acq_fn(pred_mean, pred_std, cost)
        ranks = [r for r in ranks if r not in self.stoped | self.failed]
        index = ranks[-1]
        logger.debug(f"predicted score: {pred_mean[index]:.4f}")
        return index

    def ask(self) -> dict | None:
        """Ask the optimizer for a configuration to evaluate.

        Returns:
            A dictionary with the configuration to evaluate.
        """
        if not self.ready:
            raise RuntimeError("Call setup() before ask()")

        if self.finished:
            return None

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

    def tell(self, result: dict | list[dict]):
        """Tell the optimizer the result of an evaluation.

        Args:
            result (dict | list[dict]): A dictionary with the result of an evaluation.
                If a list is provided, it is interpreted as a list of results.
        """
        if isinstance(result, dict):
            result = [result]
        for res in result:
            self._tell(res)

    def _tell(self, result: dict):
        self.tell_count += 1

        index = result["config_id"]
        fidelity = result["fidelity"]
        # cost = result["cost"]
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
        # self.costs[index] = cost
        self.history.append(result)
        self.evaled.add(index)
        self.eval_count += 1

        if self.patience is not None:
            assert self.score_history is not None
            if not np.any(self.score_history[index] < (score - self.tol)):
                self.stoped.add(index)
            self.score_history[index][fidelity % self.patience] = score

        self.finished = self._check_is_finished()

    def _check_is_finished(self):
        """Check if there is no more configurations to evaluate."""
        left = set(range(self.N)) - self.evaled - self.failed - self.stoped
        if not left:
            return True
        return False

    def ante(self):
        """Some operations to perform by the tuner before the optimization loop

        Here: refit the predictors with observed data.
        """
        if (
            self.refit
            and not self.eval_count % self.refit_interval
            and self.eval_count >= self.refit_init_steps
        ):
            self.fit_extra()

    def fit_extra(self):
        """Refit the predictors with observed data."""
        pipeline, curve = self.pipelines, self.curves
        self.perf_predictor.fit_extra(pipeline, curve)  # type: ignore

    def fit(self, X, curve, cost):
        """
        Fit the predictors with the given training data.
        """
        self.perf_predictor.fit(X, curve)  # type: ignore
        if self.cost_predictor is not None:
            self.cost_predictor.fit(X, cost)
