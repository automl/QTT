import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm

from qtt.optimizers.surrogates.dyhpo import DyHPO
from qtt.optimizers.surrogates.estimator import CostEstimator
from qtt.utils.log_utils import set_logger_verbosity

logger = logging.getLogger("QuickOptimizer")


class QuickOptimizer:
    configs: np.ndarray
    n_iter_no_change: int | None = 3
    tol = 0.0
    no_improvement_patience = 0

    def __init__(
        self,
        dyhpo: DyHPO,
        cost_estimator: Optional[CostEstimator] = None,
        metafeatures: Optional[torch.Tensor] = None,
        max_budget: int = 50,
        fantasize_steps: int = 1,
        acq_fn: str = "ei",
        explore_factor: float = 0.0,
        verbosity: int = 2,
        seed: Optional[int] = None,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.dyhpo = dyhpo
        self.cost_estimator = cost_estimator

        self.metafeatures = metafeatures

        self.max_budget = max_budget
        self.fantasize_stepsize = fantasize_steps

        self.init_conf_idx = 0
        self.init_conf_nr = 5
        self.init_conf_eval_count = 0
        self.eval_count = 0

        self.incumbent = -1
        self.incumbent_score = 0.0
        self.info_dict = dict()
        self.finished_configs = set()
        self.evaluated_configs = set()

        self.results: dict[int, List[int]] = defaultdict(list)
        self.scores: dict[int, List[float]] = defaultdict(list)
        self.costs: dict[int, List[float]] = defaultdict(list)

        self.acq_func = acq_fn
        self.explore_factor = explore_factor

    def set_configs(self, configs):  # np.ndarray | pd.DataFrame
        self.configs = configs.to_numpy()
        if self.n_iter_no_change is not None:
            self.score_history = {
                i: np.zeros(self.n_iter_no_change) for i in range(len(configs))
            }

    def set_metafeatures(
        self,
        t: Any,
    ):
        """
        Set the metafeatures of the dataset.
        """
        meta_scale_factor = 10000  # TODO: automate this
        t = torch.tensor(t, dtype=torch.float)
        self.metafeatures = t / meta_scale_factor

    def suggest(self) -> Tuple[int, int]:
        """
        Suggest the next hyperparameter configuration to evaluate.

        Returns
        -------
        best_config_index: int
            The index of the best hyperparameter configuration.
        budget: int
            The budget of the hyperparameter configuration.
        """
        # check if we still have random configurations to evaluate
        if self.init_conf_eval_count < self.init_conf_nr:
            index = self.init_conf_idx
            budget = self.fantasize_stepsize
            self.init_conf_idx += 1

        else:
            mean, std, budgets, costs = self._predict()
            ranks = self.find_suggested_config(mean, std, budgets, costs)
            ranks = [idx for idx in ranks if idx not in self.finished_configs]
            index = ranks[-1]

            # decide for what budget we will evaluate the most promising hyperparameter configuration next.
            budget = self.fantasize_stepsize
            if index in self.results:
                budget = self.results[index][-1]
                budget += self.fantasize_stepsize
                # if fantasize_stepsize is bigger than 1
                budget = min(budget, self.max_budget)
        return index, budget

    def _get_train_data(self) -> Dict[str, torch.Tensor]:
        """
        Get the training data for the surrogate model.
        Training dataconsists of the evaluated hyperparameter configurations and their
        associated larning curves.
        """

        config, target, budget, curve = [], [], [], []
        for idx in self.evaluated_configs:
            _budgets = self.results[idx]
            _scores = self.scores[idx]
            for n in range(len(_scores)):
                config.append(self.configs[idx])
                budget.append(_budgets[n])
                target.append(_scores[n])
                _curve = _scores[:n] + [0.0] * (self.max_budget - n)
                curve.append(_curve)
        # np.array(configs), np.array(targets), np.array(budgets), np.array(curves)
        config = torch.tensor(config, dtype=torch.float)
        budget = torch.tensor(budget, dtype=torch.float) / self.max_budget
        curve = torch.tensor(curve, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)

        metafeat = None
        if self.metafeatures is not None:
            metafeat = self.metafeatures.repeat(len(config), 1)

        data = dict(
            config=config,
            budget=budget,
            curve=curve,
            target=target,
            metafeat=metafeat,
        )

        return data  # type: ignore

    def _fit_surrogate(self):
        """
        Fit the surrogate model with the observed hyperparameter configurations.
        """
        data = self._get_train_data()
        self.dyhpo.fit_pipeline(data=data)

    def _predict(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict the performances of the hyperparameter configurations
        as well as the standard deviations based on the surrogate model.
        Returns:
            mean_predictions, std_predictions, hp_indices, non_scaled_budgets:
                The mean predictions and the standard deviations over
                all model predictions for the given hyperparameter
                configurations with their associated indices, scaled and
                non-scaled budgets.
        """
        # config, budget, curve, indices = self._get_candidate_configs()
        config, budget, curve = self._get_candidate_configs()

        # add fantasize steps to the budget
        t_budget = torch.tensor(budget, dtype=torch.float) + self.fantasize_stepsize
        # scale budget to [0, 1]
        t_budget /= self.max_budget
        t_config = torch.tensor(config, dtype=torch.float)

        t_curve = torch.tensor(curve, dtype=torch.float)

        t_metafeat = None
        if self.metafeatures is not None:
            t_metafeat = self.metafeatures.repeat(t_config.size(0), 1)

        test_data = {
            "config": t_config,
            "budget": t_budget,
            "curve": t_curve,
            "metafeat": t_metafeat,
        }
        train_data = self._get_train_data()

        mean, std = self.dyhpo.predict_pipeline(train_data, test_data)
        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()

        cost = None
        if self.cost_estimator is not None:
            cost = self.cost_estimator(**test_data)
            cost = cost.squeeze().cpu().detach().numpy()

        return mean, std, budget, cost

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
                    self.init_conf_eval_count += 1

            if score >= 1.0 or budget >= self.max_budget:
                self.finished_configs.add(config_id)

            if self.incumbent_score < score:
                self.incumbent = config_id
                self.incumbent_score = score
                self.no_improvement_patience = 0
            else:
                self.no_improvement_patience += 1

            if self.n_iter_no_change is not None:
                n_last_iter = self.score_history[config_id]
                if not np.any(score + self.tol > n_last_iter):
                    self.finished_configs.add(config_id)
                self.score_history[config_id] = np.roll(
                    self.score_history[config_id], 1
                )
                self.score_history[config_id][0] = score

        # Initialization phase over. Fit the surrogate model
        if self.init_conf_eval_count >= 10:
            self._fit_surrogate()

    def _get_candidate_configs(self) -> Tuple[np.ndarray, ...]:
        budgets = []
        curves = []

        for index in range(len(self.configs)):
            if index in self.results:
                budget = max(self.results[index])
                curve = self.scores[index]
            else:  # config was not evaluated before, fantasize
                budget = 1
                curve = []

            # pad the curve with zeros if it is not fully evaluated
            curve = curve + [0.0] * (self.max_budget - len(curve))

            budgets.append(budget)
            curves.append(curve)

        configs = self.configs
        budgets = np.array(budgets)
        curves = np.array(curves)

        return configs, budgets, curves

    def find_suggested_config(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        budgets: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> List[int]:
        if cost is None:
            cost = np.array(1)

        ymax = self._get_ymax_per_budget()
        ymax = ymax[budgets - 1]

        acq_values = self.acq(ymax, mean, std, cost)
        return np.argsort(acq_values).tolist()

    def acq(
        self,
        ymax: float | np.ndarray,
        mean: float | np.ndarray,
        std: float | np.ndarray,
        cost: float | np.ndarray = 1,
    ) -> float | np.ndarray:
        """
        Calculate the acquisition function value for a given hyperparameter configuration.

        Args
        ----
        ymax: float | np.ndarray
            The best value observed so far for the given fidelity.
        mean: float | np.ndarray
            The mean prediction of the surrogate model.
        std: float | np.ndarray
            The standard deviation of the surrogate model.
        cost: float | np.ndarray, default = 1
            The cost of the hyperparameter configuration.

        Returns
        -------
        acq_value: float | np.ndarray
            The acquisition function value for the given hyperparameter configurations.
        """
        fn = self.acq_func
        xi = self.explore_factor

        cost += 1e-6  # to avoid division by zero

        # Expected Improvement
        if fn == "ei":
            mask = std == 0
            std = std + mask * 1.0
            z = (mean - ymax - xi) / std
            acq_value = (mean - ymax - xi) * norm.cdf(z) + std * norm.pdf(z)
            if isinstance(acq_value, float):
                acq_value = acq_value if mask else 0.0
            else:
                acq_value[mask] = 0.0
        # Upper Confidence Bound
        elif fn == "ucb":
            acq_value = mean + xi * std
        # Thompson Sampling
        elif fn == "thompson":
            acq_value = np.random.normal(mean, std)
        elif fn == "exploit":
            # Exploitation
            acq_value = mean
        else:
            msg = f"acquisition function {fn} is not implemented"
            raise NotImplementedError(msg)

        return acq_value / cost

    def _get_ymax_per_budget(self) -> np.ndarray:
        """
        Calculate the maximum performance for each budget level.

        Returns
        -------
        ymax: np.ndarray
            The maximum performance for each budget level.
        """
        from itertools import zip_longest

        ymax = np.zeros(self.max_budget)
        scores = self.scores.values()
        for n, score in enumerate(zip_longest(*scores, fillvalue=0)):
            ymax[n] = max(score)

        ymax[ymax == 0] = ymax.max()
        ymax /= 100  # normalize to [0, 1]
        return ymax
