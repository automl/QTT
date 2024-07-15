import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm

from qtt.optimizers.optimizer import BaseOptimizer

from ..configuration import ConfigManager
from ..utils import fix_random_seeds, set_logger_verbosity
from .surrogates.dyhpo import DyHPO
from .surrogates.estimator import CostEstimator

logger = logging.getLogger("QuickOptimizer")


class QuickOptimizer(BaseOptimizer):
    def __init__(
        self,
        cm: ConfigManager,
        dyhpo: Optional[DyHPO] = None,
        cost_estimator: Optional[CostEstimator] = None,
        num_configs: int = 256,
        cost_aware: bool = False,
        metafeatures: Optional[np.ndarray] = None,
        fantasize_stepsize: int = 1,
        total_budget: int = 50,
        acq_fn: str = "ei",
        explore_factor: float = 0.0,
        tol: float = 0.0,
        n_iter_no_change: Optional[int] = 3,
        surrogate_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        verbosity: int = 2,
    ):
        super().__init__()
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)

        if seed is not None:
            fix_random_seeds(seed)

        self.cm = cm
        self.acq_func = acq_fn
        self.explore_factor = explore_factor
        self.total_budget = total_budget
        self.fantasize_stepsize = fantasize_stepsize
        self.num_configs = num_configs
        self.cost_aware = cost_aware

        self.metafeatures = None
        if metafeatures is not None:
            self.metafeatures = torch.tensor(metafeatures, dtype=torch.float)

        self.init_conf_idx = 0
        self.init_conf_nr = 5
        self.init_conf_count = 0
        self.eval_count = 0
        self.init_nr = 100

        self.inc_config = None
        self.inc_score = 0.0
        self.finished_configs = set()
        self.evaluated_configs = set()

        self.candidates = self.cm.sample_configuration(num_configs)
        self.configs = self.cm.preprocess_configurations(self.candidates)

        self.budgets = np.zeros(num_configs, dtype=np.int64)
        self.scores = np.zeros((num_configs, total_budget), dtype=np.float64)
        self.costs = np.full(num_configs, np.nan, dtype=np.float64)

        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        if n_iter_no_change is not None:
            self.score_history = np.zeros(
                (num_configs, n_iter_no_change), dtype=np.float64
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)

        if dyhpo is None:
            if surrogate_kwargs is None:
                surrogate_kwargs = self._get_default_surrogate_kwargs()
                print("surrogate_kwargs: ", surrogate_kwargs)
            self.dyhpo = DyHPO(**surrogate_kwargs)
            self.dyhpo.to(self.dev)
            self.cost_estimator = None
            if cost_aware:
                self.cost_estimator = CostEstimator(**surrogate_kwargs)
                self.cost_estimator.to(self.dev)
        else:
            self.dyhpo = dyhpo
            self.dyhpo.to(self.dev)
            self.cost_estimator = cost_estimator
            if self.cost_estimator is not None:
                self.cost_estimator.to(self.dev)

    def _get_default_surrogate_kwargs(self) -> Dict[str, Any]:
        one_hot = self.cm.get_one_hot_encoding()
        in_features = len(one_hot)
        in_meta_features = (
            len(self.metafeatures) if self.metafeatures is not None else None
        )
        surrogate_kwargs = dict(
            in_features=in_features,
            in_metafeat_dim=in_meta_features,
        )
        return surrogate_kwargs

    def _predict(self) -> Tuple[np.ndarray, ...]:
        test_data = self._get_candidate_configs()
        train_data = self._get_train_data()

        mean, std = self.dyhpo.predict_pipeline(train_data, test_data)  # type: ignore
        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()

        cost = self.costs
        if self.cost_estimator is not None:
            cost = self.cost_estimator(**test_data)
            cost = cost.squeeze().cpu().detach().numpy()

            mask = np.isnan(self.costs)
            cost = self.costs + mask * cost
        return mean, std, cost

    def _get_train_data(self):
        config, target, budget, curve = [], [], [], []
        for idx in self.evaluated_configs:
            _budget = self.budgets[idx]
            _scores = self.scores[idx].tolist()
            for n in range(_budget):
                config.append(self.configs[idx])
                budget.append(n + 1)
                target.append(_scores[n])
                _curve = _scores[:n] + [0.0] * (self.total_budget - n)
                curve.append(_curve)

        config = np.array(config)
        config = torch.tensor(config, dtype=torch.float).to(self.dev)
        budget = torch.tensor(budget, dtype=torch.float).to(self.dev)
        budget /= self.total_budget
        curve = torch.tensor(curve, dtype=torch.float).to(self.dev)
        target = torch.tensor(target, dtype=torch.float).to(self.dev)

        metafeat = None
        if self.metafeatures is not None:
            metafeat = self.metafeatures.repeat(len(config), 1).to(self.dev)

        data = dict(
            config=config,
            budget=budget,
            curve=curve,
            target=target,
            metafeat=metafeat,
        )
        return data

    def _get_train_cost_data(self):
        mask = np.isnan(self.costs)
        config = self.configs[~mask]
        cost = self.costs[~mask]

        config = torch.tensor(config, dtype=torch.float)
        cost = torch.tensor(cost, dtype=torch.float).unsqueeze(1)

        metafeat = None
        if self.metafeatures is not None:
            metafeat = self.metafeatures.repeat(len(config), 1)

        data = dict(config=config, cost=cost, metafeat=metafeat)
        for key, value in data.items():
            if value is not None:
                data[key] = value.to(self.dev)
        return data

    def _fit_surrogate(self):
        """Fit the surrogate model with the observed data."""
        data = self._get_train_data()
        self.dyhpo.fit_pipeline(data=data)  # type: ignore

        if self.cost_aware:
            data = self._get_train_cost_data()
            self.cost_estimator.fit_pipeline(data=data)  # type: ignore

    def _get_candidate_configs(self) -> dict:
        config = torch.tensor(self.configs, dtype=torch.float).to(self.dev)
        budget = torch.tensor(self.budgets, dtype=torch.float).to(self.dev)
        budget = (budget + self.fantasize_stepsize) / self.total_budget
        curve = torch.tensor(self.scores, dtype=torch.float).to(self.dev)
        metafeat = None
        if self.metafeatures is not None:
            metafeat = self.metafeatures.repeat(self.num_configs, 1).to(self.dev)

        candidates = {
            "config": config,
            "budget": budget,
            "curve": curve,
            "metafeat": metafeat,
        }
        return candidates

    def _find_most_promising_config(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        cost: np.ndarray,
    ) -> List[int]:
        ymax = self._get_ymax_per_budget()
        next_budgets = self.budgets
        next_budgets[next_budgets >= self.total_budget] = self.total_budget - 1
        ymax = ymax[self.budgets]

        acq_values = self.acq(mean, std, ymax)
        if self.cost_aware:
            cost += 1e-6  # avoid division by zero
            acq_values /= cost

        return np.argsort(acq_values).tolist()

    def acq(
        self,
        mean: float | np.ndarray,
        std: float | np.ndarray,
        ymax: float | np.ndarray,
    ) -> float | np.ndarray:
        fn = self.acq_func
        xi = self.explore_factor
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
        return acq_value

    def _get_ymax_per_budget(self) -> np.ndarray:
        ymax = self.scores.max(axis=0)
        ymax[ymax == 0] = ymax.max()
        return ymax

    def finished(self) -> bool:
        return len(self.finished_configs) == self.num_configs

    def ask(self):
        # check if we still have random configurations to evaluate
        if self.init_conf_count < self.init_conf_nr:
            index = self.init_conf_idx
            budget = self.fantasize_stepsize
            self.init_conf_idx += 1

        else:
            mean, std, cost = self._predict()
            ranks = self._find_most_promising_config(mean, std, cost)
            ranks = [idx for idx in ranks if idx not in self.finished_configs]
            index = ranks[-1]

            budget = self.budgets[index] + self.fantasize_stepsize
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
            cost = result["cost"]

            if not status:
                self.finished_configs.add(config_id)

            else:
                self.eval_count += 1
                self.evaluated_configs.add(config_id)
                self.scores[config_id] = score
                self.costs[config_id] = cost
                self.budgets[config_id] += 1

                if budget == 1:
                    self.init_conf_count += 1

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
