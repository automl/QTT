import logging
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm

from qtt.configuration.manager import ConfigManager
from qtt.utils.log_utils import set_logger_verbosity
from qtt.utils.qt_utils import QTaskStatus, QTunerResult

from qtt.optimizers.surrogates.surrogate import Surrogate

logger = logging.getLogger("QuickOptimizer")


class QuickOptimizer:
    """
    The QuickTuneOptimizer is the main element for the QuickTune optimization process.

    The QuickTuneOptimizer class provides methods for training the surrogate model,
    predicting the performances of hyperparameter configurations, suggesting the next
    hyperparameter configuration to evaluate, observing the learning curve of a hyperparameter
    configuration, and updating the information dictionary with the current HPO iteration info.

    Args
    ----
    surrogate: nn.Module
        The surrogate model to be used for the optimization process.
    config_manager: ConfigurationManager
        The configuration manager object that contains the configuration space.
    metaset: MetaSet
        The MetaSet object that contains the metafeatures of the dataset.
    num_configs: int
        The number of configurations to sample from the configuration space.
    metafeatures: Optional[torch.Tensor]
        The metafeatures of the dataset.
    max_benchmark_epochs: int, default = 50
        The maximum number of epochs to benchmark the hyperparameter configurations.
    fantasize_steps: int, default = 1
        The number of steps to fantasize the hyperparameter configurations.
    acq_func: str, default = "ei"
        The acquisition function to be used for the optimization process.
    explore_factor: float, default = 0.0
        The exploration factor for the acquisition function.
    output_path: str, default = "."
        The path to save the output files.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        config_manager: ConfigManager,
        num_configs: int,
        metafeatures: Optional[torch.Tensor] = None,
        max_budget: int = 50,
        fantasize_steps: int = 1,
        acq_fn: str = "ei",
        explore_factor: float = 0.0,
        output_path: str = ".",
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
        self.metafeatures = metafeatures
        self.surrogate = surrogate

        self.sample_configs = self.cm.sample_configuration(num_configs)
        self.candidate_configs = self.cm.preprocess_configurations(
            self.sample_configs,
            standardize=True,
        ).values

        self.max_budget = max_budget
        self.fantasize_stepsize = fantasize_steps

        self.init_conf_idx = 0
        self.init_conf_nr = 1
        self.init_conf_eval_count = 0
        self.eval_count = 0

        self.incumbent = -1
        self.incumbent_score = float("-inf")
        self.info_dict = dict()
        self.finished_configs = set()
        self.results: dict[int, List[int]] = dict()
        self.scores: dict[int, List[float]] = dict()
        self.costs: dict[int, List[float]] = dict()

        self.suggest_time_duration = 0
        self.output_path = output_path

        self.no_improvement_threshold = int(self.max_budget + 0.2 * self.max_budget)
        self.no_improvement_patience = 0
        self.acq_func = acq_fn
        self.explore_factor = explore_factor

    def set_metafeatures(
        self,
        n_samples: int,
        n_classes: int,
        n_features: int,
        n_channels: int,
    ):
        """
        Set the metafeatures of the dataset.

        Args
        ----
        metafeatures: torch.Tensor
            The metafeatures of the dataset.
        """
        meta_scale_factor = 10000  # TODO: automate this
        t = torch.tensor(
            [n_samples, n_classes, n_features, n_channels],
            dtype=torch.float,
        ).reshape(1, -1)
        self.metafeatures = t / meta_scale_factor

    def _get_train_data(self) -> Dict[str, torch.Tensor]:
        """
        Prepare the data that will be the input to the surrogate.

        Returns
        -------
        data: Dict[str, torch.Tensor]
            The data that will be the input to the surrogate.
        """

        config, target, budget, curve = self._get_history_configs()

        # normalize data
        config = torch.tensor(config, dtype=torch.float)
        budget = torch.tensor(budget, dtype=torch.float) / self.max_budget
        curve = torch.tensor(curve, dtype=torch.float) / 100
        target = torch.tensor(target, dtype=torch.float) / 100

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

    def _fit_surrogate(self, restart: bool = False):
        """
        Train the surrogate model with the observed hyperparameter configurations.
        """
        data = self._get_train_data()
        self.surrogate.train_pipeline(data=data, restart=restart)

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

        t_curve = torch.tensor(curve, dtype=torch.float) / 100

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

        mean, std, cost = self.surrogate.predict_pipeline(train_data, test_data)

        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()
        cost = cost.squeeze().cpu().detach().numpy() if cost is not None else None

        return mean, std, budget, cost

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
            logger.info(
                "Initialization phase not over yet. "
                "Returning randomly sampled configuration"
            )

            index = self.init_conf_idx
            budget = self.fantasize_stepsize
            self.init_conf_idx += 1

            return index, budget

        else:
            mean, std, budgets, costs = self._predict()

            best_indices = self.find_suggested_config(mean, std, budgets, costs)

            best_indices = [
                idx for idx in best_indices if idx not in self.finished_configs
            ]
            index = best_indices[-1]

            # decide for what budget we will evaluate the most promising hyperparameter configuration next.
            if index in self.results:
                budget = self.results[index][-1]
                budget += self.fantasize_stepsize
                # if fantasize_stepsize is bigger than 1
                budget = min(budget, self.max_budget)
            else:
                budget = self.fantasize_stepsize

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
        observe_time_start = time.time()

        score = result.score
        if result.status == QTaskStatus.ERROR:
            self.finished_configs.add(index)
            score = 0.0

        # if score >= (1 - threshold)
        # maybe accept config as finished before reaching max performance ??? TODO
        if score >= 100 or budget >= self.max_budget:
            self.finished_configs.add(index)
        
        if result.status == QTaskStatus.SUCCESS:
            if index in self.results:
                self.results[index].append(budget)
                self.scores[index].append(score)
            else:
                self.results[index] = [budget]
                self.scores[index] = [score]
                self.init_conf_eval_count += 1

        if self.incumbent_score < score:
            self.incumbent = index
            self.incumbent_score = score
            self.no_improvement_patience = 0
        else:
            self.no_improvement_patience += 1

        # initialization phase over. Now we can sample from the model.
        if self.init_conf_eval_count >= self.init_conf_nr:
            restart = True
            # restart the model if we have not seen enough evaluations
            if self.eval_count > 10:  # TODO: make this a parameter
                restart = False
            # restart the model if we have not seen any improvement for a while
            if self.no_improvement_patience > self.no_improvement_threshold:
                restart = True

            self._fit_surrogate(restart=restart)

        observe_time = time.time() - observe_time_start
        overhead_time = observe_time + self.suggest_time_duration
        return overhead_time

    def _get_candidate_configs(
        self,
    ) -> Tuple[np.ndarray, ...]:
        """
        Generate candidate configurations that will be fantasized upon.

        Returns
        -------
        configurations: List
            The hyperparameter configurations.
        hp_indices: List
            The indices of the hyperparameter configurations.
        hp_budgets: List
            The budgets of the hyperparameter configurations.
        learning_curves: List
            The learning curves of the hyperparameter configurations.
        """
        budgets = []
        curves = []

        for index in range(len(self.candidate_configs)):
            if index in self.results:
                budget = max(self.results[index])
                curve = self.scores[index]
            else:  # config was not evaluated before fantasize
                budget = 0
                curve = [0.0]

            # pad the curve with zeros if it is not fully evaluated
            curve = curve + [0.0] * (self.max_budget - len(curve))

            budgets.append(budget)
            curves.append(curve)

        configs = self.candidate_configs
        budgets = np.array(budgets)
        curves = np.array(curves)

        return configs, budgets, curves

    def _get_history_configs(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the configurations, labels, budgets and curves based on
        the history of evaluated configurations.

        Returns
        -------
        configs: List
            The hyperparameter configurations.
        train_labels: List
            The performances of the hyperparameter configurations.
        train_budgets: List
            The budgets of the hyperparameter configurations.
        train_curves: List
            The learning curves of the hyperparameter configurations.
        """
        configs = []
        targets = []
        budgets = []
        curves = []

        for hp_index in self.results:
            budget = self.results[hp_index]
            scores = self.scores[hp_index]
            config = self.candidate_configs[hp_index]

            for n in range(len(scores)):
                configs.append(config)
                budgets.append(budget[n])
                targets.append(scores[n])
                curve = scores[:n]
                curve = curve + [0.0] * (self.max_budget - len(curve))
                curves.append(curve)

        return np.array(configs), np.array(targets), np.array(budgets), np.array(curves)

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
