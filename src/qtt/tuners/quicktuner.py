import json
import logging
import os
import time
from collections import defaultdict
from typing import Callable, Optional

from qtt.optimizers.quick import QuickOptimizer
from qtt.utils.log_utils import add_log_to_file, set_logger_verbosity
from qtt.utils.qt_utils import QTunerResult, get_dataset_metafeatures
from qtt.utils.utils import setup_outputdir

logger = logging.getLogger("QuickTuner")


class QuickTuner:
    _log_to_file: bool = True
    _log_file_name: str = "quicktuner_log.txt"
    _log_file_path: str = "auto"

    def __init__(
        self,
        optimizer: QuickOptimizer,
        objective_function: Callable[..., QTunerResult],
        path: Optional[str] = None,
        path_suffix: Optional[str] = None,
        verbosity: int = 4,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)
        self.path = setup_outputdir(path, path_suffix=path_suffix)
        self.exp_dir = os.path.join(self.path, "exp")
        self._setup_log_to_file(self._log_to_file, self._log_file_path)

        self._validate_init_kwargs(kwargs)

        self.optimizer = optimizer
        self.objective_function = objective_function

        self.history = {"score": [], "cost": [], "configs": defaultdict(list)}


    def _setup_log_to_file(self, log_to_file: bool, log_file_path: str) -> None:
        if log_to_file:
            if log_file_path == "auto":
                log_file_path = os.path.join(self.path, "logs", self._log_file_name)
            log_file_path = os.path.abspath(os.path.normpath(log_file_path))
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            add_log_to_file(log_file_path, logger)

    def fit(
        self,
        data_path: str,
        train_split: str = "train",
        val_split: str = "val",
        time_limit: Optional[float] = None,
    ):
        logger.info("Starting QuickTuner fit.")
        logger.info(f"QuickTuneTool will save results to {self.path}")

        if time_limit is None:
            time_limit = float("inf")

        # if self.config.include_metafeatures:
        metafeat = get_dataset_metafeatures(data_path)
        self.optimizer.set_metafeatures(**metafeat)

        data_info = {
            "train-split": train_split,
            "val-split": val_split,
            "num_classes": metafeat["n_classes"],
        }

        orig_configs = self.optimizer.sample_configs
        configs = {i: config.get_dictionary() for i, config in enumerate(orig_configs)}
        with open(os.path.join(self.path, "configs.json"), "w") as f:
            json.dump(configs, f, indent=2, sort_keys=True)

        start_time = time.time()
        while True:
            config_id, budget = self.optimizer.suggest()
            logger.info(f"Optimizer suggests: {config_id} with budget {budget}")

            config = self.optimizer.sample_configs[config_id].get_dictionary()

            result = self.objective_function(
                budget=budget,
                config=config,
                config_id=config_id,
                data_path=data_path,
                data_info=data_info,
                output=self.exp_dir,
                verbosity=self.verbosity,
            )

            logger.info("Evaluation complete.")
            logger.info(f"Score: {result.score:.3f}% | Time: {result.time:.1f}s")

            self.optimizer.observe(config_id, budget, result)

            cost = time.time() - start_time

            self._save_results(result, cost)

            if (time.time() - start_time) > time_limit:
                logger.info("Time limit reached.")
                break

        logger.info("QuickTuner fit complete.")

        return self

    def _save_results(self, result: QTunerResult, cost: float) -> None:
        """
        Save the results of the tuning process.

        Returns:
            None
        """
        idx = result.idx
        score = result.score
        self.history["configs"][idx].append(score)
        self.history["score"].append(score)
        self.history["cost"].append(cost)

        # save to file
        with open(os.path.join(self.path, "history.json"), "w") as f:
            logger.debug("Saving history")
            json.dump(self.history, f, indent=2, sort_keys=True)

    def _validate_init_kwargs(self, kwargs: dict) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown argument: {key}")
