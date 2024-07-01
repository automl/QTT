import json
import logging
import os
import time
from collections import defaultdict
from typing import Callable, Optional, Union

from qtt.configuration.manager import ConfigManager
from qtt.optimizers import QuickOptimizer
from qtt.optimizers import RandomOptimizer
from qtt.utils.log_utils import add_log_to_file, set_logger_verbosity
from qtt.utils.qt_utils import get_dataset_metafeatures
from qtt.utils.utils import setup_outputdir

logger = logging.getLogger("QuickTuner")


class QuickTuner:
    _log_to_file: bool = True
    _log_file_name: str = "quicktuner_log.txt"
    _log_file_path: str = "auto"

    def __init__(
        self,
        optimizer: QuickOptimizer | RandomOptimizer,
        config_manager: ConfigManager,
        objective_function: Callable[..., Union[dict, list[dict]]],
        path: Optional[str] = None,
        path_suffix: Optional[str] = None,
        verbosity: int = 2,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)
        self.path = setup_outputdir(path, path_suffix=path_suffix)
        self.exp_dir = os.path.join(self.path, "exp")
        self._setup_log_to_file(self._log_to_file, self._log_file_path)

        self._validate_kwargs(kwargs)

        self.optimizer = optimizer
        self.cm = config_manager
        self.objective_function = objective_function

        self.history = {"score": [], "cost": [], "configs": defaultdict(list)}
        self.incumbent = {"config": None, "score": 0, "cost": float("inf")}

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
        num_configs: int = 256,
        train_split: str = "train",
        val_split: str = "val",
        time_limit: Optional[float] = None,
    ):
        if time_limit is not None:
            assert time_limit > 0, "Time limit must be greater than 0."

        logger.info("Starting QuickTuner fit.")
        logger.info(f"QuickTuneTool will save results to {self.path}")

        metafeat = get_dataset_metafeatures(data_path)
        self.optimizer.set_metafeatures(metafeat)

        configs = self.cm.sample_configuration(num_configs)
        self.save_configs(configs)

        self.optimizer.set_configs(self.cm.preprocess_configurations(configs))

        start_time = time.time()
        while True:
            config_id, budget = self.optimizer.suggest()
            logger.info(f"Optimizer suggests: {config_id} with budget {budget}")

            config = configs[config_id].get_dictionary()

            task_info = {
                "config_id": config_id,
                "data_path": data_path,
                "train-split": train_split,
                "val-split": val_split,
                "num_classes": metafeat[1],
                "output_dir": self.exp_dir,
                "verbosity": self.verbosity,
            }

            results = self.objective_function(
                config=config,
                budget=budget,
                task_info=task_info,
            )

            self.process_results(results)
            self.optimizer.observe(results)

            if time_limit is not None:
                time_passed = time.time() - start_time
                if time_passed > time_limit:
                    logger.info("Time limit reached.")
                    break

        logger.info("QuickTuner fit complete.")

        return self

    def process_results(self, results):
        if isinstance(results, dict):
            results = [results]

        for result in results:
            config_id = result["config_id"]
            score = result["score"]
            cost = result["cost"]
            self.history["configs"][config_id].append(score)
            self.history["score"].append(score)
            self.history["cost"].append(cost)

            if score > self.incumbent["score"]:
                self.incumbent["score"] = score
                self.incumbent["cost"] = cost
                self.incumbent["config"] = config_id
                
            if result["status"]:
                logger.info("Evaluation completed.")
                logger.info(
                    f"Score: {100 * result['score']:.3f}% | Time: {result['cost']:.1f}s"
                )
            else:
                logger.info("Evaluation failed.")
                logger.info(f"Error: {result['info']}")

        # save to file
        with open(os.path.join(self.path, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2, sort_keys=True)

    def save_configs(self, configs) -> None:
        configs = {i: config.get_dictionary() for i, config in enumerate(configs)}
        with open(os.path.join(self.path, "configs.json"), "w") as f:
            json.dump(configs, f, indent=2, sort_keys=True)

    def _validate_kwargs(self, kwargs: dict) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown argument: {key}")

    def get_incumbent(self):
        path = os.path.join(self.exp_dir, str(self.incumbent["config"]))
        path_to_model = os.path.join(path, "model_best.pth.tar")

        with open(os.path.join(self.path, "configs.json"), "r") as f:
            config_id = str(self.incumbent["config"])
            config = json.load(f)[config_id]
        
        return path_to_model, config