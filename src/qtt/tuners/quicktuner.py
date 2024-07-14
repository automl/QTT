import json
import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from qtt.utils.log_utils import add_log_to_file, set_logger_verbosity
from qtt.utils.setup import setup_outputdir

from ..optimizers import BaseOptimizer

logger = logging.getLogger("QuickTuner")


class QuickTuner:
    _log_to_file: bool = True
    _log_file_name: str = "quicktuner_log.txt"
    _log_file_path: str = "auto"
    path_suffix: Optional[str] = None

    def __init__(
        self,
        optimizer: BaseOptimizer,
        f: Callable,
        path: Optional[str] = None,
        save_freq: Optional[str] = "incumbent",
        verbosity: int = 2,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)
        self.output_path = setup_outputdir(path, path_suffix=self.path_suffix)
        self._setup_log_to_file(self._log_to_file, self._log_file_path)

        self._validate_kwargs(kwargs)
        if save_freq not in ["step", "incumbent"] and save_freq is not None:
            raise ValueError("Invalid value for 'save_freq'.")
        self.save_freq = save_freq

        self.optimizer = optimizer
        self.f = f

        # trackers
        self.inc_score: float = 0.0
        self.inc_config: dict = {}
        self.inc_cost: float = 0.0
        self.inc_info: object = None
        self.inc_id: int = -1
        self.traj: list[object] = []
        self.history: list[object] = []
        self.runtime: list[object] = []

    def reset(self):
        self.inc_score = 0.0
        self.inc_config = {}
        self.inc_cost = 0.0
        self.inc_id = -1
        self.traj = []
        self.history = []
        self.runtime = []

        self.optimizer.reset()

    def _setup_log_to_file(self, log_to_file: bool, log_file_path: str) -> None:
        if log_to_file:
            if log_file_path == "auto":
                log_file_path = os.path.join(
                    self.output_path, "logs", self._log_file_name
                )
            log_file_path = os.path.abspath(os.path.normpath(log_file_path))
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            add_log_to_file(log_file_path, logger)

    def _tune_budget_exhausted(self, fevals=None, time_budget=None):
        """Checks if the run should be terminated or continued."""
        if fevals is not None:
            if len(self.traj) >= fevals:
                return True
        if time_budget is not None:
            if time.time() - self.start >= time_budget:
                return True
        if self.optimizer.finished():
            return True
        return False

    def _save_incumbent(self):
        if not self.inc_config:
            return
        try:
            out = {}
            out["config"] = self.inc_config
            out["score"] = self.inc_score
            out["cost"] = self.inc_cost
            out["info"] = self.inc_info
            with open(os.path.join(self.output_path, "incumbent.json"), "w") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save incumbent: {e}")

    def _save_history(self):
        if not self.history:
            return
        try:
            history_path = os.path.join(self.output_path, "history.parquet.gzip")
            history_df = pd.DataFrame(
                self.history,
                columns=["config_id", "config", "fitness", "cost", "fidelity", "info"],
            )
            # Check if the 'info' column is empty or contains only None values
            if (
                history_df["info"]
                .apply(lambda x: (isinstance(x, dict) and len(x) == 0))
                .all()
            ):
                # Drop the 'info' column
                history_df = history_df.drop(columns=["info"])
            history_df.to_parquet(history_path, compression="gzip")
        except Exception as e:
            logger.warning(f"History not saved: {e!r}")

    def _log_job_submission(self, job_info: dict):
        budget = job_info["budget"]
        config_id = job_info["config_id"]
        logger.info(
            "Evaluating configuration {} with budget {}".format(config_id, budget),
        )
        logger.info(
            f"Best score seen/Incumbent score: {self.inc_score}",
        )

    def save(self):
        logger.info("Saving current state to disk...")
        self._save_incumbent()
        self._save_history()

    def run(
        self,
        task_info: dict = {},
        fevals: Optional[int] = None,
        time_budget: Optional[float] = None,
    ):
        logger.info("Starting QuickTuner fit.")
        logger.info(f"QuickTuneTool will save results to {self.output_path}")

        self.start = time.time()
        while True:
            #
            self.optimizer.ante()
            # ask for a new configuration
            job_info = self.optimizer.ask()

            _task_info = self._add_task_info(task_info)
            result = self.f(job_info, task_info=_task_info)

            self._log_result(job_info, result)
            self.optimizer.tell(result)
            #
            self.optimizer.post()
            if self._tune_budget_exhausted(fevals, time_budget):
                break

        self._log_end()

        self.save()

        return (
            np.array(self.traj),
            np.array(self.runtime),
            np.array(self.history, dtype=object),
        )

    def _log_result(self, job_info, results):
        if isinstance(results, dict):
            results = [results]

        inc_changed = False
        for result in results:
            config_id = job_info["config_id"]
            score = result["score"]
            cost = result["cost"]

            if self.inc_score < score:
                self.inc_score = score
                self.inc_cost = cost
                self.inc_id = config_id
                self.inc_config = job_info["config"].get_dictionary()

        if self.save_freq == "step" or (self.save_freq == "incumbent" and inc_changed):
            self.save()
    
    def _log_end(self):
        logger.info("Tuning complete.")
        logger.info(f"Best score: {self.inc_score}")
        logger.info(f"Best cost: {self.inc_cost}")
        logger.info(f"Best config ID: {self.inc_id}")
        logger.info(f"Best configuration: {self.inc_config}")

    def _validate_kwargs(self, kwargs: dict) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown argument: {key}")
    
    def _add_task_info(self, task_info: dict):
        _task_info = task_info.copy()
        _task_info["output-path"] = self.output_path
        return _task_info

    def get_incumbent(self):
        return self.inc_config, self.inc_score, self.inc_cost, self.inc_id
