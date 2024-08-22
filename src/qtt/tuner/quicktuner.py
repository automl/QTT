import json
import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..optimizers import BaseOptimizer
from ..utils import (
    add_log_to_file,
    config_to_serializible_dict,
    set_logger_verbosity,
    setup_outputdir,
)

logger = logging.getLogger(__name__)


class QuickTuner:
    _log_to_file: bool = True
    _log_file_name: str = "quicktuner_log.txt"
    _log_file_path: str = "auto"
    path_suffix: Optional[str] = None

    def __init__(
        self,
        optimizer: BaseOptimizer,
        f: Callable,
        path: str | None = None,
        save_freq: str = "step",
        verbosity: int = 2,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)
        self.output_path = setup_outputdir(
            path, path_suffix=self.path_suffix, timestamp=True
        )
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

    # def reset(self):
    #     self.inc_score = 0.0
    #     self.inc_config = {}
    #     self.inc_cost = 0.0
    #     self.inc_fidelity = -1
    #     self.inc_id = -1
    #     self.traj = []
    #     self.history = []
    #     self.runtime = []

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
            evals_left = fevals - len(self.traj)
            if evals_left <= 0:
                return True
            logger.info(f"Evaluations left: {evals_left}")
        if time_budget is not None:
            time_left = time_budget - (time.time() - self.start)
            if time_left <= 0:
                return True
            logger.info(f"Time left: {time_left:.2f}s")
        # if self.optimizer.finished():
        #     return True
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
            history_path = os.path.join(self.output_path, "history.csv")
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(history_path)
        except Exception as e:
            logger.warning(f"History not saved: {e!r}")

    def _log_job_submission(self, job_info: dict):
        fidelity = job_info["fidelity"]
        config_id = job_info["config_id"]
        logger.info(
            "Evaluating configuration {} with fidelity {}".format(config_id, fidelity),
        )
        logger.info(
            f"Incumbent score: {self.inc_score}",
        )

    def save(self):
        logger.info("Saving current state to disk...")
        self._save_incumbent()
        self._save_history()

    def run(
        self,
        task_info: dict | None = None,
        fevals: int | None = None,
        time_budget: float | None = None,
    ):
        logger.info("Starting QuickTuner Run...")
        logger.info(f"QuickTuneTool will save results to {self.output_path}")

        self.start = time.time()
        while True:
            self.optimizer.ante()

            # ask for a new configuration
            job_info = self.optimizer.ask()
            if job_info is None:
                break
            _task_info = self._add_task_info(task_info)

            self._log_job_submission(job_info)
            result = self.f(job_info, task_info=_task_info)

            self._log_result(result)
            self.optimizer.tell(result)

            self.optimizer.post()
            if self._tune_budget_exhausted(fevals, time_budget):
                logger.info("Budget exhausted. Stopping run...")
                break

        self._log_end()
        self.save()

        return (
            np.array(self.traj),
            np.array(self.runtime),
            np.array(self.history, dtype=object),
        )

    def _update_trackers(self, traj, runtime, history):
        self.traj.append(traj)
        self.runtime.append(runtime)
        self.history.append(history)

    def _log_result(self, results):
        if isinstance(results, dict):
            results = [results]

        inc_changed = False
        for result in results:
            config_id = result["config_id"]
            score = result["score"]
            cost = result["cost"]
            fidelity = result["fidelity"]
            config = config_to_serializible_dict(result["config"])

            logger.info(
                f"*** CONFIG: {config_id}"
                f" - SCORE: {score:.3f}"
                f" - FIDELITY: {fidelity}"
                f" - TIME-TAKEN {cost:.3f} ***"
            )

            if self.inc_score < score:
                self.inc_score = score
                self.inc_cost = cost
                self.inc_fidelity = fidelity
                self.inc_id = config_id
                self.inc_config = config
                self.inc_info = result.get("info")
                inc_changed = True

            result["config"] = config
            self._update_trackers(
                self.inc_score,
                time.time() - self.start,
                result,
            )

        if self.save_freq == "step" or (self.save_freq == "incumbent" and inc_changed):
            self.save()

    def _log_end(self):
        logger.info("Run complete!")
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

    def _add_task_info(self, task_info: dict | None) -> dict:
        _task_info = {} if task_info is None else task_info.copy()
        _task_info["output-path"] = self.output_path
        return _task_info

    def get_incumbent(self):
        return (
            self.inc_id,
            self.inc_config,
            self.inc_score,
            self.inc_fidelity,
            self.inc_cost,
        )
