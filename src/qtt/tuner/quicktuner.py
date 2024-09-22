import json
import pickle
import logging
import os
import time
from typing import Callable

import numpy as np
import pandas as pd

from ..optimizers import Optimizer
from ..utils import (
    add_log_to_file,
    config_to_serializible_dict,
    set_logger_verbosity,
    setup_outputdir,
)

logger = logging.getLogger(__name__)


class QuickTuner:
    """QuickTuner is a wrapper around an optimizer that runs the optimization loop.

    Parameters
    ----------
    optimizer : Optimizer
        An instance of an Optimizer class.
    f : Callable
        A function that takes a configuration and returns a score.
    path : str, default = None 
        Directory location to store all outputs.
        If None, a new unique time-stamped directory is chosen.
    save_freq : str, default = "step"
        Frequency of saving the state of the tuner.
        - "step": save after each evaluation.
        - "incumbent": save only when the incumbent changes.
        - None: do not save.
    verbosity : int, default = 2
        Verbosity level of the logger.
    resume : bool, default = False
        Whether to resume the tuner from a previous state.
    log_to_file : bool, default = True
        Whether to log to a file.
    log_file_name : str, default = "quicktuner.log"
        Name of the log file.
    log_file_path : str, default = "auto"
        Path to the log file.
    path_suffix : str, default = "tuner"
        Suffix to append to the output directory.
        By default, the output directory is named ``qtt/`timestamp`/tuner``.
    """
    log_to_file: bool = True
    log_file_name: str = "quicktuner.log"
    log_file_path: str = "auto"
    path_suffix: str = "tuner"

    def __init__(
        self,
        optimizer: Optimizer,
        f: Callable,
        path: str | None = None,
        save_freq: str | None = "step",
        verbosity: int = 2,
        resume: bool = False,
        **kwargs,
    ):
        if resume and path is None:
            raise ValueError("Cannot resume without specifying a path.")
        self._validate_kwargs(kwargs)

        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)

        self.output_path = setup_outputdir(path, path_suffix=self.path_suffix)
        self._setup_log_to_file(self.log_to_file, self.log_file_path)

        if save_freq not in ["step", "incumbent"] and save_freq is not None:
            raise ValueError("Invalid value for 'save_freq'.")
        self.save_freq = save_freq

        self.optimizer = optimizer
        self.optimizer.reset_path(self.output_path)
        self.f = f

        # trackers
        self.inc_score: float = 0.0
        self.inc_fidelity: int = -1
        self.inc_config: dict = {}
        self.inc_cost: float = 0.0
        self.inc_info: object = None
        self.inc_id: int = -1
        self.traj: list[object] = []
        self.history: list[object] = []
        self.runtime: list[object] = []

        self._remaining_fevals = None
        self._remaining_time = None

        if resume:
            self.load(os.path.join(self.output_path, "qt.json"))

    def _setup_log_to_file(self, log_to_file: bool, log_file_path: str) -> None:
        if not log_to_file:
            return
        if log_file_path == "auto":
            log_file_path = os.path.join(self.output_path, "logs", self.log_file_name)
        log_file_path = os.path.abspath(os.path.normpath(log_file_path))
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        add_log_to_file(log_file_path, logger)

    def _is_budget_exhausted(self, fevals=None, time_budget=None):
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
        return False

    def _save_incumbent(self, save: bool = True):
        if not self.inc_config or not save:
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

    def _save_history(self, save: bool = True):
        if not self.history or not save:
            return
        try:
            history_path = os.path.join(self.output_path, "history.csv")
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(history_path)
        except Exception as e:
            logger.warning(f"History not saved: {e!r}")

    def _log_job_submission(self, trial_info: dict):
        fidelity = trial_info["fidelity"]
        config_id = trial_info["config_id"]
        logger.info(
            f"INCUMBENT: {self.inc_id}  "
            f"SCORE: {self.inc_score}  "
            f"FIDELITY: {self.inc_fidelity}",
        )
        logger.info(f"Evaluating configuration {config_id} with fidelity {fidelity}")

    def _get_state(self):
        state = self.__dict__.copy()
        state.pop("optimizer")
        state.pop("f")
        return state

    def _save_state(self, save: bool = True):
        if not save:
            return
        # Get state
        state = self._get_state()
        # Write state to disk
        try:
            state_path = os.path.join(self.output_path, "qt.json")
            with open(state_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"State not saved: {e!r}")
        try:
            opt_path = os.path.join(self.output_path, "optimizer")
            self.optimizer.save(opt_path)
        except Exception as e:
            logger.warning(f"Optimizer state not saved: {e!r}")

    def save(self, incumbent: bool = True, history: bool = True, state: bool = True):
        logger.info("Saving current state to disk...")
        self._save_incumbent(incumbent)
        self._save_history(history)
        self._save_state(state)

    def load(self, path: str):
        logger.info(f"Loading state from {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        self.optimizer = Optimizer.load(os.path.join(self.output_path, "optimizer"))

    def run(
        self,
        task_info: dict | None = None,
        fevals: int | None = None,
        time_budget: float | None = None,
    ):
        """Run the tuner.

        Args:
            task_info (dict, optional): Additional information to pass to the objective function. Defaults to None.
            fevals (int, optional): Number of function evaluations to run. Defaults to None.
            time_budget (float, optional): Time budget in seconds. Defaults to None.

        Returns:
            np.ndarray: Trajectory of the incumbent scores.
            np.ndarray: Runtime of the incumbent evaluations.
            np.ndarray: History of all evaluations.
        """
        logger.info("Starting QuickTuner Run...")
        logger.info(f"QuickTuneTool will save results to {self.output_path}")

        self.start = time.time()
        while True:
            self.optimizer.ante()

            # ask for a new configuration
            trial = self.optimizer.ask()
            if trial is None:
                break
            _task_info = self._add_task_info(task_info)

            self._log_job_submission(trial)
            result = self.f(trial, task_info=_task_info)

            self._log_report(result)
            self.optimizer.tell(result)

            self.optimizer.post()
            if self._is_budget_exhausted(fevals, time_budget):
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

    def _log_report(self, reports):
        if isinstance(reports, dict):
            reports = [reports]

        inc_changed = False
        for report in reports:
            config_id = report["config_id"]
            score = report["score"]
            cost = report["cost"]
            fidelity = report["fidelity"]
            config = config_to_serializible_dict(report["config"])

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
                self.inc_info = report.get("info")
                inc_changed = True

            report["config"] = config
            self._update_trackers(
                self.inc_score,
                time.time() - self.start,
                report,
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
        out = {} if task_info is None else task_info.copy()
        out["output-path"] = self.output_path
        out["remaining-fevals"] = self._remaining_fevals
        out["remaining-time"] = self._remaining_time
        return out

    def get_incumbent(self):
        return (
            self.inc_id,
            self.inc_config,
            self.inc_score,
            self.inc_fidelity,
            self.inc_cost,
            self.inc_info,
        )
