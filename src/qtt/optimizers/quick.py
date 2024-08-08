import logging
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from ConfigSpace import Configuration, ConfigurationSpace
from scipy.stats import norm

from ..data import MetaDataset
from ..utils import encode_config_space, fix_random_seeds, set_logger_verbosity
from .optimizer import BaseOptimizer
from .surrogates import CostPredictor, DyHPO, Predictor

logger = logging.getLogger(__name__)


ACQ_FN = [
    "ei",
    "ucb",
    "thompson",
    "exploit",
]

IGNORE_SAVE_LOAD_VARS = [
    "cs",
    "cs_meta",
    "dev",
    "cost_predictor",
    "perf_predictor",
]


class QuickOptimizer(BaseOptimizer):
    def __init__(
        self,
        cs: ConfigurationSpace,
        meta: ConfigurationSpace | None = None,
        *,
        init_steps: int = 10,
        cost_aware: bool = False,
        acq_fn: str = "ei",
        explore_factor: float = 0.0,
        n_iter_no_change: int | None = None,
        tol: float = 1e-4,
        score_thresh: float = 0.0,
        #
        device: str | None = None,
        seed: int | None = None,
        verbosity: int = 2,
    ):
        set_logger_verbosity(verbosity, logger)
        self.verbosity = verbosity

        if seed is not None:
            fix_random_seeds(seed)
        self.seed = seed

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)

        # configuration space
        self.cs = cs
        self.meta = meta
        self.max_fidelity = int(cs["max_fidelity"].default_value)

        cs_enc, cs_splits = encode_config_space(cs)
        self.cs_encoding = cs_enc
        self.cs_splits = cs_splits
        if meta is not None:
            cs_meta_enc, _ = encode_config_space(meta)
            self.cs_meta_enc = cs_meta_enc
        else:
            self.cs_meta_enc = None

        self.metafeat: np.ndarray | None = None
        self.config_norm: pd.DataFrame | None = None
        self.metafeat_norm: pd.DataFrame | None = None

        # optimizer related parameters
        assert acq_fn in ACQ_FN, f"invalid acq-fn: {acq_fn}, choose from {ACQ_FN}"
        self.acq_fn = acq_fn
        self.explore_factor = explore_factor
        self.cost_aware = cost_aware
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.scr_thr = score_thresh

        # predictors
        model_kwargs = self.get_model_kwargs()
        self.perf_predictor = DyHPO(**model_kwargs)
        self.perf_predictor.to(self.dev)
        self.cost_predictor: Predictor | None = None
        if self.cost_aware:
            self.cost_predictor = CostPredictor(**model_kwargs)
            self.cost_predictor.to(self.dev)
        self.model_kwargs = model_kwargs

        # trackers
        self.init_steps = init_steps
        self.iteration = 0
        self.ask_count = 0
        self.tell_count = 0
        self.init_count = 0
        self.eval_count = 0
        self.candidates: list[Configuration] = []
        self.configs = []
        self.evaled = set()
        self.stoped = set()
        self.failed = set()
        self.history = []

        self._setup_run = False

    def get_model_kwargs(self) -> dict:
        config_dim = [len(s) for s in self.cs_splits]
        curve_dim = self.max_fidelity
        meta_dim = len(self.cs_meta_enc) if self.cs_meta_enc is not None else None
        kwargs = {
            "in_dim": config_dim,
            "in_curve_dim": curve_dim,
            "in_metafeat_dim": meta_dim,
        }
        return kwargs

    def _config_to_vector(self, configs: list[Configuration]) -> np.ndarray:
        df =pd.DataFrame(configs, columns=self.cs_encoding)
        df.fillna(0, inplace=True)

        if self.config_norm is not None:
            mean = self.config_norm.iloc("mean")
            std = self.config_norm.iloc("std")
            df = (df - mean) / std

        df = df[self.cs_encoding]  # assert columns order
        return df.to_numpy(dtype=float)

    def setup(
        self,
        n: int,
        metafeat: Mapping[str, int | float] | None = None,
    ):
        self.N = n
        self.fidelities: np.ndarray = np.zeros(n, dtype=int)
        self.curves: np.ndarray = np.zeros((n, self.max_fidelity), dtype=float)
        self.costs: np.ndarray = np.full(n, np.nan, dtype=float)
        if self.n_iter_no_change is not None:
            self._score_history = np.zeros((n, self.n_iter_no_change), dtype=float)

        if self.seed is not None:
            self.cs.seed(self.seed)
        self.candidates = self.cs.sample_configuration(n)
        self.configs = self._config_to_vector(self.candidates)

        if self.cs_meta_enc is not None and metafeat is None:
            raise ValueError("metafeatures not given")
        elif self.cs_meta_enc is None and metafeat is not None:
            logger.warning("metafeatures given but not used in this optimization")
        elif self.cs_meta_enc is not None and metafeat is not None:
            _meta = [v for k, v in metafeat.items() if k in self.cs_meta_enc]
            _meta = np.array(_meta)

            if self.metafeat_norm is not None:
                mean = self.metafeat_norm.loc["mean"]
                std = self.metafeat_norm.loc["std"]
                std[std == 0] = 1
                _meta = (_meta - mean) / std
                self.metafeat = _meta

        self._setup_run = True

    def _get_history_data(self):
        idx = np.array(list(self.evaled), dtype=int)
        config = self.configs[idx]
        fidelity = self.fidelities[idx] / self.max_fidelity
        curve = self.curves[idx]

        config = torch.tensor(config, dtype=torch.float, device=self.dev)
        fidelity = torch.tensor(fidelity, dtype=torch.float, device=self.dev)
        curve = torch.tensor(curve, dtype=torch.float, device=self.dev)
        metafeat = None
        if self.metafeat is not None:
            metafeat = torch.tensor(self.metafeat, dtype=torch.float, device=self.dev)

        data = {
            "config": config,
            "fidelity": fidelity,
            "curve": curve,
            "metafeat": metafeat,
        }
        return data

    def _get_candidate_data(self):
        config = torch.tensor(self.configs, dtype=torch.float, device=self.dev)
        fidelity = torch.tensor(self.fidelities, dtype=torch.float, device=self.dev)
        fidelity /= self.max_fidelity
        curve = torch.tensor(self.curves, dtype=torch.float, device=self.dev)
        metafeat = None
        if self.metafeat is not None:
            metafeat = torch.tensor(self.metafeat, dtype=torch.float, device=self.dev)

        data = {
            "config": config,
            "fidelity": fidelity,
            "curve": curve,
            "metafeat": metafeat,
        }
        return data

    def _predict(self):
        test_data = self._get_candidate_data()

        pred = self.perf_predictor.predict(test_data)  # type: ignore
        pred_mean, pred_std = pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()

        cost = self.costs
        if self.cost_predictor is not None:
            pred_cost = self.cost_predictor(**test_data)
            pred_cost = pred_cost.cpu().numpy()
            mask = np.isnan(cost)
            cost[mask] = pred_cost[mask]

        return pred_mean, pred_std, cost

    def _calc_acq_val(self, mean, std, y_max):
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
        # max score per fidelity
        y_max = self.curves.max(axis=0)
        y_max[y_max == 0] = y_max.max()

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
        return ranks[-1]

    def ask(self) -> dict:
        if not self._setup_run:
            raise RuntimeError("Call setup() before ask()")

        self.ask_count += 1
        if len(self.evaled) < self.init_steps:
            left = set(range(self.N)) - self.evaled - self.failed - self.stoped
            index = left.pop()
            fidelity = 1
        else:
            index = self._ask()
            fidelity = self.fidelities[index] + 1

        return {
            "config_id": index,
            "config": self.candidates[index],
            "fidelity": fidelity,
        }

    def tell(self, result: dict | list):
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

        if self.n_iter_no_change is not None:
            if not np.any(self._score_history[index] < (score - self.tol)):
                self.stoped.add(index)
            self._score_history[index][fidelity % self.n_iter_no_change] = score

    def ante(self):
        pass

    def post(self):
        pass

    def update(self):
        train_data = self._get_history_data()
        self.perf_predictor.update(train_data)
        # if self.cost_predictor is not None:
        #     self.cost_predictor.update(train_data)

    def fit(self, data: MetaDataset, **kwargs):
        # fit the predictors
        self.perf_predictor.fit(data, **kwargs)
        if self.cost_predictor is not None:
            self.cost_predictor.fit(data, **kwargs)

        self.config_norm = data.get_config_norm()
        self.metafeat_norm = data.get_metafeat_norm()

    def save(self, path: str | Path = ""):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        ckp = dict(vars(self))

        for key in IGNORE_SAVE_LOAD_VARS:
            ckp.pop(key)

        self.cs.to_yaml(path / "cs.yaml")
        if self.meta is not None:
            self.meta.to_yaml(path / "meta.yaml")

        self.perf_predictor.save(path)
        if self.cost_predictor is not None:
            self.cost_predictor.save(path)

        # torch.save(ckp, path / "checkpoint.pth")
        with open(path / "quick.pkl", "wb") as f:
            pickle.dump(ckp, f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        assert path.exists(), f"Path {path} does not exist"

        # load configspace(s)
        cs = ConfigurationSpace.from_yaml(path / "cs.yaml")
        cs_meta = None
        if (path / "meta.yaml").exists():
            cs_meta = ConfigurationSpace.from_yaml(path / "meta.yaml")

        # create instance
        opt = cls(cs, cs_meta)

        # load checkpoint
        with open(path / "quick.pkl", "rb") as f:
            checkpoint: dict = pickle.load(f)
        # update instance attributes
        for key, value in vars(opt).items():
            if key in IGNORE_SAVE_LOAD_VARS:
                continue
            if key in checkpoint:
                setattr(opt, key, checkpoint.get(key, value))
            else:
                logger.info(f"'{key}' not in checkpoint")
        return opt
