import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ConfigSpace import Configuration, ConfigurationSpace
from scipy.stats import norm

from qtt.config.utils import one_hot_encode_config_space
from qtt.utils import fix_random_seeds, set_logger_verbosity

from ..data.dataset import MetaDataset
from .optimizer import BaseOptimizer
from .surrogates import CostPredictor, DyHPO
from .surrogates.predictor import Predictor

logger = logging.getLogger("QuickOptimizer")


ACQ_FN = "ei", "ucb", "thompson", "exploit"


class QuickOptimizer(BaseOptimizer):
    config_norm: pd.DataFrame | None = None
    metafeat_norm: pd.DataFrame | None = None
    perf_predictor: Predictor
    cost_predictor: Predictor | None

    def __init__(
        self,
        cs: ConfigurationSpace,
        srt_mthd: str = "auto",
        metafeat_list: list[str] | None = None,
        *,
        init_steps: int = 10,
        cost_aware: bool = False,
        acq_fn: str = "ei",
        explore_factor: float = 0.0,
        n_iter_no_change: int | None = None,
        tol: float = 1e-4,
        score_thresh: float = 0.0,
        #
        max_fidelity: int | None = None,
        predictor_kwargs: dict | None = None,
        #
        device: str | None = None,
        seed: int | None = None,
        verbosity: int = 2,
    ):
        super().__init__()
        # setup
        set_logger_verbosity(verbosity, logger)
        self.verbosity = verbosity

        if seed is not None:
            fix_random_seeds(seed)
        self.seed = seed

        self.device = device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)

        # configuration space
        self.cs = cs
        self.srt_mthd = srt_mthd

        self.max_fidelity = int(cs["max_fidelity"].default_value)
        if max_fidelity is not None:
            self.max_fidelity = max_fidelity

        # optimizer related parameters
        assert (
            acq_fn in ACQ_FN
        ), f"Invalid acquisition function: {acq_fn}, choose from {ACQ_FN}"
        self.acq_fn = acq_fn
        self.explore_factor = explore_factor
        self.cost_aware = cost_aware
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.scr_thr = score_thresh
        self.metafeat_lst = metafeat_list
        self.metafeat = None

        # predictors
        one_hot, splits = one_hot_encode_config_space(cs, srt_mthd)
        if predictor_kwargs is None:
            config_dim = [len(s) for s in splits]
            curve_dim = self.max_fidelity
            meta_dim = len(self.metafeat_lst) if self.metafeat_lst is not None else None
            predictor_kwargs = {
                "in_dim": config_dim,
                "in_curve_dim": curve_dim,
                "in_metafeat_dim": meta_dim,
            }
        self.cs_one_hot = one_hot
        self.cs_splits = splits

        self.perf_predictor = DyHPO(**predictor_kwargs)
        self.cost_predictor = None
        if self.cost_aware:
            self.cost_predictor = CostPredictor(**predictor_kwargs)
        self.predictor_kwargs = predictor_kwargs

        # trackers
        self.iteration = 0
        self.ask_count = 0
        self.tell_count = 0
        self._init_count = 0
        self.init_steps = init_steps
        self.candidates: list[Configuration] = []
        self.configs = []
        self.evaled = set()
        self.stoped = set()
        self.failed = set()
        self.history = []

        self._init = False

    def _configuration_to_vector(self, cfg_lst: list[Configuration]) -> np.ndarray:
        encoded_configs = []
        for config in cfg_lst:
            config = dict(config)
            enc_config = dict()
            for hp in self.cs_one_hot:
                # categorical hyperparameters
                if len(hp.split(":")) > 1:
                    key, choice = hp.split(":")
                    val = 1 if config.get(key) == choice else 0
                else:
                    val = config.get(hp, 0)
                    if isinstance(val, bool):
                        val = int(val)
                enc_config[hp] = val
            encoded_configs.append(enc_config)

        df = pd.DataFrame(encoded_configs)

        if self.config_norm is not None:
            mean = self.config_norm.loc["mean"]
            std = self.config_norm.loc["std"]
            std[std == 0] = 1
            df = (df - mean) / std

        df = df[self.cs_one_hot]
        return df.to_numpy()

    def setup(
        self,
        n: int,
        metafeat: dict[str, int | float] | None = None,
    ):
        self.N = n
        self.fidelities: np.ndarray = np.zeros(n, dtype=np.int64)
        self.scores: np.ndarray = np.zeros((n, self.max_fidelity), dtype=np.float64)
        self.costs: np.ndarray = np.full(n, np.nan, dtype=np.float64)
        if self.n_iter_no_change is not None:
            self._score_history = np.zeros((n, self.n_iter_no_change), dtype=np.float64)

        self.candidates = self.cs.sample_configuration(n)
        self.configs = self._configuration_to_vector(self.candidates)

        if metafeat is not None:
            assert (
                self.metafeat_lst is not None
            ), "metafeat_list not provided during init"
            _meta = [v for k, v in metafeat.items() if k in self.metafeat_lst]
            _meta = np.array(_meta)

            if self.metafeat_norm is not None:
                mean = self.metafeat_norm.loc["mean"]
                std = self.metafeat_norm.loc["std"]
                std[std == 0] = 1
                _meta = (_meta - mean) / std
                self.metafeat = _meta

        self._init = True

    def _get_train_data(self):
        config, fidelity, curve, target = [], [], [], []
        for i in self.evaled:
            config.append(self.configs[i])
            fidelity.append(self.fidelities[i])
            _scores = self.scores[i]
            y = _scores[-1]
            _scores[-1] = 0.0
            curve.append(_scores)
            target.append(y)

        config = torch.tensor(config, dtype=torch.float, device=self.dev)
        fidelity = torch.tensor(fidelity, dtype=torch.float, device=self.dev)
        fidelity /= self.max_fidelity
        curve = torch.tensor(curve, dtype=torch.float, device=self.dev)
        target = torch.tensor(target, dtype=torch.float, device=self.dev)
        metafeat = None
        if self.metafeat_lst is not None:
            metafeat = torch.tensor(
                self.metafeat_lst, dtype=torch.float, device=self.dev
            )

        data = {
            "config": config,
            "fidelity": fidelity,
            "curve": curve,
            "target": target,
            "metafeat": metafeat,
        }
        return data

    def _get_test_data(self):
        config = torch.tensor(self.configs, dtype=torch.float, device=self.dev)
        fidelity = torch.tensor(self.fidelities, dtype=torch.float, device=self.dev)
        fidelity /= self.max_fidelity
        curve = torch.tensor(self.scores, dtype=torch.float, device=self.dev)
        metafeat = None
        if self.metafeat_lst is not None:
            metafeat = torch.tensor(
                self.metafeat_lst, dtype=torch.float, device=self.dev
            )

        data = {
            "config": config,
            "fidelity": fidelity,
            "curve": curve,
            "metafeat": metafeat,
        }
        return data

    def _predict(self):
        train_data = self._get_train_data()
        test_data = self._get_test_data()

        pred_mean, pred_std = self.perf_predictor(train_data, test_data)  # type: ignore
        pred_mean, pred_std = pred_mean.numpy(), pred_std.numpy()

        cost = self.costs
        if self.cost_predictor is not None:
            pred_cost = self.cost_predictor(**test_data)
            pred_cost = pred_cost.numpy()
            mask = np.isnan(cost)
            cost[mask] = pred_cost[mask]

        return pred_mean, pred_std, cost

    def _acq_fn(self, mean, std, y_max):
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

    def _find_most_promising_configs(self, mean, std, cost) -> list[int]:
        # max score per fidelity
        y_max = self.scores.max(axis=0)
        y_max[y_max == 0] = y_max.max()

        next_fidelitys = np.minimum(self.fidelities + 1, self.max_fidelity)
        y_max = y_max[next_fidelitys - 1]

        acq_values = self._acq_fn(mean, std, y_max)
        if self.cost_aware:
            cost += 1e-6  # avoid division by zero
            acq_values /= cost

        return np.argsort(acq_values).tolist()

    def _ask(self):
        pred_mean, pred_std, cost = self._predict()
        ranks = self._find_most_promising_configs(pred_mean, pred_std, cost)
        ranks = [r for r in ranks if r not in self.stoped | self.failed]
        return ranks[-1]

    def ask(self) -> dict:
        if not self._init:
            raise RuntimeError("Call setup() before ask()")

        self.ask_count += 1
        if len(self.evaled) < self.init_steps:
            rest = set(range(self.N)) - self.evaled - self.failed - self.stoped
            index = rest.pop()
            index = self._init_count
            fidelity = 1
        else:
            index = self._ask()
            fidelity = self.fidelities[index] + 1

        return {
            "config_id": index,
            "config": self.candidates[index],
            "fidelity": fidelity,
        }

    def tell(self, result: dict):
        self.tell_count += 1

        index = result["config_id"]
        score = result["score"]
        fidelity = result["fidelity"]
        cost = result["cost"]
        status = result["status"]

        if not status:
            self.failed.add(index)
            return

        if score >= 1.0 - self.scr_thr or fidelity == self.max_fidelity:
            self.stoped.add(index)

        self.scores[index, fidelity - 1] = score
        self.fidelities[index] = fidelity
        self.costs[index] = cost
        self.history.append(result)

        if self.n_iter_no_change is not None:
            if not np.any(self._score_history[index] < (score - self.tol)):
                self.stoped.add(index)
            self._score_history[index][fidelity % self.n_iter_no_change] = score

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

        # save configspace
        self.cs.to_yaml(path / "space.yaml")

        kwargs = {
            "srt_mthd": self.srt_mthd,
            "metafeat_list": self.metafeat_lst,
            "init_steps": self.init_steps,
            "cost_aware": self.cost_aware,
            "acq_fn": self.acq_fn,
            "explore_factor": self.explore_factor,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
            "score_thresh": self.scr_thr,
            "max_fidelity": self.max_fidelity,
            "predictor_kwargs": self.predictor_kwargs,
            "device": self.device,
            "seed": self.seed,
            "verbosity": self.verbosity,
        }
        # save kwargs
        with open(path / "kwargs.yaml", "w") as f:
            yaml.dump(kwargs, f)

        # save predictors
        torch.save(self.perf_predictor.state_dict(), path / "perf_predictor.pth")
        if self.cost_predictor is not None:
            torch.save(self.cost_predictor.state_dict(), path / "cost_predictor.pth")

        # save normalization data
        if self.config_norm is not None:
            self.config_norm.to_csv(path / "config_norm.csv")
        if self.metafeat_norm is not None:
            self.metafeat_norm.to_csv(path / "metafeat_norm.csv")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)

        # load kwargs
        with open(path / "kwargs.yaml", "r") as f:
            kwargs = yaml.load(f, Loader=yaml.SafeLoader)
        # load configspace
        cs = ConfigurationSpace.from_yaml(path / "space.yaml")

        # create instance
        opt = cls(cs=cs, **kwargs)

        # load predictors
        opt.perf_predictor.load_state_dict(torch.load(path / "perf_predictor.pth"))
        if opt.cost_predictor is not None:
            opt.cost_predictor.load_state_dict(torch.load(path / "cost_predictor.pth"))

        # load normalization data
        if (path / "config_norm.csv").exists():
            opt.config_norm = pd.read_csv(path / "config_norm.csv", index_col=0)
        if (path / "metafeat_norm.csv").exists():
            opt.metafeat_norm = pd.read_csv(path / "metafeat_norm.csv", index_col=0)

        return opt
