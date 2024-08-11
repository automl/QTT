from pathlib import Path

import pandas as pd
import torch
from ConfigSpace import ConfigurationSpace
from torch.utils.data import Dataset

from qtt.utils import encode_config_space


class MetaDataset(Dataset):
    metafeat = None
    cost = None
    config_norm = None
    metafeat_norm = None

    def __init__(
        self,
        root: str | Path,
        standardize: bool = True,
        to_tensor: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.standardize = standardize
        self.to_tensor = to_tensor

        self.cs = ConfigurationSpace.from_json(self.root / "space.json")
        self.config = self._load_csv(self.root / "config.csv")
        self.curve = self._load_csv(self.root / "curve.csv")

        _path = self.root / "cost.csv"
        if _path.exists():
            self.cost = self._load_csv(_path)
        _path = self.root / "meta.csv"
        if _path.exists():
            self.metafeat = self._load_csv(_path)

        # preprocess
        self._preprocess_configs()
        self._preprocess_metafeat()
        self.curve.fillna(0, inplace=True)

    def _load_csv(self, path: Path):
        return pd.read_csv(path, index_col=0)

    def _preprocess_configs(self):
        NUM = self.config.select_dtypes(include=["number"]).columns.tolist()

        df = pd.get_dummies(self.config, prefix_sep="=", dtype=int)
        df.fillna(0, inplace=True)
        NON_NUM = [col for col in df.columns if col not in NUM]

        if self.standardize:
            mean = df.mean()
            std = df.std()
            # exclude non-numerical hyperparameters
            mean[NON_NUM] = 0
            std[NON_NUM] = 1
            df = df - mean / std
            # save mean and std for later use
            self.config_norm = pd.DataFrame([mean, std], index=["mean", "std"])

        one_hot, _ = encode_config_space(self.cs)
        self.config = df[one_hot]

        self.config = df.astype(float)

    def _preprocess_metafeat(self):
        if self.metafeat is None:
            return
        mean = self.metafeat.mean()
        std = self.metafeat.std()
        # remove constant columns
        mean[std == 0] = 0
        std[std == 0] = 1
        self.metafeat = (self.metafeat - mean) / std

        # save mean and std for later use
        self.metafeat_norm = pd.DataFrame([mean, std], index=["mean", "std"])

    def __len__(self):
        return len(self.config)

    def __getitem__(self, idx):
        out = {
            "config": self.config.iloc[idx].values,
            "curve": self.curve.iloc[idx].values,
        }

        if self.cost is not None:
            out["cost"] = self.cost.iloc[idx].values
        if self.metafeat is not None:
            out["metafeat"] = self.metafeat.iloc[idx].values

        if self.to_tensor:
            out = {k: torch.tensor(v, dtype=torch.float) for k, v in out.items()}

        return out

    def get_config_norm(self):
        return self.config_norm

    def get_metafeat_norm(self):
        return self.metafeat_norm

    def get_config_dim(self):
        return len(self.config.columns)

    def get_metafeat_dim(self):
        if self.metafeat is None:
            return 0
        return len(self.metafeat.columns)
