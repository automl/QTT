import os

from ConfigSpace import ConfigurationSpace
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..config.utils import encode_config_space


class MetaDataset(Dataset):
    pipeline_norm = None
    metafeat_norm = None

    def __init__(
        self,
        root: str,
        cs: ConfigurationSpace,
        standardize: bool = True,
        to_tensor: bool = True,
    ):
        super().__init__()
        self.root = root
        self.cs = cs
        self.standardize = standardize
        self.to_tensor = to_tensor


        self.config = self.load_csv("config.csv")
        self.curve = self.load_csv("score.csv")
        self.cost = self.load_csv("cost.csv")
        self.metafeat = self.load_csv("meta.csv")

        # preprocess
        self._preprocess_configs()
        self._preprocess_metafeat()
        self.curve.fillna(0, inplace=True)

    def load_csv(self, filename: str):
        path = os.path.join(self.root, filename)
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
            self.pipeline_norm = pd.DataFrame([mean, std], index=["mean", "std"])

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
        config = self.config.iloc[idx].values
        score = self.curve.iloc[idx].values
        metafeat = self.metafeat.iloc[idx].values
        cost = self.cost.iloc[idx].values

        if self.to_tensor:
            config = torch.tensor(config, dtype=torch.float)
            score = torch.tensor(score, dtype=torch.float)
            metafeat = torch.tensor(metafeat, dtype=torch.float)
            cost = torch.tensor(cost, dtype=torch.float)

        return {"config": config, "curve": score, "metafeat": metafeat, "cost": cost}

    def get_config_norm(self):
        """ """
        return self.pipeline_norm

    def get_metafeat_norm(self):
        return self.metafeat_norm

    def get_config_dim(self):
        return len(self.config.columns)

    def get_metafeat_dim(self):
        if self.metafeat is None:
            return 0
        return len(self.metafeat.columns)

    def get_config_order(self):
        return self.config.columns.tolist()

    def get_metafeat_order(self):
        if self.metafeat is None:
            return []
        return self.metafeat.columns.tolist()
