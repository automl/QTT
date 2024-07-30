import os

from ConfigSpace import ConfigurationSpace
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..config.utils import one_hot_encode_config_space, SORT_MTHDS


class MetaDataset(Dataset):
    config_norm = None
    metafeat_norm = None
    def __init__(
        self,
        root: str,
        cs: ConfigurationSpace,
        standardize: bool = True,
        sort_mthd: str = "auto",
        to_tensor: bool = True,
    ):
        super().__init__()
        assert sort_mthd in SORT_MTHDS, "Invalid sort option"
        self.root = root
        self.cs = cs
        self.standardize = standardize
        self.sort_mthd = sort_mthd
        self.to_tensor = to_tensor

        self.configs = pd.read_csv(os.path.join(self.root, "configs.csv"), index_col=0)
        self.scores = pd.read_csv(os.path.join(self.root, "score.csv"), index_col=0)
        self.scores.fillna(0, inplace=True)

        self.cost = pd.read_csv(os.path.join(self.root, "cost.csv"), index_col=0)
        self.metafeat = pd.read_csv(os.path.join(self.root, "meta.csv"), index_col=0)

        self._preprocess_configs()
        self._preprocess_metafeat()

    def _preprocess_configs(self):
        NUM = self.configs.select_dtypes(include=["number"]).columns.tolist()

        df = pd.get_dummies(self.configs, prefix_sep="=", dtype=int)
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

        if self.sort_mthd:
            one_hot, _ = one_hot_encode_config_space(self.cs, self.sort_mthd)
            df = df[one_hot]

        self.configs = df.astype(float)

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
        return len(self.configs)

    def __getitem__(self, idx):
        config = self.configs.iloc[idx].values
        score = self.scores.iloc[idx].values
        metafeat = self.metafeat.iloc[idx].values
        cost = self.cost.iloc[idx].values

        if self.to_tensor:
            config = torch.tensor(config, dtype=torch.float)
            score = torch.tensor(score, dtype=torch.float)
            metafeat = torch.tensor(metafeat, dtype=torch.float)
            cost = torch.tensor(cost, dtype=torch.float)

        return {"config": config, "score": score, "metafeat": metafeat, "cost": cost}

    def get_config_norm(self):
        """
        
        """
        return self.config_norm
    
    def get_metafeat_norm(self):
        return self.metafeat_norm

    def get_config_dim(self):
        return len(self.configs.columns)
    
    def get_metafeat_dim(self):
        if self.metafeat is None:
            return 0
        return len(self.metafeat.columns)

    def get_config_order(self):
        return self.configs.columns.tolist()
    
    def get_metafeat_order(self):
        if self.metafeat is None:
            return []
        return self.metafeat.columns.tolist()

    # def get_dataset_info(self):
    #     info = {}
    #     info["num-samples"] = len(self.configs)
    #     info["config-dim"] = self.get_config_dim()
    #     info["config-order"] = self.get_config_order()
    #     if self.metafeat is not None:
    #         info["metafeat-dim"] = self.get_metafeat_dim()
    #         info["metafeat-order"] = self.get_metafeat_order()
    #     return info

    # def save_norm_to_file(self, path="./"):
    #     if self.metafeat_norm is not None:
    #         self.metafeat_norm.to_csv(os.path.join(path, "metafeat_norm.csv"))
    #     if self.config_norm is not None:
    #         self.config_norm.to_csv(os.path.join(path, "config_norm.csv"))