from typing import List

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace

NUM_HP = [
    "bss_reg",
    "clip_grad",
    "cotuning_reg",
    "cutmix",
    "decay_rate",
    "delta_reg",
    "drop",
    "layer_decay",
    "lr",
    "mixup",
    "mixup_prob",
    "momentum",
    "pct_to_freeze",
    "smoothing",
    "sp_reg",
    "warmup_lr",
    "weight_decay",
]


class ConfigManager:
    def __init__(
        self,
        space: ConfigurationSpace,
        meta_data: pd.DataFrame,
    ):
        self.space = space
        self.md = meta_data

    def sample_configuration(self, n: int):
        """
        Samples n configurations from the configuration space.

        Args:
            n (int): The number of configurations to sample.

        Returns:
            List[Configuration]: A list of sampled configurations.
        """
        return self.space.sample_configuration(n)

    def preprocess_configurations(
        self,
        configurations: List[Configuration],
        standardize: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocesses a list of configurations by encoding categorical and numerical hyperparameters,
        and optionally standardizing numerical hyperparameters.

        Args:
            configurations (List[Configuration]): A list of configurations to preprocess.
            standardize (bool, optional): Whether to standardize numerical hyperparameters. Defaults to False.

        Returns:
            pd.DataFrame: The preprocessed configurations as a pandas DataFrame.
        """
        one_hot_hp = self.md.columns

        encoded_configs = []
        for config in configurations:
            enc_config = dict()
            for hp in one_hot_hp:
                # categorical hyperparameters
                if len(hp.split(":")) > 1:
                    key, choice = hp.split(":")
                    val = 1 if config.get(key) == choice else 0
                else:
                    # numerical hyperparameters
                    val = config.get(hp, 0)
                    # boolean hyperparameters
                    if isinstance(val, bool):
                        val = int(val)
                    # not-active (conditional or numerical) hyperparameters
                    elif val == "None":
                        val = 0
                enc_config[hp] = val
            encoded_configs.append(enc_config)

        df = pd.DataFrame(encoded_configs)

        # reorder columns to match the order of the metadataset
        df = df[self.md.columns]

        # standardize numerical hyperparameters
        if standardize:
            mean = self.md.loc["mean"]
            std = self.md.loc["std"]
            df[NUM_HP] = (df[NUM_HP] - mean[NUM_HP]) / std[NUM_HP]

        return df

    @staticmethod
    def config_id(config: Configuration) -> str:
        """
        Generates a unique identifier for a configuration object.

        Args:
            config (Configuration): The configuration object.
        Returns:
            str: The unique identifier for the configuration object.
        """
        return str(hash(frozenset(config.get_dictionary().items())))
