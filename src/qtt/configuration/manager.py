from typing import List, Optional

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace

from .utils import get_one_hot_encoding, SORT_MTHDS


class ConfigManager:
    def __init__(
        self,
        cs: ConfigurationSpace,
        std_data: Optional[pd.DataFrame] = None,
        sort_mthd: str = "alphabet",
    ):
        assert sort_mthd in SORT_MTHDS, "Invalid sort option"
        self.cs = cs
        self.std_data = std_data
        self.sort_mthd = sort_mthd

        self.one_hot = get_one_hot_encoding(cs, sort_mthd)

    def sample_configuration(self, n: int):
        return self.cs.sample_configuration(n)

    def preprocess_configurations(self, configurations: List[Configuration]):
        encoded_configs = []
        for config in configurations:
            config = dict(config)
            enc_config = dict()
            for hp in self.one_hot:
                # categorical hyperparameters
                if len(hp.split(":")) > 1:
                    key, choice = hp.split(":")
                    val = 1 if config.get(key) == choice else 0
                else:
                    val = config.get(hp, 0) # NUM
                    if isinstance(val, bool): # BOOL
                        val = int(val)
                enc_config[hp] = val
            encoded_configs.append(enc_config)

        df = pd.DataFrame(encoded_configs)

        # standardize numerical hyperparameters
        if self.std_data is not None:
            mean = self.std_data.loc["mean"]
            std = self.std_data.loc["std"]
            std[std == 0] = 1
            df = (df - mean) / std
        
        # assert order of columns matches one-hot-encoded columns
        df = df[self.one_hot]
        return df.to_numpy()

    def get_one_hot_encoding(self):
        return self.one_hot

    @staticmethod
    def config_id(config: Configuration) -> str:
        """
        Generates a unique identifier for a configuration object.

        Args:
            config (Configuration): The configuration object.
        Returns:
            str: The unique identifier for the configuration object.
        """
        return str(hash(frozenset(dict(config).items())))
