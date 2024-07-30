import os
from pathlib import Path
import pandas as pd

from .dataset import MetaDataset

def get_meta_dataset(path: str | Path):
    path = Path(path)
    configs = pd.read_csv(path / 'configs.csv', index_col=0)
    configs.fillna(0, inplace=True)

    metafeat = pd.read_csv(path / 'metafeat.csv', index_col=0)
    metafeat.fillna(0, inplace=True)
    
    scores = pd.read_csv(path / 'scores.csv', index_col=0)
    scores.fillna(0, inplace=True)

    cost = pd.read_csv(path / 'cost.csv', index_col=0)
    cost.fillna(0, inplace=True)

    return configs, metafeat, scores, cost