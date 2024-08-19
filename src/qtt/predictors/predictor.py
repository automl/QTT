import abc
from pathlib import Path

import numpy as np


class Predictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, X) -> np.ndarray: ...

    @abc.abstractmethod
    def fit(self, X): ...

    @abc.abstractmethod
    def save(self, path: str | Path, name: str = "filename.ckpt"): ...

    @abc.abstractmethod
    def load(self, path: str | Path, name: str = "filename.ckpt"): ...
