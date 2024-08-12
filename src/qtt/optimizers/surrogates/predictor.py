import abc
from pathlib import Path


class Predictor(abc.ABC):
    @abc.abstractmethod
    def predict():
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str | Path, name: str = "predictor.ckpt"):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str | Path, name: str = "predictor.ckpt"):
        raise NotImplementedError
