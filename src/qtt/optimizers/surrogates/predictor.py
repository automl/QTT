import abc
import torch
from pathlib import Path


class Predictor(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str | Path, name: str = "predictor.pth"):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str | Path, name: str = "predictor.pth"):
        raise NotImplementedError
