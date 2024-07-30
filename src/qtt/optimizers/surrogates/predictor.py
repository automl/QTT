import abc
import torch

class Predictor(torch.nn.Module, abc.ABC):
    meta_trained: bool = False
    
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError
