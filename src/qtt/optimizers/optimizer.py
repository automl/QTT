from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """Base class for optimizers

    An optimizer is an object that can be asked for a trail using `ask` and a
    `tell` to inform the optimizer of the report from that trial.
    """
    @abstractmethod
    def ask(self) -> dict:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A config to sample.
        """
        ...
    
    @abstractmethod
    def tell(self, results: dict | list[dict]):
        """Tell the optimizer the result for an asked trial.

        Args:
            result: The result for a trial
        """
    
    @abstractmethod
    def save(self):
        """Save the current state of the optimizer"""
    
    @abstractmethod
    def load(self):
        """Load a checkpoint of the optimizer"""

    @abstractmethod
    def ante(self):
        """Pre processing before each iteration"""

    @abstractmethod
    def post(self):
        """Post processing after each iteration"""

