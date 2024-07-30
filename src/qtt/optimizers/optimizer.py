from abc import ABC, abstractmethod

class BaseOptimizer(ABC):

    @abstractmethod
    def ask(self) -> dict:
        """Get the next configuration to evaluate"""
    
    @abstractmethod
    def tell(self, results: dict | list[dict]):
        """Return the results of the evaluation"""

