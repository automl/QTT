from abc import ABC, abstractmethod

class BaseOptimizer(ABC):

    @abstractmethod
    def ask(self) -> dict:
        """Get the next pipeline to evaluate"""
    
    @abstractmethod
    def tell(self, results: dict | list[dict]):
        """Return the results of the evaluation"""
    
    @abstractmethod
    def save(self):
        """Save the current state of the optimizer"""
    
    @abstractmethod
    def load(self):
        """Load a checkpoint of the optimizer"""
    
    def ante(self):
        """Pre processing before each iteration"""
        pass
    
    def post(self):
        """Post processing after each iteration"""
        pass

