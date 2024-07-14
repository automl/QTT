from abc import ABC, abstractmethod

class BaseOptimizer(ABC):

    @abstractmethod
    def ask(self) -> dict:
        """Suggests the next configuration to evaluate"""
    
    @abstractmethod
    def tell(self, results: dict | list[dict]):
        """Processes the results of the evaluation of a configuration"""
    
    def ante(self):
        pass

    def inter(self):
        pass
    
    def post(self):
        pass

    def reset(self):
        pass
    
    def finished(self):
        raise NotImplementedError()
