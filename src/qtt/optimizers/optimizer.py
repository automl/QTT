import logging
import os
import pickle

from ..utils import setup_outputdir

logger = logging.getLogger(__name__)


class Optimizer:
    """Base class for all optimizers.

    Args:
        path (str):
            Directory location to store all outputs.
            If None, a new unique time-stamped directory is chosen.
        name (str):
            Name of the subdirectory inside path where model will be saved.
            The final model directory will be os.path.join(path, name)
            If None, defaults to the model's class name: self.__class__.__name__
    """

    model_file_name = "model.pkl"

    def __init__(
        self,
        name: str | None = None,
        path: str | None = None,
    ):
        if name is None:
            self.name = self.__class__.__name__
            logger.info(
                f"No name was specified for model, defaulting to class name: {self.name}",
            )
        else:
            self.name = name

        if path is None:
            self.path = setup_outputdir(path=self.name.lower())
            logger.info(
                f"No path was specified for predictor, defaulting to: {self.path}",
            )
        else:
            self.path = setup_outputdir(path)

        self._is_initialized = False
    
    def ante(self):
        """This method is intended for the use with a tuner.
        It allows to perform some pre-processing steps before each ask."""
        pass

    def post(self):
        """This method is intended for the use with a tuner.
        It allows to perform some post-processing steps after each tell."""
        pass

    def ask(self) -> dict:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A config to sample.
        """
        raise NotImplementedError
    
    def tell(self, report: dict | list[dict]):
        """Tell the optimizer the result for an asked trial.

        Args:
            report (dict): The result for a trial
        """
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """
        Loads the model from disk to memory.

        Args:
            path (str):
                Path to the saved model, minus the file name.
                This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
                The model file is typically located in os.path.join(path, cls.model_file_name).
            reset_paths (bool):
                Whether to reset the self.path value of the loaded model to be equal to path.
                It is highly recommended to keep this value as True unless accessing the original self.path value is important.
                If False, the actual valid path and self.path may differ, leading to strange behaviour and potential exceptions if the model needs to load any other files at a later time.
            verbose (bool):
                Whether to log the location of the loaded file.

        Returns:
            model (Optimizer):
                Loaded model object.
        """
        file_path = os.path.join(path, cls.model_file_name)
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        if reset_paths:
            model.path = path
        if verbose:
            logger.info(f"Model loaded from: {file_path}")
        return model

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """
        Saves the model to disk.

        Args:
            path (str):
                Path to the saved model, minus the file name.
                This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
                If None, self.path is used.
                The final model file is typically saved to os.path.join(path, self.model_file_name).
            verbose (bool):
                Whether to log the location of the saved file.

        Returns:
            path (str):
                Path to the saved model, minus the file name.
                Use this value to load the model from disk via cls.load(path), cls being the class of the model object, such as model = RFModel.load(path)
        """
        if path is None:
            path = self.path
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, self.model_file_name)
        # tmp = {}
        # for key, obj in vars(self).items():
        #     if hasattr(obj, "save"):
        #         obj_path = os.path.join(path, key)
        #         obj.save(obj_path)
        #         tmp[key] = obj
        #         setattr(self, key, None)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        # for key, obj in tmp.items():
        #     setattr(self, key, obj)
        if verbose:
            logger.info(f"Model saved to: {file_path}")
        return path

    def reset_path(self, path: str | None = None):
        """Reset the path of the model.
        
        Args:
            path (str):
                Directory location to store all outputs.
                If None, a new unique time-stamped directory is chosen.
        """
        if path is None:
            path = setup_outputdir(path=self.name.lower(), path_suffix=self.name)
        self.path = path