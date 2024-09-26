import copy
import logging
import os
import random
import shutil

import gpytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from qtt.predictors.models import FeatureEncoder
from qtt.utils.log_utils import set_logger_verbosity

from .data import (
    CurveRegressionDataset,
    create_preprocessor,
    get_feature_mapping,
    get_types_of_features,
)
from .predictor import Predictor
from .utils import MetricLogger, get_torch_device

logger = logging.getLogger(__name__)

DEFAULT_FIT_PARAMS = {
    "learning_rate_init": 0.001,
    "batch_size": 2048,
    "max_iter": 100,
    "early_stop": True,
    "patience": 5,
    "validation_fraction": 0.1,
    "tol": 1e-4,
}

DEFAULT_REFIT_PARAMS = {
    "learning_rate_init": 0.001,
    "batch_size": 2048,
    "max_iter": 50,
    "early_stop": True,
    "patience": 5,
    "tol": 1e-4,
}


class PerfPredictor(Predictor):
    temp_file_name: str = "temp_model.pt"
    train_data_size: int = 4096
    _fit_data = None

    def __init__(
        self,
        fit_params: dict = {},
        refit_params: dict = {},
        path: str | None = None,
        seed: int | None = None,
        verbosity: int = 2,
    ) -> None:
        super().__init__(path=path)
        self.fit_params = self._validate_fit_params(fit_params, DEFAULT_FIT_PARAMS)
        self.refit_params = self._validate_fit_params(
            refit_params, DEFAULT_REFIT_PARAMS
        )
        self.seed = seed
        self.verbosity = verbosity

        set_logger_verbosity(verbosity, logger)

    @staticmethod
    def _validate_fit_params(fit_params, default_params):
        """
        Validate hyperparameters for fitting the model.

        Args:
            fit_params (dict): Hyperparameters for fitting the model.
            default_params (dict): Default hyperparameters for fitting the model.

        Raises:
            ValueError: If fit_params is not a dictionary or contains unknown hyperparameters.

        Returns:
            dict: Validated hyperparameters.
        """
        if not isinstance(fit_params, dict):
            raise ValueError("fit_params must be a dictionary")
        for key in fit_params:
            if key not in default_params:
                raise ValueError(f"Unknown fit parameter: {key}")
        return {**default_params, **fit_params}

    def _validate_fit_data(self, pipeline: pd.DataFrame, curve: np.ndarray):
        """
        Validate data for fitting the model.

        Args:
            pipeline (pandas.DataFrame): Pipeline data.
            curve (numpy.ndarray): Curve data.

        Raises:
            ValueError: If pipeline or curve is not a pandas.DataFrame or numpy.ndarray, or if
                pipeline and curve have different number of samples, or if column names are not
                unique.

        Returns:
            tuple: Validated pipeline and curve data.
        """
        if not isinstance(pipeline, pd.DataFrame):
            raise ValueError("pipeline must be a pandas.DataFrame instance")

        if not isinstance(curve, np.ndarray):
            raise ValueError("curve must be a numpy.ndarray instance")

        if pipeline.shape[0] != curve.shape[0]:
            raise ValueError("pipeline and curve must have the same number of samples")

        if len(set(pipeline.columns)) < len(pipeline.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})"
            )

        self._curve_dim = curve.shape[1]

    def _preprocess_fit_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for fitting the model.

        Args:
            df (pandas.DataFrame): Data to preprocess.

        Returns:
            numpy.ndarray: Preprocessed data.
        """
        self.original_features = list(df.columns)

        df, self.types_of_features, self.features_to_drop = get_types_of_features(df)
        self.input_features = list(df.columns)

        self.preprocessor = create_preprocessor(
            self.types_of_features["continuous"],
            self.types_of_features["categorical"],
            self.types_of_features["bool"],
        )
        out = self.preprocessor.fit_transform(df)
        self.feature_mapping = get_feature_mapping(self.preprocessor)
        if out.shape[1] != sum(len(v) for v in self.feature_mapping.values()):
            raise ValueError(
                "Error during one-hot encoding data processing for neural network. "
                "Number of columns in df array does not match feature_mapping."
            )
        return np.array(out)

    def _validate_predict_data(self, pipeline, curve):
        """Validate data for prediction. Applies the same steps as _validate_fit_data

        Args:
            pipeline (pandas.DataFrame): Pipeline data.
            curve (numpy.ndarray): Curve data.

        Raises:
            ValueError: If pipeline or curve is not a pandas.DataFrame or numpy.ndarray, or if
                pipeline and curve have different number of samples, or if column names are not
                unique.

        Returns:
            tuple: Validated pipeline and curve data.
        """
        if not isinstance(pipeline, pd.DataFrame) or not isinstance(curve, np.ndarray):
            raise ValueError("pipeline and curve must be pandas.DataFrame instances")

        if pipeline.shape[0] != curve.shape[0]:
            raise ValueError("pipeline and curve must have the same number of samples")

        if len(set(pipeline.columns)) < len(pipeline.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})"
            )

        if curve.shape[1] != self._curve_dim:
            raise ValueError(
                "curve must have the same number of features as the curve used for fitting"
                " (expected: {self._curve_length}, got: {curve.shape[1]})"
            )

    def _preprocess_predict_data(self, df: pd.DataFrame, fill_missing=True):
        extra_features = list(set(df.columns) - set(self.original_features))
        if extra_features:
            logger.warning(
                f"Features {extra_features} were not present in training data and are dropped"
            )
            df = df.drop(columns=extra_features, errors="ignore")

        df = df.drop(columns=self.features_to_drop, errors="ignore")

        missing_features = list(set(self.input_features) - set(df.columns))
        if missing_features:
            if fill_missing:
                logger.warning(
                    f"Features {missing_features} missing in data. Missing values will be imputed."
                )
                for col in missing_features:
                    df[col] = None
            else:
                raise AssertionError(f"Features {missing_features} missing in data.")

        # process data
        X = self.preprocessor.transform(df)
        X = np.array(X)
        X = np.nan_to_num(X)
        return X

    def _get_model(self):
        params = {
            "in_dim": [
                len(self.types_of_features["continuous"]),
                len(self.types_of_features["categorical"])
                + len(self.types_of_features["bool"]),
            ],
            "in_curve_dim": self._curve_dim,
        }
        return SurrogateModel(**params)

    def _train_model(
        self,
        dataset,
        learning_rate_init: float,
        batch_size: int,
        max_iter: int,
        early_stop: bool,
        patience: int | None,
        validation_fraction: float,
        tol: float,
    ):
        """
        Train the model on the given dataset.

        Args:
            dataset: dataset to train on
            learning_rate_init: initial learning rate
            batch_size: batch size to use
            max_iter: maximum number of iterations to train for
            early_stop: if True, stop training when validation loss stops improving
            patience: number of iterations to wait before stopping training
            validation_fraction: fraction of dataset to use for validation
            tol: tolerance for determining when to stop training
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.device = get_torch_device()
        dev = self.device
        self.model.to(dev)

        optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate_init)

        patience_counter = 0
        best_iter = 0
        best_val_metric = np.inf

        if patience is not None:
            if early_stop:
                if validation_fraction <= 0 or validation_fraction >= 1:
                    raise AssertionError(
                        "validation_fraction must be between 0 and 1 when early_stop is True"
                    )
                logger.info(
                    f"Early stopping on validation loss with patience {patience} "
                    f"using {validation_fraction} of the data for validation"
                )
                train_set, val_set = random_split(
                    dataset=dataset,
                    lengths=[1 - validation_fraction, validation_fraction],
                )
            else:
                logger.info(f"Early stopping on training loss with patience {patience}")
                train_set = dataset
                val_set = None
        else:
            train_set = dataset
            val_set = None

        train_loader = DataLoader(
            train_set,
            batch_size=min(batch_size, len(train_set)),
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )
        val_loader = None
        if val_set is not None:
            val_loader = DataLoader(
                val_set,
                batch_size=min(batch_size, len(val_set)),
                num_workers=4,
            )

        cache_dir = os.path.join(self.path, ".tmp")
        os.makedirs(cache_dir, exist_ok=True)
        temp_save_file_path = os.path.join(cache_dir, self.temp_file_name)
        for it in range(1, max_iter + 1):
            self.model.train()

            train_loss = []
            header = f"TRAIN: ({it}/{max_iter})"
            metric_logger = MetricLogger(delimiter=" ")
            for batch in metric_logger.log_every(
                train_loader, max(len(train_loader) // 10, 1), header, logger
            ):
                # forward
                batch = (b.to(dev) for b in batch)
                X, curve, y = batch
                loss = self.model.train_step(X, curve, y)
                train_loss.append(loss.item())

                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log
                metric_logger.update(loss=loss.item())
                metric_logger.update(lengthscale=self.model.lengthscale)
                metric_logger.update(noise=self.model.noise)  # type: ignore
            logger.info(f"({it}/{max_iter}) Averaged stats: {str(metric_logger)}")
            val_metric = np.mean(train_loss)

            if val_loader is not None:
                self.model.eval()

                val_loss = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = (b.to(dev) for b in batch)
                        X, curve, y = batch
                        pred = self.model.predict(X, curve)
                        loss = torch.nn.functional.l1_loss(pred.mean, y)
                        val_loss.append(loss.item())
                val_metric = np.mean(val_loss)

            if patience is not None:
                if val_metric + tol < best_val_metric:
                    patience_counter = 0
                    best_val_metric = val_metric
                    best_iter = it
                    torch.save(self.model.state_dict(), temp_save_file_path)
                else:
                    patience_counter += 1
                logger.info(
                    f"VAL: {round(val_metric, 4)}  "
                    f"ITER: {it}/{max_iter}  "
                    f"BEST: {round(best_val_metric, 4)} ({best_iter})"
                )
                if patience_counter >= patience:
                    logger.log(
                        15,
                        "Stopping training..."
                        f"No improvement in the last {patience} iterations. "
                    )
                    break

        if early_stop:
            logger.info(
                f"Loading best model from iteration {best_iter} with val score {best_val_metric}"
            )
            self.model.load_state_dict(torch.load(temp_save_file_path))
        
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        # after training the gp, set its training data
        # TODO: check if this can be improved
        self.model.eval()
        size = min(self.train_data_size, len(dataset))
        loader = DataLoader(dataset, batch_size=size, shuffle=True)
        a, b, c = next(iter(loader))
        a, b, c = a.to(dev), b.to(dev), c.to(dev)
        self.model.set_train_data(a, b, c)

    def _fit(
        self,
        pipeline: pd.DataFrame,
        curve: np.ndarray,
    ):
        if self.is_fit:
            raise AssertionError("Predictor is already fit! Create a new one.")

        self._validate_fit_data(pipeline, curve)
        x = self._preprocess_fit_data(pipeline)
        train_dataset = CurveRegressionDataset(x, curve)

        self.model = self._get_model()
        self._train_model(train_dataset, **self.fit_params)

        self._model_fit = copy.deepcopy(self.model)
        self._fit_data = train_dataset

        return self

    def fit_extra(
        self,
        X: pd.DataFrame,
        curve: np.ndarray,
        fit_params: dict = {},
    ):
        if not self.is_fit:
            raise AssertionError("Model is not fitted yet")

        self._validate_predict_data(X, curve)

        x = self._preprocess_predict_data(X)

        tune_dataset = CurveRegressionDataset(x, curve)

        fit_params = self._validate_fit_params(fit_params, self.refit_params)
        self._refit_model(tune_dataset, **fit_params)

    def _refit_model(
        self,
        dataset,
        learning_rate_init,
        batch_size,
        max_iter,
        early_stop,
        patience,
        tol,
    ):
        learning_rate_init = 0.001
        logger.info("Refitting model...")
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        cache_dir = os.path.join(self.path, ".tmp")
        os.makedirs(cache_dir, exist_ok=True)
        temp_save_file_path = os.path.join(cache_dir, self.temp_file_name)

        num_workers = 4
        self.device = get_torch_device()
        dev = self.device

        self.model.to(dev)
        self.model.eval()
        torch.save(self.model.state_dict(), temp_save_file_path)

        # initial validation loss
        loader = DataLoader(
            dataset,
            batch_size=min(len(dataset), batch_size),
            num_workers=num_workers,
        )
        val_metric = []
        for batch in loader:
            batch = (b.to(dev) for b in batch)
            X, curve, y = batch
            pred = self.model.predict(X, curve)
            loss = torch.nn.functional.l1_loss(pred.mean, y)
            val_metric.append(loss.item())
        best_val_metric = np.mean(val_metric)
        logger.info(f"Initial validation loss: {best_val_metric}")
        patience_counter = 0
        best_iter = 0

        assert self._fit_data is not None
        fitting_set = self._fit_data
        logger.debug(f"Number of samples in the tuning set: {len(dataset)}")
        if len(dataset) < batch_size:
            logger.warning(
                f"Tuning-set size is small ({len(dataset)})."
                "Using all samples for training + validation. "
                f"Adding samples from training set to reach minimal sample size {batch_size}"
            )

        if patience is not None:
            if early_stop:
                logger.info(
                    f"Early stopping on validation loss with patience {patience} "
                )
            else:
                logger.info(f"Early stopping on training loss with patience {patience}")

        loader_bs = min(int(2 ** np.floor(np.log2(len(dataset) - 1))), batch_size)
        train_loader = DataLoader(
            dataset,
            batch_size=loader_bs,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            num_workers=num_workers,
        )
        extra_loader = None
        if loader_bs < self.train_data_size:
            extra_loader = DataLoader(
                fitting_set,
                batch_size=batch_size - loader_bs,
                shuffle=True,
                num_workers=num_workers,
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate_init)
        for it in range(1, max_iter + 1):
            self.model.train()

            train_loss = []
            header = f"TRAIN: ({it}/{max_iter})"
            metric_logger = MetricLogger(delimiter=" ")
            for batch in metric_logger.log_every(train_loader, 1, header, logger):
                # forward
                if extra_loader is not None:
                    b1 = next(iter(extra_loader))
                    batch = [torch.cat([b1, b2]) for b1, b2 in zip(batch, b1)]
                batch = (b.to(dev) for b in batch)
                X, curve, y = batch
                loss = self.model.train_step(X, curve, y)
                train_loss.append(loss.item())

                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log
                metric_logger.update(loss=loss.item())
                metric_logger.update(lengthscale=self.model.lengthscale)
                metric_logger.update(noise=self.model.noise)  # type: ignore
            logger.info(f"[{it}/{max_iter}]Averaged stats: {str(metric_logger)}")
            val_metric = np.mean(train_loss)

            if val_loader is not None:
                self.model.eval()

                l1 = DataLoader(
                    dataset,
                    batch_size=len(dataset),
                    shuffle=True,
                )
                batch = next(iter(l1))
                if len(dataset) < self.train_data_size:
                    l2 = DataLoader(
                        fitting_set,
                        batch_size=self.train_data_size - loader_bs,
                        shuffle=True,
                    )
                    b2 = next(iter(l2))
                    batch = [torch.cat([p, q]) for p, q in zip(batch, b2)]
                batch = (b.to(dev) for b in batch)
                a, b, c = batch
                self.model.set_train_data(a, b, c)

                val_loss = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = (b.to(dev) for b in batch)
                        X, curve, y = batch
                        pred = self.model.predict(X, curve)
                        loss = torch.nn.functional.l1_loss(pred.mean, y)
                        val_loss.append(loss.item())
                val_metric = np.mean(val_loss)

            if patience is not None:
                if val_metric + tol < best_val_metric:
                    patience_counter = 0
                    best_val_metric = val_metric
                    best_iter = it
                    torch.save(self.model.state_dict(), temp_save_file_path)
                else:
                    patience_counter += 1
                logger.info(
                    f"[{it}/{max_iter}]  "
                    f"VAL: {round(val_metric, 4)}  "
                    f"BEST: {round(best_val_metric, 4)} ({best_iter})",
                )
                if patience_counter >= patience:
                    logger.log(
                        15,
                        "Stopping training..."
                        f"No improvement in the last {patience} iterations. ",
                    )
                    break

        if patience:
            logger.info(f"Loading best model from iteration {best_iter}")
            self.model.load_state_dict(torch.load(temp_save_file_path))

        # remove cache dir
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        # after training the model, reset GPs training data
        self.model.eval()
        if len(dataset) < self.train_data_size:
            l1 = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
            l2 = DataLoader(
                fitting_set,
                batch_size=self.train_data_size - len(dataset),
                shuffle=True,
            )
            b1 = next(iter(l1))
            b2 = next(iter(l2))
            batch = [torch.cat([a, b]) for a, b in zip(b1, b2)]
            batch = (b.to(dev) for b in batch)
        else:
            loader = DataLoader(dataset, batch_size=self.train_data_size, shuffle=True)
            batch = next(iter(loader))
            batch = (b.to(dev) for b in batch)
        a, b, c = batch
        self.model.set_train_data(a, b, c)

    def predict(self, X: pd.DataFrame, curve: np.ndarray, fill_missing=True):
        if not self.is_fit:
            raise AssertionError("Model is not fitted yet")

        self._validate_predict_data(X, curve)
        x = self._preprocess_predict_data(X, fill_missing)
        curve = np.nan_to_num(curve)

        device = self.device
        self.model.eval()
        self.model.to(device)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        c = torch.tensor(curve, dtype=torch.float32, device=device)
        mean = np.array([])
        std = np.array([])
        with torch.no_grad():
            bs = 4096  # TODO: make this a parameter
            for i in range(0, x.shape[0], bs):
                pred = self.model.predict(x[i : i + bs], c[i : i + bs])
                mean = np.append(mean, pred.mean.cpu().numpy())
                std = np.append(std, pred.stddev.cpu().numpy())
        return mean, std

    def save(self, path: str | None = None, verbose=True) -> str:
        # Save on CPU to ensure the model can be loaded on a box without GPU
        if self.model is not None:
            self.model = self.model.to(torch.device("cpu"))
        path = super().save(path, verbose)
        # Put the model back to the device after the save
        if self.model is not None:
            self.model.to(self.device)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        Loads the model from disk to memory.

        The loaded model will be on the same device it was trained on (e.g., cuda/mps).
        If the device is unavailable (e.g., trained on GPU but deployed on CPU),
        the model will be loaded on `cpu`.

        Args:
            path (str): Path to the saved model, excluding the file name.
                This should typically be a directory path ending with a '/' character
                (or appropriate path separator based on OS). The model file is usually
                located at `os.path.join(path, cls.model_file_name)`.
            reset_paths (bool, optional): Whether to reset the `self.path` value of the loaded
                model to be equal to `path`. Defaults to True. Setting this to False may cause
                inconsistencies between the actual valid path and `self.path`, potentially leading
                to strange behavior and exceptions if the model needs to load other files later.
            verbose (bool, optional): Whether to log the location of the loaded file. Defaults to True.

        Returns:
            cls: The loaded model object.
        """
        model: PerfPredictor = super().load(
            path=path, reset_paths=reset_paths, verbose=verbose
        )

        verbosity = model.verbosity
        set_logger_verbosity(verbosity, logger)
        return model


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor | None,
        train_y: torch.Tensor | None,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)  # type: ignore


class SurrogateModel(torch.nn.Module):
    def __init__(
        self,
        in_dim: int | list[int],
        in_curve_dim: int,
        out_dim: int = 32,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        out_curve_dim: int = 16,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            in_dim,
            in_curve_dim,
            out_dim,
            enc_hidden_dim,
            enc_out_dim,
            enc_nlayers,
            out_curve_dim,
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = GPRegressionModel(
            train_x=None,
            train_y=None,
            likelihood=self.likelihood,
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood,
            self.gp_model,
        )

    def forward(self, pipeline, curve):
        encoding = self.encoder(pipeline, curve)
        output = self.gp_model(encoding)
        return self.likelihood(output)

    @torch.no_grad()
    def predict(self, pipeline, curve):
        return self(pipeline, curve)

    def train_step(self, pipeline, curve, y) -> torch.Tensor:
        encoding = self.encoder(pipeline, curve)
        self.gp_model.set_train_data(encoding, y, False)
        output = self.gp_model(encoding)
        loss = -self.mll(output, y)  # type: ignore
        return loss

    @torch.no_grad()
    def set_train_data(self, pipeline, curve, y) -> None:
        self.eval()
        encoding = self.encoder(pipeline, curve)
        self.gp_model.set_train_data(encoding, y, False)

    @property
    def lengthscale(self) -> float:
        return self.gp_model.covar_module.base_kernel.lengthscale.item()

    @property
    def noise(self) -> float:
        return self.gp_model.likelihood.noise.item()  # type: ignore
