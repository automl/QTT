from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset


def _custom_combiner(input_feature, category):
    return str(input_feature) + "=" + str(category)


def create_preprocessor(continous_features, categorical_features, bool_features):
    transformers = []
    if continous_features:
        continous_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("continuous", continous_transformer, continous_features))
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        feature_name_combiner=_custom_combiner,  # type: ignore
                    ),
                )
            ]
        )
        transformers.append(
            ("categorical", categorical_transformer, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        force_int_remainder_cols=False,  # type: ignore
    )


def get_feature_mapping(processor):
    feature_preserving_transforms = set(["continuous", "bool", "remainder"])
    feature_mapping = {}
    col_index = 0
    for tf_name, tf, transformed_features in processor.transformers_:
        if tf_name in feature_preserving_transforms:
            for feature in transformed_features:
                feature_mapping[feature] = [col_index]
                col_index += 1
        elif tf_name == "categorical":
            encoder = [step for (name, step) in tf.steps if name == "onehot"][0]
            for i in range(len(transformed_features)):
                feature = transformed_features[i]
                if feature in feature_mapping:
                    raise ValueError(
                        f"same feature is processed by two different column transformers: {feature}"
                    )
                encoding_size = len(encoder.categories_[i])
                feature_mapping[feature] = list(
                    range(col_index, col_index + encoding_size)
                )
                col_index += encoding_size
        else:
            raise ValueError(f"Unknown transformer {tf_name}")
    return OrderedDict([(key, feature_mapping[key]) for key in feature_mapping])


class SimpleTorchTabularDataset(Dataset):
    def __init__(self, *args):
        super().__init__()
        self.data = [torch.tensor(arg, dtype=torch.float32) for arg in args]

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return [arg[idx] for arg in self.data]


class CurveRegressionDataset(Dataset):
    """ """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__()

        true_indices = np.where(~np.isnan(y.flatten()))[0]
        self.mapping = list(true_indices)
        self.x = x
        self.y = y
        self.y_dim = y.shape[1]

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        idx = self.mapping[idx]
        true_index = idx // self.y_dim
        fidelity = idx % self.y_dim
        curve = np.concatenate(
            (self.y[true_index, :fidelity], np.zeros(self.y.shape[1] - fidelity))
        )
        target = self.y[true_index, fidelity]
        x = torch.tensor(self.x[true_index], dtype=torch.float32)
        curve = torch.tensor(curve, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return x, curve, target


def make_regression_from_series_dataset(pipeline: pd.DataFrame, curve: np.ndarray):
    """
    This method takes a pandas DataFrame `pipeline` containing features and a numpy
    array `curve` representing learning curves. For each point in the curve
    array the function creates a training sample.

    ->  [pipeline]  [y_0, ..., y_m]
        [pipeline]  [0, ..., 0]                 -> [y_0]
        [pipeline]  [y_0, 0, ..., 0]            -> [y_1]
        [pipeline]  [y_0, y_1, 0, ..., 0]       -> [y_2]
        ...
        [pipeline]  [y_0, y_1, ..., y_m-1, 0]   -> [y_m]

    The transformation produces three outputs:
    1. `X`: A DataFrame where each row corresponds to a repeated and filtered
    version of the original `pipeline` data.
    2. `curve_out`: A 2D numpy array representing the padded versions of the
    sequences in `curve`.
    3. `y`: A 1D numpy array representing the flattened and filtered values
    from `curve`.
    """
    _, m = curve.shape
    y = curve.flatten()
    curve_out = np.vstack(
        [np.concatenate((row[:i], np.zeros(m - i))) for row in curve for i in range(m)]
    )
    mask = np.isnan(y)
    y = y[~mask]
    curve_out = curve_out[~mask]
    X = pipeline.values.repeat(m, axis=0)[~mask]

    X = pd.DataFrame(X, columns=pipeline.columns)
    for col in pipeline.columns:
        X[col] = X[col].astype(pipeline[col].dtype)
    curve_out = np.array(curve_out)
    y = np.array(y)

    if X.shape[0] != curve_out.shape[0] or curve_out.shape[0] != y.shape[0]:
        raise ValueError("Data size mismatch")

    return X, curve_out, y
