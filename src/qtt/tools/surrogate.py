from ..configuration.manager import ConfigManager
from ..optimizers.surrogates import DyHPO, CostEstimator


def get_dyhpo(cm: ConfigManager):
    one_hot = cm.get_one_hot_encoding()
    n_model = len([x for x in hps if x.startswith("model")])

    n_features = len(one_hot)
    config = dict(
        in_features=[n_features, n_model],
    )

    surrogate = DyHPO(**config)
    
    return surrogate, config


def get_cost_estimator(dataset):
    n_features = dataset.num_hps
    hps = dataset.hyperparameter_names
    n_model = len([x for x in hps if x.startswith("model")])

    config = dict(
        in_features=[n_features, n_model],
    )

    surrogate = CostEstimator(**config)  # type: ignore
    
    return surrogate, config
