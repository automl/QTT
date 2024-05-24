from typing import Tuple

from ..data.dataset import MTLBMDataSet
from ..optimizers.surrogates import DyHPO, Surrogate


def get_surrogate(
    dataset: MTLBMDataSet,
    surrogate: str = "dyhpo",
) -> Tuple[Surrogate, dict]:
    if surrogate == "dyhpo":
        return get_dyhpo(dataset)
    else:
        raise ValueError(f"Unknown surrogate {surrogate}")


def get_dyhpo(dataset: MTLBMDataSet):
    n_features = dataset.get_num_hps()
    hps = dataset.get_hyperparameters_names()
    n_model = len([x for x in hps if x.startswith("model")])

    config = {
        "in_features": n_features,
        "enc_slice_ranges": [n_model],
    }

    surrogate = DyHPO(config, config)

    surrogate_config = {
        "extractor_cfg": config,
        "predictor_cfg": config,
    }

    return surrogate, surrogate_config
