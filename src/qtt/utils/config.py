from collections import defaultdict

import ConfigSpace as CS

CATEGORICAL = CS.CategoricalHyperparameter
NUMERICAL = (
    CS.UniformIntegerHyperparameter,
    CS.BetaIntegerHyperparameter,
    CS.UniformFloatHyperparameter,
    CS.BetaFloatHyperparameter,
    CS.NormalFloatHyperparameter,
    CS.NormalIntegerHyperparameter,
)
ORDINAL = CS.OrdinalHyperparameter


def to_one_hot(hp_name, sequence):
    return [f"{hp_name}={c}" for c in sequence]


def encode_config_space(cs: CS.ConfigurationSpace) -> tuple[list[str], list[list[str]]]:
    """Encode a ConfigSpace.ConfigurationSpace object into a list of one-hot
    encoded hyperparameters.

    Args:
        cs (CS.ConfigurationSpace): A ConfigSpace.ConfigurationSpace object.
    """
    type_dict = defaultdict(list)
    for hp in list(cs.values()):
        if isinstance(hp, CS.Constant):
            continue
        elif isinstance(hp, CATEGORICAL):
            one_hot = to_one_hot(hp.name, hp.choices)
        elif isinstance(hp, ORDINAL) and isinstance(hp.default_value, str):
            one_hot = to_one_hot(hp.name, hp.sequence)
        else:
            one_hot = [hp.name]

        _type = "none"
        if hp.meta is not None:
            _type = hp.meta.get("type", "none")
        type_dict[_type].extend(one_hot)

    encoding = []
    splits = []
    for key in sorted(type_dict.keys()):
        g = type_dict[key]
        g.sort()
        encoding.extend(g)
        splits.append(g)
    return encoding, splits


def config_to_vector(configs: list[CS.Configuration], one_hot):
    """Convert a list of ConfigSpace.Configuration to a list of	
    one-hot encoded dictionaries.

    Args:
        configs (list[CS.Configuration]): A list of ConfigSpace.Configuration objects.
        one_hot (list[str]): One-hot encodings of the hyperparameters.
    """
    encoded_configs = []
    for config in configs:
        config = dict(config)
        enc_config = {}
        for hp in one_hot:
            # categorical hyperparameters
            if len(hp.split("=")) > 1:
                key, choice = hp.split("=")
                val = 1 if config.get(key) == choice else 0
            else:
                val = config.get(hp, 0)  # NUM
                if isinstance(val, bool):  # BOOL
                    val = int(val)
            enc_config[hp] = val
        encoded_configs.append(enc_config)
    return encoded_configs

def config_to_serializible_dict(config: CS.Configuration) -> dict:
    """Convert a ConfigSpace.Configuration to a serializable dictionary.
    Cast all values to basic types (int, float, str, bool) to ensure
    that the dictionary is JSON serializable.

    Args:
        config (CS.Configuration): A ConfigSpace.Configuration object.
    """
    serializable_dict = dict(config)
    for k, v in serializable_dict.items():
        if hasattr(v, "item"):
            serializable_dict[k] = v.item()
    return serializable_dict