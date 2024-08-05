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


def encode_config_space(cs: CS.ConfigurationSpace):
    _dict = defaultdict(list)
    for hp in list(cs.values()):
        # 
        if isinstance(hp, CS.Constant):
            continue
        elif isinstance(hp, CATEGORICAL):
            enc = to_one_hot(hp.name, hp.choices)
        elif isinstance(hp, ORDINAL) and isinstance(hp.default_value, str):
            enc = to_one_hot(hp.name, hp.sequence)
        else:
            enc = [hp.name]
        
        _type = "none"
        if hp.meta is not None:
            _type = hp.meta.get("type", "none")
        _dict[_type].extend(enc)
    
    encoding = []
    splits = []
    for key in sorted(_dict.keys()):
        g = _dict[key]
        g.sort()
        encoding.extend(g)
        splits.append(g)
    return encoding, splits


def config_to_vector(configs: list[CS.Configuration], one_hot):
    encoded_configs = []
    for config in configs:
        config = dict(config)
        enc_config = dict()
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