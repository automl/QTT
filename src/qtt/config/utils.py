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
SORT_MTHDS = [
    "auto",
    "alphabet",
    "cat-num",
    "group",
]


def cat_to_one_hot(hp):
    return [f"{hp.name}={c}" for c in hp.choices]


def ord_to_one_hot(hp):
    return [f"{hp.name}={c}" for c in hp.sequence]


def get_hp_encoding(hp):
    if isinstance(hp, CATEGORICAL):
        return cat_to_one_hot(hp)
    elif isinstance(hp, NUMERICAL):
        return [hp.name]
    elif isinstance(hp, ORDINAL):
        if isinstance(hp.default_value, str):
            return ord_to_one_hot(hp)
        return [hp.name]
    return []


def _auto(cs: CS.ConfigurationSpace):
    _hot = [get_hp_encoding(hp) for hp in list(cs.values())]
    # unpack
    hot = [item for sublist in _hot if sublist for item in sublist]
    return hot, [hot]


def _alphabet(cs: CS.ConfigurationSpace):
    hot, splits = _auto(cs)
    hot.sort()
    return hot, splits


def _cat_num(cs: CS.ConfigurationSpace):
    cat_hp = []
    num_hp = []
    for hp in list(cs.values()):
        if isinstance(hp, CS.Constant):
            continue
        elif isinstance(hp, CATEGORICAL):
            cat_hp.extend(cat_to_one_hot(hp))
        elif isinstance(hp, NUMERICAL):
            num_hp.append(hp.name)
        # ordinal
        elif isinstance(hp, CS.OrdinalHyperparameter):
            d_type = type(hp.default_value)
            if d_type is str:
                cat_hp.extend(ord_to_one_hot(hp))
            else:
                num_hp.append(hp.name)
    cat_hp.sort()
    num_hp.sort()
    _all = cat_hp + num_hp
    _split = [cat_hp, num_hp]
    return _all, _split


def _group(cs: CS.ConfigurationSpace):
    groups = defaultdict(list)
    for hp in list(cs.values()):
        if isinstance(hp, CS.Constant):
            continue
        assert hp.meta, f"Hyperparameter {hp.name} has no meta"
        assert hp.meta.get("group"), f"Hyperparameter {hp.name} has no group in meta"

        g = hp.meta["group"]
        groups[g].extend(get_hp_encoding(hp))

    _all = []
    _groups = []
    for g in sorted(groups.keys()):
        hp_list = groups[g]
        hp_list.sort()
        _all.extend(hp_list)
        _groups.append(hp_list)
    return _all, _groups


def one_hot_encode_config_space(cs: CS.ConfigurationSpace, method: str = "auto"):
    assert method in SORT_MTHDS, f"Invalid method: {method}"
    match method:
        case "alphabet":
            return _alphabet(cs)
        case "cat-num":
            return _cat_num(cs)
        case "group":
            return _group(cs)
        case _:
            return _auto(cs)


def config_to_vector(configs: list[CS.Configuration], one_hot):
    encoded_configs = []
    for config in configs:
        config = dict(config)
        enc_config = dict()
        for hp in one_hot:
            # categorical hyperparameters
            if len(hp.split(":")) > 1:
                key, choice = hp.split(":")
                val = 1 if config.get(key) == choice else 0
            else:
                val = config.get(hp, 0)  # NUM
                if isinstance(val, bool):  # BOOL
                    val = int(val)
            enc_config[hp] = val
        encoded_configs.append(enc_config)
    return encoded_configs