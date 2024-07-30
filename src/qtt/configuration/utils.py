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
SORT_MTHDS = [
    "auto",
    "alphabet",
    "group",
]


def get_cs_dim(cs: CS.ConfigurationSpace) -> int:
    dim = 0
    for hp in list(cs.values()):
        if isinstance(hp, CS.Constant):
            continue
        elif isinstance(hp, CATEGORICAL):
            dim += len(hp.choices)
        else:
            dim += 1
    return dim


def get_one_hot_encoding(cs: CS.ConfigurationSpace, sort_mthd=None) -> list[str]:
    assert sort_mthd in SORT_MTHDS, "Invalid sort option"
    cat_hp = []
    num_hp = []
    ord_hp = []
    for hp in list(cs.values()):
        if isinstance(hp, CS.Constant):
            continue
        elif isinstance(hp, CATEGORICAL):
            for choice in hp.choices:
                cat_hp.append(f"{hp.name}:{choice}")
        elif isinstance(hp, NUMERICAL):
            num_hp.append(hp.name)
        else:
            ord_hp.append(hp.name)

    if sort_mthd == "auto":
        cat_hp.sort()
        num_hp.sort()
        ord_hp.sort()
        one_hot = cat_hp + num_hp + ord_hp
    elif sort_mthd == "alphabet":
        one_hot = cat_hp + num_hp + ord_hp
        one_hot.sort()
    return one_hot
