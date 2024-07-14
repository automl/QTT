import json

from ConfigSpace.read_and_write import json as cs_json
import torch
from qtt.data.dataset import MetaDataset
from qtt.optimizers.surrogates import DyHPO, CostEstimator
from qtt.tools.metatrain import metatrain_dyhpo, metatrain_cost_estimator

# load data
cs_path = "path/to/space.json"
cs = cs_json.read(open("space.json", "r").read())

meta_path = "path/to/data/root"
dataset = MetaDataset(meta_path, cs, cost_aware=True, include_metafeat=True)

# retrieve data to build the surrogate model
dim = dataset.get_config_dim()
metafeat_dim = dataset.get_metafeat_dim()

config = {
    "in_features": dim,
    "in_metafeat_dim": metafeat_dim,
}
# save config
json.dump(config, open("config.json", "w"))

# create, train and save DyHPO
dyhpo = DyHPO(**config)
dyhpo = metatrain_dyhpo(dyhpo, dataset)
torch.save(dyhpo.state_dict(), "dyhpo.pth")

# create, train and save cost estimator
cost_estimator = CostEstimator(**config)
cost_estimator = metatrain_cost_estimator(cost_estimator, dataset)
torch.save(cost_estimator.state_dict(), "estimator.pth")