import json

import torch
from qtt.data.dataset import MTLBMDataSet
from qtt.optimizers.surrogates import DyHPO, CostEstimator
from qtt.tools.metatrain import metatrain_dyhpo, metatrain_cost_estimator

# load data
meta_path = "path/to/data/root"
data = MTLBMDataSet(meta_path)
data.save_data_info()

# setup DyHPO
n_features = data.num_hps
hps = data.hyperparameter_names

# split the features into model (one-hot encoded) and other features (numerical)
n_model = len([x for x in hps if x.startswith("model")])
config = {"in_features": [n_model, n_features - n_model]}

# save config
json.dump(config, open("config.json", "w"))

# create, train and save DyHPO
dyhpo = DyHPO(**config)
dyhpo = metatrain_dyhpo(dyhpo, data, seed=1)
torch.save(dyhpo.state_dict(), "dyhpo.pth")

# create, train and save cost estimator
cost_estimator = CostEstimator(**config)
cost_estimator = metatrain_cost_estimator(cost_estimator, data, seed=1)
torch.save(cost_estimator.state_dict(), "estimator.pth")