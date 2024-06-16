from qtt.data.dataset import MTLBMDataSet
from qtt.optimizers.surrogates.dyhpo import DyHPO
from qtt.tools import metatrain_dyhpo

meta_path = "../data/mtlbm/micro"
data = MTLBMDataSet(meta_path)
data.save_data_info()

# setup DyHPO
n_features = data.num_hps
hps = data.hyperparameter_names
n_model = len([x for x in hps if x.startswith("model")])

dyhpo = DyHPO(in_features=[n_model, n_features - n_model])
dyhpo = metatrain_dyhpo(dyhpo, data)
