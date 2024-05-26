from qtt.data.dataset import MTLBMDataSet
from qtt.tools import get_surrogate, meta_train_surrogate

meta_path = "../data/mtlbm/micro"
data = MTLBMDataSet(meta_path)
data.save_data_info()

surrogate, config = get_surrogate(data, "dyhpo")
surrogate = meta_train_surrogate(surrogate, data)
surrogate.save_checkpoint()