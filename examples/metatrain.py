from qtt import MetaDataset, QuickOptimizer

dataset = MetaDataset("path/to/mtlbm")
cs, meta = dataset.get_space()
opt = QuickOptimizer(cs, meta, cost_aware=True)
# call fit to train the optimizer
# optionally pass parameters like lr, train_steps, batch_size
opt.fit(dataset)  # lr=1e-4, train_steps=1000, batch_size=32
opt.save("path/to/optimizer")
