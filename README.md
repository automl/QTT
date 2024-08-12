# QuickTuneTool: A Framework for Efficient Model Selection and Hyperparameter Optimization

QuickTune is a tool designed to address the challenge of selecting the optimal pretrained model and its finetuning hyperparameters for new datasets. QuickTune aims to streamline this process by using a Combined Algorithm Selection and Hyperparameter Optimization (CASH) technique within a Bayesian optimization framework.

The approach is based on three key components:
1. **Gray-Box Hyperparameter Optimization (HPO)**: We explore learning curves partially by training models for a few epochs initially and investing more time into the most promising candidates.
2. **Meta-Learning**: We utilize information from previous evaluations on related tasks to guide the search process more effectively.
3. **Cost-Awareness**: We balance the trade-off between time and performance during the search for optimal models and hyperparameters.

Find more information in the paper `Quick-Tune: Quickly Learning Which Pre Trained Model to Fine Tune and How` [ICLR2024](https://openreview.net/forum?id=tqh1zdXIra)

**At the moment only *Image Classification* is implemented.**

## Getting Started

### Installation
Create environment:
```bash
conda create -n qtt python=3.10
conda activate qtt
```

### Install from source
```bash
git clone https://github.com/automl/QTT
pip install -e QTT
```

## Basic Usage
With this repo, we provide pretrained models for quick testing. 

```python
from qtt import QuickTuner, get_pretrained_optimizer
from qtt.finetune.cv.classification import finetune_script, extract_task_info_metafeat

# specify the path to the dataset here
# it will extract the meta-features and create two dictionaries for the optimizer and tuner/finetune_script
task_info, metafeat = extract_task_info_metafeat("path/to/dataset")

# load a pretrained optimizer
optimizer = get_pretrained_optimizer("mtlbm/micro")
# setup the optimizer
n = 100  # number of configs to sample
optimizer.setup(n, metafeat)

# setup and run the tuner
qt = QuickTuner(optimizer, finetune_script)  # pass the optimizer and objective function
qt.run(task_info=task_info, time_budget=3600)  # run for 1 hour
```

The custom dataset must be in Pytorch's [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format, e.g. download the Imagenette dataset:
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvzf imagenette2-320.tgz
```

Modify the quicktuning [script](examples/quicktuning.py) in the examples folder to your needs.


## Advanced Usage
### Download the QuickTune Meta-Album-Dataset:
```bash
wget https://nc.informatik.uni-freiburg.de/index.php/s/K5gbJ72nNz873Ct/download/mtlbm.zip
unzip mtlbm.zip
```

And modify the meta-training [script](examples/create_surrogate.py) in the examples folder to your needs.


### Cite us

If you use this repository, please cite us as:
```
@inproceedings{
arango2024quicktune,
title={Quick-Tune: Quickly Learning Which Pretrained Model to Finetune and How},
author={Sebastian Pineda Arango and Fabio Ferreira and Arlind Kadra and Frank Hutter and Josif Grabocka},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tqh1zdXIra}
}
```
