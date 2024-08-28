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
We provide a simple to use script.

```python
from qtt import QuickCVCLSTuner
tuner = QuickCVCLSTuner("path/to/dataset")
tuner.run(fevals=100, time_budget=3600)
```

For more code examples take a look into the notebooks [folder](notebooks).


## Advanced Usage
### Download the QuickTune Meta-Album-Dataset:
```bash
wget https://nextcloud.tf.uni-freiburg.de/index.php/s/fQmPmB84EmwxddJ/download/mtlbm.zip
unzip mtlbm.zip
```
The Meta-Dataset consists of learning curves generated with different vision datasets of the [Meta Album](https://meta-album.github.io/). They are divided into three different groups: `micro`, ``mini`` and ``extended``.

If you want to train your own predictors, take a look at the examples folder and modify the [script](examples/metatrain.py) to your needs.

## References

The concepts and methodologies of this project are discussed in the following workshop paper:

**Title**: *Quick-Tune-Tool: A Practical Tool and its User Guide for Automatically Finetuning Pretrained Models*  

**Authors**: Ivo Rapant, Lennart Purucker, Fabio Ferreira, Sebastian Pineda Arango, Arlind Kadra, Josif Grabocka, Frank Hutter

**Conference**: AutoML 2024 Workshop

You can access the full paper and additional details on OpenReview [here](https://openreview.net/forum?id=d0Hapti3Uc).


### Further References / Citations

This project is based on the following paper. Please also consider citing this paper:
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
