# QuickTuneTool: A Framework for Efficient Model Selection and Hyperparameter Optimization

QuickTuneTool tackles the challenge of selecting the best pretrained model and fine-tuning hyperparameters for new datasets. It simplifies this process using a Combined Algorithm Selection and Hyperparameter Optimization (CASH) technique within a Bayesian optimization framework.


### Table of Contents
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
    - [Install from Source](#install-from-source)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [References](#references)
- [Citations](#citations)

The approach relies on three key components:
1. **Gray-Box HPO**: Instead of fully training all models, QuickTuneTool leverages partial learning curves by running models for only a few initial epochs and then focusing on the most promising ones.
2. **Meta-Learning**: The tool draws insights from prior evaluations on related tasks to accelerate and improve the search for optimal models and hyperparameters.
3. **Cost-Awareness**: QuickTuneTool balances the trade-off between time and performance, ensuring an efficient search for the best configurations.

For more details, check out the paper `Quick-Tune: Quickly Learning Which Pre Trained Model to Fine Tune and How` [ICLR2024](https://openreview.net/forum?id=tqh1zdXIra)

**At the moment only *Image Classification* is implemented.**

## Getting Started

### Installation

### Install from source
```bash
git clone https://github.com/automl/QTT
pip install -e QTT  # -e for editable mode
```

## Simple Usage
We provide a simple to use script.

```python
from qtt import QuickCVCLSTuner
tuner = QuickCVCLSTuner("path/to/dataset")
tuner.run(fevals=100, time_budget=3600)
```

For more code examples take a look into the notebooks [folder](notebooks).


## Advanced Usage

Please check out our documentation for more:

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
