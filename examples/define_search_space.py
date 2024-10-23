"""Define Search Space

This examples shows how to define a search space. We use
[ConfigSpace](https://github.com/automl/ConfigSpace).

This search space is defined for a computer vision classification task and includes
various hyperparameters that can be optimized.

First import the necessary modules:
"""

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    OrConjunction,
    OrdinalHyperparameter,
)

cs = ConfigurationSpace("cv-classification")

"""
## Finetuning Parameters
The finetuning parameters in this configuration space are designed to control how a
pre-trained model is fine-tuned on a new dataset. Here's a breakdown of each finetuning
parameter:

1. **`pct_to_freeze`** (Percentage of Model to Freeze):
This parameter controls the fraction of the model's layers that will be frozen during
training. Freezing a layer means that its weights will not be updated. Where `0.0`
means no layers are frozen, and `1.0` means all layers are frozen, except for the
final classification layer.

2. **`layer_decay`** (Layer-wise Learning Rate Decay):
Layer-wise decay is a technique where deeper layers of the model use lower learning
rates than layers closer to the output.

3. **`linear_probing`**:
When linear probing is enabled, it means the training is focused on updating only the
final classification layer (linear layer), while keeping the rest of the model frozen.

4. **`stoch_norm`** ([Stochastic Normalization](https://proceedings.neurips.cc//paper_files/paper/2020/hash/bc573864331a9e42e4511de6f678aa83-Abstract.html)):
Enabling stochastic normalization during training.

5. **`sp_reg`** ([Starting Point Regularization](https://arxiv.org/abs/1802.01483)):
This parameter controls the amount of regularization applied to the weights of the model
towards the pretrained model.

6. **`delta_reg`** ([DELTA Regularization](https://arxiv.org/abs/1901.09229)):
DELTA regularization aims to preserve the outer layer outputs of the target network.

7. **`bss_reg`** ([Batch Spectral Shrinkage Regularization](https://proceedings.neurips.cc/paper_files/paper/2019/hash/c6bff625bdb0393992c9d4db0c6bbe45-Abstract.html)):
Batch Spectral Shrinkage (BSS) regularization penalizes the spectral norm of the model's weight matrices.
   
8. **`cotuning_reg`** ([Co-tuning Regularization](https://proceedings.neurips.cc/paper/2020/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html)):
This parameter controls the strength of co-tuning, a method that aligns the
representation of new data with the pre-trained model's representations
"""
freeze = OrdinalHyperparameter("pct_to_freeze", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ld = OrdinalHyperparameter("layer_decay", [0.0, 0.65, 0.75])
lp = OrdinalHyperparameter("linear_probing", [False, True])
sn = OrdinalHyperparameter("stoch_norm", [False, True])
sr = OrdinalHyperparameter("sp_reg", [0.0, 0.0001, 0.001, 0.01, 0.1])
d_reg = OrdinalHyperparameter("delta_reg", [0.0, 0.0001, 0.001, 0.01, 0.1])
bss = OrdinalHyperparameter("bss_reg", [0.0, 0.0001, 0.001, 0.01, 0.1])
cot = OrdinalHyperparameter("cotuning_reg", [0.0, 0.5, 1.0, 2.0, 4.0])

"""
## Regularization Parameters

- **`mixup`**: A data augmentation technique that mixes two training samples and their labels. The value determines the strength of mixing between samples.
  
- **`mixup_prob`**: Specifies the probability of applying mixup augmentation to a given batch. A value of 0 means mixup is never applied, while 1 means it is applied to every batch.

- **`cutmix`**: Another data augmentation method that combines portions of two images and their labels.

- **`drop`** (Dropout): Dropout is a regularization technique where random neurons in a layer are "dropped out" (set to zero) during training.

- **`smoothing`** (Label Smoothing): A technique that smooths the true labels, assigning a small probability to incorrect classes.

- **`clip_grad`**: This controls the gradient clipping, which constrains the magnitude of gradients during backpropagation.

"""
mix = OrdinalHyperparameter("mixup", [0.0, 0.2, 0.4, 1.0, 2.0, 4.0, 8.0])
mix_p = OrdinalHyperparameter("mixup_prob", [0.0, 0.25, 0.5, 0.75, 1.0])
cut = OrdinalHyperparameter("cutmix", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
drop = OrdinalHyperparameter("drop", [0.0, 0.1, 0.2, 0.3, 0.4])
smooth = OrdinalHyperparameter("smoothing", [0.0, 0.05, 0.1])
clip = OrdinalHyperparameter("clip_grad", [0, 1, 10])

"""
## Optimization Parameters
"""
amp = OrdinalHyperparameter("amp", [False, True])
opt = Categorical("opt", ["sgd", "momentum", "adam", "adamw", "adamp"])
betas = Categorical(
    "opt_betas", ["(0.9, 0.999)", "(0.0, 0.99)", "(0.9, 0.99)", "(0.0, 0.999)"]
)
lr = OrdinalHyperparameter("lr", [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01])
w_ep = OrdinalHyperparameter("warmup_epochs", [0, 5, 10])
w_lr = OrdinalHyperparameter("warmup_lr", [0.0, 1e-05, 1e-06])
wd = OrdinalHyperparameter("weight_decay", [0, 1e-05, 0.0001, 0.001, 0.01, 0.1])
bs = OrdinalHyperparameter("batch_size", [2, 4, 8, 16, 32, 64, 128, 256, 512])
mom = OrdinalHyperparameter("momentum", [0.0, 0.8, 0.9, 0.95, 0.99])
sched = Categorical("sched", ["cosine", "step", "multistep", "plateau"])
pe = OrdinalHyperparameter("patience_epochs", [2, 5, 10])
dr = OrdinalHyperparameter("decay_rate", [0.1, 0.5])
de = OrdinalHyperparameter("decay_epochs", [10, 20])
da = Categorical(
    "data_augmentation",
    ["auto_augment", "random_augment", "trivial_augment", "none"],
)
aa = Categorical("auto_augment", ["v0", "original"])
ra_nops = OrdinalHyperparameter("ra_num_ops", [2, 3])
ra_mag = OrdinalHyperparameter("ra_magnitude", [9, 17])
cond_1 = EqualsCondition(pe, sched, "plateau")
cond_2 = OrConjunction(
    EqualsCondition(dr, sched, "step"),
    EqualsCondition(dr, sched, "multistep"),
)
cond_3 = OrConjunction(
    EqualsCondition(de, sched, "step"),
    EqualsCondition(de, sched, "multistep"),
)
cond_4 = EqualsCondition(mom, opt, "momentum")
cond_5 = OrConjunction(
    EqualsCondition(betas, opt, "adam"),
    EqualsCondition(betas, opt, "adamw"),
    EqualsCondition(betas, opt, "adamp"),
)
cond_6 = EqualsCondition(ra_nops, da, "random_augment")
cond_7 = EqualsCondition(ra_mag, da, "random_augment")
cond_8 = EqualsCondition(aa, da, "auto_augment")
cs.add(
    mix,
    mix_p,
    cut,
    drop,
    smooth,
    clip,
    freeze,
    ld,
    lp,
    sn,
    sr,
    d_reg,
    bss,
    cot,
    amp,
    opt,
    betas,
    lr,
    w_ep,
    w_lr,
    wd,
    bs,
    mom,
    sched,
    pe,
    dr,
    de,
    da,
    aa,
    ra_nops,
    ra_mag,
    cond_1,
    cond_2,
    cond_3,
    cond_4,
    cond_5,
    cond_6,
    cond_7,
    cond_8,
)

"""
## Model Choices

The **model choices** represent a range of state-of-the-art deep learning architectures
for image classification tasks. Each model has different characteristics in terms of
architecture, size, and computational efficiency, providing flexibility to users
depending on their specific needs and resources. Here's an overview:

- **Transformer-based models**: These models, such as BEiT and DeiT, use the transformer
architecture that has become popular in computer vision tasks. They are highly scalable
and effective for large datasets and benefit from pre-training on extensive image
corpora.
  
- **ConvNet-based models**: Models like ConvNeXt and EfficientNet are based on
convolutional neural networks (CNNs), which have long been the standard for image
classification.

- **Lightweight models**: Options such as MobileViT and EdgeNeXt are designed for
resource-constrained environments like mobile devices or edge computing. These models
prioritize smaller size and lower computational costs.
"""
model = Categorical(
    "model",
    [
        "beit_base_patch16_384",
        "beit_large_patch16_512",
        "convnext_small_384_in22ft1k",
        "deit3_small_patch16_384_in21ft1k",
        "dla46x_c",
        "edgenext_small",
        "edgenext_x_small",
        "edgenext_xx_small",
        "mobilevit_xs",
        "mobilevit_xxs",
        "mobilevitv2_075",
        "swinv2_base_window12to24_192to384_22kft1k",
        "tf_efficientnet_b4_ns",
        "tf_efficientnet_b6_ns",
        "tf_efficientnet_b7_ns",
        "volo_d1_384",
        "volo_d3_448",
        "volo_d4_448",
        "volo_d5_448",
        "volo_d5_512",
        "xcit_nano_12_p8_384_dist",
        "xcit_small_12_p8_384_dist",
        "xcit_tiny_12_p8_384_dist",
        "xcit_tiny_24_p8_384_dist",
    ],
)
cs.add(model)
