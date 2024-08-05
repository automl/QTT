from typing import Mapping, Hashable, Any

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    Integer,
    OrConjunction,
    OrdinalHyperparameter,
)

cs = ConfigurationSpace("cv-classification/pipeline")

# finetuning parameters
_meta: Mapping[Hashable, Any] = {"type": "hyperparameters"}
freeze = OrdinalHyperparameter(
    "pct_to_freeze", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], meta=_meta
)
ld = OrdinalHyperparameter("layer_decay", [0.0, 0.65, 0.75], meta=_meta)
lp = OrdinalHyperparameter("linear_probing", [False, True], meta=_meta)
sn = OrdinalHyperparameter("stoch_norm", [False, True], meta=_meta)
sr = OrdinalHyperparameter("sp_reg", [0.0, 0.0001, 0.001, 0.01, 0.1], meta=_meta)
d_reg = OrdinalHyperparameter("delta_reg", [0.0, 0.0001, 0.001, 0.01, 0.1], meta=_meta)
bss = OrdinalHyperparameter("bss_reg", [0.0, 0.0001, 0.001, 0.01, 0.1], meta=_meta)
cot = OrdinalHyperparameter("cotuning_reg", [0.0, 0.5, 1.0, 2.0, 4.0], meta=_meta)

# regularization parameters
mix = OrdinalHyperparameter("mixup", [0.0, 0.2, 0.4, 1.0, 2.0, 4.0, 8.0], meta=_meta)
mix_p = OrdinalHyperparameter("mixup_prob", [0.0, 0.25, 0.5, 0.75, 1.0], meta=_meta)
cut = OrdinalHyperparameter("cutmix", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0], meta=_meta)
drop = OrdinalHyperparameter("drop", [0.0, 0.1, 0.2, 0.3, 0.4], meta=_meta)
smooth = OrdinalHyperparameter("smoothing", [0.0, 0.05, 0.1], meta=_meta)
clip = OrdinalHyperparameter("clip_grad", [0, 1, 10], meta=_meta)

# optimization
amp = OrdinalHyperparameter("amp", [False, True], meta=_meta)
opt = Categorical("opt", ["sgd", "momentum", "adam", "adamw", "adamp"], meta=_meta)
betas = Categorical(
    "opt_betas",
    ["(0.9, 0.999)", "(0.0, 0.99)", "(0.9, 0.99)", "(0.0, 0.999)"],
    meta=_meta,
)
lr = OrdinalHyperparameter(
    "lr", [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01], meta=_meta
)
w_ep = OrdinalHyperparameter("warmup_epochs", [0, 5, 10], meta=_meta)
w_lr = OrdinalHyperparameter("warmup_lr", [0.0, 1e-05, 1e-06], meta=_meta)
wd = OrdinalHyperparameter(
    "weight_decay", [0, 1e-05, 0.0001, 0.001, 0.01, 0.1], meta=_meta
)
bs = OrdinalHyperparameter(
    "batch_size", [2, 4, 8, 16, 32, 64, 128, 256, 512], meta=_meta
)
mom = OrdinalHyperparameter("momentum", [0.0, 0.8, 0.9, 0.95, 0.99], meta=_meta)
sched = Categorical("sched", ["cosine", "step", "multistep", "plateau"], meta=_meta)
pe = OrdinalHyperparameter("patience_epochs", [2, 5, 10], meta=_meta)
dr = OrdinalHyperparameter("decay_rate", [0.1, 0.5], meta=_meta)
de = OrdinalHyperparameter("decay_epochs", [10, 20], meta=_meta)
da = Categorical(
    "data_augmentation",
    ["none", "auto_augment", "random_augment", "trivial_augment"],
    meta=_meta,
)
aa = Categorical("auto_augment", ["v0", "original"], meta=_meta)
ra_nops = Integer("ra_num_ops", (2, 3), meta=_meta)
ra_mag = Integer("ra_magnitude", (5, 10), meta=_meta)
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
)

# model
_meta = {"type": "model"}
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
    meta=_meta,
)
cs.add(model)

# max_fidelity
b = Constant("max_fidelity", 50)
cs.add(b)

cs.to_yaml("space.yaml", indent=2)


########################################################################################
######################## DEFINE THE SPACE FOR THE META-FEATURES ########################
########################################################################################

metafeat = ConfigurationSpace("cv-classification/meta-features")
num_classes = Integer("num_classes", (1, 100))
num_features = Integer("num_features", (1, 100))
num_samples = Integer("num_samples", (1, 100))
num_channels = Integer("num_channels", (1, 3))

metafeat.add(num_classes, num_features, num_samples, num_channels)

metafeat.to_yaml("metafeat.yaml", indent=2)
