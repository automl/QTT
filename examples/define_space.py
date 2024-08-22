from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    OrConjunction,
    OrdinalHyperparameter,
)

cs = ConfigurationSpace("cv-classification/pipeline")

# finetuning parameters
freeze = OrdinalHyperparameter("pct_to_freeze", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ld = OrdinalHyperparameter("layer_decay", [0.0, 0.65, 0.75])
lp = OrdinalHyperparameter("linear_probing", [False, True])
sn = OrdinalHyperparameter("stoch_norm", [False, True])
sr = OrdinalHyperparameter("sp_reg", [0.0, 0.0001, 0.001, 0.01, 0.1])
d_reg = OrdinalHyperparameter("delta_reg", [0.0, 0.0001, 0.001, 0.01, 0.1])
bss = OrdinalHyperparameter("bss_reg", [0.0, 0.0001, 0.001, 0.01, 0.1])
cot = OrdinalHyperparameter("cotuning_reg", [0.0, 0.5, 1.0, 2.0, 4.0])

# regularization parameters
mix = OrdinalHyperparameter("mixup", [0.0, 0.2, 0.4, 1.0, 2.0, 4.0, 8.0])
mix_p = OrdinalHyperparameter("mixup_prob", [0.0, 0.25, 0.5, 0.75, 1.0])
cut = OrdinalHyperparameter("cutmix", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
drop = OrdinalHyperparameter("drop", [0.0, 0.1, 0.2, 0.3, 0.4])
smooth = OrdinalHyperparameter("smoothing", [0.0, 0.05, 0.1])
clip = OrdinalHyperparameter("clip_grad", [0, 1, 10])

# optimization parameters
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

# model
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

# max_fidelity
b = Constant("max_fidelity", 50)
cs.add(b)
