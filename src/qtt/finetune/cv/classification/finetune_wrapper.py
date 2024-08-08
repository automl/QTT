import os

from . import finetune
from .utils.build_parser import build_parser

hp_list = [
    "batch_size",
    "bss_reg",
    "clip_grad",
    "cotuning_reg",
    "cutmix",
    "delta_reg",
    "drop",
    "lr",
    "mixup",
    "mixup_prob",
    "model",
    "opt",
    "pct_to_freeze",
    "sched",
    "smoothing",
    "sp_reg",
    "warmup_epochs",
    "warmup_lr",
    "weight_decay",
]

num_hp_list = [
    "clip_grad",
    "layer_decay",
]

bool_hp_list = [
    "amp",
    "linear_probing",
    "stoch_norm",
]

cond_hp_list = ["decay_rate", "decay_epochs", "patience_epochs"]

static_args = [
    "--pretrained",
    "--checkpoint_hist",
    "1",
    "--epochs",
    "50",
    "--workers",
    "8",
]

task_args = [
    "train-split",
    "val-split",
    "num-classes",
]


def finetune_script(
    job: dict,
    task_info: dict,
):
    config = dict(job["config"])
    config_id = job["config_id"]
    epochs_step = job["fidelity"]
    data_path = task_info["data-path"]
    output = task_info.get("output-path", "./output")
    verbosity = task_info.get("verbosity", 2)

    args = [data_path]
    # REGULAR HPS/ARGS
    for hp in hp_list:
        if hp in config:
            args += [f"--{hp}", str(config[hp])]

    # NUMERICAL ARGS (if the value is not 0)
    for hp in num_hp_list:
        value = config.get(hp)
        args += [f"--{hp}", str(value)]

    # BOOLEAN ARGS
    for hp in bool_hp_list:
        enabled = config.get(hp, False)
        if enabled:
            args += [f"--{hp}"]

    # CONDITIONAL ARGS
    for hp in cond_hp_list:
        option = config.get(hp, False)
        if option:
            args += [f"--{hp}", str(option)]

    # DATA AUGMENTATIONS
    data_augmentation = config.get("data_augmentation")
    if data_augmentation != "no_augment":
        if data_augmentation == "auto_augment":
            vers = config.get("auto_augment")
            args += ["--auto_augment", str(vers)]
        else:
            args += [f"--{data_augmentation}"]

    # OPTIMIZER BETAS
    opt_betas = config.get("opt_betas")
    if opt_betas:
        opt_betas = opt_betas.strip("()").split(",")
        args += ["--opt_betas", *opt_betas]

    # TASK SPECIFIC ARGS
    for arg in task_args:
        args += [f"--{arg}", str(task_info[arg])]

    args += ["--epochs_step", str(epochs_step)]
    args += ["--experiment", str(config_id)]
    args += ["--output", output]

    # OUTPUT DIRECTORY
    output = os.path.join(output, str(config_id))
    resume_path = os.path.join(output, "last.pth.tar")
    if os.path.exists(resume_path):
        args += ["--resume", resume_path]

    args += static_args

    parser = build_parser()
    args, _ = parser.parse_known_args(args)
    args.verbosity = verbosity

    try:
        results = finetune.main(args)
    except Exception as e:
        print("Error:", e)
        result = job.copy()
        result["score"] = 0
        result["cost"] = float("inf")
        result["status"] = False
        result["info"] = str(e)
        return result

    out = []
    for epoch, values in results.items():
        result = job.copy()
        score = float(values["eval_top1"]) / 100
        cost = float(values["train_time"]) + float(values["eval_time"])
        result["score"] = score
        result["cost"] = cost
        result["status"] = True
        result["info"] = values
        result["fidelity"] = epoch
        out.append(result)

    return out
