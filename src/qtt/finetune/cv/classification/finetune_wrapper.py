import os
import time

import pandas as pd
import yaml

from . import train

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
    fidelity = job["fidelity"]
    data_path = task_info["data-path"]
    output_path = task_info.get("output-path", ".")

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
    if data_augmentation == "auto_augment":
        vers = config.get("auto_augment")
        args += ["--auto_augment", str(vers)]
    elif data_augmentation == "trivial_augment":
        args += [f"--{data_augmentation}"]
    elif data_augmentation == "random_augment":
        ra_num_ops = config.get("ra_num_ops")
        ra_magnitude = config.get("ra_magnitude")
        args += ["--random_augment"]
        args += ["--ra_num_ops", str(ra_num_ops)]
        args += ["--ra_magnitude", str(ra_magnitude)]

    # OPTIMIZER BETAS
    opt_betas = config.get("opt_betas")
    if opt_betas:
        opt_betas = opt_betas.strip("()").split(",")
        args += ["--opt_betas", *opt_betas]

    # TASK SPECIFIC ARGS
    for arg in task_args:
        args += [f"--{arg}", str(task_info[arg])]

    args += ["--fidelity", str(fidelity)]
    args += ["--experiment", str(config_id)]
    args += ["--output", output_path]

    # OUTPUT DIRECTORY
    output_dir = os.path.join(output_path, str(config_id))
    resume_path = os.path.join(output_dir, "last.pth.tar")
    if os.path.exists(resume_path):
        args += ["--resume", resume_path]

    args += static_args

    parser = train.build_parser()
    args, _ = parser.parse_known_args(args)
    args_text = yaml.safe_dump(args.__dict__)

    start = time.time()
    try:
        result = train.main(args, args_text)
    except Exception as e:
        result = e
    end = time.time()
    try:
        summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
        eval_top1 = summary["eval_top1"].iloc[-1]
    except FileNotFoundError:
        result = "No summary.csv found"

    if result is not None:
        report = job.copy()
        report["score"] = 0
        report["cost"] = end - start
        report["status"] = False
        report["info"] = result
        return report

    report = job.copy()
    report["score"] = eval_top1 / 100
    report["cost"] = end - start
    report["status"] = True
    report["info"] = {"path": output_dir}

    return report
