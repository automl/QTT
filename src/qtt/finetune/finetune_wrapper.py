import os

import pandas as pd

from qtt.finetune import finetune
from qtt.finetune.utils.build_parser import build_parser
from qtt.utils.qt_utils import QTaskStatus, QTunerResult


hp_list = [
    "batch_size",
    "bss_reg",
    "clip_grad_norm",
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
    "smoothing",
    "sp_reg",
    "warmup_epochs",
    "warmup_epochs",
    "warmup_lr",
    "weight_decay",
]

bool_hp_list = [
    "amp",
    "linear_probing",
    "stoch_norm",
]

cond_hp_list = [
    "layer_decay",
    "sched",
]

static_args = [
    "--pretrained",
    "--checkpoint_hist", "1",
    "--epochs", "50",
    "--workers", "8",
]

task_args = [
    "train-split",
    "val-split",
    "num_classes",
]


def eval_finetune_conf(
    config: dict,
    budget: int,
    config_id: int,
    data_path: str,
    data_info: dict,
    output: str,
    verbosity: int = 2,
):
    experiment = str(config_id)
    args = [data_path]

    # REGULAR HPS/ARGS
    for hp in hp_list:
        if hp in config:
            args += [f"--{hp}", str(config[hp])]
    
    # CLIP GRAD NORM
    clip_grad_norm = config.get("clip_grad", "None")
    if clip_grad_norm != "None":
        args += ["--clip_grad", str(clip_grad_norm)]

    # BOOLEAN ARGS
    for hp in bool_hp_list:
        enabled = config.get(hp, False)
        if enabled:
            args += [f"--{hp}"]

    # CONDITIONAL ARGS
    for hp in cond_hp_list:
        option = config.get(hp, "None")
        if option != "None":
            args += [f"--{hp}", str(option)]
    
    # DATA AUGMENTATIONS
    data_augmentation = config.get("data_augmentation", "None")
    if data_augmentation != "None":
        if data_augmentation == "auto_augment":
            vers = config.get("auto_augment")
            args += ["--auto_augment", str(vers)]
        else:
            args += [f"--{data_augmentation}"]

    # OPTIMIZER BETAS
    opt_betas = config.get("opt_betas", "None")
    if opt_betas != "None":
        opt_betas = opt_betas.strip("()").split(",")
        args += ["--opt_betas", *opt_betas]

    # TASK SPECIFIC ARGS
    for arg in task_args:
        args += [f"--{arg}", str(data_info[arg])]

    args += ["--epochs_step", str(budget)]
    args += ["--experiment", experiment]
    args += ["--output", output]

    # OUTPUT DIRECTORY
    output_dir = os.path.join(output, experiment)
    resume_path = os.path.join(output_dir, "last.pth.tar")
    if os.path.exists(resume_path):
        args += ["--resume", resume_path]
    
    args += static_args

    parser = build_parser()
    args, _ = parser.parse_known_args(args)
    args.verbosity = verbosity

    try:
        finetune.main(args)
    except Exception as e:
        print("Error:", e)
        result = QTunerResult(
            idx=config_id,
            score=-1,
            time=-1,
            status=QTaskStatus.ERROR,
            info=str(e),
        )
        return result

    # read last line of txt
    summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
    eval_top1 = float(summary["eval_top1"].iloc[-1])
    train_time = float(summary["train_time"].iloc[-1])
    eval_time = float(summary["eval_time"].iloc[-1])
    time = train_time + eval_time
    
    result = QTunerResult(
        idx=config_id,
        score=eval_top1,
        time=time,
        status=QTaskStatus.SUCCESS,
    )

    return result
