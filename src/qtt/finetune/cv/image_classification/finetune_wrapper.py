import os

from . import finetune
from .utils.build_parser import build_parser


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
    "num_classes",
]


def finetune_script(
    budget: int,
    config: dict,
    task_info: dict,
):
    config_id = task_info["config_id"]
    data_path = task_info["data_path"]
    output_dir = task_info["output_dir"]
    verbosity = task_info["verbosity"]
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
        args += [f"--{arg}", str(task_info[arg])]

    args += ["--epochs_step", str(budget)]
    args += ["--experiment", str(config_id)]
    args += ["--output", output_dir]

    # OUTPUT DIRECTORY
    output_dir = os.path.join(output_dir, str(config_id))
    resume_path = os.path.join(output_dir, "last.pth.tar")
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
        result = dict(
            config_id=config_id,
            budget=budget,
            score=0,
            cost=float("inf"),
            status=False,
            info=str(e),
        )
        return result

    out = []
    for epoch, values in results.items():
        score = float(values["eval_top1"]) / 100
        cost = float(values["train_time"]) + float(values["eval_time"])
        budget = int(values["budget"])
        result = dict(
            config_id=config_id,
            budget=budget,
            score=score,
            cost=cost,
            status=True,
        )
        out.append(result)
    return out
