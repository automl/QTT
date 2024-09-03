import os
from pathlib import Path

from torchvision.datasets import ImageFolder

from .finetune_wrapper import finetune_script

__all__ = ["finetune_script"]


def extract_task_info_metafeat(
    root: str | Path, train_split: str = "train", val_split: str = "val"
):
    root = Path(root)
    assert root.exists(), f"dataset-path: {root} does not exist."

    num_samples = 0
    num_classes = 0
    num_features = 128  # fixed for now
    num_channels = 3

    # trainset
    train_path = os.path.join(root, train_split)
    if os.path.exists(train_path):
        trainset = ImageFolder(train_path)
        num_samples += len(trainset)
        num_channels = 3 if trainset[0][0].mode == "RGB" else 1
        num_classes = len(trainset.classes)

    # valset
    val_path = os.path.join(root, val_split)
    if os.path.exists(val_path):
        valset = ImageFolder(val_path)
        num_samples += len(valset)

    metafeat = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "num_features": num_features,
        "num_channels": num_channels,
    }

    task_info = {
        "data-path": str(root),
        "train-split": train_split,
        "val-split": val_split,
        "num-classes": num_classes,
    }

    return task_info, metafeat


def load_best_model(path: str):
    import yaml
    from timm.models import create_model, load_checkpoint

    from qtt.finetune.cv.classification.utils import (
        export_model_after_finetuning,
        prepare_model_for_finetuning,
    )

    args = yaml.safe_load(open(os.path.join(path, "args.yaml"), "r"))

    model = create_model(
        args["model"],
        num_classes=args["num_classes"],
    )

    feat_flag = args["bss_reg"] or args["delta_reg"] or args["cotuning_reg"]
    source_flag = args["delta_reg"] or args["cotuning_reg"]
    model, _ = prepare_model_for_finetuning(
        model,
        args["num_classes"],
        return_features=feat_flag,
        return_source_output=source_flag,
    )

    load_checkpoint(model, os.path.join(path, "model_best.pth.tar"))
    model = export_model_after_finetuning(model)

    return model
