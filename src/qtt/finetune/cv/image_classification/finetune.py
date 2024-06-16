#!/usr/bin/env python3
"""ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Taken from https://github.com/rwightman
Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""

import copy
import logging
import logging.handlers
import os
import time
from argparse import Namespace
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.backends.cuda as cuda
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.utils
from .utils.custom_timm import create_loader
from .utils.finetuning_stategies import (
    BatchSpectralShrinkage,
    BehavioralRegularization,
    CoTuningLoss,
    Relationship,
    SPRegularization,
    convert_to_stoch_norm,
)
from .utils.utils import (
    compute_gradient_norm,
    extend_metrics,
    get_dataset_path,
    get_icgen_dataset_info_json,
    get_number_of_classes,
    prepare_model_for_finetuning,
)
from qtt.utils.log_utils import set_logger_verbosity
from timm import utils
from timm.data import (
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    create_dataset,
    resolve_data_config,
)
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import (
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from timm.models import (
    create_model,
    load_checkpoint,
    model_parameters,
    resume_checkpoint,
    safe_model_name,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp  # type: ignore
    from apex.parallel import DistributedDataParallel as ApexDDP  # type: ignore
    from apex.parallel import convert_syncbn_model  # type: ignore

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb  # type: ignore

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError:
    has_functorch = False

try:
    from syne_tune import Reporter  # type: ignore

    has_synetune = True
except ImportError:
    has_synetune = False


logger = logging.getLogger("finetune")


def main(args: Namespace):
    verbosity = args.verbosity if hasattr(args, "verbosity") else 1
    set_logger_verbosity(verbosity, logger)
    if torch.cuda.is_available():
        cuda.matmul.allow_tf32 = True
        cudnn.benchmark = True
    args.prefetcher = not args.no_prefetcher
    device = utils.init_distributed_device(args)
    if args.distributed:
        logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            project_name = args.project_name
            wandb.init(project=project_name, name=args.experiment, config=args)
        else:
            logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == "apex":
            assert has_apex, "AMP impl specified as APEX but APEX is not installed."
            use_amp = "apex"
            assert args.amp_dtype == "float16"
        else:
            assert (
                has_native_amp
            ), "Please update PyTorch to a version with native AMP (or use APEX)."
            use_amp = "native"
            assert args.amp_dtype in ("float16", "bfloat16")
        if args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.test_mode:
        print("------ Test Mode active ------")
        args.epochs = 1

    if has_synetune and args.report_synetune:
        global report
        current_dir = os.path.dirname(os.path.realpath(__file__))
        report = Reporter()
        args.data_dir = os.path.join(current_dir, args.data_dir)
        print(args.data_dir)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    if args.dataset_augmentation_path:
        try:
            dataset_aug_path = get_dataset_path(
                args.dataset_augmentation_path, args.dataset
            )
            # dataset_path = get_dataset_path(args.data_dir, args.dataset)
            icgen_dataset_info = get_icgen_dataset_info_json(
                dataset_aug_path, args.dataset
            )

            actual_num_classes = icgen_dataset_info["number_classes"]
            original_num_classes = get_number_of_classes(dataset_aug_path)
            args.num_classes = actual_num_classes
            extra_dataset_info = {"icgen_dataset_info": icgen_dataset_info}

            assert args.dataset.startswith(
                "tfdsicgn"
            ), "dataset_augmentation_path is only supported for tfds_icgn datasets"

            print(f"Number of classes in original dataset: {original_num_classes}")
            print(f"Number of classes sampled from: {actual_num_classes}")
            print(
                f"Number of train samples: {icgen_dataset_info['number_train_samples_per_class']*actual_num_classes}"
            )
            print(
                f"Number of test samples: {icgen_dataset_info['number_test_samples_per_class']*actual_num_classes}"
            )

        except Exception as e:
            raise ValueError(f"Error reading dataset information: {e}")
    else:
        assert (
            args.num_classes is not None
        ), "num_classes not inferred from data, please set it manually"
        # we don't need sub-sampling the datasets (only used when ICGen augmentations are used and dataset_augmentation_path is set)
        extra_dataset_info = {}

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
    )

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    _verbose = args.verbosity > 2
    data_config = resolve_data_config(vars(args), model=model, verbose=_verbose)

    # correct data config mean
    if args.in_chans == 1:
        data_config["mean"] = data_config["mean"][:1]
        data_config["std"] = data_config["std"][:1]

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)  # type: ignore

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ""  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == "apex":
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)  # type: ignore

    if args.lr is None:
        global_batch_size = args.batch_size * args.world_size
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = (
                "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
            )
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) "
                f"and global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
            )

    if args.linear_probing:
        pct_to_freeze = 1.0
    else:
        pct_to_freeze = args.pct_to_freeze

    return_features = False
    return_source_output = False

    if args.bss_reg or args.delta_reg or args.cotuning_reg:
        return_features = True
    if args.delta_reg or args.cotuning_reg:
        return_source_output = True

    # to do: add the regularization weight to all methods
    if args.stoch_norm:
        model = convert_to_stoch_norm(model)

    model, head_name = prepare_model_for_finetuning(
        model,
        num_classes=args.num_classes,
        pct_to_freeze=pct_to_freeze,
        return_features=return_features,
        return_source_output=return_source_output,
    )

    backbone_regularizations = []
    if args.sp_reg > 0.0:
        backbone_regularizations.append(SPRegularization(model, head_name, args.sp_reg))
    if args.bss_reg > 0.0:
        backbone_regularizations.append(
            BatchSpectralShrinkage(regularization_weight=args.bss_reg)
        )
    if args.delta_reg > 0.0:
        source_model = copy.deepcopy(model)
        backbone_regularizations.append(
            BehavioralRegularization(source_model, regularization_weight=args.delta_reg)
        )
    if args.cotuning_reg > 0.0:
        compute_relationship = True
        source_model = copy.deepcopy(model)
        backbone_regularizations.append(CoTuningLoss(args.cotuning_reg))
    else:
        compute_relationship = False

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        assert device.type == "cuda"
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == "cuda":
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if utils.is_primary(args):
            logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,  # type: ignore
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == "apex":
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb
            )
        # NOTE: EMA model does not need to be wrapped by DDP

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode="L" if in_chans == 1 else "RGB",
        **extra_dataset_info,
    )

    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        input_img_mode="L" if in_chans == 1 else "RGB",
        **extra_dataset_info,
    )
    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher:
            assert (
                not num_aug_splits
            )  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]
    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        # in_chans=in_chans,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.auto_augment,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
        trivial_augment=args.trivial_augment,
        rand_augment=args.random_augment,
        ra_num_ops=args.ra_num_ops,
        ra_magnitude=args.ra_magnitude,
        persistent_workers=args.persistent_workers,
    )

    eval_workers = args.workers
    if args.distributed and ("tfds" in args.dataset or "wds" in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader(
        dataset_eval,  # type: ignore
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config["crop_pct"],
        pin_memory=args.pin_mem,
        device=device,
        persistent_workers=args.persistent_workers,
    )

    # relationship for the co-tuning loss
    if compute_relationship:
        file_name = (
            os.path.split(args.data_dir)[1]
            + "_"
            + args.model
            + "_"
            + args.dataset_augmentation_path.replace("/", "_")
            + "_"
            + args.dataset.replace("/", "_")
            + ".npy"
        )
        output_dir = utils.get_outdir(
            os.path.join(args.output, "..", "relationships")
            if args.output
            else "./output/relationships"
        )
        relationship = Relationship(
            loader_train,
            source_model,
            device,
            cache=os.path.join(output_dir, file_name),
        )
    else:
        relationship = None
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(
            num_splits=num_aug_splits, smoothing=args.smoothing
        )
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing, target_threshold=args.bce_target_thresh
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = utils.get_outdir(
            args.output if args.output else "./output/train", exp_name
        )

        if has_synetune and args.report_synetune:
            # report = Reporter()
            output_dir = args.st_checkpoint_dir
            os.makedirs(output_dir, exist_ok=True)

        else:
            report = None

        decreasing = True if eval_metric == "loss" else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),  # type: ignore
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    # if args.initial_test:
    #     eval_metrics = validate(
    #         model,
    #         loader_eval,
    #         validate_loss_fn,
    #         args,
    #         amp_autocast=amp_autocast,  # type: ignore
    #         return_features=return_features,
    #         return_source_output=return_source_output,
    #     )
    # else:
    #     eval_metrics = {"loss": -1, "top1": -1, "top5": -1}

    # if utils.is_primary(args):
    #     logger.info(
    #         f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
    #     )

    #     initial_general_log = {
    #         "initial_eval_loss": eval_metrics["loss"],
    #         "initial_eval_top1": eval_metrics["top1"],
    #         "initial_eval_top5": eval_metrics["top5"],
    #         "device_count": device_count,
    #     }

    #     with open(os.path.join(output_dir, "initial_general_log.yml"), "w") as f:
    #         yaml.dump(initial_general_log, f, default_flow_style=False)
    out = OrderedDict()

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.epochs_step != -1 and epoch == args.epochs_step:
                break

            utils.random_seed(args.seed + epoch, args.rank)
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)  # type: ignore
            elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)  # type: ignore

            if args.linear_probing and (epoch == (num_epochs - start_epoch) // 2):
                model, head_name = prepare_model_for_finetuning(
                    model,
                    pct_to_freeze=args.pct_to_freeze,
                    return_features=return_features,
                    return_source_output=return_source_output,
                    change_head=False,
                )

            train_metrics = train_one_epoch(
                epoch=epoch,
                model=model,
                loader=loader_train,
                optimizer=optimizer,
                loss_fn=train_loss_fn,
                args=args,
                device=device,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,  # type: ignore
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                return_features=return_features,
                return_source_output=return_source_output,
                relationship=relationship,
                backbone_regularizations=backbone_regularizations,
                head_name=head_name,
            )

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if utils.is_primary(args):
                    logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == "reduce")

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,  # type: ignore
                return_features=return_features,
                return_source_output=return_source_output,
            )

            if has_synetune and args.report_synetune:
                report(epoch=epoch + 1, eval_accuracy=eval_metrics["top1"])  # type: ignore

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    utils.distribute_bn(
                        model_ema, args.world_size, args.dist_bn == "reduce"
                    )

                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,  # type: ignore
                    log_suffix=" (EMA)",
                    return_features=return_features,
                    return_source_output=return_source_output,
                )
                eval_metrics = ema_eval_metrics

            if output_dir is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, "summary.csv"),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            budget = epoch + 1
            out[epoch] = {"budget": budget}
            out[epoch].update([("train_" + k, v) for k, v in train_metrics.items()])
            out[epoch].update([("eval_" + k, v) for k, v in eval_metrics.items()])

            if not args.test_mode and saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric
                )

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if np.isnan(train_metrics["loss"]):
                break

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    if best_metric is not None:
        logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))

    return out


@extend_metrics
def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
    return_features=False,
    return_source_output=False,
    relationship=None,
    backbone_regularizations=[],
    head_name=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    head_grad_norm = 0.0
    backbone_grad_norm = 0.0
    num_logs = 0.0

    model.train()
    end = time.time()
    num_batches_per_epoch = len(loader)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch

    for batch_idx, (input, target) in enumerate(loader):

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            if return_features:
                if return_source_output:
                    output, source_output, features = output
                else:
                    output, features = output

            loss = loss_fn(output, target)
            # print("0.", loss)
            for backbone_regularization in backbone_regularizations:
                if isinstance(backbone_regularization, SPRegularization):
                    loss += backbone_regularization()
                if isinstance(backbone_regularization, BatchSpectralShrinkage):
                    loss += backbone_regularization(features)
                if isinstance(backbone_regularization, BehavioralRegularization):
                    output = backbone_regularization.source_model(input)
                    if len(output) == 2:
                        output_source, layer_outputs_source = output
                    elif len(output) == 3:
                        output_source, _, layer_outputs_source = output
                    loss += backbone_regularization(layer_outputs_source, features)
                if isinstance(backbone_regularization, CoTuningLoss):
                    source_label = (
                        torch.from_numpy(relationship[target.cpu().numpy()])  # type: ignore
                        .to(device)
                        .float()
                    )
                    loss += backbone_regularization(source_output, source_label)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        if device.type == "cuda":
            torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            num_logs += 1
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            temp_backbone_grad_norm, temp_head_grad_norm = compute_gradient_norm(
                model, head_name
            )
            backbone_grad_norm = (num_logs - 1) * backbone_grad_norm / num_logs + (
                1 / num_logs
            ) * temp_backbone_grad_norm
            head_grad_norm = (num_logs - 1) * head_grad_norm / num_logs + (
                1 / num_logs
            ) * temp_head_grad_norm  # type: ignore

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if utils.is_primary(args):
                # fixing the batch_idx and last_idx

                logger.debug(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:#.4g} ({loss.avg:#.3g})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "BackboneGradNorm: {backbone_grad_norm:.3e}  "
                    "HeadGradNorm: {head_grad_norm:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx + 1,
                        len(loader),
                        100.0 * (batch_idx + 1) / len(loader),
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                        head_grad_norm=head_grad_norm,
                        backbone_grad_norm=backbone_grad_norm,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

        if np.isnan(losses_m.avg):
            break
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict(
        [
            ("loss", losses_m.avg),
            ("head_grad_norm", head_grad_norm),
            ("backbone_grad_norm", backbone_grad_norm),
        ]
    )


@extend_metrics
def validate(
    model,
    loader,
    loss_fn,
    args,
    device=torch.device("cuda"),
    amp_autocast=suppress,
    log_suffix="",
    return_features=False,
    return_source_output=False,
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if args.test_mode:
                if batch_idx > 1:
                    print(
                        "--------- test_mode set to True: breaking epoch for test reasons ---------"
                    )
                    break

            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if return_features:
                    if return_source_output:
                        output, _, _ = output
                    else:
                        (
                            output,
                            _,
                        ) = output
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                logger.debug(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )
    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )

    return metrics


if __name__ == "__main__":
    from .utils.build_parser import build_parser

    parser = build_parser()
    args = parser.parse_args()

    main(args)