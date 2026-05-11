import json
import os
import sys
from datetime import datetime
from functools import partial
from math import isfinite
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from metrics import accuracy, calculate_metrics, mIoU
from text_loss import text_loss_factory, text_weight_sched_factory
from utils import (
    BaikalLoss,
    EmbeddingMixup,
    NoScaler,
    ScalerGradNormReturn,
    SchedulerArgs,
    log_formatter,
    save_model_state,
)

try:
    from apex.optimizers import FusedLAMB  # noqa: F401

    apex_available = True
except ImportError:
    logger.error("Nvidia apex not available")
    apex_available = False
try:
    from lion_pytorch import Lion

    lion_available = True
except ImportError:
    logger.error("Lion not available")
    lion_available = False


WANDB_AVAILABLE = False
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    logger.error("wandb not available")


def wandb_available(turn_off=False):
    """If wandb is available.

    Args:
        turn_off (bool, optional): set wandb to be unavailble manually.

    Returns:
        bool: wandb is available
    """
    global WANDB_AVAILABLE
    if turn_off:
        WANDB_AVAILABLE = False
    return WANDB_AVAILABLE


tqdm = partial(tqdm, leave=True, position=0)  # noqa: F405


def setup_tracking_and_logging(args, rank, append_model_path=None, log_wandb=True):
    """Set up logging and tracking for an experiment.

    Args:
        args (DotDict): Parsed command-line arguments
        rank (int): The rank of the current process
        append_model_path (str, optional): Path of an existing model, by default None
        log_wandb (bool, optional): Whether to log to wandb, by default True

    Returns:
        str: folder, where all the run data is saved.

    Raises:
        AssertionError: If `dataset` or `model` is `None`.

    Notes:
        This function sets up logger to stdout and file, as well as MLflow tracking for an experiment.
        For wandb logger, provide .wandb.apikey in the current directory.
    """
    dataset, model, epochs = args.dataset.replace(os.sep, "_").lower(), args.model.replace(os.sep, "_"), args.epochs
    run_folder = os.path.join(
        args.results_folder,
        args.experiment_name,
        args.task.replace("-", ""),
        dataset,
        f"{args.run_name.replace(os.sep, '_')}_{model}_{datetime.now().strftime('%d.%m.%Y_%H:%M:%S')}",
    )
    assert dataset is not None and model is not None

    if os.name == "nt":
        run_folder = run_folder.replace("@", "_").replace(" ", "_").replace(":", ".")

    if append_model_path is not None:
        run_folder = os.path.dirname(append_model_path)
        if "run_name" not in args or args.run_name is None:
            args.run_name = run_folder.split(os.sep)[-1].split("_")[0]
    elif args.distributed:
        obj_list = [None]
        if rank == 0:
            obj_list[0] = run_folder
        dist.broadcast_object_list(obj_list, src=0)
        run_folder = obj_list[0]
        if rank == 0:
            os.makedirs(run_folder, exist_ok=True)
        dist.barrier()
    elif rank == 0:
        os.makedirs(run_folder, exist_ok=True)

    assert "%" not in args.run_name, f"found '%' in run_name '{args.run_name}'. This messes with string formatting..."

    if args.debug:
        args.log_level = "debug"

    # logger to stdout & file
    log_name = args.task.replace("-", "")
    if args.task not in ["pre-train", "fine-tune", "fine-tune-head"]:
        log_name += f"_{dataset}_{datetime.now().strftime('%d.%m.%Y_%H:%M:%S')}"
    log_file = os.path.join(run_folder, f"{log_name}.log")
    logger.remove()
    logger.configure(extra=dict(run_name=args.run_name, rank=rank, world_size=args.world_size))
    logger.add(sys.stderr, format=log_formatter, enqueue=True, colorize=True, level=args.log_level.upper())
    logger.add(log_file, format=log_formatter, enqueue=True, colorize=True, level=args.log_level.upper())
    logger.info(f"Run folder '{run_folder}'")

    if rank == 0:
        logger.info(f"{args.task.replace('-', '').capitalize()} {model} on {dataset} for {epochs} epochs")

        global WANDB_AVAILABLE
        WANDB_AVAILABLE = WANDB_AVAILABLE and log_wandb and os.path.isfile(".wandb.apikey") and args.wandb
        if WANDB_AVAILABLE:
            with open(".wandb.apikey", "r") as f:
                __wandb_api_key = f.read().strip()
            wandb.login(key=__wandb_api_key)
            if args.wandb_run_id is not None:
                wandb_args = dict(project=args.experiment_name, resume="must", id=args.wandb_run_id)
            else:
                wandb_args = dict(
                    project=args.experiment_name,
                    name=args.run_name.replace("_", "-").replace(" ", "-"),
                    config={"logfile": log_file, **dict(args)},
                    job_type=args.task,
                    tags=[model, dataset],
                    resume="allow",
                    id=args.wandb_run_id,
                )
            wandb.init(**wandb_args)
            args["wandb_run_id"] = wandb.run.id
            if rank == 0:
                logger.info(f"wandb run initialized with id {args['wandb_run_id']}.")
        else:
            logger.info(
                f"Not using wandb. (args.wandb={args.wandb}, .wandb.apikey exists={os.path.isfile('.wandb.apikey')},"
                f" function declaration log_wandb={log_wandb})"
            )

    if args.distributed:
        dist.barrier()

    return run_folder


def setup_model_optim_sched_scaler(model, device, epochs, args, head_only=False):
    """Set up model, optimizer, and scheduler with automatic mixed precision (amp) and distributed data parallel (DDP).

    Args:
        model (nn.Module): the loaded model
        device (torch.device): the current device
        epochs (int): total number of epochs to learn for (for scheduler)
        args: further arguments
        head_only (bool, optional): train only the linear head. Default: False

    Returns:
        tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler, ScalerGradNormReturn]: model, optimizer, scheduler, scaler

    """
    model = model.to(device)

    if head_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = model.head.parameters()
    else:
        params = model  # model.named_parameters() use model itself for now and let timm do the work...

    if args.opt == "lion" and not lion_available:
        args.opt = "fusedlamb"
        logger.warning("Falling back from lion to fusedlamb")
    if args.opt == "fusedlamb" and not apex_available:
        args.opt = "adamw"
        logger.warning("Falling back from fusedlamb to adamw")
    if args.opt == "lion":
        optimizer = Lion(params, lr=args["lr"], weight_decay=args["weight_decay"])
    else:
        optimizer = create_optimizer(args, params)

    scaler = ScalerGradNormReturn() if args.amp else NoScaler()

    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     ema_model = ModelEma(model, decay=args.model_ema_decay, resume='')

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device])

    if args.compile_model:
        model = torch.compile(model)

    # scheduler = optim.lr_scheduler.LambdaLR(optimizer,
    #                                         lr_lambda=scheduler_function_factory(**args))
    sched_args = SchedulerArgs(args.sched, args.epochs, args.min_lr, args.warmup_lr, args.warmup_epochs)
    scheduler, _ = create_scheduler(sched_args, optimizer)

    return model, optimizer, scheduler, scaler


def setup_criteria_mixup(args, dataset=None, **criterion_kwargs):
    """Set up further objects that are needed for training.

    Args:
        args: arguments
        dataset (torch.data.Dataset, optional): dataset that implements images_per_class, for class weights (Default value = None)
        criterion_kwargs: further arguments for the criterion
        **criterion_kwargs:

    Returns:
        tuple[nn.Module, nn.Module, Mixup]: criterion, val_criterion, mixup

    """
    weight = None
    if args.loss_weight != "none":
        if dataset is not None and hasattr(dataset, "images_per_class"):
            ipc = dataset.images_per_class
            total_ims = sum(ipc)

            if args.loss_weight == "linear":
                weight = torch.tensor([total_ims / (ims * args.n_classes) for ims in ipc])
            elif args.loss_weight == "log":
                p_c = torch.tensor([ims / total_ims for ims in ipc])
                log_p_c = torch.where(p_c > 0, p_c.log(), torch.zeros_like(p_c))
                entr = -(p_c * log_p_c).sum()
                weight = -log_p_c / entr
            elif args.loss_weight == "sqrt":
                p_c = torch.tensor([ims / total_ims for ims in ipc])
                weight = 1 / (p_c.sqrt() * p_c.sqrt().sum())

        else:
            logger.warning("Could not find images_per_class in dataset. Using uniform weights.")

    if (args.aug_cutmix_alpha or args.multi_label or args.aug_mixup_alpha) and args.ignore_index >= 0:
        if weight is None:
            weight = torch.ones(args.n_classes)
        weight[args.ignore_index] = 0
    if args.multi_label:
        if args.loss == "ce":
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight, **criterion_kwargs)
            val_criterion = nn.BCEWithLogitsLoss(pos_weight=weight, **criterion_kwargs)
        else:
            raise NotImplementedError(
                f"Only BCEWithLogitsLoss (ce) is implemented for multi-label classification, not {args.loss}."
            )
    else:
        if args.loss == "ce":
            loss_cls = nn.CrossEntropyLoss
        elif args.loss == "baikal":
            loss_cls = BaikalLoss
        else:
            raise NotImplementedError(f"'{args.loss}'-loss is not implemented.")
            # criterion = loss_cls(weight=weight, **criterion_kwargs)
            # val_criterion = loss_cls(weight=weight, **criterion_kwargs)
    criterion = loss_cls(
        label_smoothing=args.label_smoothing,
        ignore_index=args.ignore_index if weight is None else -100,
        weight=weight,
        **criterion_kwargs,
    )  # LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    # criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    val_criterion = loss_cls(
        label_smoothing=args.label_smoothing,
        ignore_index=args.ignore_index if weight is None else -100,
        weight=weight,
        **criterion_kwargs,
    )  # LabelSmoothingCrossEntropy(smoothing=0.)

    mixup_kwargs = dict(
        mixup_alpha=args.aug_mixup_alpha,
        cutmix_alpha=args.aug_cutmix_alpha,
        label_smoothing=args.label_smoothing,
        num_classes=args.n_classes,
        ignore_index=args.ignore_index,
    )
    if abs(args.aug_cutmix_alpha) + abs(args.aug_mixup_alpha) > 0.0 or (
        args.label_smoothing and args.multi_label
    ):  # Use Mixup function for label-smooting when using multi-labels
        mixup = EmbeddingMixup(**mixup_kwargs)
    else:
        mixup = None

    return criterion, val_criterion, mixup


def _train(
    model,
    train_loader,
    optimizer,
    rank,
    epochs,
    device,
    mixup,
    criterion,
    world_size,
    scheduler,
    args,
    val_loader,
    val_criterion,
    model_folder,
    scaler,
    do_metrics_calculation=True,
    start_epoch=0,
    show_tqdm=True,
    topk=(1, 5),
    acc_dict_key=None,
    unsupervised_loader=None,
    online_teacher=None,
):
    """Train the model.

    Args:
        model:
        train_loader:
        optimizer:
        rank:
        epochs:
        device:
        mixup:
        criterion:
        world_size:
        scheduler:
        args:
        val_loader:
        val_criterion:
        model_folder:
        scaler:
        do_metrics_calculation:  (Default value = True)
        start_epoch:  (Default value = 0)
        show_tqdm:  (Default value = True)
        topk:  (Default value = (1)
        5):
        acc_dict_key:  (Default value = None)

    Returns:
        dict: evaluation metrics at the end of training

    """
    if acc_dict_key is None:
        acc_dict_key = f"{args.perf_metric}{{}}"
    training_start = time()
    topk = tuple(k for k in topk if k <= args.n_classes)
    time_spend_training = time_spend_validating = 0
    assert (
        online_teacher is None or unsupervised_loader is None
    ), "Code for online teacher + unsupervised loader is not implemented yet"
    current_best_acc = 0.0
    if rank == 0:
        logger.info(f"Dataloader has {len(train_loader)} batches")

        logger.debug("Starting training with the following settings:")
        logger.debug(f"criterion: {criterion}")
        logger.debug(f"train_loader: {train_loader}, sampler: {train_loader.sampler}")
        logger.debug(f"unsupervised loader: {unsupervised_loader}")
        logger.debug(f"dataset: {train_loader.dataset}")
        logger.debug(f"optimizer: {optimizer}")
        logger.debug(f"device: {device}")
        logger.debug(f"start epoch: {start_epoch}, epochs: {epochs}")
        logger.debug(f"scaler: {scaler}")
        logger.debug(f"max_grad_norm: {args.max_grad_norm}")
        # logger.debug(f"model_ema:\n{model_ema}\n{model_ema.decay}\n{model_ema.device}")
        if mixup:
            logger.debug(
                f"mixup: {mixup}; mixup_alpha: {mixup.mixup_alpha}, cutmix_alpha: {mixup.cutmix_alpha},"
                f" cutmix_minmax: {mixup.cutmix_minmax}, prob: {mixup.mix_prob}, switch_prob: {mixup.switch_prob},"
                f" label_smoothing: {mixup.label_smoothing}, num_classes: {mixup.num_classes}, correct_lam:"
                f" {mixup.correct_lam}, mixup_enabled: {mixup.mixup_enabled}"
            )
        else:
            logger.debug(f"mixup: {mixup}")

    text_criterion = text_loss_factory(args.text_loss_function)
    text_loss_weight_function = text_weight_sched_factory(args.text_loss_schedule, args.text_loss_lambda, epochs)

    for epoch in range(start_epoch, epochs):
        with logger.contextualize(epoch=str(epoch + 1)):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            set_ep_func = getattr(train_loader.dataset, "set_epoch", None)
            if callable(set_ep_func):
                train_loader.dataset.set_epoch(epoch)
                val_loader.dataset.set_epoch(epoch)

            epoch_time, epoch_stats = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                rank,
                epoch,
                device,
                mixup,
                criterion,
                world_size,
                scheduler,
                scaler,
                args,
                text_criterion,
                text_loss_weight_function,
                topk,
                "train/" + acc_dict_key,
                show_tqdm,
                unsupervised_loader=unsupervised_loader,
                online_teacher=online_teacher,
            )
            time_spend_training += epoch_time

            val_time, val_stats = _evaluate(
                model,
                val_loader,
                epoch,
                rank,
                device,
                val_criterion,
                args,
                topk,
                "val/" + acc_dict_key,
            )
            time_spend_validating += val_time

            if rank == 0:
                logger.info(f"total_time={time() - training_start}s")

            if rank == 0:
                top1_val_acc = val_stats["val/" + acc_dict_key.format(1)]
                # print metadata for grafana
                metadata = {
                    "epoch": epoch + 1,
                    "progress": (epoch + 1) / args.epochs,
                    **val_stats,
                    **epoch_stats,
                }
                # filter out Nan and infinity values
                metadata = {k: v for k, v in metadata.items() if isfinite(v)}
                print(json.dumps(metadata), flush=True)
                logger.debug(f"printed metadata: {json.dumps(metadata)}")
                if WANDB_AVAILABLE:
                    wandb.log(metadata, step=epoch + 1)

                # saving current state
                if top1_val_acc > current_best_acc or (epoch + 1) % args.save_epochs == 0:
                    reason = "top" if top1_val_acc > current_best_acc else ""  # min(...) will be the top-1 accuracy
                    if reason == "top":
                        current_best_acc = top1_val_acc
                        logger.info(f"found a new best model with {args.perf_metric}: {current_best_acc}")
                    kwargs = dict(
                        model_state=model.state_dict(),
                        stats=metadata,
                        optimizer_state=optimizer.state_dict(),
                        additional_reason=reason,
                        regular_save=(epoch + 1) % args.save_epochs == 0,
                    )
                    if scheduler:
                        kwargs["scheduler_state"] = scheduler.state_dict()
                    save_model_state(
                        model_folder, epoch + 1, args, **kwargs, max_interm_ep_states=args.keep_interm_states
                    )

    if rank == 0:
        end_time = time()
        logger.info(
            f"training done: total time={end_time - training_start}, "
            f"time spend training={time_spend_training}, "
            f"time spend validating={time_spend_validating}"
        )

    results = {**val_stats, **epoch_stats, f"val/best_{acc_dict_key.format(1)}": current_best_acc}

    if rank == 0:
        save_model_state(
            model_folder,
            epoch + 1,
            args,
            model_state=model.state_dict(),
            stats=results,
            additional_reason="final",
            regular_save=False,
            max_interm_ep_states=args.keep_interm_states,
        )

    if do_metrics_calculation:
        # Calculate efficiency metrics
        inp = next(iter(train_loader))[0].to(device)
        metrics = calculate_metrics(
            args,
            model,
            rank=rank,
            input=inp,
            device=device,
            did_training=True,
            all_metrics=False,
            world_size=world_size,
            key_start="eval/",
        )

        if rank == 0:
            logger.info(f"Efficiency metrics: {json.dumps(metrics)}")
    return results


def _mask_preds(preds, cls_masks, mask_val=-100):
    """Mask the predictions by the mask.

    Args:
        preds: model predictions
        cls_masks: class masks
        mask_val:  (Default value = -100)

    Returns:
        torch.Tensor: masked predictions

    """
    if cls_masks is None:
        return preds
    return torch.where(cls_masks.bool(), mask_val, preds)


def _evaluate(
    model,
    val_loader,
    epoch,
    rank,
    device,
    val_criterion,
    args,
    topk=(1, 5),
    acc_dict_key=None,
):
    """Evaluate the model.

    Args:
        model (nn.Module): the model to evaluate
        val_loader (DataLoader): loader for evaluation data
        epoch (int): the current epoch (for logger & tracking)
        rank (int): this processes rank (don't log n times)
        device (torch.device): device to evaluate on
        val_criterion (nn.Module): validation loss
        args (DotDict): further arguments
        topk (tuple[int], optional, optional): top-k accuracy, by default (1, 5)
        acc_dict_key (str, optional, optional): key for the accuracy dictionary, by default name of the performance metric. 'val_' will be prepended.

    Returns:
        tuple[float, float, dict, dict]: validation time, validation loss, validation accuracies, additional information

    """
    if not acc_dict_key:
        acc_dict_key = f"{args.perf_metric}{{}}"

    topk = (1,) if args.perf_metric.lower() == "miou" else tuple(k for k in topk if k <= args.n_classes)
    model.eval()
    val_loss = 0
    val_accs = {acc_dict_key.format(k): 0.0 for k in topk}
    val_start = time()
    n_iters = 0
    iterator = (
        tqdm(val_loader, total=len(val_loader), desc=f"Validating epoch {epoch + 1}")
        if rank == 0 and args.tqdm
        else val_loader
    )
    for batch_data in iterator:
        xs, ys = batch_data[:2]
        cls_masks = batch_data[2].to(device, non_blocking=True) if len(batch_data) == 3 else None

        if args.debug:
            logger.debug(f"y_max = {ys.max()}, y_min = {ys.min()}, num_classes={args.n_classes}")

        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.eval_amp):
            preds = model(xs)
            preds = _mask_preds(preds, cls_masks)

        if args.multi_label:
            # labels are float for BCELoss
            ys = ys.float()
        val_loss += val_criterion(preds.transpose(1, -1), ys.transpose(1, -1) if len(ys.shape) > 1 else ys).item()
        if args.perf_metric.lower() == "acc":
            for key, val in accuracy(
                preds, ys, topk=topk, dict_key=acc_dict_key, ignore_index=args.ignore_index
            ).items():
                val_accs[key] += val
        elif args.perf_metric.lower() == "miou":
            val_accs[list(val_accs.keys())[0]] += mIoU(
                preds, ys, num_classes=args.n_classes, ignore_index=args.ignore_index
            )
        else:
            raise ValueError(f"Unknown performance metric {args.perf_metric}")
        n_iters += 1

    if args.distributed:
        dist.barrier()

    val_end = time()
    iterations = n_iters
    val_accs = {key: val / iterations for key, val in val_accs.items()}
    val_loss = val_loss / iterations

    if args.distributed:
        gather_tensor = torch.Tensor([val_loss, *[val_accs[acc_dict_key.format(k)] for k in topk]]).to(device)
        logger.debug(f"allreduce accuracies: {gather_tensor}")
        dist.barrier()
        dist.all_reduce(gather_tensor, op=dist.ReduceOp.AVG)
        gather_tensor = gather_tensor.tolist()
        val_loss = gather_tensor[0]
        for i, k in enumerate(topk):
            val_accs[acc_dict_key.format(k)] = gather_tensor[i + 1]

    val_accs["val/loss"] = val_loss

    if rank == 0:
        log_s = f"val/time={val_end - val_start}s"
        for key, val in val_accs.items():
            log_s += f", {key}={val:.4f}"
        logger.info(log_s)

    logger.debug("evaluate: done")

    return val_end - val_start, val_accs


def _train_one_epoch(
    model,
    train_loader,
    optimizer,
    rank,
    epoch,
    device,
    mixup,
    criterion,
    world_size,
    scheduler,
    scaler,
    args,
    text_criterion,
    text_loss_weight_function,
    topk=(1, 5),
    acc_dict_key=None,
    show_tqdm=True,
    unsupervised_loader=None,
    online_teacher=None,
):
    """Train the model for one epoch.

    Args:
        model:
        train_loader:
        optimizer:
        rank:
        epoch:
        device:
        mixup:
        criterion:
        scheduler:
        scaler:
        args:
        topk:  (Default value = (1, 5)
        acc_dict_key:  (Default value = None)
        show_tqdm:  (Default value = True)

    Returns:
        tuple[float, float, dict]: time spend in training, epoch loss, epoch accuracies

    """
    if not acc_dict_key:
        acc_dict_key = f"{args.perf_metric}{{}}"

    if args.perf_metric.lower() == "miou":
        topk = (1,)

    model.train()
    iterator = (
        tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch + 1}")
        if rank == 0 and show_tqdm
        else train_loader
    )

    if not args.amp:
        scaler = NoScaler()

    epoch_loss = epoch_cls_loss = epoch_txt_loss = epoch_w_adapt = 0
    epoch_accs = {acc_dict_key.format(k): 0.0 for k in topk}
    epoch_start = time()
    grad_norms = []
    n_iters = 0
    text_loss_weight = text_loss_weight_function(epoch)
    if hasattr(train_loader.dataset, "epoch"):
        train_loader.dataset.epoch = epoch

    logger.debug(
        f"start training: epoch: {epoch}, batches: {len(train_loader)}, distributed: {args.distributed},"
        f" text_loss_weight: {text_loss_weight}, adaptive text loss: {args.adaptive_text_loss_weighting}"
    )
    for i, data in enumerate(iterator):
        if unsupervised_loader is not None and i % len(unsupervised_loader) == 0:
            unsupervised_iter = iter(unsupervised_loader)
        if unsupervised_loader is not None:
            unsuper_xs, _, unsuper_text_labels = next(unsupervised_iter)
        else:
            unsuper_xs = unsuper_text_labels = None

        if len(data) == 3:
            xs, ys, text_labels = data
        else:
            xs, ys = data
            text_labels = None
            if epoch == 0 and i == 0:
                logger.warning(
                    "No text labels for the current dataset! Will not use text-guidance for this training run!"
                )
        optimizer.zero_grad()
        n_iters += 1

        if mixup:
            xs, ys, text_labels = mixup(xs, ys, text_labels)
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            if text_labels is not None:
                text_labels = text_labels.to(device, non_blocking=True)
            if unsuper_xs is not None:
                unsuper_xs, _, unsuper_text_labels = mixup(unsuper_xs, None, unsuper_text_labels)
                unsuper_xs = unsuper_xs.to(device, non_blocking=True)
                unsuper_text_labels = unsuper_text_labels.to(device, non_blocking=True)
        else:
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            if text_labels is not None:
                text_labels = text_labels.to(device, non_blocking=True)
            if unsuper_xs is not None:
                unsuper_xs = unsuper_xs.to(device, non_blocking=True)
                unsuper_text_labels = unsuper_text_labels.to(device, non_blocking=True)

        if online_teacher:
            with torch.no_grad():
                teacher_embeddings = online_teacher(xs).detach()
        else:
            teacher_embeddings = None

        with torch.amp.autocast("cuda", enabled=args.amp):
            last_interm, preds, text_preds = model(xs, return_all_vals=True)

            if unsuper_xs is not None:
                unsuper_interm, _, unsuper_text_preds = model(unsuper_xs, return_all_vals=True)

            if args.debug and (i == 0 or i == len(train_loader) - 1):
                logger.debug(
                    f"shapes: x: {xs.shape}, y: {ys.shape}, text_labels:"
                    f" {text_labels.shape if text_labels is not None else None}, last_interm: {last_interm.shape},"
                    f" preds: {preds.shape}, text_preds: {text_preds.shape}"
                )
                logger.debug(
                    f"Gradient functions: last_interm: {last_interm.grad_fn}, preds: {preds.grad_fn}, text_preds:"
                    f" {text_preds.grad_fn}"
                )
                if unsuper_xs is not None:
                    logger.debug(
                        f"unsuper: x: {unsuper_xs.shape}, text_labels:"
                        f" {unsuper_text_labels.shape if unsuper_text_labels is not None else None}, last_interm:"
                        f" {unsuper_interm.shape}, text_preds: {unsuper_text_preds.shape}"
                    )
            cls_loss = criterion(preds.transpose(1, -1), ys.transpose(1, -1) if len(ys.shape) > 1 else ys)
            if text_labels is not None or teacher_embeddings is not None:
                if args.text_loss_function.lower() == "borlan":
                    text_loss = text_criterion(text_preds, ys)
                elif args.text_loss_function.lower() == "vl2lite":
                    text_loss = text_criterion(text_preds, text_labels, teacher_embeddings)
                else:
                    text_loss = text_criterion(
                        text_preds, teacher_embeddings if teacher_embeddings is not None else text_labels
                    )
            else:
                text_loss = 0.0

            super_bs = xs.shape[0]
            if unsuper_xs is not None and args.text_loss_function.lower() != "borlan" and text_labels is not None:
                unsuper_text_loss = text_criterion(unsuper_text_preds, unsuper_text_labels)
            if args.adaptive_text_loss_weighting and text_loss_weight > 0.0 and text_labels is not None:
                cls_grad = torch.autograd.grad(cls_loss, last_interm, retain_graph=True)[0].norm()
                txt_grad = torch.autograd.grad(text_loss, last_interm, retain_graph=True)[0].norm()
                if unsuper_xs is not None:
                    unsuper_grad = torch.autograd.grad(unsuper_text_loss, unsuper_interm, retain_graph=True)[0].norm()
                    unsuper_bs = unsuper_xs.shape[0]
                else:
                    unsuper_grad = 0.0
                    unsuper_bs = 0
                w_adapt = (
                    (cls_grad + args.opt_eps)
                    / (
                        txt_grad * super_bs / (super_bs + unsuper_bs)
                        + (unsuper_grad * unsuper_bs / (super_bs + unsuper_bs))
                        + args.opt_eps
                    )
                ).item()
                # logger.debug(f"adaptive weight: {w_adapt}: {cls_grad}, txt_grad: {txt_grad}")
            else:
                w_adapt = 1.0

            if unsuper_xs is not None:
                text_loss = text_loss * super_bs / (super_bs + unsuper_bs) + unsuper_text_loss * unsuper_bs / (
                    super_bs + unsuper_bs
                )

            loss = (
                text_loss_weight * w_adapt * text_loss + (1 - text_loss_weight) * cls_loss
                if text_labels is not None
                else cls_loss
            )
            if args.debug and (i == 0 or i == len(train_loader) - 1):
                logger.debug(f"Loss at step {i}/{len(train_loader)}: {loss}")

        if not isfinite(loss.item()):
            logger.error(f"Got loss value {loss.item()}. Stopping training.")
            logger.info(f"input has nan: {xs.isnan().any().item()}")
            logger.info(f"target has nan: {ys.isnan().any().item()}")
            logger.info(f"output has nan: {preds.isnan().any().item()}")
            logger.info(f"cls loss has nan: {cls_loss.isnan().any().item()}")
            logger.info(f"text loss has nan: {text_loss.isnan().any().item()}")
            for name, param in model.named_parameters():
                if param.isnan().any().item():
                    logger.error(f"parameter {name} has a nan value")
            if len(grad_norms) > 0:
                grad_norms = torch.Tensor(grad_norms)
                logger.info(
                    f"Gradient norms until now: min={grad_norms.min().item()}, 20th"
                    f" %tile={torch.quantile(grad_norms, .2).item()}, mean={torch.mean(grad_norms)}, 80th"
                    f" %tile={torch.quantile(grad_norms, .8).item()}, max={grad_norms.max()}"
                )
            sys.exit(1)

        if args.debug and (i == 0 or i == len(train_loader) - 1):
            logger.debug(f"Scaling loss at step {i}/{len(train_loader)}")
        iter_grad_norm = scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            clip_grad=args.max_grad_norm if args.max_grad_norm > 0.0 else None,
        ).cpu()

        if args.gather_stats_during_training and isfinite(iter_grad_norm):
            grad_norms.append(iter_grad_norm)

            if args.aug_cutmix:
                ys = ys.argmax(dim=-1)  # for accuracy with CutMix, just use the argmax for both

            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_txt_loss += 0.0 if isinstance(text_loss, float) else text_loss.item()
            epoch_w_adapt += w_adapt
            if args.perf_metric.lower() == "acc":
                accuracies = accuracy(preds, ys, topk=topk, dict_key=acc_dict_key, ignore_index=args.ignore_index)
                for key in accuracies:
                    epoch_accs[key] += accuracies[key]
            elif args.perf_metric.lower() == "miou":
                epoch_accs[list(epoch_accs.keys())[0]] += mIoU(
                    preds, ys, num_classes=args.n_classes, ignore_index=args.ignore_index
                )
            if args.debug and (i == 0 or i == len(train_loader) - 1):
                logger.debug(
                    f"gathering training stats (local; step {i}/{len(train_loader)}): {epoch_accs}; {grad_norms}"
                )

    if args.distributed:
        logger.debug("Pre dist-barrier")
        dist.barrier()
        logger.debug("Post dist-barrier")
    epoch_end = time()

    iterations = n_iters
    # epoch_accs = {key: val / iterations for key, val in epoch_accs.items()}
    epoch_loss = epoch_loss / iterations
    epoch_cls_loss = epoch_cls_loss / iterations
    epoch_txt_loss = epoch_txt_loss / iterations
    epoch_w_adapt = epoch_w_adapt / iterations
    grad_norm_avrg = -1
    inf_grads = iterations - len(grad_norms)
    if len(grad_norms) > 0 and args.gather_stats_during_training:
        logger.debug("compufing gradient statistics")
        grad_norm_max = max(grad_norms)
        grad_norms = torch.Tensor(grad_norms)
        grad_norm_20 = torch.quantile(grad_norms, 0.2).item()
        grad_norm_80 = torch.quantile(grad_norms, 0.8).item()
        grad_norm_avrg = torch.mean(grad_norms)

    if args.distributed:
        # grad norm is already synchronized
        gather_tensor = torch.Tensor(
            [
                epoch_loss,
                epoch_cls_loss,
                epoch_txt_loss,
                epoch_w_adapt,
                *[epoch_accs[acc_dict_key.format(k)] for k in topk],
            ]
        ).to(device)
        logger.debug(f"Communicating gather tensor: {gather_tensor}")
        dist.barrier()
        dist.all_reduce(gather_tensor)
        gather_tensor = (gather_tensor / world_size).tolist()
        epoch_loss = gather_tensor[0]
        epoch_cls_loss = gather_tensor[1]
        epoch_txt_loss = gather_tensor[2]
        epoch_w_adapt = gather_tensor[3]
        for i, k in enumerate(topk):
            epoch_accs[acc_dict_key.format(k)] = gather_tensor[i + 4]

    logger.debug(f"preparing output dict: {epoch_accs}")
    lr = optimizer.param_groups[0]["lr"]
    epoch_accs["train/lr"] = lr
    epoch_accs["train/loss"] = epoch_loss
    epoch_accs["train/cls-loss"] = epoch_cls_loss
    if args.text_loss_lambda > 0:
        epoch_accs["train/txt-loss"] = epoch_txt_loss
        epoch_accs["train/text-loss-weight"] = text_loss_weight
        if args.adaptive_text_loss_weighting:
            epoch_accs["train/adaptive-text-weight"] = epoch_w_adapt

    if rank == 0:
        if args.gather_stats_during_training:
            print_s = f"train/time={epoch_end - epoch_start}s"
            logger.info(print_s)
            if len(grad_norms) > 0:
                logger.info(
                    f"grad norm avrg={grad_norm_avrg}, grad norm max={grad_norm_max}, "
                    f"inf grad norm={inf_grads}, grad norm 20%={grad_norm_20}, grad norm 80%={grad_norm_80}"
                )
            else:
                logger.warning(f"inf grad norm={inf_grads}")
                logger.error("100% of update steps with infinite grad norms!")
        else:
            logger.info(f"train/time={epoch_end - epoch_start}s")

    if scheduler:
        if isinstance(scheduler, optim.lr_scheduler.LambdaLR):
            scheduler.step()
        else:
            scheduler.step(epoch)

    logger.debug("train one epoch: done")
    if args.gather_stats_during_training:
        return epoch_end - epoch_start, epoch_accs
    return epoch_end - epoch_start, {}
