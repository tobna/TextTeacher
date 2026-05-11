import os
import random
import sys

import numpy as np
import timm
import torch
from loguru import logger

from engine import (
    _train,
    setup_criteria_mixup,
    setup_model_optim_sched_scaler,
    setup_tracking_and_logging,
    wandb_available,
)
from load_dataset import prepare_dataset
from models import load_pretrained, prepare_model
from online_teacher import make_online_teacher
from utils import ddp_cleanup, ddp_setup, log_args, prep_kwargs, set_filter_warnings


def finetune(model, dataset, epochs, val_dataset=None, head_only=False, **kwargs):
    """Finetune a pretrained model on a given dataset for a specified number of epochs.

    Args:
        model (str): Path to the pretrained model state file (in .tar format).
        dataset (str): Name of the dataset to finetune on.
        val_dataset (str, optional): Name of the validation dataset. (Default value = None)
        epochs (int): Number of epochs to train for.
        head_only (bool, optional): Whether to train only the head of the model. Default: False.
        **kwargs (dict): Further arguments for model setup, training, evaluation,...

    Notes:
        This function assumes that the model was pretrained on a different dataset using a different set of hyperparameters.
        It fine-tunes the model on a new dataset by loading the pretrained weights and training for the specified number of
        epochs. The function supports distributed training using the PyTorch DistributedDataParallel module.
    """
    set_filter_warnings()

    # Add defaults & make keys properties
    args = prep_kwargs(kwargs)

    if val_dataset is None:
        val_dataset = dataset

    args.val_dataset = val_dataset
    args.dataset = dataset
    args.epochs = epochs

    args.distributed, device, world_size, rank, gpu_id = ddp_setup()
    args.world_size = world_size
    try:
        torch.cuda.set_device(device)
    except RuntimeError as e:
        logger.error(
            f"Could not set device {device} as current device; "
            f"environment parameters: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}; SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')}, "
            f"GPU_DEVICE_ORDINAL={os.environ.get('GPU_DEVICE_ORDINAL')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"SLURM_STEP_NODELIST={os.environ.get('SLURM_STEP_NODELIST')}, "
            f"SLURMD_NODENAME={os.environ.get('SLURMD_NODENAME')}"
        )
        raise e
    args.batch_size = int(args.batch_size / world_size)

    if args.seed is not None:
        # fix the seed for reproducibility
        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    save_state = torch.load(model, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    parent_folder = os.path.dirname(model)
    args.model = old_args.model
    run_folder = setup_tracking_and_logging(args, rank)
    if rank == 0:
        if not os.path.exists(os.path.join(run_folder, "parent_run")):
            os.symlink(parent_folder, os.path.join(run_folder, "parent_run"), target_is_directory=True)
        logger.info(
            f"environment parameters: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}; SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')}, "
            f"GPU_DEVICE_ORDINAL={os.environ.get('GPU_DEVICE_ORDINAL')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"SLURM_STEP_NODELIST={os.environ.get('SLURM_STEP_NODELIST')}, "
            f"SLURMD_NODENAME={os.environ.get('SLURMD_NODENAME')}"
        )

    if args.seed:
        logger.info(f"setting manual seed '{seed}' (arg: {args.seed} + rank: {rank})")

    # get the datasets & dataloaders
    args.multi_label = False
    train_loader, args.n_classes, args.ignore_index, args.has_text_embeddings, unsuper_loader = prepare_dataset(
        dataset, args, rank=rank
    )
    val_loader, _val_classes, _, __, ___ = prepare_dataset(val_dataset, args, train=False, rank=rank)
    assert (
        args.n_classes == _val_classes
    ), f"Training and validation datasets have different numbers of classes: {args.n_classes} vs {_val_classes}"

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=True)
    # model_name = old_args.model

    if rank == 0:
        logger.info(f"The model was pretrained on {old_args.dataset} for {save_state['epoch']} epochs.")

    model, optimizer, scheduler, scaler = setup_model_optim_sched_scaler(
        model, device, epochs, args, head_only=head_only
    )
    online_teacher = make_online_teacher(args.online_teacher, device)

    # log all devices
    logger.info(f"training on {device} -> {torch.cuda.get_device_name(device) if args.device != 'cpu' else ''}")
    if rank == 0:
        logger.info(f"torch version {torch.__version__}")
        logger.info(f"timm version {timm.__version__}")
        logger.info(f"full set of old arguments: {old_args}")
        log_args(args)

    if args.seed:
        torch.manual_seed(seed)

    criterion, val_criterion, mixup = setup_criteria_mixup(args)
    if rank == 0:
        logger.info(f"Run info at: '{run_folder}'")

    res = _train(
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
        run_folder,
        scaler,
        do_metrics_calculation=True,
        show_tqdm=args.tqdm,
        unsupervised_loader=unsuper_loader,
        online_teacher=online_teacher,
    )

    if rank == 0:
        best_acc_key = sorted([key for key in res.keys() if key.startswith("val/best_")])[0]
        logger.info(
            f"Run '{run_folder.split(os.sep)[-1]}' is done. Top-1 validation accuracy: {res[best_acc_key] * 100:.2f}%"
        )

    ddp_cleanup(args=args, sync_old_wandb=wandb_available(), rank=rank)


def pretrain(model, dataset, epochs, val_dataset=None, **kwargs):
    """Train or pretrain a model.

    Args:
        model (str): Name of the model to train.
        dataset (str): Name of the dataset to train the model on.
        epochs (int): Number of training epochs.
        val_dataset (str, optional, optional): Name of the validation dataset, by default None
        **kwargs (dict): Additional keyword arguments.

    Notes:
        This function sets up logger, prepares the model, and trains the model on the given dataset.
    """
    set_filter_warnings()

    # Add defaults & make args properties
    args = prep_kwargs(kwargs)

    if val_dataset is None:
        val_dataset = dataset

    args.val_dataset = val_dataset
    args.dataset = dataset
    args.model = model
    args.epochs = epochs

    args.distributed, device, world_size, rank, gpu_id = ddp_setup(args.cuda)
    args.world_size = world_size

    # sleep(rank * 5)
    # logger.debug(f'running environment commands for rank {rank}')
    # os.system('env')
    # os.system('nvidia-smi')
    # sleep((world_size - rank) * 5)

    logger.debug(
        f"rank params: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}; gpu params: "
        f"SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')}, "
        f"GPU_DEVICE_ORDINAL={os.environ.get('GPU_DEVICE_ORDINAL')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    if args.cuda:
        try:
            torch.cuda.set_device(device)
        except RuntimeError as e:
            logger.error(f"Could not set device {device} as current device: {e}")
            logger.error(
                f"environment parameters: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}; SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')}, "
                f"GPU_DEVICE_ORDINAL={os.environ.get('GPU_DEVICE_ORDINAL')}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                f"SLURM_STEP_NODELIST={os.environ.get('SLURM_STEP_NODELIST')}, "
                f"SLURMD_NODENAME={os.environ.get('SLURMD_NODENAME')}"
            )
            raise e

    args.batch_size = int(args.batch_size / world_size)

    run_folder = setup_tracking_and_logging(args, rank)
    if rank % world_size == 0:
        logger.info(
            f"environment parameters: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}; SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')}, "
            f"GPU_DEVICE_ORDINAL={os.environ.get('GPU_DEVICE_ORDINAL')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"SLURM_STEP_NODELIST={os.environ.get('SLURM_STEP_NODELIST')}, "
            f"SLURMD_NODENAME={os.environ.get('SLURMD_NODENAME')}"
        )

    if args.seed is not None:
        # fix the seed for reproducibility
        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"setting manual seed '{seed}' (arg: {args.seed} + rank: {rank})")

    # get the datasets & dataloaders
    args.multi_label = False
    train_loader, args.n_classes, args.ignore_index, args.has_text_embeddings, unsuper_loader = prepare_dataset(
        dataset, args, rank=rank
    )
    val_loader, _val_classes, _, __, ___ = prepare_dataset(val_dataset, args, train=False, rank=rank)
    if args.n_classes != _val_classes:
        logger.warning(
            f"Training and validation datasets have different numbers of classes: {args.n_classes} vs {_val_classes}"
        )
    args.text_embed_dim = train_loader.dataset.text_dim

    # setup model with amp & DDP
    if isinstance(model, str):
        if model.startswith("ViT") and "_" not in model:
            model += f"_{args.imsize}"
        model_name = model
        model = prepare_model(model, args)
    if not model_name:
        model_name = type(model).__name__

    model, optimizer, scheduler, scaler = setup_model_optim_sched_scaler(model, device, epochs, args)

    online_teacher = make_online_teacher(args.online_teacher, device)

    # log all devices
    logger.info(f"training on {device} -> {torch.cuda.get_device_name(device) if device != 'cpu' else ''}")
    if rank == 0:
        logger.info(f"python version {sys.version}")
        logger.info(f"torch version {torch.__version__}")
        logger.info(f"timm version {timm.__version__}")
        log_args(args)

    if args.seed:
        torch.manual_seed(seed)

    criterion, val_criterion, mixup = setup_criteria_mixup(args)

    if rank == 0:
        logger.info(f"Run info at: '{run_folder}'")

    res = _train(
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
        run_folder,
        scaler,
        do_metrics_calculation=True,
        show_tqdm=args.tqdm,
        unsupervised_loader=unsuper_loader,
        online_teacher=online_teacher,
    )

    if rank == 0:
        best_acc_key = [key for key in res.keys() if key.startswith("val/best_")][0]
        logger.info(
            f"Run '{run_folder.split(os.sep)[-1]}' is done. Top-1 validation accuracy: {res[best_acc_key] * 100:.2f}%"
        )

    ddp_cleanup(args=args, sync_old_wandb=wandb_available(), rank=rank)
