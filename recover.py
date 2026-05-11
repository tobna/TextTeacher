"""Continue pretraining / finetuning after something went wrong."""

import torch
from loguru import logger

from engine import (
    _train,
    setup_criteria_mixup,
    setup_model_optim_sched_scaler,
    setup_tracking_and_logging,
)
from load_dataset import prepare_dataset
from models import load_pretrained
from online_teacher import make_online_teacher
from utils import ddp_cleanup, ddp_setup, log_args, prep_kwargs


def continue_training(model, **kwargs):
    """Continue training a model from a saved state.

    Args:
        model (str): path to saved state.
        **kwargs: additional keyword arguments.

    """
    model_path = model
    save_state = torch.load(model, map_location="cpu")

    # state is of the form
    #
    # state = {'epoch': epochs,
    #          'model_state': model.state_dict(),
    #          'optimizer_state': optimizer.state_dict(),
    #          'scheduler_state': scheduler.state_dict(),
    #          'args': dict(args),
    #          'run_name': run_name,
    #          'stats': metrics}

    args = prep_kwargs(save_state["args"])

    args.distributed, device, world_size, rank, gpu_id = ddp_setup()
    torch.cuda.set_device(device)

    if "world_size" in args and args.world_size is not None:
        global_bs = args.batch_size * args.world_size
    else:
        # assume global bs is given in kwargs
        global_bs = kwargs["batch_size"]
    args.batch_size = int(global_bs / world_size)
    args.world_size = world_size

    if "dataset" in args and args.dataset is not None:
        dataset = args.dataset
    else:
        # get default dataset for the task
        dataset = "ImageNet21k" if args.task == "pre-train" else "ImageNet"
        args.dataset = dataset

    if "val_dataset" in args and args.val_dataset is not None:
        val_dataset = args.val_dataset
    else:
        val_dataset = dataset
        args.val_dataset = val_dataset

    start_epoch = save_state["epoch"]
    if "epochs" in args and args.epochs is not None and args.epochs != start_epoch:
        epochs = args.epochs
    else:
        epochs = kwargs["epochs"]

    run_folder = setup_tracking_and_logging(args, rank, append_model_path=model_path)
    logger.info(f"Logging run information to '{run_folder}'")

    # get the datasets & dataloaders
    args.multi_label = False
    train_loader, args.n_classes, args.ignore_index, args.has_text_embeddings, unsuper_loader = prepare_dataset(
        dataset, args, rank=rank
    )
    val_loader = prepare_dataset(val_dataset, args, train=False, rank=rank)[0]

    # model_name = args.model

    logger.info(f"Loading pretrained model state from: {model_path}")
    model, args, _, __ = load_pretrained(model_path, args)

    logger.info("Recreating optimizer and scheduler states")
    model, optimizer, scheduler, scaler = setup_model_optim_sched_scaler(model, device, epochs, args)
    online_teacher = make_online_teacher(args.online_teacher, device=device)

    try:
        optimizer.load_state_dict(save_state["optimizer_state"])
    except ValueError as e:
        logger.error(f"Could not load optimizer state: {e}")
        logger.error(
            f"optimizer state: {optimizer.state_dict().keys()}, param groups: {optimizer.state_dict()['param_groups']}"
        )
        logger.error(
            f"saved state: {save_state['optimizer_state'].keys()}, param groups:"
            f" {save_state['optimizer_state']['param_groups']}"
        )
        raise e

    scheduler.load_state_dict(save_state["scheduler_state"])

    # log all devices
    logger.info(f"training on {device} -> {torch.cuda.get_device_name(device) if args.device != 'cpu' else ''}")
    if rank == 0:
        logger.info(f"torch version {torch.__version__}")
        log_args(args)

    if args.seed:
        torch.manual_seed(args.seed)

    criterion, val_criterion, mixup = setup_criteria_mixup(args)

    if rank == 0:
        logger.info(f"start training at epoch {start_epoch}")
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
        scaler=scaler,
        do_metrics_calculation=True,
        start_epoch=start_epoch,
        show_tqdm=args.tqdm,
        unsupervised_loader=unsuper_loader,
        online_teacher=online_teacher,
    )

    if rank == 0:
        best_acc_key = [key for key in res.keys() if key.startswith("val/best_")][0]
        logger.info(f"Run '{args.run_name}' is done. Top-1 validation accuracy: {res[best_acc_key] * 100:.2f}%")
    ddp_cleanup(args=args, rank=rank)
