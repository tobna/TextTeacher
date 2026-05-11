"""Module to evaluate trained models."""

import os
import random
from contextlib import nullcontext
from datetime import datetime
from time import time

import numpy as np
import torch
from loguru import logger
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from engine import (
    _evaluate,
    setup_criteria_mixup,
    setup_model_optim_sched_scaler,
    setup_tracking_and_logging,
    wandb_available,
)
from load_dataset import prepare_dataset
from metrics import calculate_metrics
from models import load_pretrained
from utils import (
    RepeatedDataset,
    ddp_cleanup,
    ddp_setup,
    get_cpu_name,
    log_args,
    prep_kwargs,
    set_filter_warnings,
)


def evaluate_metrics(model, dataset, **kwargs):
    """Evaluate efficiency metrics for a given model.

    Args:
        model (str): path to model state .tar
        dataset (str): name of the dataset to evaluate on
        **kwargs: further arguments

    """
    set_filter_warnings()
    model_path = model
    args = prep_kwargs(kwargs)
    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        rank = 0
        args.compile_model = False

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    args.model = old_args.model
    args.dataset = dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name
    args.wandb_run_id = old_args.wandb_run_id
    setup_tracking_and_logging(args, rank=rank, append_model_path=model_path, log_wandb=args.wandb_run_id is not None)

    train_loader, args.n_classes, args.ignore_index, args.multi_label, _ = prepare_dataset(dataset, args)

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=True)
    old_args["eval_imsize"] = args.imsize
    args.model = model_name = old_args.model
    args.dataset = dataset
    args.epochs = 5

    model, optim, _, scaler = setup_model_optim_sched_scaler(model, device, epochs=10, args=args, head_only=False)

    if rank == 0:
        logger.info(
            f"Evaluate metrics for model {model_name} on {dataset}. "
            f"It was {old_args.task.replace('-','')}d on {old_args.dataset} for {save_state['epoch']} "
            "epochs."
        )
        # logger.info(f"full set of arguments: {args}")
        logger.info(f"full set of training arguments: {old_args}")
        logger.info(f"full set of eval-metrics arguments: {args}")

    logger.info(
        f"evaluating on {device} -> {torch.cuda.get_device_name(device) if device.type != 'cpu' else get_cpu_name()}"
    )
    metrics = calculate_metrics(
        args, model, rank=rank, device=device, optim=optim, scaler=scaler, train_loader=train_loader, key_start="eval/"
    )
    if rank == 0:
        logger.info(f"Metrics: {metrics}")
        if wandb_available():
            import wandb

            wandb.log(metrics)


def per_class_accuracy(model, **kwargs):
    """Calculate per-class model training accuracy.

    Args:
      model (str): path to model state .tar
      **kwargs: further arguments

    """
    set_filter_warnings()
    model_path = model
    args = prep_kwargs(kwargs)

    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        world_size = 1
        rank = 0
        args.compile_model = False
    args.batch_size = int(args.batch_size / world_size)
    assert world_size == 1 and rank == 0, "Per-class accuracy evaluation is not supported in distributed mode."

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])

    if "dataset" in args and args.dataset is not None:
        assert args.dataset in [
            "train",
            "val",
        ], 'Per-class accuracy evaluation only supported for "train" or "val" dataset'
        eval_on_train = args.dataset == "train"
    else:
        eval_on_train = False

    if "val_dataset" not in args or args.val_dataset is None:
        if "val_dataset" in old_args and old_args.val_dataset is not None:
            args.val_dataset = old_args.val_dataset
        else:
            args.val_dataset = old_args.dataset

    args.dataset = args.val_dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name

    args.model = model_name = old_args.model
    args.dataset = args.val_dataset + ("-train" if eval_on_train else "-val")
    args.wandb_run_id = old_args.wandb_run_id
    run_folder = setup_tracking_and_logging(args, rank, append_model_path=model_path, log_wandb=False)

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=False)
    model = model.to(device)

    logger.info(
        f"Evaluate per-class accuracy of model {model_name}."
        f"It was pretrained on {old_args.dataset} for {save_state['epoch']} epochs."
        f"Evaluating on {args.val_dataset}'s {'train' if eval_on_train else 'validation'} set."
    )

    loader, args.n_classes, args.ignore_index, args.multi_label, unsuper_loader = prepare_dataset(
        args.val_dataset, args, train=eval_on_train
    )

    logger.info("start evaluation")
    logger.info(f"Run info at: '{run_folder}'")

    eval_time, conf_matrix = _confusion_matrix(model=model, loader=loader, rank=rank, device=device, args=args)

    logger.info(f"Evaluation done in {eval_time}s")
    eval_stats_text = {
        key: [
            f"{key:>8d}",
            f"{sum(conf_matrix[key]):>6d}",
            f"{conf_matrix[key][key]/sum(conf_matrix[key])*100 if sum(conf_matrix[key]) > 0 else 0:>2.2f}",
        ]
        for key in range(len(conf_matrix))
    }
    top_k_heading = "class\ttotal\tacc"
    logger.info(
        f"Per-class accuracy:\n\t{top_k_heading}\n\t"
        + "\n\t".join(["\t".join(eval_stats_text[key]) for key in range(len(conf_matrix))])
    )

    conf_mat_out_file = f"confusion_matrix_{args.val_dataset}_{str(datetime.now()).replace(' ', '_')}.csv"
    with open(os.path.join(run_folder, conf_mat_out_file), "w") as f:
        for confusions in conf_matrix:
            f.write(",".join([str(x) for x in confusions]) + "\n")

    logger.info(f"Confusion matrix written to {conf_mat_out_file}")

    ddp_cleanup(args=args, rank=rank)


def _confusion_matrix(model, loader, rank, device, args):
    """Calculate the full confusion matrix for a given model on a dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The dataloader containing the data.
        rank (int): Rank of the process if running in distributed mode. Defaults to 0.
        device (torch.nn.Module): The device to use for computations (e.g., CPU or GPU).
        args (DotDict): Arguments parsed from command line.
        topk (tuple[int]): The top-k accuracy values to track. Defaults to (1, 5).

    Returns:
        tuple: A tuple containing (elapsed_time, per_class_accuracies).
        float: Time taken to calculate per-class accuracy metrics.
        dict: Dictionary containing per-class accuracy metrics.
        - key (int): Class index.
        - value (dict): Dictionary containing accuracy metrics for the class.
        - total (int): Total number of samples for the class.
        - acc1 (float): Top-1 accuracy for the class.
        - ... (float): Additional top-k accuracies for the specified topk values.

    """
    model.eval()
    iterator = tqdm(loader, total=len(loader), desc="per-class accuracy") if rank == 0 and args.tqdm else loader
    confusion_matrix = [[0 for _ in range(args.n_classes)] for __ in range(args.n_classes)]
    start = time()
    for xs, ys in iterator:
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
            preds = model(xs)
        preds = list(preds)
        ys = list(ys)
        for pred, y in zip(preds, ys):
            label_index = y.item() if y.view(-1).shape[0] == 1 else y.argmax().item()
            pred_index = pred.view(-1).argmax().item()
            confusion_matrix[label_index][pred_index] += 1

        end = time()

    return end - start, confusion_matrix


def evaluate(model, dataset=None, val_dataset=None, **kwargs):
    """Evaluate model accuracy.

    Args:
        model (str): path to model state .tar
        dataset (str, optional): name of the dataset to evaluate on (Default value = None)
        val_dataset (str, optional): name of the dataset to evaluate on (Default value = None)
        **kwargs: further arguments
    Note:
        If `val_dataset` is not provided, the model will be evaluated on `dataset`.

    """
    set_filter_warnings()
    model_path = model
    args = prep_kwargs(kwargs)
    if val_dataset is None:
        val_dataset = dataset
    args.dataset = dataset
    args.val_dataset = val_dataset
    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        world_size = 1
        rank = 0
        args.compile_model = False
    args.batch_size = int(args.batch_size / world_size)

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    args.model = old_args.model
    args.dataset = dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name
    args.wandb_run_id = old_args.wandb_run_id
    run_folder = setup_tracking_and_logging(
        args, rank, append_model_path=model_path, log_wandb=args.wandb_run_id is not None
    )

    val_loader, args.n_classes, args.ignore_index, args.multi_label, unsuper_loader = prepare_dataset(
        val_dataset, args, train=False
    )

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=True)
    model = model.to(device)
    args.model = model_name = old_args.model
    args.dataset = dataset

    if rank == 0:
        logger.info(
            f"Evaluate model {model_name} on {val_dataset}. "
            f"It was pretrained on {old_args.dataset} for {save_state['epoch']} epochs."
        )

    if args.distributed:
        model = DDP(model)

    if args.compile_model:
        model = torch.compile(model)

    # log all devices
    logger.info(
        f"evaluating on {device} -> {torch.cuda.get_device_name(device) if device.type != 'cpu' else get_cpu_name()}"
    )
    if rank == 0:
        logger.info(f"torch version {torch.__version__}")
        logger.info(f"full set of arguments: {args}")
        logger.info(f"full set of old arguments: {old_args}")

    if args.seed:
        torch.manual_seed(args.seed)

    val_criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    if rank == 0:
        logger.info("start evaluation")
        logger.info(f"Run info at: '{run_folder}'")

    if rank == 0:
        val_time, val_stats = _evaluate(
            model.to(device),
            val_loader,
            epoch=save_state["epoch"] - 1,
            rank=rank,
            device=device,
            val_criterion=val_criterion,
            args=args,
            acc_dict_key=f"eval_{val_dataset}/{args.perf_metric}{{}}",
        )
        log_s = f"Evaluation done in {val_time}s"
        for key, val in val_stats.items():
            log_s += f", {key}={val:.4f}"
        logger.info(log_s)
        if wandb_available():
            import wandb

            wandb.log(val_stats)
    else:
        _evaluate(
            model.to(device),
            val_loader,
            epoch=save_state["epoch"] - 1,
            rank=rank,
            device=device,
            val_criterion=val_criterion,
            args=args,
            acc_dict_key=f"eval_{val_dataset}/{args.perf_metric}{{}}",
        )

    ddp_cleanup(args=args, rank=rank)


def evaluate_loss_per_example(model, dataset, **kwargs):
    """Evaluate loss per training example.

    Args:
        model (str): path to model state .tar
        dataset (str): name of the dataset to evaluate on
        **kwargs: further arguments
    Note:
        If `val_dataset` is not provided, the model will be evaluated on `dataset`.

    """
    set_filter_warnings()
    model_path = model
    args = prep_kwargs(kwargs)
    args.dataset = dataset
    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        world_size = 1
        rank = 0
        args.compile_model = False
    args.batch_size = int(args.batch_size / world_size)

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    args.model = old_args.model
    args.dataset = dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name
    run_folder = setup_tracking_and_logging(args, rank, append_model_path=model_path, log_wandb=False)

    if rank == 0 and args.shuffle:
        logger.info("Setting args.shuffle to false")

    args.shuffle = False
    loader, args.n_classes, args.ignore_index, args.multi_label, unsuper_loader = prepare_dataset(
        dataset, args, train=True
    )
    rep_dataset = RepeatedDataset(loader.dataset, args.batch_size)
    loader = DataLoader(
        rep_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=loader.collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=True)
    model = model.to(device)
    args.model = model_name = old_args.model
    args.dataset = dataset

    if rank == 0:
        logger.info(
            f"Evaluate loss per example with {model_name} on {dataset} (train set). "
            f"Model was pretrained on {old_args.dataset} for {save_state['epoch']} epochs."
        )

    if args.distributed:
        model = DDP(model)

    if args.compile_model:
        model = torch.compile(model)

    # log all devices
    logger.info(
        f"evaluating on {device} -> {torch.cuda.get_device_name(device) if device.type != 'cpu' else get_cpu_name()}"
    )
    if rank == 0:
        logger.info(f"torch version {torch.__version__}")
        logger.info(f"full set of arguments: {args}")
        logger.info(f"full set of old arguments: {old_args}")

    if args.seed:
        torch.manual_seed(args.seed)

    criterion, _, __ = setup_criteria_mixup(old_args, dataset, reduce=False)
    lpe_file = os.path.join(run_folder, "loss_per_example.csv")
    if args.world_size is not None and args.world_size > 1:
        lpe_file = lpe_file.replace(".csv", f"_rank{rank}.csv")
    logger.info(f"Writing loss per example to {lpe_file}")
    with open(lpe_file, "w+") as f:
        f.write("idx,label,loss mean,loss std,loss min,loss max\n")

    if rank == 0:
        logger.info("start evaluation")
        logger.info(f"Run info at: '{run_folder}'")

    for d_idx, data in enumerate(tqdm(loader, desc="Loss per example")):

        xs, ys = data

        assert all([y == ys[0] for y in ys]), f"Labels not the same: {ys}"
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.eval_amp):
            preds = model(xs)
            losses = criterion(preds, ys)
            mean_loss = losses.mean().item()
            loss_std = losses.std().item()
            min_loss = losses.min().item()
            max_loss = losses.max().item()
        with open(lpe_file, "a") as f:
            f.write(f"{d_idx},{ys[0].item()},{mean_loss},{loss_std},{min_loss},{max_loss}\n")

    ddp_cleanup(args=args, rank=rank)


def extract_embeddings(model, dataset=None, val_dataset=None, n_images=10000, **kwargs):
    set_filter_warnings()
    model_path = model

    args = prep_kwargs(kwargs)
    if val_dataset is None:
        val_dataset = dataset
    args.dataset = val_dataset
    args.val_dataset = val_dataset
    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        world_size = 1
        rank = 0
        args.compile_model = False
    args.batch_size = int(args.batch_size / world_size)
    args.n_images = n_images
    args.world_size = world_size
    assert world_size == 1, "Multinode embedding extraction not supported"

    if args.seed is None:
        logger.info("Setting seed to 42")
        args.seed = 42

    torch.manual_seed(args.seed)

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    args.model = old_args.model
    args.dataset = val_dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name
    args.wandb_run_id = old_args.wandb_run_id
    args.shuffle = False
    args.model_path = model_path
    args.augment_strategy = "minimal"
    run_folder = setup_tracking_and_logging(args, rank, append_model_path=model_path, log_wandb=False)

    logger.info(f"Extracting embeddings of {args.model} at {args.model_path}")

    val_loader, args.n_classes, args.ignore_index, args.multi_label, unsuper_loader = prepare_dataset(
        val_dataset, args, train=True
    )

    model, args, old_args, save_state = load_pretrained(model_path, args, new_dataset_params=False)

    embedding_folder = os.path.join(run_folder, "embeddings", model_path.split(os.sep)[-1].split(".pt")[0])
    os.makedirs(embedding_folder, exist_ok=True)
    log_args(args)
    logger.info(f"Saving embeddings to: {embedding_folder}")

    ds = val_loader.dataset
    assert isinstance(ds, ImageFolder), "Only works with ImageFolder datasets"
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), k=n_images)
    ds = Subset(ds, indices)
    loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers)

    for i, (x, y, txt_emb) in enumerate(tqdm(loader, disable=not args.tqdm, desc="Generating embeddings")):
        batch_indices = range(i * args.batch_size, i * args.batch_size + x.shape[0])
        ds_indices = [ds.indices[idx] for idx in batch_indices]
        index_paths = [ds.dataset.samples[idx][0].split(os.sep)[-1] for idx in ds_indices]

        with torch.amp.autocast("cuda", enabled=args.eval_amp):
            embeddings, _, __ = model(x, return_all_vals=True)

        embeddings = embeddings.unbind(0)
        for id, emb in zip(index_paths, embeddings):
            np.save(os.path.join(embedding_folder, f"{id}.emb.npy"), emb.detach().numpy())


def per_class_emdedding_distribution(model, dataset=None, val_dataset=None, n_images=10000, **kwargs):
    set_filter_warnings()
    model_path = model

    args = prep_kwargs(kwargs)
    if val_dataset is None:
        val_dataset = dataset
    args.dataset = val_dataset
    args.val_dataset = val_dataset
    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        world_size = 1
        rank = 0
        args.compile_model = False
    args.batch_size = int(args.batch_size / world_size)
    args.n_images = n_images
    args.world_size = world_size
    assert world_size == 1, "Multinode embedding extraction not supported"

    if args.seed is None:
        logger.info("Setting seed to 42")
        args.seed = 42

    torch.manual_seed(args.seed)

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    args.model = old_args.model
    args.dataset = val_dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name
    args.wandb_run_id = old_args.wandb_run_id
    args.shuffle = False
    args.model_path = model_path
    args.augment_strategy = "minimal"
    run_folder = setup_tracking_and_logging(args, rank, append_model_path=model_path, log_wandb=False)

    logger.info(f"Extracting embeddings of {args.model} at {args.model_path}")

    val_loader, args.n_classes, args.ignore_index, args.multi_label, unsuper_loader = prepare_dataset(
        val_dataset, args, train=False
    )

    model, args, old_args, save_state = load_pretrained(model_path, args, new_dataset_params=False)

    embedding_folder = os.path.join(run_folder, "fid_data", model_path.split(os.sep)[-1].split(".pt")[0])
    os.makedirs(embedding_folder, exist_ok=True)
    log_args(args)
    logger.info(f"Saving data to: {embedding_folder}")

    ds = val_loader.dataset
    random.seed(args.seed)

    data_store = [None for _ in range(args.n_classes)]
    n_samples = len(ds)
    approx_samples_per_class = 2 * n_samples // args.n_classes
    samples_per_class = [0 for _ in range(args.n_classes)]

    for i, (x, y) in enumerate(tqdm(val_loader, disable=not args.tqdm, desc="Generating embeddings")):
        with torch.amp.autocast("cuda", enabled=args.eval_amp):
            embeddings, _, __ = model(x, return_all_vals=True)

        for lbl, emb in zip(y.unbind(0), embeddings.unbind(0)):
            # data_store[lbl].append(emb.detach().numpy())
            emb = emb.detach().numpy()[0]
            if data_store[lbl] is None:
                data_store[lbl] = np.zeros((approx_samples_per_class, emb.shape[-1]), dtype=emb.dtype)
            if samples_per_class[lbl] >= data_store[lbl].shape[0]:
                data_store[lbl] = np.concatenate(
                    (data_store[lbl], np.zeros((approx_samples_per_class, emb.shape[-1]), dtype=emb.dtype)), axis=0
                )
            data_store[lbl][samples_per_class[lbl]] = emb
            samples_per_class[lbl] += 1

    logger.info("saving outputs")
    for lbl, embs in enumerate(tqdm(data_store, desc="compute statistics for class")):
        # embs = np.stack(embs, axis=0)
        embs = embs[: samples_per_class[lbl] + 1]
        mean = np.mean(embs, axis=0)
        cov = np.cov(embs, rowvar=False)
        # logger.info(f"class: {lbl} -> mean: {mean.tolist()}, cov: {cov.tolist()}")
        np.save(os.path.join(embedding_folder, f"cls_{lbl}_mean.npy"), mean)
        np.save(os.path.join(embedding_folder, f"cls_{lbl}_cov.npy"), cov)
