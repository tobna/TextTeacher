"""Utils and small helper functions."""

import collections.abc
import json
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from itertools import repeat
from math import cos, pi, sqrt

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from timm.data import Mixup
from timm.utils import NativeScaler, dispatch_clip_grad
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import paths_config
from config import default_kwargs, get_default_kwargs  # noqa: F401  # is used in prep_kwargs


class BaikalLoss(_WeightedLoss):
    r"""Implementation of the Baikal Loss from the Paper [Improved Training Speed, Accuracy, and Data Utilization Through Loss Function Optimization](https://arxiv.org/abs/1905.11528).

    $$
        L = - 1/n \\sum_{i = 1}^n w_0 * \\log(w_1 * p_i) - w_2 \\frac{t_i}{p_i}
    $$
    """

    __constants__ = ["reduction", "ignore_index", "label_smoothing"]
    ignore_index: int
    label_smoothing: float

    def __init__(
        self,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        cma_weights: tuple[float] = None,
        logits_input: bool = True,
    ):
        """Instanciate the Baikal loss.

        The interface is the same as nn.CrossEntropyLoss, with the addition of the CMA weights.

        Args:
            weight (torch.Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size C and floating point dtype. Defaults to None.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. When `reduction` is not `None`, the loss is reduced over non-ignored classes. Note that `ignore_index` ignores samples where the majority label is the ignored one. Defaults to -100.
            reduction (str, optional): Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the weighted mean of the output is taken, `'sum'`: the output will be summed. Defaults to `'mean'`.
            label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). Defaults to 0.0.
            cma_weights (tuple[float] | str, optional): Weights for the cma version of the Baikal loss. `None`: uses the plain cma loss. `'cma'`: uses the weights for the BaikalCMA loss. `tuple[float]`: Given a tuple of length 3, can use user-provided weights according to the equation above. Defaults to None.
            logits_input (bool, optional): The input is logits. We will apply softmax for you. Defaults to True.

        """
        super(BaikalLoss, self).__init__(weight=weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if cma_weights is None:
            cma_weights = (1.0, 1.0, 1.0)
        elif cma_weights == "cma":
            cma_weights = (2.7278 * 0.9863, 1.5352, 1.1135 * 1.3716 / 0.8411)
        assert len(cma_weights) == 3 and isinstance(
            cma_weights, tuple
        ), "CMA weights have to be a tuple of length 3, None, or 'cma'"
        self.cma_weights = cma_weights
        self.logits_input = logits_input

    def forward(self, inp, trgt):
        """Calculate Baikal Loss.

        Args:
            inp (torch.Tensor): model predictions
            trgt (torch.Tensor): correct target labels

        Returns:
            torch.Tensor: loss value

        """
        hard_trgt = trgt
        if len(inp.shape) == len(trgt.shape) + 1:  # trgt is B x ... (one-hot)
            trgt = torch.zeros_like(inp).scatter(1, trgt.unsqueeze(-1), 1)
        else:
            hard_trgt = trgt.argmax(dim=-1)
        # trgt is B x ... x C and hard_trgt is B x ...

        if self.label_smoothing > 0:
            trgt = trgt * (1 - self.label_smoothing) + self.label_smoothing / inp.shape[-1]

        # inp is B x ... x C
        if self.logits_input:
            log_loss = -(self.cma_weights[1] * inp).log_softmax(dim=-1).mean(dim=-1)  # loss term: -log(p) -> B x ...
            inp = inp.softmax(dim=-1)
        else:
            log_loss = -(self.cma_weights[1] * inp).log(dim=-1).mean(dim=-1)  # loss term: -log(p) -> B x ...
        frac_diff = torch.where(trgt > 0, trgt / inp, torch.zeros_like(trgt)).sum(dim=-1)  # loss term: t/p -> B x ...

        if self.ignore_index >= 0:
            mask = hard_trgt != self.ignore_index
            log_loss = log_loss[mask]
            frac_diff = frac_diff[mask]
            hard_trgt = hard_trgt[mask]

        if self.weight is not None:
            loss_weights = self.weight[hard_trgt]
            log_loss = log_loss * loss_weights
            frac_diff = frac_diff * loss_weights
        else:
            loss_weights = None

        if self.reduction == "mean":
            log_loss = log_loss.mean()
            frac_diff = frac_diff.mean()
            if loss_weights is not None:
                log_loss /= loss_weights.mean()
                frac_diff /= loss_weights.mean()
        elif self.reduction == "sum":
            log_loss = log_loss.sum()
            frac_diff = frac_diff.sum()

        return self.cma_weights[0] * log_loss + self.cma_weights[2] * frac_diff


class RepeatedDataset(Dataset):
    """Dataset that repeats the given dataset a number of times."""

    def __init__(self, dataset, num_repeats):
        """Create repeated dataset.

        Args:
            dataset (Dataset): dataset to repeat.
            num_repeats (int): number of repeats.

        """
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __getitem__(self, idx):
        return self.dataset[idx // self.num_repeats]

    def __len__(self):
        return len(self.dataset) * self.num_repeats


@dataclass
class SchedulerArgs:
    """Class for scheduler arguments."""

    sched: str
    epochs: int
    min_lr: float
    warmup_lr: float
    warmup_epochs: int
    cooldown_epochs: int = 0


def scheduler_function_factory(
    epochs, sched, warmup_epochs=0, lr=None, min_lr=0.0, warmup_sched=None, warmup_lr=None, offset=0, **kwargs
):
    """Create a scheduler factor function.

    Args:
        sched (str): the learning rate schedule type
        epochs (int): length of the full schedule
        warmup_epochs (int, optional): number of epochs reserved for warmup (Default value = 0)
        lr (float, optional): learning rate (has to be given, when warmup or min_lr are set) (Default value = None)
        min_lr (float, optional): minimum learning rate (Default value = 0.0)
        warmup_sched (str, optional): the type of schedule during warmup (Default value = None)
        warmup_lr (float, optional): (starting) learning rate during warmup (Default value = None)
        offset (int, optional): offset for the schedule (to be the same as the timm scheduler) (Default value = 0)
        **kwargs: unused

    Returns:
        function: scheduler function

    """
    sched = sched.lower()

    def warmup_f(ep):
        return 1.0

    if warmup_epochs > 0:
        assert warmup_lr is not None, "Need warmup_lr, but got None"
        warmup_lr_factor = warmup_lr / lr
        if warmup_sched == "linear":

            def warmup_f(ep):
                return warmup_lr_factor + (1 - warmup_lr_factor) * max(ep, 0.0) / warmup_epochs

        elif warmup_sched == "const":

            def warmup_f(ep):
                return warmup_lr_factor

        else:
            raise NotImplementedError(f"Warmup schedule {warmup_sched} not implemented")

    epochs = epochs - warmup_epochs + offset
    if sched == "cosine":
        # cos from 0 to pi
        def main_f(ep):
            return cos(pi * ep / epochs) / 2 + 0.5

    elif sched == "const":

        def main_f(ep):
            return 1.0

    else:
        raise NotImplementedError(f"Schedule {sched} is not implemented.")

    # rescale and add min_lr
    min_lr_fact = min_lr / lr

    def main_f_with_min_lr(ep):
        return (1 - min_lr_fact) * main_f(ep) + min_lr_fact

    return lambda ep: (
        warmup_f(ep + offset) if ep + offset < warmup_epochs else main_f_with_min_lr(ep + offset - warmup_epochs)
    )


def one_hot(x, num_classes, ignore_index=None, on_value=1.0, off_value=0.0):
    x = x.long().view(-1, 1)
    if x == ignore_index:
        return torch.full((x.size()[0], num_classes), 1 / num_classes, device=x.device)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


class EmbeddingMixup(Mixup):
    def __init__(self, *args, ignore_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def encoding_mixup(self, encoding, lam=1.0):
        encoding2 = encoding.flip(0)
        return lam * encoding + (1 - lam) * encoding2

    def __call__(self, x, target, encoding=None):
        assert len(x) % 2 == 0, "Batch size should be even when using this"
        if self.mode == "elem":
            lam = self._mix_elem(x)
        elif self.mode == "pair":
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        if target is not None:
            target = self.mixup_target(target, lam=lam)
        if encoding is not None:
            encoding = self.encoding_mixup(encoding, lam=lam)
        return x, target, encoding

    def mixup_target(self, target, lam=1.0):
        off_value = self.label_smoothing / self.num_classes
        on_value = 1.0 - self.label_smoothing + off_value
        y1 = one_hot(target, self.num_classes, on_value=on_value, off_value=off_value)
        y2 = one_hot(target.flip(0), self.num_classes, on_value=on_value, off_value=off_value)
        return y1 * lam + y2 * (1.0 - lam)


class MultiTargetMixup(Mixup):
    """MixUp and CutMix for multi-target classification."""

    def multi_target_mixup(self, target, num_classes, lam=1.0, smooting=0.0, class_mask=None):
        """Mixup the targets.

        For internal use mainly.

        Args:
            target (torch.Tensor): target batch
            num_classes (int): number of classes (total)
            lam (float, optional): Relative mixup weight. Defaults to 1.0.
            smooting (float, optional): Label smoothing factor. Defaults to 0.0.
            class_mask (torch.Tensor, optional): Mask out some of the classes (shape: num_classes). Defaults to None.

        Returns:
            torch.Tensor: mixed target tensor

        """
        assert len(target.shape) == 2 and target.shape[-1] == num_classes, "target has to be one/multiple-hot encoded"
        if class_mask is not None:
            assert (
                len(class_mask.shape) == 2 and class_mask.shape[-1] == num_classes
            ), "class mask has to be one/multiple-hot encoded"
            num_classes = num_classes - class_mask.sum(dim=-1, keepdim=True)
        off_value = smooting / num_classes
        if class_mask is not None:
            off_value = off_value * (1 - class_mask)

        n_correct = target.sum(dim=-1, keepdim=True)
        on_value = 1 / n_correct - smooting + off_value
        y1 = target * on_value + off_value
        y2 = y1.flip(0)
        return y1 * lam + y2 * (1 - lam)

    def __call__(self, x, target, class_mask=None):
        """Mixup the input and target.

        Args:
            x (torch.Tensor): inout batch of images
            target (torch.Tensor): input batch of targets (B x num_classes)
            class_mask (torch.Tensor, optional): Classes to mask out (shape: num_classes). Defaults to None.

        Returns:
            torch.Tensor, torch.Tensor: mixed input and target

        """
        assert len(x) % 2 == 0, "Batch size should be even when using this"
        if self.mode == "elem":
            lam = self._mix_elem(x)
        elif self.mode == "pair":
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = self.multi_target_mixup(target, self.num_classes, lam, self.label_smoothing, class_mask)
        return x, target


class DotDict(dict):
    """Extension of a Python dictionary to access its keys using dot notation."""

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item, default=None):
        """Get item from.

        Args:
            item: key
            default (optional): default value. Defaults to None.

        Returns:
            value

        """
        if item not in self:
            return default
        return self.get(item)


def prep_kwargs(kwargs):
    """Prepare the arguments and add defaults.

    Args:
        kwargs (dict[str, Any]): dict of kwargs

    Returns:
        DotDict: prepared kwargs

    """
    if "defaults" not in kwargs:
        kwargs["defaults"] = "DeiTIII"
    defaults = get_default_kwargs(kwargs["defaults"])
    for k, v in defaults.items():
        if k not in kwargs or kwargs[k] is None:
            kwargs[k] = v

    if "results_folder" not in kwargs:
        kwargs["results_folder"] = paths_config.results_folder  # globals()[var_name]

    if kwargs["results_folder"].endswith("/"):
        kwargs["results_folder"] = kwargs["results_folder"][:-1]

    if "val_dataset" not in kwargs and "dataset" in kwargs:
        kwargs["val_dataset"] = kwargs["dataset"]

    return DotDict(kwargs)


def denormalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Invert the normlize operation.

    Args:
        x (torch.Tensor): images to de-normalize
        mean (tuple, optional): normalization mean. Defaults to (0.485, 0.456, 0.406).
        std (tuple, optional): normalization std. Defaults to (0.229, 0.224, 0.225).

    Returns:
        torch.Tensor: de-normalized images

    """
    operation = transforms.Normalize(
        mean=[-mu / sigma for mu, sigma in zip(mean, std)], std=[1 / sigma for sigma in std]
    )
    return operation(x)


def log_formatter(record):
    if "run_name" not in record["extra"]:
        return (
            "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | name TBD > ?/? | <level>{level: <8}</level> |"
            " <c>{name}</c>.<c>{function}</c>:<c>{line}</c> - {message}\n"
        )
    if "epoch" in record["extra"]:
        return (
            "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | <m>{extra[run_name]}</m> >"
            " <m>{extra[rank]}</m>/<m>{extra[world_size]}</m> @ epoch <y>{extra[epoch]: >3}</y> | <level>{level:"
            " <8}</level> | <c>{name}</c>.<c>{function}</c>:<c>{line}</c> - {message}\n"
        )
    return (
        "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | <m>{extra[run_name]}</m> >"
        " <m>{extra[rank]}</m>/<m>{extra[world_size]}</m> | <level>{level: <8}</level> |"
        " <c>{name}</c>.<c>{function}</c>:<c>{line}</c> - {message}\n"
    )


def ddp_setup(use_cuda=True):
    """Set up the distributed environment.

    Args:
        use_cuda:  (Default value = True)

    Returns:
        tuple: A tuple containing the following elements:
        * bool: Whether the training is distributed.
        * torch.device: The device to use for distributed training.
        * int: The total number of processes in the distributed setup.
        * int: The global rank of the current process in the distributed setup.
        * int: The local rank of the current process on its node.

    Notes:
        The 'nccl' backend is used.

    """
    logger.remove()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    num_gpus = int(os.getenv("WORLD_SIZE", 1))
    distributed = "RANK" in os.environ and num_gpus > 1
    logger.add(sys.stderr, format=log_formatter, colorize=True, enqueue=True)
    if distributed:
        try:
            dist.init_process_group("nccl")
        except ValueError as e:
            logger.error(
                f"Value Error on nccl init. Env params: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')},"
                f" SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')},"
                f" GPU_DEVICE_ORDINAL={os.environ.get('GPU_DEVICE_ORDINAL')} for process:"
                f" RANK={os.getenv('RANK')} (LOCAL_RANK={os.getenv('LOCAL_RANK')}) of"
                f" WORLD_SIZE={os.getenv('WORLD_SIZE')}"
            )
            raise e
        assert torch.cuda.is_available() or not use_cuda, "CUDA is not available"
        assert (
            len(str(os.environ.get("SLURM_STEP_GPUS")).split(","))
            == len(str(os.environ.get("CUDA_VISIBLE_DEVICES")).split(","))
            == len(str(os.environ.get("GPU_DEVICE_ORDINAL")).split(","))
            == num_gpus
        ) or not use_cuda, (
            f"SLURM GPU setup is incorrect: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')},"
            f" SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS')},"
            f" GPU_DEVICE_ORDINAL={os.environ.get('GPU_DEVICE_ORDINAL')} for process:"
            f" RANK={rank} (LOCAL_RANK={local_rank}) of WORLD_SIZE={num_gpus}"
        )
    return distributed, torch.device(f"cuda:{local_rank}") if use_cuda else "cpu", num_gpus, rank, local_rank


def ddp_cleanup(args, sync_old_wandb=False, rank=0):
    """Clean the distributed setup after use.

    Args:
        args (DotDict): arguments
        sync_old_wandb (bool, optional): Whether to sync and remove wandb runs older than 100 hours (>3 days). Defaults to False.
        rank (int, optional): The rank of the current process, so only one process syncs wandb. Defaults to 0.

    """
    if sync_old_wandb and rank == 0:
        os.system("wandb sync --clean --clean-old-hours 100 --clean-force")

    if args.distributed:
        logger.info("waiting for all processes to finish")
        dist.barrier()
        dist.destroy_process_group()
        logger.info("exiting now")
        exit(0)


def set_filter_warnings():
    """Filter out some warnings to reduce spam."""
    # filter DataLoader number of workers warning
    warnings.filterwarnings(
        "ignore", ".*worker processes in total. Our suggested max number of worker in current system is.*"
    )

    # Filter datadings only varargs warning
    warnings.filterwarnings("ignore", ".*only accepts varargs so.*")

    # Filter warnings from calculation of MACs & FLOPs
    # warnings.filterwarnings("ignore", ".*No handlers found:.*")

    # Filter warnings from gather
    warnings.filterwarnings("ignore", ".*is_namedtuple is deprecated, please use the python checks instead.*")

    # Filter warnings from meshgrid
    warnings.filterwarnings("ignore", ".*in an upcoming release, it will be required to pass the indexing.*")

    # Filter warnings from timm when overwriting models
    warnings.filterwarnings("ignore", ".*UserWarning: Overwriting .*")


def remove_prefix(state_dict, prefix="module."):
    """Remove a prefix from the keys in a state dictionary.

    Args:
        state_dict (dict[str, Any]): The state dictionary to remove the prefix from.
        prefix (str, optional): The prefix to remove from the keys. Default is 'module.'.

    Returns:
        dict[str, Any]: A new dictionary with the prefix removed from the keys.

    Examples:
        >>> state_dict = {'module.layer1.weight': 1, 'module.layer1.bias': 2}

        >>> remove_prefix(state_dict)

        {'layer1.weight': 1, 'layer1.bias': 2}

    """
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def prime_factors(n):
    """Calculate the prime factors of a given integer.

    Args:
        n (int): The integer to find the prime factors of.

    Returns:
        list[int]: The prime factors of n.

    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def linear_regession(points):
    """Calculate a linear interpolation of the points.

    Args:
        points (dict[float, float]): points to interpolate in the format points[x] = y

    Returns:
        function: A function that interpolates the points.

    """
    N = len(points)
    x = []
    y = []
    for x_i, y_i in points.items():
        x.append(x_i)
        y.append(y_i)
    x = np.array(x)
    y = np.array(y)

    a = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x * x).sum() - x.sum() ** 2)
    b = (y.sum() - a * x.sum()) / N
    return lambda z: a * z + b


def save_model_state(
    model_folder,
    epoch,
    args,
    model_state,
    regular_save=True,
    stats=None,
    val_accs=None,
    epoch_accs=None,
    additional_reason="",
    max_interm_ep_states=2,
    **kwargs,
):
    """Save the model state.

    Args:
        model_folder: Folder to save model in
        epoch: current epoch
        args: arguments to guide and save
        model_state: state of the model
        regular_save: Is this a regular or a special save? (Default value = True)
        stats: model stats to save (Default value = None)
        val_accs: model accuracy to save (Default value = None)
        epoch_accs: training accuracy to save (Default value = None)
        additional_reason: save reason; in case it's not just a regular save interval. Would be "top" or "final", for example. (Default value = "")
        max_interm_ep_states: Number of regular epoch states to keep (Default value = 2)
        **kwargs: Further arguments to save

    """
    # make args dict, not DotDict to be able to save it
    state = {"epoch": epoch, "model_state": model_state, "run_name": args.run_name, "args": dict(args)}
    if stats is None:
        stats = {}
    if val_accs is not None:
        stats = {**stats, **val_accs}
    if epoch_accs is not None:
        stats = {**stats, **epoch_accs}
    state["stats"] = stats
    state = {**state, **kwargs}
    logger.info(f"saving model state at epoch {epoch} ({additional_reason})")
    regular_file_name = (
        f"ep_{epoch}{f'_{args.perf_metric}_{int(100 * min(val_accs.values()))}' if val_accs is not None else ''}.pt"
    )
    save_name = additional_reason + ".pt" if len(additional_reason) > 0 else regular_file_name
    outfile = os.path.join(model_folder, save_name)
    torch.save(state, outfile)
    if len(additional_reason) > 0 and regular_save:
        shutil.copyfile(outfile, os.path.join(model_folder, regular_file_name))

    # remove intermediate epoch states (all but the last max_interm_ep_states)
    if max_interm_ep_states > 0:
        epoch_states = [f for f in os.listdir(model_folder) if f.startswith("ep_") and f.endswith(".pt")]
        epoch_states = sorted(epoch_states, key=lambda x: int(x.split("_")[1].split(".")[0]))
        if len(epoch_states) > max_interm_ep_states:
            for f in epoch_states[:-max_interm_ep_states]:
                os.remove(os.path.join(model_folder, f))
                logger.debug(f"removed intermediate epoch state {f}")


def log_args(args, rank=0):
    if rank == 0:
        logger.info("full set of arguments: " + json.dumps(dict(args), sort_keys=True))
        # keys = sorted(list(args.keys()))
        # for key in keys:
        #     logger.info(f"arg: {key} = {args[key]}")


class ScalerGradNormReturn(NativeScaler):
    """A wrapper around PyTorch's NativeScaler that returns the gradient norm."""

    def __str__(self) -> str:
        return f"{type(self).__name__}(_scaler: {self._scaler})"

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode="norm", parameters=None, create_graph=False):
        """Scale and backpropagate through the loss tensor, and return the gradient norm of the selected parameters.

        Does an optimizer step.

        Args:
            loss (torch.Tensor): The loss tensor to scale and backpropagate through.
            optimizer (torch.optim.Optimizer): The optimizer to use for the optimization step.
            clip_grad (float, optional): The maximum allowed norm of the gradients. If None, no clipping is performed.
            clip_mode (str, optional): The mode used for clipping the gradients. Only used if `clip_grad` is not None. Possible values are 'norm'
                (clipping the norm of the gradients) and 'value' (clipping the value of the gradients). (default='norm')
            parameters (iterable[torch.nn.Parameter], optional): The parameters to compute the gradient norm for. If None, the gradient norm is not computed.
            create_graph (bool, optional): Whether to create a computation graph for computing second-order gradients. (default=False)

        Returns:
            float: The gradient norm of the selected parameters.

        """
        self._scaler.scale(loss).backward(create_graph=create_graph)

        # always unscale the gradients, since it's being done anyway
        self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        if parameters is not None:
            grads = [p.grad for p in parameters if p.grad is not None]
            device = grads[0].device
            grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2)
        else:
            grad_norm = -1
        if clip_grad is not None:
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()
        return grad_norm


class NoScaler:
    """Dummy gradient scaler that doesn't scale gradients.

    This scaler performs a simple backward pass with the given loss, and then updates the model's parameters
    with the given optimizer. The resulting gradient norm is computed and returned.

    """

    def __str__(self) -> str:
        return f"{type(self).__name__}()"

    def __call__(self, loss, optimizer, parameters=None, **kwargs):
        """Perform backward pass with the given loss, updates the model's parameters with the given optimizer, and computes the resulting gradient norm.

        Args:
            loss (torch.Tensor): The loss tensor that the gradients will be computed from.
            optimizer (torch.optim.Optimizer): The optimizer that will be used to update the model's parameters.
            parameters (iterable[torch.Tensor], optional): An iterable of model parameters to compute gradients. If None, returns -1.
            **kwargs: Additional keyword arguments; nothing will be done with these.

        Returns:
            float: The gradient norm computed after the optimizer step, if parameters is not None.

        """
        loss.backward()
        if parameters is not None:
            grads = [p.grad for p in parameters if p.grad is not None]
            device = grads[0].device
            grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2)
        else:
            grad_norm = -1
        optimizer.step()
        return grad_norm


def get_cpu_name():
    """Get the name of the CPU."""
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":")[1].strip()
    return "unknown"


# From PyTorch internals
def _ntuple(n):
    """Make a function to create n-tuples.

    Args:
        n (int): tuple length

    Returns:
        function: function to create n-tuples

    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Calculate the smallest number >= v that is divisible by divisor.

    This function is primarily used to ensure that the output of a layer
    is divisible by a certain number, typically to align with hardware
    optimizations or memory layouts.

    Args:
        v (int): The input value.
        divisor (int, optional): The divisor. Defaults to 8.
        min_value (int, optional): The minimum value to return. If None, defaults to the divisor.
        round_limit (float, optional): A threshold for rounding down. If the result of rounding down is less than round_limit * v, the next multiple of the divisor is returned instead. Defaults to 0.9.

    Returns:
        int: The smallest number >= v that is divisible by divisor.

    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def grad_cam_reshape_transform(tensor):
    """Transform the tensor for Grad-CAM calculation.

    Args:
        tensor (torch.Tensor): input tensor

    Returns:
        torch.Tensor: reshaped tensor without [CLS] token.

    """
    n_squ = tensor.shape[1]
    result = tensor[:, 1:] if int(sqrt(n_squ)) ** 2 != n_squ else tensor
    bs, n, dim = result.shape
    result = result.reshape(bs, int(sqrt(n)), int(sqrt(n)), dim)

    # Bring the channels to the first dimension,
    # like in CNNs.
    return result.transpose(2, 3).transpose(1, 2)
