import math
from functools import partial
from math import cos

import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F

from infonce_loss import info_nce

_loss_registry = {}
_sched_registry = {}


def _register_loss(func):
    """A decorator to register a function in the global operation_registry, using the function's own name as the key."""
    operation_name = func.__name__  # Get the name of the function
    if not operation_name.endswith("_loss"):
        raise ValueError(f"Can only register losses. Those have to end with '_loss'. Got: {operation_name}.")
    operation_name = operation_name[:-5]
    if operation_name.startswith("_"):
        operation_name = operation_name[1:]
    _loss_registry[operation_name] = func
    return func


def _register_schedule(func):
    """A decorator to register a function in the global operation_registry, using the function's own name as the key."""
    operation_name = func.__name__  # Get the name of the function
    if not operation_name.endswith("_sched"):
        raise ValueError(f"Can only register schedules. Those have to end with '_sched'. Got: {operation_name}.")
    operation_name = operation_name[:-6]
    if operation_name.startswith("_"):
        operation_name = operation_name[1:]
    _sched_registry[operation_name] = func
    return func


def text_loss_factory(loss_name, **kwargs):
    """Return the desired loss function. Can be called with additional kwargs that will be passed to the loss.

    Args:
        loss_name(str): Name of the loss function.
        kwargs: Additional kwargs to be passed to the loss function

    Return:
        callable: text loss function
    """
    if loss_name not in _loss_registry:
        raise NotImplementedError(f"loss {loss_name} is not implemented. Choose one of: {_loss_registry.keys()}")

    if len(kwargs) == 0:
        return _loss_registry[loss_name]

    return partial(_loss_registry[loss_name], **kwargs)


def text_weight_sched_factory(sched_name, max_weight, end_epoch):
    """Return the desired weigth schedule.

    A weight schedule is a function from 0 <= epoch < num_epochs to 0 <= text weight <= 1.
    """
    if sched_name.startswith("jump-"):
        _, start_trans, end_trans, rel_min_weight = sched_name.split("-")
        start_trans, end_trans = int(start_trans), int(end_trans)
        rel_min_weight = float(rel_min_weight)
        logger.info(
            f"{sched_name}: Text weight schedule will be {max_weight} until epoch {start_trans}. Then transition"
            f" linearly to {max_weight  * rel_min_weight} ({rel_min_weight} * {max_weight}) until epoch {end_trans}."
        )
        return partial(
            _jump_sched,
            max_weight=max_weight,
            min_weight=max_weight * rel_min_weight,
            start_trans=start_trans,
            end_trans=end_trans,
        )
    if sched_name not in _sched_registry:
        raise NotImplementedError(
            f"Schedule {sched_name} is not implemented. Choose one of: {_sched_registry.keys()} or a"
            " jump-<start>-<end>-<rel_diff> schedule."
        )

    return partial(_sched_registry[sched_name], max_weight=max_weight, end_epoch=end_epoch)


def _jump_sched(epoch, max_weight, start_trans, min_weight, end_trans):
    if epoch <= start_trans:
        return max_weight
    if epoch > end_trans:
        return min_weight
    return min_weight + (max_weight - min_weight) * (epoch - end_trans) / (start_trans - end_trans)


@_register_schedule
def _const_sched(epoch, max_weight, end_epoch):
    if epoch < end_epoch:
        return max_weight
    return 0.0


@_register_schedule
def _lin_sched(epoch, max_weight, end_epoch):
    if epoch >= end_epoch:
        return 0.0
    return (end_epoch - epoch) / end_epoch * max_weight


@_register_schedule
def _cos_sched(epoch, max_weight, end_epoch):
    if epoch >= end_epoch:
        return 0.0
    return cos(epoch / end_epoch * math.pi / 2.0) * max_weight


@_register_schedule
def _shiftcos_sched(epoch, max_weight, end_epoch):
    if epoch >= end_epoch:
        return 0.0
    return (cos((epoch / end_epoch + 1) * math.pi / 2.0) + 1) * max_weight


@_register_schedule
def _fullcos_sched(epoch, max_weight, end_epoch):
    if epoch >= end_epoch:
        return 0.0
    return 0.5 * max_weight * (cos(math.pi * epoch / end_epoch) + 1.0)


@_register_schedule
def _cntrcos_sched(epoch, max_weight, end_epoch):
    if epoch < end_epoch / 3:
        return max_weight
    if epoch >= end_epoch * 2 / 3:
        return 0.0
    return _fullcos_sched(epoch - end_epoch / 3, max_weight=max_weight, end_epoch=end_epoch / 3)


@_register_schedule
def _vl2_sched(epoch, max_weight, end_epoch):
    weight = epoch / (end_epoch * 8)
    return 1 - weight


@_register_loss
def _clip_loss(model_output, text_encodings):
    logits = model_output @ text_encodings.T
    labels = torch.arange(model_output.size(0), device=logits.device)
    loss_I = F.cross_entropy(logits, labels)
    loss_T = F.cross_entropy(logits.T, labels)
    return (loss_I + loss_T) / 2.0


@_register_loss
def _normclip_loss(model_output, text_encodings):
    model_output = F.normalize(model_output, dim=-1)
    text_encodings = F.normalize(text_encodings, dim=-1)
    return _clip_loss(model_output, text_encodings)


@_register_loss
def _l2_loss(model_output, text_encodings):
    return F.mse_loss(model_output, text_encodings)


@_register_loss
def _cos_loss(model_output, text_encodings):
    cossim = F.cosine_similarity(model_output, text_encodings, dim=-1)
    return (1 - cossim).mean()


@_register_loss
def _kl_loss(model_output, text_encodings):
    return F.kl_div(
        model_output.log_softmax(dim=0),
        text_encodings.log_softmax(dim=0),
        log_target=True,
        reduction="batchmean",
    )


@_register_loss
def _dist_loss(model_output, text_encodings):
    model_dists = model_output.unsqueeze(0) - model_output.unsqueeze(1)
    model_dists = (model_dists / model_dists.norm()).norm(dim=-1)
    text_dists = text_encodings.unsqueeze(0) - text_encodings.unsqueeze(1)
    text_dists = (text_dists / text_dists.norm()).norm(dim=-1)
    return (model_dists - text_dists).norm()


@_register_loss
def _lncos_loss(model_output, text_encodings, eps=1e-12):
    cossim = F.cosine_similarity(model_output, text_encodings, dim=-1)
    return (-(0.5 * cossim + 0.5 + eps).log()).mean()


@_register_loss
def _infonce_loss(model_output, text_encodings):
    return info_nce(model_output, text_encodings)


_MEANS = None
_COVS = None


@_register_loss
def _borlan_loss(query, label):
    global _MEANS, _COVS
    if _MEANS is None:
        logger.info("Loading BorLan means and covs (from /fscratch). borlan loss only works with ImageNet for now!")
        filename = "/fscratch/nauen/text_encodings/ImageNet-BorLan/imagenet_BertL.pt"
        all_text_features = torch.load(filename)

        all_means = []
        all_covs = []
        for c in range(all_text_features.size(0)):
            cls_text_features = all_text_features[c].cpu().numpy()
            mean = np.mean(cls_text_features, axis=0)  # 1024
            cov = np.cov(cls_text_features.T)  # 1024 x 1024
            mean = torch.from_numpy(mean)
            cov = torch.from_numpy(cov)
            cov = cov.diag()
            all_means.append(mean)
            all_covs.append(cov)
        _MEANS = torch.stack(tuple(all_means), dim=0).float().cuda()
        _COVS = torch.stack(tuple(all_covs), dim=0).float().cuda()
    T = 0.07
    query_mean = query.mm(_MEANS.permute(1, 0).float())  # N*K
    covs = _COVS / T
    query_cov_query = 0.5 * query.pow(2).mm(covs.permute(1, 0))
    logits = query_mean + query_cov_query

    # apply temperature
    logits /= T
    ce_loss = F.cross_entropy(logits, label, reduction="none")

    # label under mixup
    if len(label.shape) > 1:
        label = label.argmax(dim=-1)

    key_covs = covs[label]
    jcl_loss = (0.5 * torch.sum(query.pow(2).mul(key_covs), dim=1)) / T
    # return (F.cross_entropy(logits, labels, reduction='none')*mask).mean() + jcl_loss
    loss = ce_loss + jcl_loss
    return loss.mean()


@_register_loss
def _vl2litetext_loss(output, text_enc, T=2.0):
    output = F.normalize(output, dim=-1, p=2.0)
    text_enc = F.normalize(text_enc, dim=-1, p=2.0)
    txt_sims = text_enc @ text_enc.T / T
    img_sims = output @ text_enc.T / T
    return T**2 * F.kl_div(
        img_sims.log_softmax(dim=0),
        txt_sims.log_softmax(dim=0),
        log_target=True,
        reduction="batchmean",
    )


def smooth_l1(tensor):
    return torch.where(tensor.abs() < 1, tensor**2 / 2, tensor.abs() - 0.5)


@_register_loss
def _vl2lite_loss(output, text_enc, teacher_enc, T=2.0):
    output = F.normalize(output, dim=-1, p=2.0)
    text_enc = F.normalize(text_enc, dim=-1, p=2.0)
    teacher_enc = F.normalize(teacher_enc, dim=-1, p=2.0)
    txt_sims = teacher_enc @ text_enc.T / T
    img_sims = output @ text_enc.T / T
    img_pairwise_dist = (output.unsqueeze(0) - output.unsqueeze(1)).norm(dim=-1)
    teacher_pairwise_dist = (teacher_enc.unsqueeze(0) - teacher_enc.unsqueeze(1)).norm(dim=-1)
    return (
        T**2
        / 2
        * F.kl_div(
            img_sims.log_softmax(dim=0),
            txt_sims.log_softmax(dim=0),
            log_target=True,
            reduction="batchmean",
        )
        + smooth_l1(img_pairwise_dist - teacher_pairwise_dist).mean() / 2
    )


if __name__ == "__main__":
    print("losses:", _loss_registry.keys())
    print("schedules:", _sched_registry.keys())
