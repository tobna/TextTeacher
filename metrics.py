import math
import traceback
from contextlib import nullcontext
from math import prod
from time import time
from typing import Dict

import psutil
import torch.cuda
import torch.distributed as dist
from fvcore.nn import FlopCountAnalysis
from loguru import logger
from torcheval.metrics import functional as TeF
from torchprofile import profile_macs
from tqdm.auto import tqdm


def mIoU(output, target, num_classes, ignore_index=-100):
    """Calculate the mean intersection over union (mIoU).

    Args:
        output (torch.Tensor): The model output of shape (batch_size, ...,  classes).
        target (torch.Tensor): The labels of shape (batch_size, ...) or (batch_size, ..., classes).
        num_classes (int): The number of classes.
        ignore_index (int, optional): The label to ignore, by default -100.

    Returns:
        float: The mean intersection over union.

    """
    with torch.inference_mode():
        trgt_view = target.view(-1) if len(target.shape) < len(output.shape) else target.view(-1, num_classes)
        confusion = TeF.multiclass_confusion_matrix(output.view(-1, num_classes), trgt_view, num_classes)
        if ignore_index >= 0:
            confusion[ignore_index, :] = (
                0  # ignore everything with label ignore_index (but not wrongly predicted to be ignore_index)
            )
        intersection = confusion.diag()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
        iou = intersection / union
        return iou.nanmean().item()


def accuracy(output, target, topk=1, dict_key="acc{}", ignore_index=-100) -> float | Dict:
    """Calculate top-k accuracy.

    Args:
        output (torch.Tensor): The model output of shape (batch_size, ...,  classes).
        target (torch.Tensor): The labels of shape (batch_size, ...) or (batch_size, ..., classes).
        topk (int or list[int] or tuple[int], optional): The k value(s) for top-k accuracy, by default 1.
        dict_key (str, optional): The format for the keys in the returned dictionary, by default "acc{}".
        ignore_index (int, optional): The label to ignore, by default -100.

    Returns:
        float | dict[str, float]: The top-k accuracy or a dictionary of top-k accuracies.

    Notes:
        If `topk` is an integer, the function returns the top-k accuracy as a float.
        If `topk` is a list or tuple, the function returns a dictionary of top-k accuracies.
        The dictionary keys are formatted as specified by `dict_key.format(k)`.

    """
    if len(target.shape) >= len(output.shape):
        # target is one/multiple-hot encoded
        target = target.view(-1, target.shape[-1])
        output = output.view(-1, output.shape[-1])
        N = target.shape[0]

        ret_dict = {}
        n_correct = target.sum(dim=-1)
        for k in topk:
            top_guesses = output.topk(k).indices
            pred_top_k = torch.zeros_like(target).scatter_(1, top_guesses, 1)
            per_sample_acc = (target * pred_top_k).sum(dim=-1) / n_correct.clamp(max=k)
            ret_dict[dict_key.format(k)] = per_sample_acc.sum().item() / N
        return ret_dict

    max_k = topk if isinstance(topk, int) else max(topk)
    top_guesses = output.topk(max_k).indices
    consonance = torch.logical_and(top_guesses == target.unsqueeze(-1), target.unsqueeze(-1) != ignore_index).reshape(
        -1, max_k
    )
    N = int(consonance.shape[0] - target.view(-1).eq(ignore_index).sum().item())
    if isinstance(topk, int):
        return consonance.reshape(-1).sum().item() / N
    return {dict_key.format(k): consonance[:, :k].reshape(-1).sum().item() / N for k in topk}


def calculate_metrics(
    args,
    model,
    rank=0,
    input=None,
    device=None,
    did_training=False,
    world_size=1,
    all_metrics=True,
    n_ims=1,
    optim=None,
    scaler=None,
    train_loader=None,
    key_start="eval/",
):
    """Calculate all metrics.

    Args:
        args: training arguments; in particular set args.eval_amp
        model (torch.nn.Module): model to analyze
        rank (int, optional): rank of this process (Default value = 0)
        input (torch.Tensor, optional): input batch (Default value = None)
        device (torch.device, optional): device to calculate throughput on (Default value = None)
        did_training (bool, optional): call after training to measure peak memory usage (Default value = False)
        world_size (int, optional): number of processes/GPUs for peak memory usage (Default value = 1)
        all_metrics (bool, optional): flag to calculate all metrics. If false, only number of parameters and memory usage is calculated. (Default value = True)
        n_ims (int, optional): number of images to consider for macs and flops (Default value = 1)

    Returns:
        dict: dictionary of metrics

    """
    assert 0 <= rank < world_size, f"Incompatible rank and world size; not 0 <= rank={rank} < world_size={world_size}"
    if rank != 0:
        max_mem_allocated(device, world_size)
        return {}

    assert (
        input is not None or train_loader is not None
    ), f"Set either input tensor or train_loader to have some data for metrics calculation"
    if input is None:
        input = next(iter(train_loader))[0].to(device)

    if input.size(0) == 1 and args.batch_size > 1:
        input = input[0]

    logger.info(f"Calculating metrics on input of shape {input.shape}.")

    metrics = {key_start + "number of parameters": number_of_params(model)}
    if input is None:
        return metrics

    if did_training:
        if world_size == 1:
            metrics[key_start + "peak_memory"] = max_mem_allocated(device, world_size)
        else:
            peak_mem_total, peak_mem_single = max_mem_allocated(device, world_size)
            metrics[key_start + "peak_memory_total"] = peak_mem_total
            metrics[key_start + "peak_memory_single"] = peak_mem_single

    if not all_metrics:
        return metrics

    model.eval()

    metrics[key_start + "macs"] = macs(
        args, model._orig_mod if hasattr(model, "_orig_mod") else model, input, n_ims=n_ims
    )
    try:
        metrics[key_start + "flops"] = flops(
            args, model._orig_mod if hasattr(model, "_orig_mod") else model, input, n_ims=n_ims
        )
    except RuntimeError as e:
        metrics[key_start + "flops"] = metrics[key_start + "macs"]
        logger.warning(f"Failed to calculate flops: {e}")
        logger.warning("Setting flops equal to macs!")

    if device is None:
        return metrics

    if args.cuda:
        for bs, inference_mem in inference_memory(args, model, input, device).items():
            metrics[key_start + f"inference_memory_@{bs}"] = inference_mem

    if optim is None or scaler is None or train_loader is None:
        logger.info(
            f"Skipping training time calculation, since one of these is None: optimizer={optim}, scaler={scaler},"
            f" train_loader={train_loader}"
        )
    else:
        logger.info(f"Calculating training time for 500 steps at batch size {args.batch_size}")
        train_time = training_time(
            args, model=model, optim=optim, scaler=scaler, data_loader=train_loader, device=device, max_iters=500
        )
        metrics[key_start + "training_time"] = {"batch_size": args.batch_size, "step_time_ms": train_time}

    tp_bs, tp_val = throughput(args, model, input, device)
    metrics[key_start + "throughput"] = {"batch_size": tp_bs, "value": tp_val}

    return metrics


def number_of_params(model):
    """Get the number of parameters from the model.

    Args:
        model (torch.nn.Module): the model

    Returns:
        int: number of parameters

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macs(args, model, input, n_ims=1):
    """Calculate the MACs (multiply-accumulate operations) of the model for a given input.

    Args:
        args: training arguments
        n_ims (int, optional): number of images to look at (Default value = 1)
        model (torch.nn.Module): the model
        input (torch.Tensor): the input tensor in batch format

    Returns:
        int: the number of MACs

    """
    if n_ims is not None:
        input = input[:n_ims]
    with torch.amp.autocast("cuda") if args.eval_amp and args.cuda else nullcontext():
        return profile_macs(model, input)


def max_mem_allocated(device, world_size=1, reset_max=False):
    """Return the max memory allocated during training.

    Use **this before** calling *throughput*, as that resets the statistics.

    Args:
        device (torch.Device): the device to look at in this process
        world_size (int, optional): the number of GPUs (processes) used in total -> stats are gathered from all GPUs (Default value = 1)
        reset_max (bool, optional): if true, resets the max memory allocated to zero. Subsequent calls will return the max memory allocated after this call. (Default value = False)

    Returns:
        int: the max memory allocated during training

    """
    max_mem_gpu = torch.cuda.max_memory_allocated(device)
    if reset_max:
        torch.cuda.reset_peak_memory_stats(device)

    if world_size == 1:
        return max_mem_gpu

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, max_mem_gpu)
    return sum(gathered), max(gathered)


def inference_memory(args, model, input, device, batch_sizes=(1, 16, 32, 64, 128)):
    """Return the memory needed for inference at different batch sizes.

    Args:
        args: training arguments; in particular set args.eval_amp
        model (torch.nn.Module): the model to evaluate
        input (torch.Tensor): batch of input data; no batch size bigger than the size of this batch are tested
        device (torch.Device): the device to test on
        batch_sizes (list[int], optional): list of batch sizes to test (Default value = (1, 16, 32, 64, 128)

    Returns:
        dict: dictionary of batch size to inference memory allocated

    """
    vram_allocated = {}

    for bs in sorted(batch_sizes, reverse=True):
        if input.shape[0] < bs:
            continue
        input = input[:bs]
        # reset statistics

        if args.compile_model:
            # force compilation first
            model(input)

        torch.cuda.reset_peak_memory_stats(device)

        with torch.amp.autocast("cuda") if args.eval_amp else nullcontext(), torch.no_grad():
            try:
                model(input)
                vram_allocated[bs] = max_mem_allocated(device, reset_max=True)
            except torch.cuda.OutOfMemoryError:
                pass
    return vram_allocated


def _add_handler(inputs, outputs):
    """Number of FLOPs for an addition operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs[0])
    # print(in_shapes, out_shape)
    # assert in_shapes[0][1:] == in_shapes[1][1:] and (in_shapes[0][0] == in_shapes[1][0] or in_shapes[1][0] == 1), \
    #     f"Got incompatible shapes for adding: {in_shapes}"
    return prod(out_shape)


def _mul_handler(inputs, outputs):
    """Number of FLOPs for a multiplication operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)
    # assert len(in_shapes[1]) <= 1 or len(in_shapes[0]) <= 1 or in_shapes[1][1:] == [1, 1] or (len(in_shapes[0]) == len(in_shapes[1]) and all(x == y == out or (x == 1 and y == out) or (y == 1 and x == out) for x, y, out in zip(in_shapes[0], in_shapes[1], out_shapes[0]))), \
    #     f"mul_handler found in_shapes: {in_shapes} -> {out_shapes[0]}"
    # print(f"in: {in_shapes}\t->\tout: {out_shapes}")
    return prod(out_shapes[0])


def _softmax_handler(inputs, outputs):
    """Number of FLOPs for a softmax operation.

    approximate times 5 for flops from exp, sum, and mult (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)
    # print(f"in: {in_shapes}\t->\tout: {out_shapes}")

    # approximate times 5 for flops from exp, sum, and mult (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)
    return prod(out_shapes[0]) * 5


def _gelu_handler(inputs, outputs):
    """Number of FLOPs for a gelu operation.

    approximate times * 8 for mult, add, tanh, and pow (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs[0])

    # approximate times * 8 for mult, add, tanh, and pow (taken from https://github.com/google-research/electra/blob/master/flops_computation.py)
    return prod(out_shape) * 8


def _div_handler(inputs, outputs):
    """Number of FLOPs for a division operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)

    return prod(out_shapes[0])


def _norm_handler(inputs, outputs):
    """Number of FLOPs for a normalization operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)[0]
    in_shapes = _get_cval_shape(inputs)[0]

    # flops come from squaring each input (M*N) and adding all of them up (M*N - 1)
    norm_dims = [1]
    batch_dims = [1]
    for dim in set(in_shapes):
        if dim == 1:
            continue
        in_cnt = in_shapes.count(dim)
        out_cnt = out_shapes.count(dim)
        assert in_cnt >= out_cnt, f"Found {dim} more in out shape ({out_shapes}) then in shape ({in_shapes})"
        batch_dims += [dim for _ in range(out_cnt)]
        norm_dims += [dim for _ in range(in_cnt - out_cnt)]

    return prod(batch_dims) * (2 * prod(norm_dims) - 1)


def _cumsum_handler(inputs, outputs):
    """Number of FLOPs for a cumsum operation.

    in cumsum_dim: 0 + 1 + ... + n-1 = n(n-1)/2
    for each of the batch dims (entries prod(all_dims) / cumsum_dim)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)[0]
    in_shapes = _get_cval_shape(inputs)[0]
    assert out_shapes == in_shapes, f"cumsum: {out_shapes} != {in_shapes}"

    # assume worst case
    cumsum_dim = max(in_shapes)
    # in cumsum_dim: 0 + 1 + ... + n-1 = n(n-1)/2
    # for each of the batch dims (entries prod(all_dims) / cumsum_dim
    return int(prod(in_shapes) * (cumsum_dim - 1) / 2)


def _pow_handler(inputs, outputs):
    """Number of FLOPs for a power operation.

    assume pow <= 4 -> ~ 3 mults

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shapes = _get_cval_shape(outputs)[0]

    # print(f"pow map: {in_shapes} -> {out_shapes}")

    # for now assume pow <= 4 -> ~ 3 mults
    return 3 * prod(out_shapes)


def _sin_cos_handler(inputs, outputs):
    """Number of FLOPs for a sin/cos operation.

    approximate each of these operations (on GPU) to be just 1 FLOP
    taken from https://foldingathome.org/support/faq/flops/

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximate each of these operations (on GPU) to be just 1 FLOP
    # taken from https://foldingathome.org/support/faq/flops/
    return prod(out_shape)


def _log_handler(inputs, outputs):
    """Number of FLOPs for a log operation.

    approximation operation costing 20 FLOPS
    taken from https://foldingathome.org/support/faq/flops/

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximation operation costing 20 FLOPS
    # taken from https://foldingathome.org/support/faq/flops/
    return 20 * prod(out_shape)


def _exp_handler(inputs, outputs):
    """Number of FLOPs for an exp operation.

    approximation operation costing 20 FLOPS
    taken from https://foldingathome.org/support/faq/flops/

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximation operation costing 20 FLOPS
    # taken from https://foldingathome.org/support/faq/flops/
    return 20 * prod(out_shape)


def _sigmoid_handler(inputs, outputs):
    """Number of FLOPs for a normalization operation.

    approximation: number of flops for exp + 2

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]

    # approximation: number of flops for exp + 2
    return 2 * prod(out_shape) + _exp_handler(inputs, outputs)


def _sum_handler(inputs, outputs):
    """Number of FLOPs for a summation operation.

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    sum_dims = [1]
    batch_dims = [1]
    for dim in set(in_shape):
        if dim == 1:
            continue
        in_cnt = in_shape.count(dim)
        out_cnt = out_shape.count(dim)
        assert in_cnt >= out_cnt, f"Found {dim} more in out shape ({out_shape}) then in shape ({in_shape})"
        batch_dims += [dim for _ in range(out_cnt)]
        sum_dims += [dim for _ in range(in_cnt - out_cnt)]

    return prod(batch_dims) * (prod(sum_dims) - 1)


def _rfft2_handler(inputs, outputs):
    """Number of FLOPs for an FFT operation.

    FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = in_shape[d_i_1] * in_shape[d_i_2]
    return int(prod(in_shape) * 2.5 * math.log2(N))


def _irfft2_handler(inputs, outputs):
    """Number of FLOPs for an inverse FFT operation.

    FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 2.5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = out_shape[d_i_1] * out_shape[d_i_2]
    return int(prod(out_shape) * 2.5 * math.log2(N))


def _fft2_handler(inputs, outputs):
    """Number of FLOPs for an FFT operation.

    FLOPS are approximate 5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    out_shape = _get_cval_shape(outputs)[0]
    in_shape = _get_cval_shape(inputs)[0]

    # assume w and h dims are next to each other
    # by default assume last two dimensions
    d_i_1, d_i_2 = -2, -1
    for i, (d_in, d_out) in enumerate(zip(in_shape, out_shape)):
        if d_in != d_out:
            d_i_1 = i
            d_i_2 = i - 1
            break

    # FLOPS are approximate 5 * N * log_2(N) (taken from http://www.fftw.org/speed/method.html -> Cooley-Tukey algorithm)
    N = out_shape[d_i_1] * out_shape[d_i_2]
    return int(prod(out_shape) * 5 * math.log2(N))


def _scaled_dot_product_attention_handler(inputs, outputs):
    qkv_shape = _get_cval_shape(inputs)[0]
    flops = prod(qkv_shape)  # scale q by sqrt d
    flops += prod(qkv_shape) * qkv_shape[-2] * 2  # Q x K^T matrix multiplication
    flops += prod(qkv_shape[:-1]) * qkv_shape[-2] * 5  # softmax calculation on QK^T (factor 5 from softmax operation)
    flops += prod(qkv_shape) * qkv_shape[-2] * 2  # A x V matrix multiplication
    return flops


def _mean_handler(inputs, outputs):
    """Number of FLOPs for a mean operation.

    mean of N elements takes N flops (N-1 for sum and 1 to divide by len)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    in_shape = _get_cval_shape(inputs)[0]

    # mean of N elements takes N flops (N-1 for sum and 1 to divide by len)
    return prod(in_shape)


def _avg_pool2d_handler(inputs, outputs):
    """Number of FLOPs for an average pool operation.

    mean of N elements takes N flops (N-1 for sum and 1 to divide by len)

    Args:
        inputs (list[torch._C.Value]): Inputs to the operation
        outputs (list[torch._C.Value]): Outputs of the operation

    Returns:
        int: Number of FLOPs
    """
    # take the mean; the same way as in mean_handler
    return _mean_handler(inputs, outputs)


def _get_cval_shape(val):
    """Get the shapes from a jit value object.

    Taken from https://github.com/facebookresearch/fvcore/blob/fd5043ff8b2e6790f5bd7c9632695c68986cc658/fvcore/nn/jit_handles.py#L23

    Args:
        val (torch._C.Value | list[torch._C.Value]): jit value object or list of those.

    Returns:
        list: shape

    """
    if isinstance(val, list):
        return [_get_cval_shape(x) for x in val]

    if val.isCompleteTensor():
        return val.type().sizes()
    return None


def flops(args, model, input, per_module=False, n_ims=1):
    """Return the number of floating point operations (FLOPs) needed for a given input.

    This function is broken, when working with timm models -> returns 0.
    The output should in theory be 2*MACs(), but it might report MACs straight up...
    Further investigation needed.

    Args:
        args: training arguments; in particular set args.eval_amp
        n_ims (int, optional): number of images to look at (Default value = 1)
        model (torch.nn.Module): the model to analyze
        input (torch.Tensor): the input to give to the model
        per_module (bool, optional): flag to return stats by submodule (Default value = False)

    Returns:
        int | dict: the number of FLOPs or a dictionary of FLOPs per submodule

    """
    if n_ims is not None:
        input = input[:n_ims]

    fca = FlopCountAnalysis(model, input)
    fca.set_op_handle("aten::add", _add_handler)
    fca.set_op_handle("aten::add_", _add_handler)
    fca.set_op_handle("aten::mul", _mul_handler)
    fca.set_op_handle("aten::mul_", _mul_handler)
    fca.set_op_handle("aten::softmax", _softmax_handler)
    fca.set_op_handle("aten::gelu", _gelu_handler)
    fca.set_op_handle("aten::bernoulli_", None)
    fca.set_op_handle("aten::div_", _div_handler)
    fca.set_op_handle("aten::div", _div_handler)
    fca.set_op_handle("aten::norm", _norm_handler)
    fca.set_op_handle("aten::cumsum", _cumsum_handler)
    fca.set_op_handle("aten::pow", _pow_handler)
    fca.set_op_handle("aten::sin", _sin_cos_handler)
    fca.set_op_handle("aten::cos", _sin_cos_handler)
    fca.set_op_handle("aten::sum", _sum_handler)
    fca.set_op_handle("aten::fft_rfft2", _rfft2_handler)
    fca.set_op_handle("aten::fft_irfft2", _irfft2_handler)
    fca.set_op_handle("aten::fft_fft2", _fft2_handler)
    fca.set_op_handle("aten::mean", _mean_handler)
    fca.set_op_handle("aten::sub", _add_handler)
    fca.set_op_handle("aten::rsub", _add_handler)
    fca.set_op_handle("aten::reciprocal", _div_handler)
    fca.set_op_handle("aten::avg_pool2d", _avg_pool2d_handler)
    fca.set_op_handle("aten::adaptive_avg_pool1d", _avg_pool2d_handler)
    fca.set_op_handle("aten::log", _log_handler)
    fca.set_op_handle("aten::exp", _exp_handler)
    fca.set_op_handle("aten::sigmoid", _sigmoid_handler)
    fca.set_op_handle("aten::scatter_add", _add_handler)
    fca.set_op_handle("aten::log_softmax", _softmax_handler)
    fca.set_op_handle("aten::square", _mean_handler)
    fca.set_op_handle("aten::scaled_dot_product_attention", _scaled_dot_product_attention_handler)

    # these operations are ignored, because 0 FLOPS
    fca.set_op_handle("aten::expand_as", None)
    fca.set_op_handle("aten::clamp_min", None)
    fca.set_op_handle("aten::view_as_complex", None)
    fca.set_op_handle("aten::real", None)
    fca.set_op_handle("aten::eye", None)
    fca.set_op_handle("aten::repeat_interleave", None)
    fca.set_op_handle("aten::scatter_reduce", None)
    fca.set_op_handle("aten::fill_", None)
    fca.set_op_handle("aten::ones_like", None)
    fca.set_op_handle("aten::topk", None)
    fca.set_op_handle("aten::expand", None)
    fca.set_op_handle("aten::reshape", None)
    fca.set_op_handle("aten::permute", None)
    fca.set_op_handle("aten::unbind", None)

    with torch.amp.autocast("cuda") if args.eval_amp and args.cuda else nullcontext():
        if per_module:
            return fca.by_module()
        try:
            return fca.total()
        except IndexError as e:
            logger.error(f"IndexError {e} when calculating flops. Might come from timm model.")
            traceback.print_exc()
            return -1


def throughput(args, model, input, device, iters=100):
    """Calculate the throughput of a given model.

    Throughput is given for the biggest batch_size, that fits into memory. Images from input are repeated to get to this
    batch_size.
    Internally resets the max allocated memory, so only use this **after** *max_mem_allocated*.

    Args:
        args: training arguments; in particular set args.eval_amp
        model (torch.nn.Module): the model to analyze
        input (torch.Tensor): the batch of images to start with
        device (torch.cuda.device): the device to measure throughput with
        iters (int, optional): the number of iterations to test with (for more accurate numbers) (Default value = 100)

    Returns:
        tuple[int, int]: the optimal batch size and the throughput in images per second

    """
    # dev_properties = torch.cuda.get_device_properties(device)
    # total_mem = dev_properties.total_memory
    #
    # # reset max mem allocated
    # torch.cuda.reset_peak_memory_stats(device)
    #
    # n_ims = input.shape[0]
    # if n_ims > 4:
    #     n_ims = 4
    #     input = input[:4]
    # with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
    #     with torch.no_grad():
    #         try:
    #             model(input)
    #         except IndexError as e:
    #             logger.error(f"Index error {e} when calculating throughput. Might come from timm with amp.")
    #             return -1, -1
    # max_alloc = max_mem_allocated(device, reset_max=True)
    #
    # memory_allocated = {n_ims: max_alloc}
    #
    # if max_alloc <= (total_mem-250_000) // 2:
    #     input = torch.cat((input, input), dim=0)
    #     n_ims *= 2
    # else:
    #     input = input[:input.shape[0]//2]
    #     n_ims = n_ims // 2
    #
    # with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
    #     with torch.no_grad():
    #         model(input)
    # max_alloc = max_mem_allocated(device, reset_max=True)
    #
    # memory_allocated[n_ims] = max_alloc
    # pred_double = linear_regession(memory_allocated)(2*n_ims)
    #
    # torch.cuda.empty_cache()
    # while pred_double <= total_mem - 500_000_000:
    #     input = torch.cat((input, input), dim=0)
    #     n_ims = int(input.shape[0])
    #     try:
    #         with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
    #             with torch.no_grad():
    #                 model(input)
    #     except torch.cuda.OutOfMemoryError:
    #         break
    #     except RuntimeError as e:
    #         logger.error(f"RuntimeError '{e}' when calculating throughput (@{n_ims}). "
    #                       f"Might come from out- or input tensor size >2**31 (max int32_t).")
    #         logger.error(f"Stacktrace:\n"
    #                       f"{''.join(traceback.TracebackException.from_exception(e).format())}")
    #         break
    #     max_alloc = max_mem_allocated(device, reset_max=True)
    #     memory_allocated[n_ims] = max_alloc
    #     pred_double = linear_regession(memory_allocated)(2 * n_ims)
    #
    # reg_line = linear_regession(memory_allocated)
    # b = reg_line(0)
    # a = reg_line(1) - b
    #
    # assert input.shape[0] == n_ims, f"Found input of shape {input.shape}. Should have {n_ims} images."
    # test_bs = set([int(2 * n_ims / i) for i in range(1, 9)] +
    #               [int((total_mem - offset - b) / a) for offset in [250_000_000, 100_000_000, 0]])
    # test_bs = {2 ** math.floor(math.log2(bs)) for bs in test_bs if bs > 4}
    # test_bs = {bs - (bs % 16) for bs in test_bs}.union(test_bs)

    bs = min(1024, args.batch_size)
    # print(f"test batch sizes: {test_bs}")
    if input.shape[0] < bs:
        input = torch.cat((input for _ in range(int(bs / input.shape[0]) + 1)), dim=0)
    input = input[:bs]

    results = []
    n_decr = 0
    while True:
        # for bs in sorted(list(test_bs)):
        #     if input.shape[0] < bs:
        #         diff = bs - input.shape[0]
        #         input = torch.cat((input, input[:diff]), dim=0)
        #     else:
        #         input = input[:bs]
        #     n_ims = input.shape[0]
        logger.info(f"thoughput calculation: test batch size {bs}")
        if args.cuda:
            try:
                tp = _measure_throughput_cuda(model, input, iters, args.eval_amp, use_tqdm=args.tqdm)
            except RuntimeError as e:
                if "canUse32BitIndexMath" in str(e):
                    logger.info(f"throughput calculation: tensor too large @ {bs}")
                else:
                    logger.info(f"throughput calculation: CUDA OOM @ {bs}")
                break
        else:
            tp = _measure_throughput_cpu(model, input, iters, use_tqdm=args.tqdm)
            logger.debug(f"used {_get_ram_usage()} MiB of {_get_ram_total()} MiB RAM")
        if len(results) > 1 and (tp < 0.98 * results[-1][1] or tp <= 0.95 * max(res_tp for _, res_tp in results)):
            n_decr += 1
        else:
            n_decr = 0
        logger.debug(f"decreasing trend for {n_decr} steps in a row")
        results.append((bs, tp))
        logger.info(f"throughput calculation: throughput @ {bs} = {tp} images/second")
        if not args.cuda and _get_ram_usage() > 0.5 * _get_ram_total():
            logger.info(f"throughput calculation: used more than 50% of total RAM @ {bs}; stopping further calculation")
            break
        if n_decr >= 2:
            logger.info(
                "decreasing throughput trend for 3 sizes:"
                f" {' -> '.join([f'{tp_r:.2f} @ {bs_r}' for bs_r, tp_r in results[-3:]])}; stopping further calculation"
            )
            break
        if len(results) >= 2 and results[-1][1] < 0.5 * results[-2][1]:
            logger.info(f"throughput calculation: throughput dropped below 50% of previous value @ {bs}; stopping now")
            break
        input = torch.cat((input, input), dim=0)
        bs *= 2
    # print(f"results {results}")
    top_bs, top_tp = max(results, key=lambda x: x[1]) if len(results) > 0 else (-1, -1)
    if 10 * iters / (top_bs * top_tp) <= 2 * 60 * 60:  # only measure again, if it takes less than 2 hours
        try:
            if args.cuda:
                top_tp = _measure_throughput_cuda(model, input[:top_bs], iters * 10, args.eval_amp, use_tqdm=args.tqdm)
            else:
                top_tp = _measure_throughput_cpu(model, input[:top_bs], iters * 10, use_tqdm=args.tqdm)
        except RuntimeError:
            pass
    return top_bs, top_tp


def _measure_throughput_cuda(model, input, iters=1000, eval_amp=False, use_tqdm=False):
    """Measure the throughput of a PyTorch model on CUDA.

    Args:
        model (torch.nn.Module): The PyTorch model to measure throughput for.
        input (torch.Tensor): The input tensor of shape (batch_size, ...) for the model.
        iters (int, optional): The number of iterations to run to measure the throughput, by default 1000.
        eval_amp (bool, optional): Whether to evaluate using Automatic Mixed Precision (AMP) mode, by default False.
        use_tqdm (bool, optional): Show a progress bar using tqdm, by default False.

    Returns:
        float: The throughput in images per second.

    """
    # total_time = 0
    samples = []
    iterator = range(iters)
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"throughput calculation @ {input.shape[0]}", total=iters)
    with torch.no_grad(), torch.amp.autocast("cuda") if eval_amp else nullcontext():
        for _ in iterator:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            __ = model(input)
            ender.record()
            torch.cuda.synchronize()
            # total_time += starter.elapsed_time(ender) / 1000  # ms -> s
            samples.append(starter.elapsed_time(ender) / 1000)  # ms -> s
    # return iters * input.shape[0]/total_time
    samples = samples[int(len(samples) / 10) :]
    return len(samples) * input.shape[0] / sum(samples)


def _measure_throughput_cpu(model, input, iters=1000, use_tqdm=False):
    """Measure the throughput of a PyTorch model on CPU.

    Args:
        model (torch.nn.Module): The PyTorch model to measure throughput for.
        input (torch.Tensor): The input tensor of shape (batch_size, ...) for the model.
        iters (int, optional): The number of iterations to run to measure the throughput, by default 1000.
        use_tqdm (bool, optional): Show a progress bar using tqdm, by default False.

    Returns:
        float: The throughput in images per second.

    """
    samples = []
    iterator = range(iters)
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"throughput calculation @ {input.shape[0]}", total=iters)
    for _ in iterator:
        with torch.no_grad():
            start = time()
            __ = model(input)
            end = time()
        samples.append(end - start)
    samples = samples[int(len(samples) / 10) :]
    return len(samples) * input.shape[0] / sum(samples)


def _get_ram_usage():
    """Return the RAM usage of this process in MiB."""
    # Get current process
    process = psutil.Process()
    # Get memory usage info in bytes
    mem_info = process.memory_info()
    # Convert bytes to MiB
    return mem_info.rss / (1024 * 1024)


def _get_ram_total():
    """Return the total RAM of this system in MiB."""
    return psutil.virtual_memory().total / 1024**2


def training_time(args, model, optim, scaler, data_loader, device, max_iters=200):
    from engine import setup_criteria_mixup

    criterion, _, mixup = setup_criteria_mixup(args)
    measure_steps = min(max_iters, len(data_loader))
    start_measure = int(measure_steps / 10)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i, (xs, ys) in tqdm(enumerate(data_loader), total=measure_steps, desc="Training time calculation"):
        if i == start_measure:
            starter.record()
        if i == measure_steps:
            break
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        if mixup:
            xs, ys = mixup(xs, ys)

        optim.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            preds = model(xs)
            loss = criterion(preds.transpose(1, -1), ys.transpose(1, -1) if len(ys.shape) > 1 else ys) + (
                model.get_internal_loss() if hasattr(model, "get_internal_loss") else model.module.get_internal_loss()
            )

        scaler(
            loss,
            optim,
            parameters=model.parameters(),
            clip_grad=args.max_grad_norm if args.max_grad_norm > 0.0 else None,
        )
    ender.record()
    torch.cuda.synchronize()

    return starter.elapsed_time(ender) / (measure_steps - start_measure)
