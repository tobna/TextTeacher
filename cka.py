import argparse
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Sampler, Subset
from torch_cka import CKA

from load_dataset import prepare_dataset
from models import load_pretrained
from utils import prep_kwargs

parser = argparse.ArgumentParser()
parser.add_argument("-m1", "--model1", type=str, required=True, help="Model 1 weights")
parser.add_argument("-m2", "--model2", type=str, default=None, help="Model 1 weights")
parser.add_argument("-o", "--outfile", type=str, required=True, help="Where to save the cka plot")
parser.add_argument("-bs", "--batch-size", type=int, default=16, help="Batch size")
parser.add_argument("-l", "--layers", type=str, nargs="*", default=["mlp", "attn"], help="Type of layers to extract")
parser.add_argument("--subset", action=argparse.BooleanOptionalAction, default=False, help="Use subset only")
parser.add_argument("--bare", action=argparse.BooleanOptionalAction, default=False, help="Output bare plot only")
parser.add_argument("--cb-only", action=argparse.BooleanOptionalAction, default=False, help="Create only the colorbar")
args = parser.parse_args()
logger.remove()
logger.add(sys.stderr, colorize=True, enqueue=True, level="INFO")

if not args.cb_only:
    model1, _, model_args, __ = load_pretrained(args.model1, prep_kwargs({}))
    if args.model2 is None:
        args.model2 = args.model1
        model2 = deepcopy(model1)
    else:
        model2, _, __, ___ = load_pretrained(args.model2, prep_kwargs({}))

    logger.info(f"using batch size {args.batch_size}")

    loader, n_classes, _, __, ___ = prepare_dataset(
        "imagenet", prep_kwargs({"batch_size": args.batch_size, "num_workers": 10}), train=False, rank=0
    )
    ds = loader.dataset
    if args.subset:
        ds = Subset(ds, list(range(0, len(ds), 100)))

    class FixedRandomSampler(Sampler):
        """
        A Sampler that generates a single, fixed random permutation of indices.
        This order is the same every time the script is run.
        """

        def __init__(self, data_source, seed=None):
            super().__init__(data_source)
            if seed is not None:
                self.seed = seed
            self.data_source = data_source
            self.num_samples = len(self.data_source)
            rng = np.random.default_rng(seed)

            self.indices = list(range(self.num_samples))
            rng.shuffle(self.indices)  # Shuffle the list in place

        def __iter__(self):
            # Return the fixed, pre-shuffled list of indices
            return iter(self.indices)

        def __len__(self):
            return self.num_samples

    loader = DataLoader(ds, num_workers=10, batch_size=args.batch_size, sampler=FixedRandomSampler(ds, 42))
    model1_layers = [n for n, _ in model1.named_modules() if any([n.endswith(lay) for lay in args.layers])]
    model2_layers = [n for n, _ in model2.named_modules() if any([n.endswith(lay) for lay in args.layers])]
    logger.info(f"Model1 layers: {model1_layers}")
    logger.info(f"Model2 layers: {model2_layers}")
    cka = CKA(
        model1,
        model2,
        model1_name=os.sep.join(args.model1.split(os.sep)[-2:]),
        model2_name=os.sep.join(args.model2.split(os.sep)[-2:]),
        model1_layers=model1_layers,
        model2_layers=model2_layers,
        device="cuda",
    )
    cka.compare(loader)
    cka_out = cka.export()
    print(cka_out.keys())
    cka_matrix = cka_out["CKA"].numpy()
    print(cka_matrix)
    print(np.min(cka_matrix), np.max(cka_matrix))
else:
    if args.model2 is None:
        args.model2 = args.model1
    cka_matrix = np.array([[0.0, 0.5], [0.7, 1.0]])
    model1_layers = model2_layers = ["hu"]

fig, ax = plt.subplots(figsize=(6, 6) if args.bare else (7, 6))
im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap="plasma")
if not args.bare:
    ax.set_xlabel(os.sep.join(args.model1.split(os.sep)[-2:]), fontsize=8)
    ax.set_ylabel(os.sep.join(args.model2.split(os.sep)[-2:]), fontsize=8)
    ax.set_xticks(list(range(len(model1_layers))), model1_layers, fontsize=6, rotation="vertical")
    ax.set_yticks(list(range(len(model2_layers))), model2_layers, fontsize=6)
    cbar = fig.colorbar(im, ax=ax)
else:
    ax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)
fig.tight_layout(pad=0.1)
if not args.cb_only:
    plt.savefig(args.outfile)

if args.bare or args.cb_only:
    # make a colorbar
    fig, ax = plt.subplots(figsize=(8, 1), layout="constrained")
    fig.colorbar(im, cax=ax, orientation="horizontal").ax.tick_params(labelsize=25)
    colorbar_out = os.sep.join(args.outfile.split(os.sep)[:-1] + ["colorbar.pdf"])
    logger.info(f"Saving colorbar to {colorbar_out}")
    plt.savefig(colorbar_out)

np_file_name = ".".join(args.outfile.split(".")[:-1]) + "_matrix.npy"
np.save(np_file_name, cka_matrix)
