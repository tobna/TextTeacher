from utils import prep_kwargs
from matplotlib import pyplot as plt
import math

from load_dataset import prepare_dataset


def load_images(dataset, **kwargs):
    args = prep_kwargs(kwargs)
    args.dataset = dataset

    args.aug_normalize = False

    loader, args.n_classes, args.ignore_index, args.multi_label, _ = prepare_dataset(dataset, args)
    images = next(iter(loader))[0]

    images = images.permute(0, 2, 3, 1).numpy()
    images = [images[i] for i in range(images.shape[0])]

    rows = math.ceil(math.sqrt(len(images) / 2))
    ims_per_row = len(images) // rows

    fig, axs = plt.subplots(rows, ims_per_row)
    axs = [ax for row in axs for ax in row]
    for img, ax in zip(images, axs):
        ax.imshow(img)
    fig.suptitle(f"Examples from {dataset}")
    fig.tight_layout(pad=0)

    plt.show()
