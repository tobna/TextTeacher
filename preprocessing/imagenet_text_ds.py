import json
import os
from functools import partial
from time import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

_ROOT_CAPTIONS = "/netscratch/nauen/ImageNet21k_captions"
_ROOT_ENCODINGS = "/fscratch/nauen/text_encodings"
_ROOT_IMAGENET = "/ds-sds/images/imagenet"


class ImageText(Dataset):
    """Dataset of image captions from imagenet."""

    def __init__(self, caption_model, folder=_ROOT_CAPTIONS, root_dataset="imagenet") -> None:
        """Build the caption dataset.

        Args:
            caption_model (str): Model that was used to caption imagenet. Caption files should start with imagenet_<caption_model>_.
            folder (str): Root folder, where all caption files live.

        """
        super().__init__()
        self.folder = folder
        self.caption_model = caption_model

        files = [
            f
            for f in os.listdir(self.folder)
            if f.startswith(f"{root_dataset}_{caption_model}_") and f.endswith(".csv")
        ]

        self.mult = max(int(f.split("_")[-1].split(".")[0]) for f in files)
        self.files = [f for f in files if f.endswith(f"_{self.mult}.csv")]

        if len(self.files) < self.mult:
            print(f"WARNING: Found less then {self.mult} ({len(self.files)}) files endling with '_{self.mult}.csv'")

        self.data = {}
        for file in tqdm(self.files, desc="Extracting encoding files"):
            with open(os.path.join(self.folder, file), "r") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    img, cap = line.strip().split(":\t")
                except ValueError as e:
                    print(
                        f"WARNING: Found irregular line '{line.replace('\n', '\\n')}' in file '{file}' L {i+1} => {e}"
                    )
                    continue
                self.data[img] = cap
        self.index = sorted(list(self.data.keys()))

    def __getitem__(self, index):
        return self.index[index], self.data[self.index[index]]

    def __len__(self):
        return len(self.index)


def has_text_embedding(image_name, encoding_folder):
    """Does this (ImageNet) file have a text encoding?"""
    return os.path.isfile(os.path.join(encoding_folder, f"{image_name.split(os.sep)[-1]}.emb.npy"))


class ImageTextEncodingDataset(ImageFolder):
    """Dataset of Image and caption encoding."""

    def __init__(
        self,
        image_folder,
        transform=None,
        encoding_model="bert-large",
        text_captions="ImageNet-Dragonfly",
        eps=1e-6,
        normalize_embeddings=False,
    ):
        """Create the image-caption encoding dataset.

        Args:
            image_folder (str): Folder to load the IamgeFolder dataset from.
            transform (callable, optional): image data augmentation.
            encoding_model (str, optional): model which was used to encode.
            text_captions (str, optional): text caption dataset. Usually is 'ImageNet-<caption model>'.
            eps (float, optional): small epsilon to add to not divide by zero.

        """
        super().__init__(
            image_folder,
            transform=transform,
            # is_valid_file=partial(has_text_embedding, encoding_folder=self.encoding_folder),
        )
        self.encoding_model = encoding_model
        self.text_captions = text_captions
        self.normalize_embeddings = normalize_embeddings
        self.encoding_folder = os.path.join(_ROOT_ENCODINGS, text_captions, encoding_model)
        self.samples = [
            (path, trgt)
            for path, trgt in tqdm(self.samples, desc="Filtering samples")
            if has_text_embedding(path, self.encoding_folder)
        ]
        # if os.path.isfile("misc_ds/imagenet_has_no_enc.json"):
        #     with open("misc_ds/imagenet_has_no_enc.json", "r") as f:
        #         hne = set(json.load(f))
        #     self.samples = [
        #         (path, trgt)
        #         for path, trgt in tqdm(self.samples, desc="Filter samples")
        #         if path.split(os.sep)[-1] not in hne
        #     ]
        # else:
        #     raise KeyError(f"No misc_ds/imagenet_has_no_enc.json file")
        assert len(self.samples) > 0, f"Found no samples with text caption."

        self.eps = eps
        if os.path.isfile(os.path.join(self.encoding_folder, "stats.npy")):
            normalizer = np.load(os.path.join(_ROOT_ENCODINGS, text_captions, encoding_model, "stats.npy"))
            self.norm_mean, self.norm_std = normalizer
            self.norm_std = np.absolute(self.norm_std) + eps
        else:
            print(f"WARNING: dataset {self.encoding_folder} has no normalizer (stats.npy).")
            self.norm_mean, self.norm_std = 0.0, 1.0

        self.text_dim = (
            int(self.__getitem__(0)[1].shape[0]) if isinstance(self.norm_mean, float) else int(self.norm_mean.shape[0])
        )

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image, target = super().__getitem__(index)
        id = path.split(os.sep)[-1]
        emb_file = os.path.join(self.encoding_folder, f"{id}.emb.npy")
        if not os.path.isfile(emb_file):
            embedding = torch.zeros(self.text_dim, dtype=torch.float)
            print(f"WARNING: No encoding for file {path.split(os.sep)[-1]}")
        else:
            embedding = np.load(emb_file)
            embedding = (embedding - self.norm_mean) / self.norm_std  # Normalize embeddings to have mean zero and std 1
            embedding = torch.from_numpy(embedding)
            if self.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return image, embedding


if __name__ == "__main__":
    ds = ImageText("BLIP-L")
    print(len(ds.index))

    for i in range(5):
        print(ds[i])

    exit()

    ds = ImageTextEncodingDataset("/ds-sds/images/imagenet/train")
    print(len(ds))

    for i in range(5):
        print(ds[i])

    start = time()
    embedding_mean = 0
    for i in range(1000):
        embedding_mean += ds[i][1]
    embedding_mean /= 1000
    print(f"needed {(time() - start) / 1000}s per sample for loading")
    print(f"mean embedding = {embedding_mean}")
