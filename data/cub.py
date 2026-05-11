import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, get_worker_info


class CUB2011(Dataset):
    BASE_FOLDER = "CUB_200_2011/CUB_200_2011"
    BASE_CAPTION_FOLDER = "/fscratch/nauen/text_encodings/"

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        text_captions=None,
        encoding_model=None,
        normalize_embedding_mean=True,
        normalize_embedding_std="full",
        eps=1e-6,
    ):
        self.root = Path(root) / self.BASE_FOLDER
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.text_captions = text_captions
        self.encoding_model = encoding_model
        if text_captions is not None and encoding_model is not None:
            encodings_folder = Path(self.BASE_CAPTION_FOLDER) / text_captions / encoding_model
            self.encoding_folder = encodings_folder
            if not encodings_folder.is_dir():
                logger.error(f"Can't open encoding directory: {encodings_folder}")
                raise ValueError(f"Can't open encoding directory: {encodings_folder}")

            self._zipfile = str(encodings_folder / "all_encodings.zip")
            logger.info(f"CUB2011 using encodings: '{self._zipfile}'")
            stats_file = encodings_folder / "stats.npy"
            if stats_file.is_file():
                normalizer = np.load(str(stats_file))
                self.norm_mean, self.norm_std = normalizer
                self.norm_std = np.absolute(self.norm_std) + eps
                if not normalize_embedding_mean:
                    self.norm_mean = 0.0
                if normalize_embedding_std == "mean":
                    self.norm_std = float(self.norm_std.mean())
                elif normalize_embedding_std == "none":
                    self.norm_std = 1.0
            else:
                logger.warning(f"WARNING: Dataset {encodings_folder} has no normalizer (stats.npy).")

        else:
            self._zipfile = None
            self.encoding_folder = None
            self.norm_mean = 0
            self.norm_std = 1
        self._worker_embeddings = {}

        paths = pd.read_csv(self.root / "images.txt", sep=" ", names=["id", "path"])
        labels = pd.read_csv(self.root / "image_class_labels.txt", sep=" ", names=["id", "label"])
        splits = pd.read_csv(self.root / "train_test_split.txt", sep=" ", names=["id", "is_training"])
        data = paths.merge(labels, on="id")
        data = data.merge(splits, on="id")

        if self.train:
            self.data = data[data.is_training == 1]
        else:
            self.data = data[data.is_training == 0]

        logger.debug(f"CUB: Loaded {len(self.data)} {'training' if self.train else 'test'} images")

    def __len__(self):
        return len(self.data)

    def get_emb_for_path(self, path):
        wrkr_id = get_worker_info()
        if wrkr_id not in self._worker_embeddings:
            self._worker_embeddings[wrkr_id] = zipfile.ZipFile(self._zipfile, "r")
        zf = self._worker_embeddings[wrkr_id]
        emb = np.load(zf.open(f"{path}.emb.npy"))
        emb = (emb - self.norm_mean) / self.norm_std  # Normalize embeddings to have mean zero and std 1
        return torch.from_numpy(emb)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = self.root / "images" / sample.path
        target = sample.label - 1  # Apparently the targets start at 1
        image = Image.open(path)

        if self.transform:
            try:
                image = self.transform(image.convert("RGB"))
            except RuntimeError as e:
                logger.error(f"Error transforming image {image}: {e}")
                raise e

        if self.target_transform:
            target = self.target_transform(image)

        if self._zipfile:
            encoding = self.get_emb_for_path(sample.path)
            return image, target, encoding

        return image, target
