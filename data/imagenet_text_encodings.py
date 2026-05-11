import os
import subprocess
import zipfile

import numpy as np
import torch
from datadings.reader import MsgpackReader
from datadings.torch import CompressedToPIL
from datadings.torch import Dataset as DDDs
from datadings.torch import get_worker_info
from loguru import logger
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

_ROOT_ENCODINGS = "/fscratch/nauen/text_encodings"


def has_text_embedding(image_name, encoding_folder):
    """Does this (ImageNet) file have a text encoding?"""
    return os.path.isfile(os.path.join(encoding_folder, f"{image_name.split(os.sep)[-1]}.emb.npy"))


def is_class_embeddings(encoding_folder):
    if os.path.isfile(os.path.join(encoding_folder, "all_encodings.zip")):
        with zipfile.ZipFile(os.path.join(encoding_folder, "all_encodings.zip")) as zf:
            file_list = zf.namelist()
    else:
        file_list = subprocess.check_output(f"ls {encoding_folder} -1f | head", shell=True).decode("utf-8").split("\n")
    return not all(
        [encoding.strip().endswith(".JPEG.emb.npy") for encoding in file_list if encoding.endswith(".emb.npy")]
    )


class ImageTextEncodingDataset(ImageFolder):
    """Dataset of Image and caption encoding."""

    def __init__(
        self,
        image_folder,
        transform=None,
        target_transform=None,
        encoding_model="bert-large",
        text_captions="ImageNet-Dragonfly",
        eps=1e-6,
        normalize_embedding_mean=True,
        normalize_embedding_std="none",
        debug=False,
        cache_text_embeddings=True,
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
            target_transform=target_transform,
            # is_valid_file=partial(has_text_embedding, encoding_folder=self.encoding_folder),
        )
        self._emb_cache = None
        self.encoding_model = encoding_model
        self.text_captions = text_captions
        self.encoding_folder = os.path.join(_ROOT_ENCODINGS, text_captions, encoding_model)
        self.class_embeddings = is_class_embeddings(self.encoding_folder)

        self.zip_embeddings = os.path.isfile(os.path.join(self.encoding_folder, "all_encodings.zip"))
        if self.zip_embeddings:
            self._zip = os.path.join(self.encoding_folder, "all_encodings.zip")
            embeddings = np.load(self._zip)
            embedding_keys = set(embeddings.keys())
            self._worker_embeddings = {}
            self.samples = [
                (path, trgt)
                for path, trgt in tqdm(self.samples, desc="Filtering samples (zip)")
                if self.class_embeddings or path.split(os.sep)[-1] + ".emb" in embedding_keys
            ]

        else:
            # self.samples = self.samples[:1000]
            if debug:
                self.samples = self.samples[:1000]
                logger.warning("USING ONLY 1000 SAMPLES FROM IMAGENET!")
            self.samples = [
                (path, trgt)
                for path, trgt in tqdm(self.samples, desc="Filtering samples (files)")
                if self.class_embeddings or has_text_embedding(path, self.encoding_folder)
            ]
        assert len(self.samples) > 0, "Found no samples with text caption."
        logger.info(
            f"image text encoding dataset has {len(self.samples)} samples; using"
            f" {'zip' if self.zip_embeddings else 'file'} mode"
        )
        logger.info("Using class embeddings" if self.class_embeddings else "Using image embeddings")
        logger.info(
            f"Using mean normalization: {normalize_embedding_mean}; using std normalization: {normalize_embedding_std}"
        )

        self.eps = eps
        if os.path.isfile(os.path.join(self.encoding_folder, "stats.npy")):
            normalizer = np.load(os.path.join(_ROOT_ENCODINGS, text_captions, encoding_model, "stats.npy"))
            self.norm_mean, self.norm_std = normalizer
            self.norm_std = np.absolute(self.norm_std) + eps
            if not normalize_embedding_mean:
                self.norm_mean = 0.0
            if normalize_embedding_std == "mean":
                self.norm_std = float(self.norm_std.mean())
            elif normalize_embedding_std == "none":
                self.norm_std = 1.0
        else:
            logger.warning(f"WARNING: Dataset {self.encoding_folder} has no normalizer (stats.npy).")
            self.norm_mean, self.norm_std = 0.0, 1.0
        logger.debug(f"Normalizer mean: {self.norm_mean}, Normalizer std: {self.norm_std}")

        # logging.info(f"INFO: Preload all {len(self.samples)} text embeddings")
        # self._text_embeddings = {path: np.load(get_emb_file_from_path(path)) for path, _ in self.samples}
        # with Pool(12) as pool:
        #     result = pool.map(get_emb_for_path, [path for path, _ in self.samples])
        # logging.info("INFO: Preloading done")
        # self._text_embeddings = {key: val for key, val in result}

        if cache_text_embeddings and not debug:
            self._emb_cache = {}
            logger.debug("Caching text embeddings")
            for i, (path, _) in enumerate(self.samples):
                self.get_emb_for_path(path)
                if ((i + 1) % int(len(self) / 10)) == 0 or i == 0:
                    logger.info(
                        f"Caching embedding {len(self._emb_cache)}={i + 1}/{len(self.samples)} of"
                        f" {'train' if 'train' in image_folder else 'val'} set"
                    )
            logger.info(f"Cached {len(self._emb_cache)} text embeddings for {len(self.samples)} images")

        self.text_dim = (
            int(self.__getitem__(0)[-1].shape[0]) if isinstance(self.norm_mean, float) else int(self.norm_mean.shape[0])
        )

    def get_emb_for_path(self, path):
        id = path.split(os.sep)[-1]
        if self.class_embeddings:
            id = id.split("_")[0]
        if self._emb_cache is not None and id in self._emb_cache:
            emb = self._emb_cache[id]
        else:
            if self.zip_embeddings:
                wrkr_id = get_worker_info()
                if wrkr_id not in self._worker_embeddings:
                    self._worker_embeddings[wrkr_id] = zipfile.ZipFile(self._zip, "r")
                zf = self._worker_embeddings[wrkr_id]
                emb = np.load(zf.open(f"{id}.emb.npy"))

            else:
                emb_file = os.path.join(self.encoding_folder, f"{id}.emb.npy")
                try:
                    emb = np.load(emb_file)
                except OSError as e:
                    logger.warning(f"Failed to load embedding '{emb_file}'.")
                    raise e
            if self._emb_cache is not None:
                self._emb_cache[id] = emb
        emb = (emb - self.norm_mean) / self.norm_std  # Normalize embeddings to have mean zero and std 1
        return torch.from_numpy(emb)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image, target = super().__getitem__(index)
        embedding = self.get_emb_for_path(path)

        return image, target, embedding


class ImageTextEncodingDatasetDatadings(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None,
        target_transform=None,
        encoding_model="bert-large",
        text_captions="ImageNet-Dragonfly",
        eps=1e-6,
        normalize_embedding_mean=True,
        normalize_embedding_std="none",
    ):
        """Create the image-caption encoding dataset.

        Args:
            image_folder (str): Folder to load the IamgeFolder dataset from.
            transform (callable, optional): image data augmentation.
            encoding_model (str, optional): model which was used to encode.
            text_captions (str, optional): text caption dataset. Usually is 'ImageNet-<caption model>'.
            eps (float, optional): small epsilon to add to not divide by zero.

        """
        super().__init__()
        self.encoding_model = encoding_model
        self.text_captions = text_captions
        self.transform = transform
        self.decoder = CompressedToPIL()
        self.target_transform = target_transform
        self.encoding_folder = os.path.join(_ROOT_ENCODINGS, text_captions, encoding_model)
        self.class_embeddings = is_class_embeddings(self.encoding_folder)
        _ds = DDDs(MsgpackReader(dataset_path))
        self.base_ds_path = dataset_path
        self._ds = {}
        logger.debug(f"ds: {_ds}, of len {len(_ds)}")

        self.zip_embeddings = os.path.isfile(os.path.join(self.encoding_folder, "all_encodings.zip"))
        if self.zip_embeddings:
            self._zip = os.path.join(self.encoding_folder, "all_encodings.zip")
            embeddings = np.load(self._zip)
            embedding_keys = set(embeddings.keys())
            self._worker_embeddings = {}
            self._samples = [
                i
                for i, sample in enumerate(tqdm(_ds, desc="Filtering samples for having an embedding (zip)"))
                # (path, trgt)
                # for path, trgt in tqdm(self.samples, desc="Filtering samples (zip)")
                if self.class_embeddings or sample["key"].split(os.sep)[-1] + ".emb" in embedding_keys
            ]

        else:
            # self.samples = self.samples[:1000]
            if debug:
                self._samples = self._samples[:1000]
                logger.warning("USING ONLY 1000 SAMPLES FROM IMAGENET!")
            # self.samples = self.samples[:1000]
            self._samples = [
                i
                for i, sample in enumerate(tqdm(_ds, desc="Filtering samples for having an embedding (file)"))
                if self.class_embeddings or has_text_embedding(sample["key"], self.encoding_folder)
            ]
        # self._samples = list(range(len(_ds)))
        assert len(self._samples) > 0, "Found no samples with text caption."
        logger.info(f"{len(self._samples)} with encodings in dataset (from {len(_ds)} total samples)")

        self.eps = eps
        if os.path.isfile(os.path.join(self.encoding_folder, "stats.npy")):
            normalizer = np.load(os.path.join(_ROOT_ENCODINGS, text_captions, encoding_model, "stats.npy"))
            self.norm_mean, self.norm_std = normalizer
            self.norm_std = np.absolute(self.norm_std) + eps
            if not normalize_embedding_mean:
                self.norm_mean = 0.0
            if normalize_embedding_std == "mean":
                self.norm_std = float(self.norm_std.mean())
            elif normalize_embedding_std == "none":
                self.norm_std = 1.0
        else:
            logger.warning(f"WARNING: Dataset {self.encoding_folder} has no normalizer (stats.npy).")
            self.norm_mean, self.norm_std = 0.0, 1.0

        self.text_dim = (
            int(self.__getitem__(0)[-1].shape[0]) if isinstance(self.norm_mean, float) else int(self.norm_mean.shape[0])
        )
        logger.debug(f"text dim: {self.text_dim}")

    def get_emb_for_path(self, path):
        id = path.split(os.sep)[-1]
        if self.zip_embeddings:
            wrkr_id = get_worker_info()
            if wrkr_id not in self._worker_embeddings:
                self._worker_embeddings[wrkr_id] = zipfile.ZipFile(self._zip, "r")
            zf = self._worker_embeddings[wrkr_id]
            emb = (
                np.load(zf.open(f"{id.split('_')[0]}.emb.npy"))
                if self.class_embeddings
                else np.load(zf.open(f"{id}.emb.npy"))
            )

        else:
            if self.class_embeddings:
                emb_file = os.path.join(self.encoding_folder, f"{id.split('_')[0]}.emb.npy")
            else:
                emb_file = os.path.join(self.encoding_folder, f"{id}.emb.npy")
            try:
                emb = np.load(emb_file)
            except OSError as e:
                logger.warning(f"Failed to load embedding '{emb_file}'.")
                raise e
        emb = (emb - self.norm_mean) / self.norm_std  # Normalize embeddings to have mean zero and std 1
        return torch.from_numpy(emb)

    def __len__(self):
        return len(self._samples)

    def _get_dd_sample(self, i):
        worker = get_worker_info()

        if worker not in self._ds:
            self._ds[worker] = MsgpackReader(self.base_ds_path)

        return self._ds[worker][i]

    def __getitem__(self, index):
        ds_idx = self._samples[index]
        try:
            sample = self._get_dd_sample(ds_idx)
        except ValueError as e:
            logger.error(f"error reading sample {index} ({ds_idx}): {e}")
            raise e
        image = self.decoder(sample["image"])
        path = sample["key"]
        target = sample["label"]
        embedding = self.get_emb_for_path(path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target, embedding
