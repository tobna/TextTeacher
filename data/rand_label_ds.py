import zlib

from loguru import logger
from torch.utils.data import Dataset


def _hash(data):
    return zlib.adler32(str(data).encode("utf-8"))


class RandomLabels(Dataset):
    """Dataset with random labels."""

    def __init__(self, ds, n_classes, noise_level=1.0):
        """Make labels of dataset random, based on index hash.

        Args:
            ds (Dataset): Base dataset.
            n_classes (int): number of random classes
            noise_level (float): how many samples to change (relative)

        """
        super().__init__()
        self.ds = ds
        self.n_classes = n_classes
        self.noise_level = noise_level
        logger.info(f"Adding {noise_level * 100:.1f}% noise to {ds}")

    def __len__(self):
        return len(self.ds)

    @property
    def text_dim(self):
        return self.ds.text_dim

    def __getitem__(self, idx):
        data = list(self.ds[idx])
        if _hash(f"rand {idx}") % 1000 <= self.noise_level * 1000:
            data[1] = _hash(f"label {idx}") % self.n_classes  # data is: [image, label (, embedding)]
        return tuple(data)
