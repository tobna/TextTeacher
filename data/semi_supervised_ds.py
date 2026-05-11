import zlib

from loguru import logger
from torch.utils.data import Dataset


def _hash(data):
    return zlib.adler32(str(data).encode("utf-8"))


class SemiDS(Dataset):
    """Make a dataset semi-supervised by overwriting a proportion of labels with a dummy label."""

    def __init__(self, ds, num_classes: int, no_label_prob: float) -> None:
        super().__init__()
        assert 0 <= no_label_prob <= 1
        self.ds = ds
        self.no_lbl_cls = num_classes
        self.no_label_prob = no_label_prob
        self.num_classes = num_classes + 1
        self.granularity = 10000

    def __getitem__(self, index):
        data = self.ds.__getitem__(index)
        if _hash(f"no label for {index}?") % self.granularity <= self.no_label_prob * self.granularity:
            data = list(data)
            data[1] = self.no_lbl_cls
        return tuple(data)

    def __len__(self):
        return len(self.ds)

    @property
    def text_dim(self):
        return self.ds.text_dim


class SubDS(Dataset):
    """Make a dataset semi-supervised by overwriting a proportion of labels with a dummy label."""

    def __init__(self, ds, include_prob: float, reverse=False) -> None:
        super().__init__()
        assert 0 <= include_prob <= 1
        self.ds = ds
        self.include_prob = include_prob
        self.granularity = 10000
        if reverse:
            self.indices = [
                i
                for i in range(len(ds))
                if _hash(f"no label for {i}?") % self.granularity > (1 - include_prob) * self.granularity
            ]
        else:
            self.indices = [
                i
                for i in range(len(ds))
                if _hash(f"no label for {i}?") % self.granularity <= include_prob * self.granularity
            ]
        logger.debug(f"SubDS choosing indices: {self.indices[:20]}...")

    def __getitem__(self, index):
        return self.ds.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)

    @property
    def num_classes(self):
        return self.ds.num_classes

    @property
    def text_dim(self):
        return self.ds.text_dim
