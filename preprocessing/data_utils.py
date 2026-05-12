from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderWithKey(Dataset):
    def __init__(self, folder):
        super().__init__()

        print(f"looking for files in {Path(folder)}")
        self.files = list(Path(folder).glob("*.jpg"))
        print(f"found {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.files[index]
        image = Image.open(path)

        return dict(image=image, key=path)


class CUB2011(Dataset):
    BASE_FOLDER = "CUB_200_2011/CUB_200_2011"

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = Path(root) / self.BASE_FOLDER
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        paths = pd.read_csv(self.root / "images.txt", sep=" ", names=["id", "path"])
        labels = pd.read_csv(self.root / "image_class_labels.txt", sep=" ", names=["id", "label"])
        splits = pd.read_csv(self.root / "train_test_split.txt", sep=" ", names=["id", "is_training"])
        data = paths.merge(labels, on="id")
        data = data.merge(splits, on="id")

        if self.train:
            self.data = data[data.is_training == 1]
        else:
            self.data = data[data.is_training == 0]

        print(f"CUB: Loaded {len(self.data)} {'training' if self.train else 'test'} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = self.root / "images" / sample.path
        target = sample.label - 1  # Apparently the targets start at 1
        image = Image.open(path)

        if self.transform:
            try:
                image = self.transform(image.convert("RGB"))
            except RuntimeError as e:
                print(f"Error transforming image {image}: {e}")
                raise e

        if self.target_transform:
            target = self.target_transform(image)

        return dict(image=image, key=sample.path, label=target)
