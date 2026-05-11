from random import uniform

import msgpack
import torch
import torchvision
from datadings.torch import CompressedToPIL
from datadings.torch import Dataset as DDDataset
from PIL import ImageFilter
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    GaussianBlur,
    Grayscale,
    InterpolationMode,
    Normalize,
    RandomChoice,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomSolarize,
    Resize,
    ToTensor,
)
from torchvision.transforms import functional as F

from data import borlan_augments as BLA

_image_and_target_transforms = [
    torchvision.transforms.RandomCrop,
    torchvision.transforms.RandomHorizontalFlip,
    torchvision.transforms.CenterCrop,
    torchvision.transforms.RandomRotation,
    torchvision.transforms.RandomAffine,
    torchvision.transforms.RandomResizedCrop,
    torchvision.transforms.RandomRotation,
]


def apply_dense_transforms(x, y, transforms: torchvision.transforms.transforms.Compose):
    """Apply some transfomations to both image and target.

    Args:
        x (torch.Tensor): image
        y (torch.Tensor): target (image)
        transforms (torchvision.transforms.transforms.Compose): transformations to apply

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x, y) with applyed transformations

    """
    for trans in transforms.transforms:
        if isinstance(trans, torchvision.transforms.RandomResizedCrop):
            params = trans.get_params(x, trans.scale, trans.ratio)
            x = F.resized_crop(x, *params, trans.size, trans.interpolation, antialias=trans.antialias)
            y = F.resized_crop(y.unsqueeze(0), *params, trans.size, 0).squeeze(0)  # nearest neighbor interpolation
        elif isinstance(trans, Resize):
            pre_shape = x.shape
            x = trans(x)
            if x.shape != pre_shape:
                y = F.resize(y.unsqueeze(0), trans.size, 0, trans.max_size, trans.antialias).squeeze(
                    0
                )  # nearest neighbor interpolation
        elif any(isinstance(trans, simul_transform) for simul_transform in _image_and_target_transforms):
            xy = torch.cat([x, y.unsqueeze(0).float()], dim=0)
            xy = trans(xy)
            x, y = xy[:-1], xy[-1].long()
        elif isinstance(trans, torchvision.transforms.ToTensor):
            if not isinstance(x, torch.Tensor):
                x = trans(x)
        else:
            x = trans(x)

    return x, y


def get_hf_transform(transform_f, trgt_transform_f=None, image_key="image"):
    """Convert the transform function to a huggingface compatible transform function.

    Args:
        transform_f (callable): Image transform.
        trgt_transform (callable, optional): Target transform. Defaults to None.
        image_key (str, optional): Key for the image in the hf ds return dict. Defaults to "image".
    """

    def _transform(samples):
        try:
            samples[image_key] = [transform_f(im) for im in samples[image_key]]
            if trgt_transform_f is not None:
                samples["label"] = [trgt_transform_f(tgt) for tgt in samples["label"]]
        except TypeError as e:
            print(f"Type error when transforming samples: {samples}")
            raise e
        return samples

    return _transform


class DDDecodeDataset(DDDataset):
    """Datadings dataset with image decoding before transform."""

    def __init__(self, *args, transform=None, target_transform=None, transforms=None, **kwargs):
        """Create datadings dataset.

        Args:
            transform (callable, optional): Image transform. Overrides transforms['image']. Defaults to None.
            target_transform (callable, optional): Label transform. Overrides transforms['label']. Defaults to None.
            transforms (dict[str, callable], optional): Dict of transforms for each key. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        if transforms is None:
            transforms = {}
        self._decode_transform = transform if transform is not None else transforms.get("image", None)
        self._decode_target_transform = (
            target_transform if target_transform is not None else transforms.get("label", None)
        )

        self.ctp = CompressedToPIL()

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img, lbl = sample["image"], sample["label"]
        if isinstance(img, bytes):
            img = self.ctp(img)
        if self._decode_transform is not None:
            img = self._decode_transform(img)
        if self._decode_target_transform is not None:
            lbl = self._decode_target_transform(lbl)
        return img, lbl


def minimal_augment(args, test=False):
    """Minimal Augmentation set for images.

    Contains only resize, crop, flip, to tensor and normalize.

    Args:
        args (DotDict): Arguments: aug_resize, aug_crop, aug_flip, aug_normalize to turn on/off the respective augmentation.
        test (bool, optional): On the test set? Defaults to False.

    Returns:
        List: Augmentation list

    """
    augs = []
    augs.append(ToTensor())

    if args.aug_resize:
        augs.append(Resize(args.imsize, interpolation=InterpolationMode.BICUBIC))

    if test and args.aug_crop:
        augs.append(CenterCrop(args.imsize))
    elif args.aug_crop:
        augs.append(RandomCrop(args.imsize, padding=4, padding_mode="reflect"))

    if not test and args.aug_flip:
        augs.append(RandomHorizontalFlip(p=0.5))

    if args.aug_normalize:
        augs.append(
            Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            )
        )

    return augs


def three_augment(args, as_list=False, test=False):
    """Create the data augmentation.

    Parameters
    ----------

    Args:
        arguments
    as_list : bool
        return list of transformations, not composed transformation
    test : bool
        In eval mode? If False => train mode

    Returns:
    -------
    torch.nn.Module | list[torch.nn.Module]
        composed transformation of list of transformations

    """
    augs = []
    augs.append(ToTensor())

    if args.aug_resize:
        augs.append(Resize(args.imsize, interpolation=InterpolationMode.BICUBIC))

    if test and args.aug_crop:
        augs.append(CenterCrop(args.imsize))
    elif args.aug_crop:
        augs.append(RandomCrop(args.imsize, padding=4, padding_mode="reflect"))

    if not test:
        if args.aug_flip:
            augs.append(RandomHorizontalFlip(p=0.5))

        augs_choice = []
        if args.aug_grayscale:
            augs_choice.append(Grayscale(num_output_channels=3))
        if args.aug_solarize:
            augs_choice.append(RandomSolarize(threshold=0.5, p=1.0))
        if args.aug_gauss_blur:
            # TODO: check kernel size?
            augs_choice.append(GaussianBlur(kernel_size=7, sigma=(0.2, 2.0)))
            # augs_choice.append(QuickGaussBlur())

        if len(augs_choice) > 0:
            augs.append(RandomChoice(augs_choice))

        if args.aug_color_jitter_factor > 0.0:
            augs.append(
                ColorJitter(
                    args.aug_color_jitter_factor,
                    args.aug_color_jitter_factor,
                    args.aug_color_jitter_factor,
                )
            )

    if args.aug_normalize:
        augs.append(
            Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            )
        )

    if as_list:
        return augs
    return Compose(augs)


def borlan_augment(args, test=False, as_list=False):
    if test:
        return three_augment(args=args, test=True, as_list=as_list)

    assert not as_list, "Can't do BorLan transform as list"

    return BLA.TransformTrain(resize_size=int(args.imsize * 256 / 224), crop_size=args.imsize)


def segment_augment(args, test=False):
    """Create the data augmentation for segmentation.

    No cropping in this part, as cropping has to be done for the image and labels simultaneously.

    Args:
        args (DotDict): arguments
        test (bool, optional): In eval mode? If False => train mode. Defaults to False.

    Returns:
        list[torch.nn.Module]: list of transformations

    """
    augs = []

    if test:
        augs.append(ResizeUp(args.imsize))
        augs.append(CenterCrop(args.imsize))
    else:
        augs.append(RandomResizedCrop(args.imsize, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)))

    if not test and args.aug_flip:
        augs.append(RandomHorizontalFlip(p=0.5))

    if args.aug_color_jitter_factor > 0.0:
        augs.append(
            ColorJitter(
                args.aug_color_jitter_factor,
                args.aug_color_jitter_factor,
                args.aug_color_jitter_factor,
            )
        )

    if args.aug_normalize:
        augs.append(
            Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            )
        )

    return augs


class QuickGaussBlur:
    """Gaussian blur transformation using PIL ImageFilter."""

    def __init__(self, sigma=(0.2, 2.0)):
        """Create Gaussian blur operator.

        Args:
        -----
        sigma : tuple[float, float]
            range of sigma for blur

        """
        self.sigma_min, self.sigma_max = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=uniform(self.sigma_min, self.sigma_max)))


class RemoveTransform:
    """Remove data from transformation.

    To use with default collate function.
    """

    def __call__(self, x, y=None):
        if y is None:
            return [1]
        return [1], y

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def collate_borlan(data):
    images, labels, text_embeddings = zip(*data)
    labels = [i for i in labels for _ in range(len(images[0]))]
    text_embeddings = [emb for emb in text_embeddings for _ in range(len(images[0]))]
    images = [i for imset in images for i in imset]
    return torch.stack(images, dim=0), torch.tensor(labels), torch.stack(text_embeddings, dim=0)


def collate_imnet(data, image_key="image"):
    """Collate function for imagenet(1k / 21k) with datadings.

    Args:
    ----
    data : list[dict[str, Any]]
        images for a batch

    Returns:
    -------
    tuple[torch.Tensor, torch.Tensor]
        images, labels

    """
    if isinstance(data[0][image_key], torch.Tensor):
        ims = torch.stack([d[image_key] for d in data], dim=0)
    else:
        ims = [d[image_key] for d in data]
    labels = torch.tensor([d["label"] for d in data])
    # keys = [d['key'] for d in data]
    return ims, labels  # , keys


def collate_listops(data):
    """Collate function for ListOps with datadings.

    Args:
    ----
    data : list[tuple[torch.Tensor, torch.Tensor]]
        list of samples

    Returns:
    -------
    tuple[torch.Tensor, torch.Tensor]
        images, labels

    """
    return data[0][0], data[0][1]


def no_param_transf(self, sample):
    """Call transformation without extra parameter.

    To use with datadings QuasiShuffler.

    Args:
    ----
    self : object
        use this as a method ( <obj>.<method_name> = MethodType(no_param_transf, <obj>) )
    sample : Any
        sample to transform

    Returns:
    -------
    Any
        transformed sample

    """
    if isinstance(sample, tuple):
        # sample of type (name (str), data (bytes encoded))
        sample = sample[1]
    if isinstance(sample, bytes):
        # decode msgpack bytes
        sample = msgpack.loads(sample)
    params = self._rng(sample)
    for k, f in self._transforms.items():
        sample[k] = f(sample[k], params)
    return sample


class ToOneHotSequence:
    """Convert a sequence of grayscale values (range 0 to 1) to a one-hot encoded sequence based on 8-bit rounded values."""

    def __call__(self, x, y=None):
        # x is 1 x 32 x 32
        x = (255 * x).round().to(torch.int64).view(-1)
        assert x.max() < 256, f"Found max value {x.max()} in {x}."
        x = torch.nn.functional.one_hot(x, num_classes=256).float()
        if y is None:
            return x
        return x, y

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ResizeUp(Resize):
    """Resize up if image is smaller than target size."""

    def forward(self, img):
        w, h = img.shape[-2], img.shape[-1]
        if w < self.size or h < self.size:
            img = super().forward(img)
        return img
