import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from datadings.torch import CompressedToPIL


class AlbumTorchCompose(A.Compose):
    """Compose albumentation augmentations in a way that works with PIL images and datadings."""

    def __init__(self, *args, **kwargs):
        """Pass to A.Compose."""
        super().__init__(*args, **kwargs)
        self.to_pil = CompressedToPIL()

    def __call__(self, image, mask=None, **kwargs):
        if isinstance(image, bytes):
            image = self.to_pil(image)
            if mask is not None and len(mask) == 0:
                mask = None
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if mask is not None and not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask is None:
            return super().__call__(image=image, **kwargs)["image"]
        return super().__call__(image=image, mask=mask, **kwargs)


class PILToNP(A.DualTransform):
    """Convert PIL image to numpy array."""

    def apply(self, image, **params):
        return np.array(image)

    def apply_to_mask(self, image, **params):
        return np.array(image)

    def get_transform_init_args_names(self):
        return ()


class AlbumCompressedToPIL(A.DualTransform):
    """Convert compressed image to PIL image."""

    def apply(self, img, **params):
        return self.to_pil(img)

    def apply_to_mask(self, img, **params):
        return self.to_pil(img)

    def get_transform_init_args_names(self):
        return ()


def minimal_augment(args, test=False):
    """Get minimal augmentations for training or testing.

    Args:
        args (argparse.Namespace): arguments
        test (bool, optional): if True, return test augmentations. Defaults to False.

    Returns:
        List: Augmentation list
    """
    augs = []

    if args.aug_resize:
        augs.append(A.SmallestMaxSize(args.imsize, interpolation=cv2.INTER_CUBIC))

    if test and args.aug_crop:
        augs.append(A.CenterCrop(args.imsize, args.imsize))
    elif args.aug_crop:
        augs.append(A.RandomCrop(args.imsize, args.imsize))

    if not test and args.aug_flip:
        augs.append(A.HorizontalFlip(p=0.5))

    if args.aug_normalize:
        augs.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    augs.append(ToTensorV2())
    return augs


def three_augment(args, as_list=False, test=False):
    """Create the data augmentation.

    Args:
        args (Namespace): arguments
        as_list (bool): return list of transformations, not composed transformation
        test (bool): In eval mode? If False => train mode

    Returns:
    torch.nn.Module | list[torch.nn.Module]: composed transformation or list of transformations

    """
    augs = []

    if args.aug_resize:
        augs.append(A.SmallestMaxSize(args.imsize, interpolation=cv2.INTER_CUBIC))

    if test and args.aug_crop:
        augs.append(A.CenterCrop(args.imsize, args.imsize))
    elif args.aug_crop:
        augs.append(A.RandomCrop(args.imsize, args.imsize, pad_if_needed=True, border_mode=cv2.BORDER_REFLECT))

    if not test:
        if args.aug_flip:
            augs.append(A.HorizontalFlip(p=0.5))

        augs_choice = []
        if args.aug_grayscale:
            augs_choice.append(A.ToGray(p=1, num_output_channels=3))

        if args.aug_solarize:
            augs_choice.append(A.Solarize(p=1, threshold_range=(0.5, 0.5)))

        if args.aug_gauss_blur:
            augs_choice.append(A.GaussianBlur(p=1, sigma_limit=(0.2, 2.0), blur_limit=(7, 7)))

        if len(augs_choice) > 0:
            augs.append(A.OneOf(augs_choice))

        if args.aug_color_jitter_factor > 0.0:
            augs.append(
                A.ColorJitter(
                    brightness=args.aug_color_jitter_factor,
                    contrast=args.aug_color_jitter_factor,
                    saturation=args.aug_color_jitter_factor,
                    hue=0.0,
                )
            )

    if args.aug_normalize:
        augs.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    augs.append(ToTensorV2())

    if as_list:
        return augs
    return AlbumTorchCompose(augs)
