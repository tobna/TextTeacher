"""Module to load the datasets, using torch and datadings."""

import os

import torchvision.transforms as tv_transforms
from data.rand_label_ds import RandomLabels
from datadings.reader import MsgpackReader
from loguru import logger
from timm.data import create_transform
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torchvision.datasets import (
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageFolder,
    OxfordIIITPet,
    StanfordCars,
)

from data.cub import CUB2011
from data.data_utils import (
    DDDecodeDataset,
    ToOneHotSequence,
    borlan_augment,
    collate_borlan,
    minimal_augment,
    segment_augment,
    three_augment,
)
from data.imagenet_text_encodings import (
    ImageTextEncodingDataset,
    ImageTextEncodingDatasetDatadings,
)
from data.samplers import RASampler
from data.semi_supervised_ds import SubDS
from paths_config import ds_path


def prepare_dataset(dataset_name, args, transform=None, train=True, rank=None):
    """Load a dataset from disk, different formats are used for different datasets.

    Supported datasets: CIFAR10, ImageNet, ImageNet21k

    Args:
        dataset_name (str): name of the dataset
        args: further arguments
        transform (list[Module] | str, optional): transformations to use on the data; the list gets composed, or give args.augment_strategy (Default value = None)
        train (bool, optional): use the training split (or test/validation split) (Default value = True)
        rank (int, optional): global rank of this process in distributed training (Default value = None)

    Returns:
        DataLoader: data loader for the dataset
        int: number of classes in the dataset
        int: ignore index for the dataset
        bool: whether the dataset is multi-label

    """
    compose = tv_transforms.Compose
    collate = None
    if transform is None:
        logger.debug(f"creating transform: {args.augment_strategy}")
        if args.augment_engine == "torchvision":
            if args.augment_strategy == "3-augment":
                transform = three_augment(args, as_list=False, test=not train)
            elif args.augment_strategy == "differentiable-transform":
                from data.distilled_dataset import differentiable_augment

                transform = differentiable_augment(args, as_list=False, test=not train)
            elif args.augment_strategy == "none":
                transform = []
            elif args.augment_strategy == "lm_one_hot":
                transform = [
                    tv_transforms.Grayscale(num_output_channels=1),
                    tv_transforms.ToTensor(),
                    ToOneHotSequence(),
                ]
            elif args.augment_strategy == "segment-augment":
                transform = segment_augment(args, test=not train)
            elif args.augment_strategy == "minimal":
                transform = minimal_augment(args, test=not train)
            elif args.augment_strategy == "deit":
                if train:
                    transform = create_transform(
                        input_size=args.imsize,
                        is_training=True,
                        color_jitter=args.aug_color_jitter_factor,
                        auto_augment=args.auto_augment_strategy,
                        interpolation="bicubic",
                        re_prob=args.aug_random_erase_prob,
                        re_mode=args.aug_random_erase_mode,
                        re_count=args.aug_random_erase_count,
                    )
                else:
                    transform = three_augment(args, test=True)  # only do resize, centercrop, and normalize
            elif args.augment_strategy == "borlan":
                transform = borlan_augment(args, test=not train, as_list=False)
                collate = collate_borlan if train else None
            else:
                raise NotImplementedError(
                    f"Augmentation strategy {args.augment_strategy} is not implemented for {args.augment_engine} (yet)."
                )
        elif args.augment_engine == "albumentations":
            from data import album_transf as ATf

            compose = ATf.AlbumTorchCompose

            if args.augment_strategy == "3-augment":
                transform = ATf.three_augment(args, as_list=False, test=not train)
            elif args.augment_strategy == "minimal":
                transform = ATf.minimal_augment(args, test=not train)
            else:
                raise NotImplementedError(
                    f"Augmentation strategy {args.augment_strategy} is not implemented for {args.augment_engine} (yet)."
                )

    dataset_name_case_sensitive = dataset_name  # keep the original name for AnimalNet folder
    dataset_name = dataset_name.lower()
    ignore_index = -100
    multi_label = False
    has_text_embeddings = False
    unsuper_ds = None

    if isinstance(transform, list):
        transform = compose(transform)

    logger.debug(f"creating dataset: {dataset_name}")
    if dataset_name == "imagenet":
        if train:
            dataset = ImageTextEncodingDataset(
                os.path.join(ds_path("imagenet1k"), "train"),
                transform=transform,
                encoding_model=args.encoding_model,
                text_captions=args.text_captions,
                normalize_embedding_mean=args.normalize_embedding_mean,
                normalize_embedding_std=args.normalize_embedding_std,
                debug=args.debug,
            )
        else:
            dataset = ImageFolder(os.path.join(ds_path("imagenet1k"), "val"), transform=transform)

        n_classes = 1000
        has_text_embeddings = True

    elif dataset_name.startswith("rand-imagenet-"):
        rand_noise = float(dataset_name.split("-")[-1])

        if train:
            dataset = ImageTextEncodingDataset(
                os.path.join(ds_path("imagenet1k"), "train"),
                transform=transform,
                encoding_model=args.encoding_model,
                text_captions=args.text_captions,
                normalize_embedding_mean=args.normalize_embedding_mean,
                normalize_embedding_std=args.normalize_embedding_std,
                debug=args.debug,
            )
        else:
            dataset = ImageFolder(os.path.join(ds_path("imagenet1k"), "val"), transform=transform)

        n_classes = 1000
        has_text_embeddings = True
        dataset = RandomLabels(dataset, n_classes=n_classes, noise_level=rand_noise)

    elif dataset_name.startswith("semi-imagenet-"):
        label_prob = float(dataset_name.split("-")[-1])

        if train:
            base_dataset = ImageTextEncodingDataset(
                os.path.join(ds_path("imagenet1k"), "train"),
                transform=transform,
                encoding_model=args.encoding_model,
                text_captions=args.text_captions,
                normalize_embedding_mean=args.normalize_embedding_mean,
                normalize_embedding_std=args.normalize_embedding_std,
                debug=args.debug,
            )
            dataset = SubDS(base_dataset, include_prob=label_prob)
            unsuper_ds = SubDS(base_dataset, include_prob=1 - label_prob, reverse=True)
        else:
            dataset = ImageFolder(os.path.join(ds_path("imagenet1k"), "val"), transform=transform)

        n_classes = 1000
        has_text_embeddings = True

    elif dataset_name.startswith("sub-imagenet-"):
        include_prob = float(dataset_name.split("-")[-1])
        if train:
            dataset = ImageTextEncodingDataset(
                os.path.join(ds_path("imagenet1k"), "train"),
                transform=transform,
                encoding_model=args.encoding_model,
                text_captions=args.text_captions,
                normalize_embedding_mean=args.normalize_embedding_mean,
                normalize_embedding_std=args.normalize_embedding_std,
                debug=args.debug,
            )
            dataset = SubDS(dataset, include_prob=include_prob)
        else:
            dataset = ImageFolder(os.path.join(ds_path("imagenet1k"), "val"), transform=transform)

        n_classes = 1000
        has_text_embeddings = True

    elif dataset_name == "imagenet21k":
        if train:
            dataset = ImageTextEncodingDatasetDatadings(
                os.path.join(ds_path("imagenet21k"), "train.msgpack"),
                transform=transform,
                encoding_model=args.encoding_model,
                text_captions=args.text_captions.replace("ImageNet-", "ImageNet21k-"),
                normalize_embedding_mean=args.normalize_embedding_mean,
                normalize_embedding_std=args.normalize_embedding_std,
            )
        else:
            dataset = DDDecodeDataset(
                MsgpackReader(os.path.join(ds_path("imagenet21k"), "val.msgpack")), transform=transform
            )
        n_classes = 10_450
        has_text_embeddings = True

    elif dataset_name == "cub2011":
        dataset = CUB2011(
            ds_path("cub"),
            train=train,
            transform=transform,
            encoding_model=args.encoding_model if train and args.text_loss_lambda > 0 else None,
            text_captions=args.text_captions if train and args.text_loss_lambda > 0 else None,
            normalize_embedding_mean=args.normalize_embedding_mean,
            normalize_embedding_std=args.normalize_embedding_std,
        )
        n_classes = 200
        has_text_embeddings = True

    elif dataset_name == "fgvc-aircraft":
        dataset = FGVCAircraft(
            root=ds_path("aircraft"),
            split="train" if train else "test",
            annotation_level="variant",
            download=False,
            transform=transform,
        )
        n_classes = 100
        has_text_embeddings = False

    elif dataset_name == "stanford-cars":
        dataset = StanfordCars(
            root=ds_path("stanford_cars"),
            split="train" if train else "test",
            download=False,
            transform=transform,
        )
        n_classes = 196
        has_text_embeddings = False

    elif dataset_name == "oxford-pet":
        dataset = OxfordIIITPet(
            root=ds_path("oxford_pet"),
            split="trainval" if train else "test",
            download=False,
            transform=transform,
        )
        n_classes = 37
        has_text_embeddings = False

    elif dataset_name == "flowers102":
        dataset = Flowers102(
            root=ds_path("flowers102"),
            split="train" if train else "test",
            download=False,
            transform=transform,
        )
        n_classes = 102
        has_text_embeddings = False

    elif dataset_name == "food-101":
        dataset = Food101(
            root=ds_path("food101"),
            split="train" if train else "test",
            download=False,
            transform=transform,
        )
        n_classes = 101
        has_text_embeddings = False

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented (yet).")

    if args.aug_repeated_augment_repeats > 1 and train:
        # use repeated augment sampler from DeiT
        sampler = RASampler(
            dataset,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=args.shuffle,
            num_repeats=args.aug_repeated_augment_repeats,
        )
    elif args.weighted_sampler:
        assert hasattr(
            dataset, "per_sample_weights"
        ), f"Dataset {type(dataset)} should implement per_sample_weights function, but does not."

        sampler = WeightedRandomSampler(dataset.per_sample_weights(), num_samples=len(dataset) // args.world_size)
    elif args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank, shuffle=train and args.shuffle)
        logger.debug(
            "Creating distributed sampler with"
            f" {dict(num_replicas=args.world_size, rank=rank, shuffle=train and args.shuffle)} on {dataset} of length"
            f" {len(dataset)}"
        )
    else:
        sampler = None

    loader_batch_size = args.batch_size
    if dataset_name.startswith("listops"):
        loader_batch_size = 1
    elif args.augment_strategy == "borlan":
        loader_batch_size = int(args.batch_size / 2)

    loader_kwargs = dict(
        batch_size=loader_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=train,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=False,
        collate_fn=collate,
        shuffle=None if sampler else train and args.shuffle,
        sampler=sampler,
    )
    if unsuper_ds is not None:
        loader_kwargs["batch_size"] = loader_batch_size // 2
        loader_kwargs["num_workers"] = args.num_workers // 2
    logger.debug(f"creating data loader (train={train}) with: {loader_kwargs}")

    data_loader = DataLoader(dataset, **loader_kwargs)
    logger.debug(f"loader has {len(data_loader)} batches")
    if unsuper_ds is not None:
        unsuper_loader = DataLoader(unsuper_ds, **loader_kwargs)
        logger.debug(f"unsuper loader has {len(data_loader)} batches")
    else:
        unsuper_loader = None

    return data_loader, n_classes, ignore_index, has_text_embeddings, unsuper_loader
