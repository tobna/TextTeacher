import os

user = "nauen"  # nauen
# PATH: /netscratch/<user>
base_folder = os.path.join(os.sep, "netscratch", user)
# PATH: /netscratch/<user>/EfficientCVBench
results_folder = os.path.join(base_folder, "EfficientCVBench")
# PATH: /netscratch/<user>/slurm
slurm_output_folder = os.path.join(os.sep, "netscratch", user, "slurm")


_ds_paths = {
    "ade20k": "/ds/images/ADE20k1",
    "aircraft": "/ds-sds/images",
    "animalnet": "/netscratch/nauen/datasets/AnimalNet2k",
    "backgroundnet": "/fscratch/nauen/datasets/INSegment_v1_f1",
    "cifar": "/ds/images/CIFAR",
    "cityscapes": "/ds/images/Cityscapes",
    "counteranimal": "/ds-sds/images/CounterAnimal/LAION-final",
    "cub": "/ds/images/CUB200",
    "distill_imagenet": os.path.join(os.sep, "netscratch", "raue", "transformers_distilled_dataset"),
    "flowers102": "/ds-sds/images",
    "food101": "/ds-sds/images",
    "fornet": "/ds-sds/images/ForNet",
    "imagenet10d_tree": "/fscratch/nauen/datasets/imagenette10d",
    "imagenet1k": "/ds-sds/images/imagenet",
    "imagenet21k": "/ds-sds/images/imagenet21k",
    "imagenet21k_tree": "/netscratch/nauen/datasets/in21k_plus",
    "imagenet21kleaf": "/netscratch/nauen/datasets",
    "imagenet9": "/ds-sds/images/ImageNet-9/bg_challenge/",
    "imagenette": "/fscratch/nauen/imagenette",
    "imagenette10d": "/fscratch/nauen/datasets/imagenette10d",
    "imdb": "/ds/text/IMDB",
    "isic_cancer": "/ds-sds/images/isic/nicolas",
    "oxford_pet": "/ds-sds/images",
    "pathfinder": "...",
    "places365": "ds/images/Places365",
    "recombnet": "/fscratch/nauen/datasets/INSegment_v1_f1",
    "stanford_cars": "/ds-sds/images",
    "tinyimagenet": "/netscratch/nauen/datasets/foraug_old/TinyINSegment",
}


def ds_path(dataset, args=None):
    """Get the (base) path for any dataset.

    Args:
    -----
        dataset (str): The dataset I'm looking for.
        args (DotDict, optional): Run args. If args.custom_dataset_path is set, this one is always returned.

    Returns:
    --------
        str: Path to the dataset root folder.
    """
    if args is not None and "custom_dataset_path" in args and args.custom_dataset_path is not None:
        return args.custom_dataset_path
    return _ds_paths[dataset]
