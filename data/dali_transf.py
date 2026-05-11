from nvidia.dali import fn, pipeline_def, types

# see https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_dali_proxy.html


@pipeline_def
def minimal_augment(args, test=False):
    """Minimal Augmentation set for images.

    Contains only resize, crop, flip, to tensor and normalize.

    Args:
        args (DotDict): Arguments: aug_resize, aug_crop, aug_flip, aug_normalize to turn on/off the respective augmentation.
        test (bool, optional): On the test set? Defaults to False.

    Returns:
        images: augmented images.

    """
    images = fn.external_source(name="images", no_copy=True)

    if args.aug_resize:
        images = fn.resize(images, size=args.imsize, mode="not_smaller")

    if test and args.aug_crop:
        images = fn.crop(images, crop=(args.imsize, args.imsize), crop_pos_x=0.5, crop_pos_y=0.5)
    elif args.aug_crop:
        images = fn.crop(
            images,
            crop=(args.imsize, args.imsize),
            crop_pos_x=fn.random.uniform(range=(0, 1)),
            crop_pos_y=fn.random.uniform(range=(0, 1)),
        )

    # if not test and args.aug_flip:
    #     images = fn.flip(images, horizontal=fn.random.coin_flip())

    # if args.aug_normalize:
    # images = fn.normalize(
    #     images,
    #     mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    #     std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    #     dtype=types.FLOAT,
    # )
    return fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(args.imsize, args.imsize),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255] if args.aug_normalize else [0, 0, 0],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255] if args.aug_normalize else [1, 1, 1],
        mirror=fn.random.coin_flip(probability=0.5) if args.aug_flip and not test else False,
    )


def dali_solarize(images, threshold=128):
    """Solarize implementation for nvidia DALI.

    Args:
        images (DALI Tensor): Images to solarize.
        threshold (int, optional): Threshold for solarization. Defaults to 128.

    Returns:
        images: solarized images.

    """
    inv_images = types.Constant(255).uint8() - images
    mask = (images >= threshold) * types.Constant(1).uint8()
    return mask * inv_images + (types.Constant(1).uint8() ^ mask) * images


@pipeline_def(enable_conditionals=True)
def three_augment(args, test=False):
    """3-augment data augmentation pipeline for nvidia DALI.

    Args:
        args (namespace): augmentation arguments.
        test (bool, optional): Test (or train) split. Defaults to False.

    Returns:
        images: augmented images.

    """
    images = fn.external_source(name="images", no_copy=True)

    if args.aug_resize:
        images = fn.resize(images, size=args.imsize, mode="not_smaller")

    if test and args.aug_crop:
        images = fn.crop(images, crop=(args.imsize, args.imsize), crop_pos_x=0.5, crop_pos_y=0.5)
    elif args.aug_crop:
        images = fn.crop(
            images,
            crop=(args.imsize, args.imsize),
            crop_pos_x=fn.random.uniform(range=(0, 1)),
            crop_pos_y=fn.random.uniform(range=(0, 1)),
        )

    if not test:
        choices = []
        # choice = fn.random.choice(3)
        # print(images.layout())
        choice_ps = [1 * args.aug_grayscale, 1 * args.aug_solarize, 1 * args.aug_gauss_blur]
        choice_ps = [c / sum(choice_ps) for c in choice_ps]
        choice = fn.random.choice(
            [0, 1, 2],
            p=choice_ps,
        )

        if choice == 0:
            images = fn.color_space_conversion(
                fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY),
                image_type=types.GRAY,
                output_type=types.RGB,
            )

        elif choice == 1:
            images = dali_solarize(images, threshold=128)
        elif choice == 2:
            images = fn.gaussian_blur(images, window_size=7, sigma=fn.random.uniform(range=(0.2, 2.0)))

        if len(choices) > 0:
            images = fn.random.choice(choices)

        if args.aug_color_jitter_factor > 0.0:
            images = fn.color_twist(
                images,
                brightness=fn.random.uniform(
                    range=(1 - args.aug_color_jitter_factor, 1 + args.aug_color_jitter_factor)
                ),
                contrast=fn.random.uniform(range=(1 - args.aug_color_jitter_factor, 1 + args.aug_color_jitter_factor)),
                saturation=fn.random.uniform(
                    range=(1 - args.aug_color_jitter_factor, 1 + args.aug_color_jitter_factor)
                ),
            )

    return fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(args.imsize, args.imsize),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255] if args.aug_normalize else [0, 0, 0],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255] if args.aug_normalize else [1, 1, 1],
        mirror=fn.random.coin_flip(probability=0.5) if args.aug_flip and not test else False,
    )
