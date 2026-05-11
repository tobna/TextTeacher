"""Default configuration parameters.

Attributes:
    default_kwargs (dict): Default hyperparameters for the training process.
    slurm_defaults (dict): Default values for SLURM batch job settings.

"""

from paths_config import user

default_kwargs = {
    "amp": True,
    "aug_color_jitter_factor": 0.3,
    "aug_crop": True,
    "aug_cutmix_alpha": 1.0,
    "aug_flip": True,
    "aug_gauss_blur": True,
    "aug_grayscale": True,
    "aug_mixup_alpha": 0.0,
    "aug_normalize": True,
    "aug_random_erase_count": 1,
    "aug_random_erase_mode": "pixel",
    "aug_random_erase_prob": 0.25,
    "aug_repeated_augment_repeats": 1,
    "aug_rand_rot": 0,
    "aug_resize": True,
    "aug_solarize": True,
    "augment_strategy": "3-augment",
    "augment_engine": "torchvision",
    "auto_augment_strategy": "rand-m9-mstd0.5-inc1",
    "batch_size": 2048,
    "compile_model": False,
    "cuda": True,
    "custom_dataset_path": None,
    "debug": False,
    "drop_path_rate": 0.05,
    "dropout": 0.0,
    "eval_amp": True,
    "experiment_name": "none",
    "fused_attn": True,
    "gather_stats_during_training": True,
    "imsize": 224,
    "input_dim": None,
    "keep_interm_states": 2,
    "label_smoothing": 0.1,
    "layer_scale_init_values": 1e-4,
    "layer_scale": True,
    "log_level": "info",
    "loss": "ce",
    "loss_weight": "none",
    "lr": 3e-3,
    "max_grad_norm": 1.0,
    "max_seq_len": None,
    "min_lr": 1e-5,
    "momentum": 0.0,
    "num_heads": None,
    "num_workers": 44,
    "opt_eps": 1e-7,
    "opt": "fusedlamb",
    "perf_metric": "acc",
    "pin_memory": False,
    "pre_norm": False,
    "prefetch_factor": 2,
    "qkv_bias": True,
    "run_name": None,
    "save_epochs": 10,
    "sched": "cosine",
    "seed": None,
    "shuffle": True,
    "tqdm": True,
    "wandb": True,
    "warmup_epochs": 5,
    "warmup_lr": 1e-6,
    "warmup_sched": "linear",
    "weight_decay": 0.02,
    "weighted_sampler": False,
    "encoding_model": "bert-large",
    "text_captions": "ImageNet-CoCa",
    "normalize_embedding_mean": True,
    "normalize_embedding_std": "none",
    "text_loss_lambda": 0.0,
    "text_loss_function": "clip",
    "text_loss_schedule": "const",
    "freeze_text_head": False,
    "adaptive_text_loss_weighting": True,
    "online_teacher": None,
}


deit_kwargs = {
    "batch_size": 1024,
    "num_workers": 10,
    "opt": "adamw",
    "weight_decay": 0.05,
    "lr": 1e-3,
    "max_grad_norm": 0.0,
    "opt_eps": 1e-8,
    "augment_strategy": "deit",
    "aug_mixup_alpha": 0.8,
    "aug_repeated_augment_repeats": 3,
}


def get_default_kwargs(settings="deitiii"):
    if settings.lower() == "deitiii":
        return default_kwargs
    if settings.lower() == "deit":
        return {**default_kwargs, **deit_kwargs}
    raise NotImplementedError(f"No such defaults setting: {settings}")


slurm_defaults = {
    "after_job": None,
    "container_image": f"/netscratch/{user}/images/custom_ViT_v2.4.sqsh",
    "container_mounts": (
        f'/netscratch/{user}:/netscratch/{user},/fscratch/{user}:/fscratch/{user},/ds-sds:/ds-sds:ro,/ds:/ds:ro,"`pwd`":"`pwd`"'
    ),
    "container_workdir": '"`pwd`"',
    "cpus_per_task": 24,
    "exclude": None,
    "export": f"ALL,NLTK_DATA=/netscratch/$USER/NLTK_DATA/,TQDM_DISABLE=1,HF_HOME=/fscratch/{user}/HF_HOME/",
    "job_name": None,
    "mem_per_gpu": 48,
    "nodes": 1,
    "ntasks": 4,
    "partition": ["A100-80GB", "A100-40GB", "A100-RP", "H100", "H100-RP", "H200", "H200-SDS"],
    "task_prolog": None,
    "time": "1-0",
}
