#!/usr/bin/env python3

"""Parse args and call the correct script inside slurm container.

Outside the container, on the head-node, create and call the correct srun command.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime

from config import slurm_defaults
from paths_config import results_folder, slurm_output_folder

_EXPNAMES = [
    "EfficientCVBench",
    "taylor_shift",
    "vit_fpga",
    "dense_trafo",
    "trafo_distill",
    "BiAtt",
    "better_imagenet",
    "rand_lbl",
    "test",
    "pixel_segment",
    "recombine_imagenet",
    "multi_fornet",
    "norm_attn",
    "cancer21k",
    "text_guidance",
]


def base_parser():
    """Create the argument parser with all the choices for the training / evaluation scripts."""
    parser = argparse.ArgumentParser("Transformer training and evaluation.")

    # Main
    group = parser.add_argument_group("Main")
    group.add_argument(
        "-t",
        "--task",
        nargs="?",
        choices=[
            "pre-train",
            "fine-tune",
            "fine-tune-head",
            "eval",
            "parser-test",
            "eval-metrics",
            "continue",
            "per-class-accuracy",
            "per-example-loss",
            "load-images",
            "extract-embeddings",
            "fid-data",
        ],
        required=True,
        help="Task to perform.",
    )
    group.add_argument(
        "-m",
        "--model",
        nargs="?",
        type=str,
        required=True,
        help="Model to use. Either model name for a new model or weights and dicts to load for fine-tuning.",
    )
    group.add_argument("-ds", "--dataset", nargs="?", type=str, help="Dataset to train on.")
    group.add_argument(
        "-valds", "--val-dataset", nargs="?", type=str, help="Validation dataset. Defaults to same as training."
    )
    group.add_argument("-ep", "--epochs", nargs="?", type=int, help="Number of epochs to train.")
    group.add_argument(
        "-run",
        "--run-name",
        nargs="?",
        type=str,
        help="A name for the run. If not give, the model name is used instead.",
    )
    group.add_argument(
        "--defaults", nargs="?", choices=["DeiT", "DeiTIII"], default="DeiTIII", help="Default settings to use."
    )

    # Further model parameters
    group = parser.add_argument_group("Further model parameters")
    group.add_argument("--drop-path-rate", nargs="?", type=float, help="Drop path rate for ViT models.")
    group.add_argument("--layer-scale-init-values", nargs="?", type=float, help="LayerScale initial values.")
    group.add_argument("--layer-scale", action=argparse.BooleanOptionalAction, help="Use layer scale?")
    group.add_argument(
        "--qkv-bias",
        action=argparse.BooleanOptionalAction,
        help="Use bias in linear transformation to queries, keys, and values?",
    )
    group.add_argument("--pre-norm", action=argparse.BooleanOptionalAction, help="Use norm first architecture?")
    group.add_argument("--dropout", nargs="?", type=float, help="Model dropout.")
    group.add_argument("-heads", "--num-heads", nargs="?", type=int, help="Number of parallel attention heads.")
    group.add_argument("--input-dim", nargs="?", type=int, help="Dimensionality of text encoding.")
    group.add_argument("--max-seq-len", nargs="?", type=int, help="Maximum sequence length for text data.")
    group.add_argument(
        "--fused-attn",
        action=argparse.BooleanOptionalAction,
        help="Use fused attention (for ViT with Timm's attention only)?",
    )
    group.add_argument(
        "--perf-metric", nargs="?", choices=["acc", "mIoU"], help="Performance metric to use for evaluation."
    )
    # group.add_argument("-no_model_ema", action="store_true",
    #                    help="Don't use an exponential moving average for model parameters")
    # group.add_argument("-model_ema_decay", nargs='?', type=float, default=default_kwargs["model_ema_decay"],
    #                    help="Decay rate for exponential moving average of model parameters")

    # Experiment management
    group = parser.add_argument_group("Experiment management")
    group.add_argument("--seed", nargs="?", type=int, help="Manual RNG seed.")
    group.add_argument(
        "-exp",
        "--experiment-name",
        nargs="?",
        choices=_EXPNAMES,
        default="text_guidance",
        help="Name for the experiment. Is used for grouping of runs.",
    )
    group.add_argument(
        "--save-epochs", nargs="?", type=int, help="Number of epochs after which to save the full training state."
    )
    group.add_argument(
        "--keep-interm-states",
        nargs="?",
        type=int,
        help="Number of intermediate states to keep. All others (earlier ones) will be deleted automatically.",
    )
    group.add_argument(
        "--custom-dataset-path", nargs="?", type=str, help="Overwrite the path to any dataset to this path."
    )
    group.add_argument(
        "--results-folder",
        nargs="?",
        default=results_folder,
        type=str,
        help="Folder to put script results (mlflow data, models, etc.).",
    )
    group.add_argument(
        "--gather-stats-during-training",
        action=argparse.BooleanOptionalAction,
        help="Gather training statistics from all GPUs?",
    )
    group.add_argument("--tqdm", action=argparse.BooleanOptionalAction, help="Show tqdm for every epoch?")
    group.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, help="Debug mode: lots of intermediate prints."
    )
    group.add_argument("--wandb", action=argparse.BooleanOptionalAction, help="Use external logging via Wandb?")
    group.add_argument("--log-level", choices=["info", "debug"], help="Log level", metavar="LEVEL")

    # Speedup
    group = parser.add_argument_group("Speedup")
    group.add_argument("--amp", action=argparse.BooleanOptionalAction, help="Use automatic mixed precision?")
    group.add_argument(
        "--eval-amp", action=argparse.BooleanOptionalAction, help="Use automatic mixed precision during evaluation?"
    )
    group.add_argument("--compile-model", action=argparse.BooleanOptionalAction, help="Use torch.compile?")
    group.add_argument("--cuda", action=argparse.BooleanOptionalAction, help="Use cuda?")

    # Data loading
    group = parser.add_argument_group("Data loading")
    group.add_argument("-bs", "--batch-size", nargs="?", type=int, help="Batch size over all graphics cards (togeter).")
    group.add_argument("--num-workers", nargs="?", type=int, help="Number of dataloader worker threads. Should be >0.")
    group.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, help="Use pin_memory of torch Dataloader?"
    )
    group.add_argument(
        "--prefetch-factor",
        nargs="?",
        type=int,
        help="Prefetch factor for dataloader workers (how many batches to fetch)",
    )
    group.add_argument("--shuffle", action=argparse.BooleanOptionalAction, help="Shuffle the training data?")
    group.add_argument(
        "--weighted-sampler",
        action=argparse.BooleanOptionalAction,
        help="Use a class-weighted sampler to sample evenly from all classes (train and val)?",
    )

    # Optimizer
    group = parser.add_argument_group("Optimizer")
    group.add_argument("--opt", nargs="?", type=str, help="Optimizer to use.")
    group.add_argument("--weight-decay", nargs="?", type=float, help="Weight decay factor for use in optimizer.")
    group.add_argument("-lr", "--lr", nargs="?", type=float, help="Initial learning rate.")
    group.add_argument(
        "--max-grad-norm", nargs="?", type=float, help="Maximum norm for the gradients (used for cutoff)."
    )
    group.add_argument("--warmup-epochs", nargs="?", type=int, help="Number of epochs of linear warmup.")
    group.add_argument("--label-smoothing", nargs="?", type=float, help="Label smoothing factor.")
    group.add_argument("--loss", nargs="?", choices=["ce", "baikal"], type=str, help="Loss function to use.")
    group.add_argument(
        "--loss-weight", nargs="?", type=str, choices=["none", "linear", "log", "sqrt"], help="Per class loss weight."
    )
    group.add_argument("--sched", nargs="?", choices=["cosine", "const"], help="Learning rate schedule.")
    group.add_argument("--min-lr", nargs="?", type=float, help="Minimum learning rate to be hit by scheduler.")
    group.add_argument("--warmup-lr", nargs="?", type=float, help="Warmup learning rate.")
    group.add_argument("--warmup-sched", nargs="?", choices=["linear", "const"], help="Schedule for warmup")
    group.add_argument(
        "--opt-eps", nargs="?", type=float, help="Epsilon value added in the optimizer to stabilize training."
    )
    group.add_argument("--momentum", nargs="?", type=float, help="Optimizer momentum.")

    # Data augmentation
    group = parser.add_argument_group("Data augmentation")
    group.add_argument("--augment-strategy", nargs="?", type=str, help="Data augmentation strategy.")
    group.add_argument("--aug-rand-rot", nargs="?", type=int, help="Random rotation limit.")
    group.add_argument(
        "--aug-flip", action=argparse.BooleanOptionalAction, help="Use data augmentation: horizontal flip?"
    )
    group.add_argument(
        "--aug-crop",
        action=argparse.BooleanOptionalAction,
        help="Use data augmentation: cropping. This may break the skript?",
    )
    group.add_argument("--aug-resize", action=argparse.BooleanOptionalAction, help="Use data augmentation: resize?")
    group.add_argument(
        "--aug-grayscale", action=argparse.BooleanOptionalAction, help="Use data augmentation: grayscale?"
    )
    group.add_argument("--aug-solarize", action=argparse.BooleanOptionalAction, help="Use data augmentation: solarize?")
    group.add_argument(
        "--aug-gauss-blur", action=argparse.BooleanOptionalAction, help="Use data augmentation: gaussian blur?"
    )
    group.add_argument(
        "--aug-cutmix-alpha",
        type=float,
        help="Alpha value for using CutMix. CutMix is active when aug_cutmix_alpha > 0.",
    )
    group.add_argument(
        "--aug-mixup-alpha", type=float, help="Alpha value for using Mixup. Mixup is active when aug_mixup_alpha > 0."
    )
    group.add_argument(
        "--aug-color-jitter-factor",
        nargs="?",
        type=float,
        help="Factor to use for the data augmentation: color jitter.",
    )
    group.add_argument(
        "--aug-normalize", action=argparse.BooleanOptionalAction, help="Use data augmentation: Normalization?"
    )
    group.add_argument(
        "--aug-repeated-augment-repeats",
        type=int,
        help="Number of image repeats with repeat-augment from DeiT. 1 is not using repeat-augment.",
    )
    group.add_argument("--imsize", nargs="?", type=int, help="Image size given to the model -> imsize x imsize.")
    group.add_argument(
        "--augment-engine",
        nargs="?",
        choices=["torchvision", "albumentations"],
        help="Which data augmentation engine to use.",
    )

    # Text guidance
    parser.add_argument_group("Text guidance")
    group.add_argument("--encoding-model", type=str, help="Model used for the text encodings")
    group.add_argument("--text-captions", type=str, help="Image captions to be used")
    group.add_argument(
        "--normalize-embedding-mean", action=argparse.BooleanOptionalAction, help="Normalize text embeddings mean?"
    )
    group.add_argument(
        "--normalize-embedding-std", choices=["none", "mean", "full"], help="Normalize text embeddings std?"
    )
    group.add_argument("--text-loss-function", type=str, help="Which loss function to use to align the text loss")
    group.add_argument(
        "--text-loss-lambda", type=float, help="Strength of the text loss relative to the classification loss"
    )
    group.add_argument("--text-loss-schedule", type=str, help="Schedule for the text_loss_lambda")
    group.add_argument(
        "--adaptive-text-loss-weighting",
        action=argparse.BooleanOptionalAction,
        help="Use adaptive weighting strategy to counteract loss magnitude differences.",
    )
    group.add_argument("--freeze-text-head", action=argparse.BooleanOptionalAction, help="Freeze the text head")
    group.add_argument("--online-teacher", type=str, help="Do online knowledge distillation from teacher")

    return parser


def partition_choices():
    """Automatically create a list of all possible slurm partitions."""
    potential = list(set([l.split(" ")[0] for l in os.popen("sinfo")]))  # noqa: E741
    if len(potential) <= 2:
        # return ['A100-SDS', 'A100-40GB', 'A100-80GB']
        return slurm_defaults["partition"]
    return [p[:-1] if "*" in p else p for p in potential if p != "PARTITION"]  # + ["A100-SDS,A100-40GB"]


def slurm_parser(parser=None):
    """Add srun arguments to the given parser.

    Args:
        parser (argparse.ArgumentParser, optional): base parser to extend; default is parser from *base_parser*

    Returns:
        parser (argparse.ArgumentParser): extended parser

    """
    if parser is None:
        parser = base_parser()
    group = parser.add_argument_group("Slurm arguments")
    group.add_argument(
        "--partition",
        nargs="*",
        default=slurm_defaults["partition"],
        choices=partition_choices(),
        help="Slurm partition to use",
    )
    group.add_argument(
        "--container-image",
        nargs="?",
        default=slurm_defaults["container_image"],
        type=str,
        help="Path to slurm container image (.sqsh)",
    )
    group.add_argument(
        "--container-workdir",
        nargs="?",
        default=slurm_defaults["container_workdir"],
        type=str,
        help="Working directory in container",
    )
    group.add_argument(
        "--container-mounts",
        nargs="?",
        default=slurm_defaults["container_mounts"],
        type=str,
        help="All slurm mounts separated by ','.",
    )
    group.add_argument(
        "--job-name",
        nargs="?",
        default=slurm_defaults["job_name"],
        type=str,
        help="Slurm job name. Will default to '<model> <task> <dataset>'.",
    )
    group.add_argument(
        "--nodes", nargs="?", default=slurm_defaults["nodes"], type=int, help="Number of cluster nodes to use."
    )
    group.add_argument(
        "--ntasks", nargs="?", default=slurm_defaults["ntasks"], type=int, help="Number of GPUs to use for the job."
    )
    group.add_argument(
        "--cpus-per-task",
        nargs="?",
        default=slurm_defaults["cpus_per_task"],
        type=int,
        help="Number of CPUs per task/GPU.",
    )
    group.add_argument(
        "--mem-per-gpu",
        nargs="?",
        default=slurm_defaults["mem_per_gpu"],
        type=int,
        help="Ram per GPU (in Gb) to use. Will be given as total mem in srun command.",
    )
    group.add_argument(
        "--task-prolog",
        nargs="?",
        default=slurm_defaults["task_prolog"],
        type=str,
        help="Shell script for task prolog (installing packages, etc.).",
    )
    group.add_argument("--time", nargs="?", default=slurm_defaults["time"], type=str, help="Slurm time limit.")
    group.add_argument(
        "--export",
        nargs="?",
        default=slurm_defaults["export"],
        type=str,
        help="Additional environment variables to export.",
    )
    group.add_argument("--exclude", nargs="?", default=slurm_defaults["exclude"], type=str, help="Nodes to exclude.")
    group.add_argument(
        "--after-job", nargs="?", default=slurm_defaults["after_job"], type=int, help="Job ID to wait for."
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Run using srun instead of sbatch. This will print the output into the terminal, not the slurm output file."
            " The logfile will still be created as usual."
        ),
        default=False,
    )

    group = parser.add_argument_group("Run locally")
    group.add_argument("--local", action="store_true", help="Run locally; not in slurm", default=False)

    return parser


def parse_args(args=None, parser=None):
    """Parse args from *base_parser* and insert defaults.

    Args:
        args:  (Default value = None)
        parser:  (Default value = None)

    Returns:
        dict: parsed arguments

    """
    if args is None:
        parser = base_parser()
        args = parser.parse_args()
        args = dict(vars(args))

    check_arg_completeness(args, parser)

    return args


def check_arg_completeness(args, parser):
    """Check completeness of arguments.

    Args:
        args (dict): arguments to check
        parser (argparse.ArgumentParser): for raising the parser error
    Note:
        will raise a parser error if the arguments are not complete.
    """
    if args["task"] in ["pre-train", "fine-tune", "fine-tune-head"]:
        if "run_name" not in args or args["run_name"] is None or len(args["run_name"]) == 0:
            parser.error(f"-run_name is required for task {args['task']}")

        if "experiment_name" not in args or args["experiment_name"] is None or len(args["experiment_name"]) == 0:
            parser.error(f"-experiment_name is required for task {args['task']}. Choose from {_EXPNAMES}")

        if "epochs" not in args or args["epochs"] is None:
            parser.error(f"-epochs is required for task {args['task']}")

    if ("dataset" not in args or args["dataset"] is None) and args["task"] in [
        "pre-train",
        "fine-tune",
        "fine-tune-head",
        "eval-metrics",
    ]:
        parser.error(f"-dataset is required for task {args['task']}")

    if (
        ("val_dataset" not in args or args["val_dataset"] is None)
        and ("dataset" not in args or args["dataset"] is None)
        and args["task"] in ["eval"]
    ):
        parser.error(f"-dataset or -val_dataset is required for task {args['task']}")

    if args["aug_repeated_augment_repeats"] is not None and args["aug_repeated_augment_repeats"] < 1:
        parser.error(
            "number of repeats for repeated augment has to be >= 1, but got -aug_repeated_augment_repeats ="
            f" {args['aug_repeated_augment_repeats']}"
        )


def inside_slurm():
    """Test for being inside a slurm container.

    Works by testing for environment variable 'RANK'.
    """
    return "RANK" in os.environ


# TODO: fix ./runscript.tmp: 18: Syntax error: Unterminated quoted string
def create_runscript(args, file_name=None):
    """Create a run script for a distributed training job using SLURM.

    Args:
        args (dict): A dictionary containing various arguments for the job, including parameters for SLURM and for training.
        file_name (str, optional, optional): The name of the file to create. Defaults to "runscript.tmp".

    Returns:
        str: The name of the created file.
        str: Additional command line arguments for sbatch.

    Example:
        >>> args = {"model": "vit_large_patch16_384", "task": "pre-train", "batch_size": 256, ...}

        >>> file_name = "my_run_script.sh"

        >>> create_runscript(args, file_name)

    """
    for key, val in slurm_defaults.items():
        if key not in args and val is not None:
            args[key] = val

    if "run_name" not in args or args["run_name"] is None:
        model_str = args["model"]
        if model_str.endswith(".pt"):
            model_str = os.path.dirname(model_str)
        run_name = args["task"] + " " + model_str.split(os.sep)[-1].split("_")[0]
    else:
        run_name = args["run_name"]
    job_name = run_name.replace(" ", "_").replace("/", "_").replace(">", "_").replace("<", "_")
    if file_name is None:
        file_name = (
            f"experiments/sbatch/run_{args['task']}_{job_name}_at_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.sbatch"
        )

    task_args = ""
    # slurm_command = "echo run distributed:\necho python3 main.py {0}\n\nsrun -K \\\n"  # "  --gpus-per-task=1 \\\n  --gpu-bind=none \\\n"
    srun_command = "\nsrun -K \\\n"
    sbatch_commands = (  # outfile name is job name, date, job id, node name
        "#!/bin/bash\n\n#SBATCH"
        f" --output={slurm_output_folder}/%x-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-%j-%N.out\n"
    )
    sbatch_cmd_args = ""  # additional command line arguments for sbatch
    python_command = "  python3 main.py {0}\n"
    for key, val in args.items():
        if key == "local":
            continue
        if key == "interactive":
            continue
        if key in slurm_defaults:
            # it's a parameter for srun
            # slurm has - instead of _
            key = key.replace("_", "-")
            if key == "mem-per-gpu":
                # convert mem-per-gpu to mem
                # slurm_command += f"  --mem={val * args['ntasks'] // args['nodes']}G \\\n"  # that amount of memory is assigned on each node
                key = "mem"
                val = f"{val * args['ntasks'] // args['nodes']}G"  # that amount of memory is assigned on each node
                # continue
            if key == "job-name" and val is None:
                # # default jobname is '<task> <model> <dataset>'
                # model_str = args["model"]
                # task = args["task"]
                # if task == "pre-train":
                #     # it's just the model name...
                #     model = model_str.split("_")[0]
                # else:
                #     # it's a path to the tar file
                #     if not model_str.startswith(res_folder):
                #         model = "<vit model>"
                #     else:
                #         model = model_str[len(res_folder) :].split("_")[1].split(" ")[0]
                # if "dataset" in args and args["dataset"] is not None:
                #     dataset = args["dataset"]
                # else:
                #     dataset = ""
                val = run_name
            if key == "job-name" and not val.startswith('"'):
                val = f'"{val}"'
            if key in ["task-prolog", "nodes", "exclude", "after-job"] and val is None:
                continue
            if key == "task-prolog":
                srun_command += f'  --{key}="{val}" \\\n'
                continue
            if key == "after-job":
                sbatch_cmd_args += f"--dependency=afterany:{val} "
                continue
            if key == "partition" and isinstance(val, list):
                val = ",".join(val)
            if key == "ntasks":
                if args["nodes"] == 1:
                    # slurm_command += f"  --gpus={val} \\\n"
                    sbatch_commands += f"#SBATCH --gpus={val}\n"
                else:
                    assert (
                        val % args["nodes"] == 0
                    ), f"Number of tasks ({val}) must be a multiple of the number of nodes ({args['nodes']})."
                    # slurm_command += f"  --gpus-per-node={val // args['nodes']} \\\n"
                    sbatch_commands += f"#SBATCH --gpus-per-node={val // args['nodes']}\n"
                    sbatch_commands += "#SBATCH --ntasks-per-node=8\n"
            if "container" in key or key == "export":
                srun_command += f"  --{key}={val} \\\n"
            else:
                sbatch_commands += f"#SBATCH --{key}={val}\n"
            # slurm_command += f"  --{key}={val} \\\n"
        else:
            # it's a parameter for the training
            if val is None:
                continue
            if key in ["results_folder"] and val == globals()[key]:
                continue
            key = key.replace("_", "-")
            if isinstance(val, bool):
                if val:
                    task_args += f"--{key} "
                else:
                    task_args += f"--no-{key} "
                continue
            if isinstance(val, str):
                task_args += f'--{key} "{val}" '
            else:
                task_args += f"--{key} {val} "

    # slurm_command += "python3 main.py {0}\n"
    # os.umask(0)  # make it possible to create an executable file
    # with open(file_name, "w+", opener=lambda pth, flgs: os.open(pth, flgs, 0o777)) as f:
    #     f.write(slurm_command.format(task_args))
    with open(file_name, "w+") as f:
        f.write(sbatch_commands + srun_command + python_command.format(task_args))

    # delete all runscripts older than a month
    n_old_files = int(os.popen("find experiments/sbatch/ -type f -mtime +30 | wc -l").read())
    if n_old_files > 0:
        print(f"Deleting {n_old_files} old runscripts.")
        os.system("find experiments/sbatch/ -type f -mtime +30 -delete")

    return file_name, sbatch_cmd_args


if __name__ == "__main__":
    if not inside_slurm():
        # Make execution script and execute it
        parser = slurm_parser()
        args = vars(parser.parse_args())

        if not args["local"]:
            script_name, cmd_args = create_runscript(args)
            # os.system("./" + script_name)  # run srun to execute this script in slurm cluster
            # -> the following lines will be executed there
            if args["interactive"]:
                os.system(f"python3 srun-sbatch.py {script_name}")
            else:
                command = f"sbatch {cmd_args} {script_name}"
                # os.system(f"sbatch {cmd_args} {script_name}")  # sbatch to queue the job on the cluster
                sbatch_out = subprocess.check_output(command, shell=True, text=True)
                print(f"Running '{command}' >\n{sbatch_out}")
                if args["task"] in ["pre-train", "fine-tune", "continue"]:
                    try:
                        job_id = int(sbatch_out.strip().split("\n")[-1].split(" ")[-1])
                        slurm_args = {}
                        outfile = None
                        with open(script_name, "r") as f:
                            script = f.readlines()
                        for line in script:
                            line = line.strip()
                            if line.startswith("#SBATCH") or "--container" in line:
                                line = line.split("--")[-1]
                                key = line.split("=")[0]
                                val = "=".join(line.split("=")[1:])
                                if key in ["partition", "nodes", "ntasks", "cpus-per-task", "export", "exclue", "mem"]:
                                    slurm_args[key] = val
                                elif key == "output":
                                    outfile = val

                        if "mem" in slurm_args:
                            slurm_args["mem-per-gpu"] = int(int(slurm_args.pop("mem")[:-1]) / int(slurm_args["ntasks"]))

                        with open("autocontinue-jobs.jsonl", "a+") as f:
                            f.write(json.dumps({"jobid": job_id, "slurm": slurm_args, "logfile": outfile}) + "\n")
                        os.system("sed -i '/^$/d' autocontinue-jobs.jsonl")

                    except Exception as e:
                        print(f"Could not create autocontinue entry: {e}")
                        raise e
            exit(0)

        # local execution is wanted
        for key in list(args.keys()):
            if key.replace("_", "-") in slurm_defaults:
                args.pop(key)
        args = parse_args(args, parser)

    else:
        args = parse_args()

    args["branch"] = subprocess.check_output(["git", "branch", "--show-current"]).strip().decode("utf-8")
    args["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

    if args["task"] == "pre-train":
        from train import pretrain

        pretrain(**args)

    elif args["task"] == "fine-tune":
        from train import finetune

        finetune(**args)

    elif args["task"] == "fine-tune-head":
        from train import finetune

        finetune(**args, head_only=True)

    elif args["task"] == "parser-test":
        from utils import prep_kwargs, log_args
        from copy import copy

        kwargs = prep_kwargs(copy(args))
        log_args(kwargs)
        # keys = sorted(list(args.keys()))
        # fill_len = max(len(k) for k in keys)
        # for key in keys:
        #     print(f"{key + ' ' * (fill_len - len(key))} = {args[key]} -> {kwargs[key]}")

    elif args["task"] == "eval-metrics":
        from evaluate import evaluate_metrics

        evaluate_metrics(**args)

    elif args["task"] == "eval":
        from evaluate import evaluate

        evaluate(**args)

    elif args["task"] == "continue":
        from recover import continue_training

        continue_training(**args)

    elif args["task"] == "per-class-accuracy":
        from evaluate import per_class_accuracy

        per_class_accuracy(**args)

    elif args["task"] == "per-example-loss":
        from evaluate import evaluate_loss_per_example

        evaluate_loss_per_example(**args)

    elif args["task"] == "load-images":
        from test import load_images

        load_images(**args)

    elif args["task"] == "extract-embeddings":
        from evaluate import extract_embeddings

        extract_embeddings(**args)

    elif args["task"] == "fid-data":
        from evaluate import per_class_emdedding_distribution

        per_class_emdedding_distribution(**args)

    else:
        raise NotImplementedError(f"Task {args['task']} is not implemented.")
