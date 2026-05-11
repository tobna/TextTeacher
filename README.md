[![Project Page](https://img.shields.io/badge/Project%20Page-darkred)](https://tobias.nauen-it.de/publications/text-teacher/)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Precomputed%20Embeddings-yellow?logo=huggingface)](https://huggingface.co/datasets/TNauen/ImageNet-Caption-Encodings)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

# TextTeacher: What Can Language Teach About Images?

> **Tobias Christian Nauen, Stanislav Frolov, Brian B. Moser, Federico Raue, Ahmed Anwar, Andreas Dengel**

TextTeacher is a simple auxiliary training objective that injects semantic knowledge from a frozen text encoder into image classification training — and then discards it entirely at inference.
The result is a plain, fast vision model with no added latency, no multimodal components, and no text dependency at deployment.

**On ImageNet with standard ViT backbones**, TextTeacher improves accuracy by up to **+2.7 p.p.**, yields consistent transfer gains (**+1.0 p.p.** on average across six benchmarks), and boosts robustness under 50% label noise (**+8.4 p.p.**) — all at negligible compute overhead.
It outperforms online vision knowledge distillation from DINOv2-L, matching its accuracy at **~66% of the wall-clock time**.

## News

- [11.03.2026] We release training code and a [project page](https://nauen-it.de/publications/text-teacher) for TextTeacher 🌐

## Requirements

This project builds on [timm](https://github.com/huggingface/pytorch-image-models) and [PyTorch](https://pytorch.org/).
Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Precomputed Caption Embeddings

To avoid re-encoding captions at every run, we provide precomputed text embeddings for ImageNet on HuggingFace:
**[TNauen/ImageNet-Caption-Encodings](https://huggingface.co/datasets/TNauen/ImageNet-Caption-Encodings)**

Download and point `paths_config.py` at the embedding files.

## Usage

After **cloning this repository**, you can train models with the TextTeacher objective.
By default, jobs are submitted to a SLURM cluster via `sbatch`.
Add `--interactive` to use `srun`, or `--local` to run directly on your machine.

### General Preparation

```bash
chmod a+x main.py
```

Adjust `paths_config.py` for your system — set `results_folder`, `slurm_output_folder`, and per-dataset paths in `_ds_paths`.

To enable Weights & Biases tracking, create `.wandb.apikey` with your W&B API key and pass `--wandb` when running.

### Dataset Preparation

Dataset paths are configured in `paths_config.py`.

Caption embeddings are loaded from the path configured under `text_encodings` in `paths_config.py`.

### Training with TextTeacher

```bash
./main.py -t pre-train -m <model_name> -ep <epochs> -ds <dataset> \
  -run "<run description>" -exp <experiment_name> \
  --text-loss-lambda 0.5 --text-captions ImageNet-CoCa
```

Key TextTeacher arguments:

| Argument                         | Description                                                                                               |
| :------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `--text-loss-lambda`             | Weight of the text auxiliary loss (e.g. `0.5`). Set to `0.0` to train without TextTeacher.                |
| `--text-captions`                | Caption embedding set to use (e.g. `ImageNet-CoCa`).                                                      |
| `--text-loss-function`           | Alignment loss: `clip` (default), `infonce`, `cos`, `l2`, `normclip`, and more.                           |
| `--text-loss-schedule`           | Weight schedule over training: `const` (default), `lin`, `cos`, `fullcos`, or `jump-<start>-<end>-<rel>`. |
| `--adaptive-text-loss-weighting` | Normalise loss magnitudes automatically (enabled by default).                                             |
| `--freeze-text-head`             | Freeze the text projection head during training.                                                          |
| `--online-teacher`               | Use an online vision teacher instead of text (e.g. `DINO-v2L`, `CLIP-L`, `CoCa-L`).                       |

### Pretraining (baseline, no TextTeacher)

```bash
./main.py -t pre-train -m <model_name> -ep <epochs> -ds <dataset> \
  -run "<run description>" -exp <experiment_name>
```

### Finetuning

```bash
./main.py -t fine-tune -m <checkpoint.tar> -ep <epochs> -ds <dataset> \
  -run "<run description>" -exp <experiment_name> -lr <lr>
```

### Evaluation

```bash
# Accuracy
./main.py -t eval -m <checkpoint.tar> -ds <dataset>

# Efficiency metrics (FLOPs, memory, throughput)
./main.py -t eval-metrics -m <checkpoint.tar> -ds <dataset>
```

### Continuing Interrupted Runs

```bash
./main.py -t continue -m <checkpoint.tar>
```

### Further Arguments

| Arg                 | Short  | Description                                                                                                                                           |
| :------------------ | :----- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--task`            | `-t`   | Task: `pre-train`, `fine-tune`, `fine-tune-head`, `eval`, `eval-metrics`, `continue`, `extract-embeddings` (image embeddings for analysis), and more. |
| `--model`           | `-m`   | Model name or path to a `.tar` checkpoint.                                                                                                            |
| `--dataset`         | `-ds`  | Dataset to use.                                                                                                                                       |
| `--epochs`          | `-ep`  | Number of training epochs.                                                                                                                            |
| `--run-name`        | `-run` | Name or description of this run.                                                                                                                      |
| `--experiment-name` | `-exp` | Experiment group name.                                                                                                                                |
| `--lr`              |        | Learning rate.                                                                                                                                        |
| `--batch-size`      | `-bs`  | Total batch size across all GPUs.                                                                                                                     |
| `--imsize`          |        | Image resolution.                                                                                                                                     |
| `--wandb`           |        | Enable W&B logging.                                                                                                                                   |
| `--local`           |        | Run locally instead of submitting to SLURM.                                                                                                           |
| `--interactive`     |        | Submit via `srun` (prints to terminal).                                                                                                               |

```bash
./main.py --help
```

## Architecture Overview

```
main.py               CLI entry point, task dispatcher, SLURM submission
train.py              Training pipelines (pre-train, fine-tune) with TextTeacher integration
evaluate.py           Accuracy, efficiency evaluation, and image embedding extraction for analysis
engine.py             Core training/eval loop, optimizer/scheduler, DDP, W&B logging
models.py             Model instantiation and checkpoint loading
text_loss.py          Text alignment losses and weight schedules
infonce_loss.py       InfoNCE contrastive loss implementation
online_teacher.py     Online vision teachers (CLIP, DINOv2, CoCa) for comparison
architectures/        model implementations (ViT, DeiT, Swin, etc.)
data/                 Dataset loaders, including imagenet_text_encodings.py
config.py             Default hyperparameters (~50 settings including TextTeacher defaults)
paths_config.py       Path configuration for datasets, embeddings, and output folders
metrics.py            FLOPs, MACs, memory, and throughput calculations
```

## Relation to WTF Benchmark

TextTeacher builds on the training infrastructure from our earlier [WTF Benchmark](https://github.com/tobna/WhatTransformerToFavor) (WACV 2025), which established standardized baselines for efficient vision transformers.
The benchmark codebase has been extended with the TextTeacher objective, text loss infrastructure, and caption embedding support.

## License

We release this code under the [MIT license](./LICENSE).

## Citation

If you use this codebase or results, please cite:

```bibtex
@article{Nauen2026TextTeacher,
  author = {Nauen, Tobias Christian and Frolov, Stanislav and Moser, Brian B.
            and Raue, Federico and Anwar, Ahmed and Dengel, Andreas},
  title  = {TextTeacher: What Can Language Teach About Images?},
  year   = {2026},
}
```

If you additionally use the benchmark infrastructure, please also cite:

```bibtex
@inproceedings{Nauen2024WTFBenchmark,
  author    = {Nauen, Tobias Christian and Palacio, Sebastian and Raue, Federico and Dengel, Andreas},
  title     = {Which Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
  month     = {February},
  year      = {2025},
  pages     = {6955-6966}
}
```
