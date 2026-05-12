# Preprocessing

This folder contains the scripts for building the caption embeddings used by TextTeacher.
The pipeline has three stages:

1. **Caption** — generate a text description for each training image
2. **Encode** — encode each caption with a frozen text encoder
3. **Normalize** — compute per-dimension mean and std across all encodings

If you want to skip this and use our precomputed embeddings for ImageNet, download them from HuggingFace:
**[TNauen/ImageNet-Caption-Encodings](https://huggingface.co/datasets/TNauen/ImageNet-Caption-Encodings)**

## Environment Setup

Before running any script, export the following environment variables:

```bash
export HF_TOKEN=<your_huggingface_token>   # required for gated models (PaliGemma, LLaVA, Dragonfly)
export HF_HOME=<path/to/hf_cache/>         # where Hugging Face downloads model weights
```

`HF_TOKEN` is a personal access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Step 1: Generate Captions

`make_captions.py` runs a captioning model over a dataset and writes one `key:\tcaption` entry per line to CSV files.

```bash
python3 make_captions.py \
  --dataset <dataset> \
  --model <model> \
  --dataset_root <path/to/datasets/> \
  --save_path <path/to/captions/> \
  [--workers <N> --id <i>]   # for parallel sharding: run N jobs with id 0..N-1
  [--continue]               # resume interrupted runs
```

**`--dataset`** choices: `imagenet`, `imagenet-val`, `imagenet21k`, `folder`, `cub2011`

**`--model`** choices:

| Key            | Model                                            |
| :------------- | :----------------------------------------------- |
| `CoCa`         | `coca_ViT-L-14` via OpenCLIP (MSCOCO fine-tuned) |
| `BLIP-L`       | `Salesforce/blip-image-captioning-large`         |
| `BLIP-B`       | `Salesforce/blip-image-captioning-base`          |
| `GIT-L`        | `microsoft/git-large`                            |
| `GIT-B`        | `microsoft/git-base`                             |
| `BLIP2`        | `Salesforce/blip2-opt-2.7b`                      |
| `PaliGemma`    | `google/paligemma-3b-mix-224`                    |
| `LLavaMistral` | `llava-hf/llava-v1.6-mistral-7b-hf`              |
| `Dragonfly`    | `Dragonfly`                                      |

The output is one or more CSV files named `<dataset>_<model>_<id>_of_<N>.csv` in `--save_path`.

For single-GPU interactive runs on SLURM:

```bash
./run_caption.sh --dataset imagenet --model CoCa --save_path <path/to/captions/> [...]
```

For large-scale parallel captioning, use the provided SLURM array job scripts:

- **`sbatch_caption_in`** — captions ImageNet-1k across 101 shards (10 concurrent), using the `image_captioning_v2` container:

  ```bash
  sbatch sbatch_caption_in --save_path <path/to/captions/>
  ```

- **`sbatch_caption`** — captions with CoCa across 4 shards, using the `image_captioning_coca` container (suitable for smaller datasets):

  ```bash
  sbatch sbatch_caption --dataset <dataset> --save_path <path/to/captions/>
  ```

Both scripts pass `$SLURM_ARRAY_TASK_ID` as `-id` and use `-c` to resume interrupted runs automatically.

## Step 2: Encode Captions

`encode_text.py` reads the CSV captions, encodes each one with a text encoder, and saves per-image `.emb.npy` files inside an uncompressed zip archive (`all_encodings.zip`).

```bash
python3 encode_text.py \
  --model <encoder> \
  --dataset <Dataset>-<CaptionModel> \
  --outfolder <path/to/encodings/ImageNet-CoCa/<encoder>/> \
  [--batch_size 64]
```

**`--model`** choices (text encoders):

| Key                                      | Model                                 |
| :--------------------------------------- | :------------------------------------ |
| `openai/clip-vit-base-patch16`           | CLIP ViT-B/16 text encoder            |
| `openai/clip-vit-large-patch14`          | CLIP ViT-L/14 text encoder            |
| `all-mpnet-base-v2`                      | MPNet (sentence-transformers)         |
| `all-MiniLM-L6-v2` / `all-MiniLM-L12-v2` | MiniLM (sentence-transformers)        |
| `all-roberta-large-v1`                   | RoBERTa-large (sentence-transformers) |
| `bert-large-cased` / `bert-base-cased`   | BERT (HuggingFace)                    |

**`--dataset`** is `<Dataset>-<CaptionModel>`, e.g. `ImageNet-CoCa`, `ImageNet-BLIP-L`, `ImageNet-val-CoCa`.
This tells the script which CSV files to read from `imagenet_text_ds.py`'s `_ROOT_CAPTIONS` path.

The output folder should be structured as:

```
<encodings_root>/
  ImageNet-CoCa/
    BERT-L/
      all_encodings.zip   ← one .emb.npy per image inside
```

To run on a SLURM cluster:

```bash
sbatch encode-text.sh --model bert-large-cased \
  --dataset ImageNet-CoCa \
  --outfolder <path/to/encodings/ImageNet-CoCa/BERT-L/>
```

## Step 3: Compute Normalizer Stats

`make_encoding_normalizer.py` computes the per-dimension mean and std across all encodings in a folder and saves them as `stats.npy`. These are used at training time to normalize the text targets to zero mean and unit std.

```bash
python3 make_encoding_normalizer.py --folder <path/to/encodings/ImageNet-CoCa/<encoder>/>
```

This reads `all_encodings.zip` (or individual `.emb.npy` files) and writes `stats.npy` into the same folder.

To run on a SLURM cluster:

```bash
./normalize-encodings.sh <path/to/encodings/ImageNet-CoCa/<encoder>/>
```

## Configuring Paths

Edit the top of `imagenet_text_ds.py` to point to your local paths:

```python
_ROOT_CAPTIONS  = "/path/to/captions/"    # folder containing the CSV files from Step 1
_ROOT_ENCODINGS = "/path/to/encodings/"   # root of the encodings tree (containing ImageNet-CoCa/ etc.)
```

The training code's path configuration (`paths_config.py` in the repo root) is separate and documented there.

## Expected Output Structure

After completing all three steps, your encodings folder should look like:

```
<encodings_root>/
  ImageNet-CoCa/
    BERT-L/
      all_encodings.zip
      stats.npy
```

This corresponds to the `ImageNet-CoCa` caption set with the BERT-L text encoder, which is the configuration used in the paper.
