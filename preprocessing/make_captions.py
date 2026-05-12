# transfomers compatibility


from pathlib import Path

import transformers
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

# Re-attach to top-level transformers namespace
transformers.BeamSearchScorer = BeamSearchScorer
transformers.LogitsProcessorList = LogitsProcessorList
transformers.TopPLogitsWarper = TopPLogitsWarper
transformers.TopKLogitsWarper = TopKLogitsWarper
transformers.RepetitionPenaltyLogitsProcessor = RepetitionPenaltyLogitsProcessor
transformers.MinLengthLogitsProcessor = MinLengthLogitsProcessor
transformers.MaxLengthCriteria = MaxLengthCriteria
transformers.StopStringCriteria = StopStringCriteria
transformers.EosTokenCriteria = EosTokenCriteria
transformers.StoppingCriteriaList = StoppingCriteriaList


print("START FILE")
import argparse
import math

import torch
from datadings.reader import MsgpackReader
from datadings.torch import Compose, CompressedToPIL, Dataset
from tqdm.auto import tqdm

from caption_CoCa import caption_CoCa
from caption_transformers import caption_transformers, model_name_dict
from data_utils import CUB2011, ImageFolderWithKey

try:
    from caption_dragonfly import caption_dragonfly
except ImportError as e:
    print("\033[93mCould not import \033[0m\033[91mDragonfly\033[0m")
    print(e)
    caption_dragonfly = None


def collate_single(data):
    return data[0]


def collate_multiple(datas):
    out_data = {"images": [data["image"] for data in datas], "keys": [data["key"] for data in datas]}
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Captioning")
    parser.add_argument("-d", "--dataset", type=str, required=True, nargs="?", help="Dataset to use")
    parser.add_argument(
        "-m",
        "--model",
        choices=["CoCa", "Dragonfly", *list(model_name_dict.keys())],
        required=True,
        nargs="?",
        help="Model to use",
    )
    parser.add_argument(
        "-r", "--dataset_root", type=str, default="/ds-sds/images/", nargs="?", help="Root directory of dataset"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=1, nargs="?", help="Number of processes for image captioning"
    )
    parser.add_argument("-id", type=int, default=0, nargs="?", help="Id of this process")
    parser.add_argument(
        "-save",
        "--save_path",
        type=str,
        default="/netscratch/nauen/ImageNet21k_captions",
        nargs="?",
        help="Path to save captions",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode (only process 10 images)")
    parser.add_argument(
        "--debug_prompt",
        type=str,
        nargs="?",
        default=None,
        help="Prompt for debugging (only for Dragonfly, PaliGemma and LLavaMistral)",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1, nargs="?", help="Batch size for image captioning")
    parser.add_argument("-c", "--continue", dest="_continue", action="store_true", help="Skip already processed images")

    args = parser.parse_args()
    args.save_path = Path(args.save_path)
    print(f"args: {args}")

    def caption_f(imgs, model, prompt=None):
        if model == "CoCa":
            caps = caption_CoCa(imgs)
        elif model == "Dragonfly":
            caps = caption_dragonfly(imgs, prompt=prompt)
        else:
            caps = caption_transformers(model, imgs, prompt=prompt)
        caps = [cap.replace("\n", " ").replace("  ", " ") for cap in caps]
        return caps

    print(f"load dataset: {args.dataset}")
    dataset = args.dataset.lower()
    if dataset == "imagenet21k":
        reader = MsgpackReader(f"{args.dataset_root}imagenet21k/train.msgpack")
        dataset = Dataset(reader, transforms={"image": Compose([CompressedToPIL()])})
    elif dataset == "imagenet-val":
        reader = MsgpackReader(f"{args.dataset_root}imagenet/msgpack/val.msgpack")
        dataset = Dataset(reader, transforms={"image": Compose([CompressedToPIL()])})
    elif dataset == "imagenet":
        reader = MsgpackReader(f"{args.dataset_root}imagenet/msgpack/train.msgpack")
        dataset = Dataset(reader, transforms={"image": Compose([CompressedToPIL()])})
    elif dataset == "folder":
        dataset = ImageFolderWithKey(args.dataset_root)
    elif dataset == "cub2011":
        dataset = CUB2011(args.dataset_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print("make subset")
    if args.workers > 1:
        partlen = math.ceil(len(dataset) // args.workers)
        start_id = args.id * partlen
        end_id = start_id + partlen
        if args.id == args.workers - 1:
            end_id = len(dataset)
        dataset = torch.utils.data.Subset(dataset, list(range(start_id, end_id)))

    print("make dataloader")
    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=10,
        collate_fn=collate_single if args.batch_size == 1 else collate_multiple,
    )

    if args.workers == 1 and not args.debug:
        dataset = tqdm(dataset)

    total_images = len(dataset)

    already_processed = set()
    savefile = f"{args.dataset}_{args.model}_{args.id}_of_{args.workers}.csv"
    savefile = args.save_path / savefile

    args.save_path.mkdir(exist_ok=True)
    if args._continue and savefile.is_file():
        print("load continued files")
        with open(savefile, "r") as infile:
            for line in infile:
                already_processed.add(line.split(":\t")[0])
        print(f"Skipping {len(already_processed)} already processed images")
    else:
        print(f"creating save file: {savefile}")
        with open(savefile, "w") as f:
            pass

    print(f"saving images to {savefile}")
    for i, data in enumerate(dataset):
        if args.batch_size == 1:
            keys = [data["key"]]
            imgs = [data["image"]]
        else:
            keys = data["keys"]
            imgs = data["images"]

        if args._continue and not args.debug:
            k_ims = [(key, img) for key, img in zip(keys, imgs) if key not in already_processed]
            if len(k_ims) == 0:
                continue
            keys, imgs = zip(*k_ims)

        if args.debug and args.model in ["PaliGemma", "LLavaMistral", "Dragonfly"]:
            caps = caption_f(imgs, args.model, args.debug_prompt)
        else:
            caps = caption_f(imgs, args.model)
        for key, cap in zip(keys, caps):
            with open(savefile, "a") as outfile:
                outfile.write(f"{key}:\t{cap}\n")
            if args.debug:
                print(f"{key}:\t{cap}")
        if args.workers > 0 and (i + 1) % 100 == 0:
            print(f"{args.id}/{args.workers}\tProcessed {(i+1)*args.batch_size}/{total_images*args.batch_size} images")
        if (i + 1) * args.batch_size >= 10 and args.debug:
            break
    print(f"{args.id}/{args.workers}\tDone processing {total_images*args.batch_size} images")
