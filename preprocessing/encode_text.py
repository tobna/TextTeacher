import argparse
import os
import zipfile
from zipfile import ZipFile

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPTokenizerFast
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

from imagenet_text_ds import ImageText


class CLIPEmbedder:
    def __init__(self, clip_model, device="cpu") -> None:
        self.clip_model = clip_model
        self.model = CLIPModel.from_pretrained(clip_model).to(device)
        self.processor = CLIPTokenizerFast.from_pretrained(clip_model)
        self.device = device

    def preprocess(self, text):
        if isinstance(text, str):
            text = [text]
        # text = [t[:92] for t in text]
        # tokens = [self.processor.tokenize(t) for t in text]
        # tokens = [
        #     t[:75] if len(t) > 75 else t for t in tokens
        # ]  # truncate to 75 tokens + <|start|> + <|end|> makes 77 tokens max seqence length for CLIP
        # text = [" ".join([tkn[:-4] for tkn in t_seq]) for t_seq in tokens]
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        inp_ids = inputs["input_ids"]
        # print(inp_ids)
        attn_mask = inputs["attention_mask"]
        # print(attn_mask)
        # print(attn_mask.shape)
        if inp_ids.shape[-1] > 77:
            # truncate input tokens to 77 (max)
            inp_ids = inp_ids[:, :77]
            attn_mask = attn_mask[:, :77]
            inp_ids[:, -1] = 49407
        inp_shape = inp_ids.size()
        embeddings = self.model.text_model.embeddings(input_ids=inp_ids)

        causal_mask = _create_4d_causal_attention_mask(inp_shape, embeddings.dtype, device=embeddings.device)
        attn_mask = _prepare_4d_attention_mask(attn_mask, embeddings.dtype)

        return dict(inputs_embeds=embeddings, attention_mask=attn_mask, causal_attention_mask=causal_mask)

    def __call__(self, text):
        model_input = self.preprocess(text)

        embeddings = self.model.text_model.encoder(**model_input).last_hidden_state
        embeddings = self.model.text_model.final_layer_norm(embeddings)
        eos_idx = model_input["attention_mask"].argmax(dim=-1)
        return embeddings[0, -1] if embeddings.shape[0] == 1 else embeddings[:, -1]  # get <eos> token(s) of the batch

    def __str__(self) -> str:
        return (
            f"TextEmbedder(pretrained_weights={self.clip_model},"
            f" embedding_dim={self.embedding_dim}\n\t{str(self.model.text_model).replace('\n', '\n\t')})"
        )

    @property
    def embedding_dim(self):
        return self.model.text_model.config.projection_dim

    @property
    def max_length(self):
        return self.model.config.text_config.max_position_embeddings


class BERTEmbedder:
    def __init__(self, model, device="cpu"):
        model = model.split("/")[-1]
        self.bert_model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertModel.from_pretrained(model).to(device)
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token

    @property
    def embedding_dim(self):
        return self.model.config.hidden_size

    def preprocess(self, text):
        if isinstance(text, str):
            text = [text]

        return self.tokenizer(text, return_tensors="pt", padding=True)

    def __call__(self, text):
        embeddings = self.preprocess(text).to(device).to(self.device)
        # print(self.tokenizer.convert_ids_to_tokens(embeddings["input_ids"][0]))
        with torch.no_grad():
            embeddings = self.model(**embeddings).last_hidden_state
        return embeddings[:, 0]  # get embedding of [CLS] token


class SentenceEmbedder:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, trust_remote_code=True).to(device)

    @property
    def embedding_dim(self):
        return self.model.get_sentence_embedding_dimension()

    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("encode text for dataset")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=[
            "all-mpnet-base-v2",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
            "intfloat/multilingual-e5-large-instruct",
            "bert-large-cased",
            "bert-base-cased",
            "all-MiniLM-L12-v2",
            "all-MiniLM-L6-v2",
            "all-roberta-large-v1",
            "google-bert/bert-large-uncased",
            "google-bert/bert-base-uncased",
            "Qwen/Qwen3-Embedding-8B",
            "nvidia/llama-embed-nemotron-8b",
        ],
        default="openai/clip-vit-base-patch32",
        help="Text encoder model from Huggingface transformers.",
    )
    parser.add_argument(
        "--dataset",
        "-ds",
        choices=[
            f"{ds}-{cap_mod}"
            for ds in ["ImageNet", "ImageNet-val", "ImageNet21k", "CUB2011"]
            for cap_mod in [
                "Dragonfly",
                "CoCa",
                "Lbl&CoCa",
                "BLIP-L",
                "PaliGemma",
                "Labels",
                "CoCa-Qwen3-32B-tags",
                "Dragonfly-Qwen3-32B-tags",
            ]
        ],
        required=True,
        help="Dataset to encode.",
    )
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="Encoding batch size.")
    parser.add_argument("--outfolder", "-o", type=str, required=True, help="Folder to save embeddings in.")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if "clip" in args.model:
        model = CLIPEmbedder(args.model, device=device)
    elif args.model.startswith("google-bert"):
        model = BERTEmbedder(args.model, device=device)
    else:
        print(f"Loading model {args.model} from sentence-transformers")
        model = SentenceEmbedder(args.model, device=device)

    print(f"using {args.model} with embedding dimension {model.embedding_dim}.")

    if args.dataset.lower().split("-")[1] == "val":
        dataset = "-".join(args.dataset.lower().split("-")[:2])
        cap_model = "-".join(args.dataset.split("-")[2:])
    else:
        dataset = "-".join(args.dataset.lower().split("-")[:1])
        cap_model = "-".join(args.dataset.split("-")[1:])
    ds = ImageText(caption_model=cap_model, root_dataset=dataset)

    print(f"taking captions of {cap_model} at {dataset}")

    os.makedirs(args.outfolder, exist_ok=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=40)
    with ZipFile(os.path.join(args.outfolder, "all_encodings.zip"), "w", compression=zipfile.ZIP_STORED) as zf:
        for ids, texts in tqdm(loader):
            with torch.no_grad():
                batch_embeddings = model(texts)

            for id, embedding in zip(ids, batch_embeddings):
                with zf.open(f"{id}.emb.npy", "w") as zf_file:
                    np.save(zf_file, embedding.numpy(force=True))
