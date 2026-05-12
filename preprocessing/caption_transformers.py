import contextlib
import os
import sys

import torch

pipe_cache = {}


class DummyFile:
    def write(self, x):
        pass

    def flush(self):
        pass


@contextlib.contextmanager
def no_output():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


def _get_pipeline(model_name):
    print(f"Loading model {model_name} from HuggingFace...")
    from transformers import pipeline

    pipe = pipeline("image-to-text", model=model_name, device=0 if torch.cuda.is_available() else -1, max_new_tokens=40)

    print("Model loaded.")
    return pipe


def _get_model_and_processor(model_name):
    print(f"Loading model and processor {model_name} from HuggingFace...")
    if "paligemma" in model_name:
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        model_loader = PaliGemmaForConditionalGeneration
        processor_loader = AutoProcessor
        model_kwargs = dict(torch_dtype=torch.bfloat16, revision="bfloat16")
    elif "llava" in model_name:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        model_loader = LlavaNextForConditionalGeneration
        processor_loader = LlavaNextProcessor
        model_kwargs = dict(torch_dtype=torch.float32)

    model = model_loader.from_pretrained(
        model_name,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        **model_kwargs,
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    processor = processor_loader.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.pad_token_id = processor.tokenizer.eos_token_id
    processor.pad_token_id = processor.tokenizer.eos_token_id

    print("Model loaded.")
    return (model, processor)


model_name_dict = {
    "GIT-L": "microsoft/git-large",
    "GIT-B": "microsoft/git-base",
    "BLIP-L": "Salesforce/blip-image-captioning-large",
    "BLIP-B": "Salesforce/blip-image-captioning-base",
    "BLIP2": "Salesforce/blip2-opt-2.7b",
    "PaliGemma": "google/paligemma-3b-mix-224",
    "LLavaMistral": "llava-hf/llava-v1.6-mistral-7b-hf",
}


model_prompt_dict = {
    "PaliGemma": "caption and describe in detail",
    "LLavaMistral": "[INST] <image>\nDescribe the images content in detail [/INST]",
}


def caption_transformers(model, imgs, prompt=None):
    if model in ["PaliGemma", "LLavaMistral"]:
        if prompt is None:
            prompt = model_prompt_dict[model]
        return _caption_model_processor(model, imgs, prompt=prompt)
    return _caption_pipeline(model, imgs)


def _caption_pipeline(model, imgs):
    global pipe_cache
    if model not in pipe_cache:
        pipe_cache[model] = _get_pipeline(model_name_dict[model])

    # generate caption
    with torch.no_grad() and no_output():
        caps = pipe_cache[model](imgs)
    caps = [cap[0]["generated_text"] if isinstance(cap, list) else cap["generated_text"] for cap in caps]
    return caps


def _caption_model_processor(model, imgs, prompt):
    global pipe_cache
    model_name = model_name_dict[model]
    if model_name not in pipe_cache:
        pipe_cache[model_name] = _get_model_and_processor(model_name)

    model, processor = pipe_cache[model_name]
    inputs = processor(text=[prompt for img in imgs], images=imgs, return_tensors="pt").to(model.device)
    input_lens = [inputs["input_ids"][i].shape[-1] for i in range(len(imgs))]

    with torch.inference_mode():
        gens = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    gens = [gen[input_len:] for input_len, gen in zip(input_lens, gens)]
    caps = [processor.decode(gen, skip_special_tokens=True) for gen in gens]

    return caps
