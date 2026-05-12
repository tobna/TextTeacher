import os
import warnings

import torch
from transformers import AutoProcessor, AutoTokenizer

dragonfly_cache = None


def _get_dragonfly():
    from Dragonfly.src.dragonfly.models.modeling_dragonfly import DragonflyForCausalLM
    from Dragonfly.src.dragonfly.models.processing_dragonfly import DragonflyProcessor

    model_name = "togethercomputer/Llama-3-8B-Dragonfly-v1"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", token=os.environ.get("HF_TOKEN"))
    image_processor = clip_processor.image_processor
    processor = DragonflyProcessor(
        image_processor=image_processor, tokenizer=tokenizer, image_encoding_style="llava-hd"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DragonflyForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device, token=os.environ.get("HF_TOKEN")
        ).eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    processor.pad_token_id = tokenizer.eos_token_id
    return (model, tokenizer, processor)


def caption_dragonfly(imgs, prompt="Shortly describe the images content."):
    global dragonfly_cache
    if dragonfly_cache is None:
        dragonfly_cache = _get_dragonfly()
    model, tokenizer, processor = dragonfly_cache

    dragonfly_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = processor(
        text=[dragonfly_prompt for _ in imgs], images=imgs, max_length=2048, return_tensors="pt", is_generate=True
    )
    input_lens = [inputs["input_ids"][i].shape[-1] for i in range(len(imgs))]
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        gens = model.generate(
            **inputs, max_new_tokens=128, eos_token_id=tokenizer.encode("<|eot_id|>"), do_sample=False, use_cache=True
        )

    gens = [gen[input_len:] for input_len, gen in zip(input_lens, gens)]
    caps = processor.batch_decode(gens, skip_special_tokens=True)
    return caps
