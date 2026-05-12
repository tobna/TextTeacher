import open_clip
import torch


coca_setup = None


def prepare_CoCa(
    model_name="coca_ViT-L-14",
):
    print(f"Loading model {model_name} from OpenCLIP...")
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained="mscoco_finetuned_laion2B-s13B-b90k",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("Model loaded.")
    return dict(model=model, transform=transform)


def caption_CoCa(img):
    global coca_setup
    if coca_setup is None:
        coca_setup = prepare_CoCa()

    if isinstance(img, (list, tuple)):
        return [caption_CoCa(im) for im in img]

    im = coca_setup["transform"](img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = coca_setup["model"].generate(im)

    cap = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
    return cap
