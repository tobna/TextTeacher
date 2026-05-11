from transformers import AutoModel, AutoModelForZeroShotImageClassification
from loguru import logger

try:
    import open_clip

    OPENCLIP_AVAILABLE = True
except Exception as e:
    logger.error(f"Could not import open_clip: {e}")
    OPENCLIP_AVAILABLE = False


def make_online_teacher(teacher, device):
    if teacher is None:
        return None

    if "-" not in teacher:
        logger.error(f"Teacher format should be Family-Variant, but got {teacher}")
        raise ValueError(f"Teacher format should be Family-Variant, but got {teacher}")

    logger.info(f"Using online teacher: {teacher}")

    fam, var = teacher.split("-")
    if fam.lower() == "clip":
        return CLIPEncoder(var, device)
    if fam.lower() == "dino":
        return DINOEncoder(var, device)
    if fam.lower() == "coca":
        return CoCaEncoder(var, device)
    logger.error("Current teacher families are: CLIP, DINO, CoCa")
    raise ValueError("Current teacher families are: CLIP, DINO, CoCa")


class CoCaEncoder:
    models = {"L": ("coca_ViT-L-14", 768)}

    def __init__(self, variant, device):
        assert OPENCLIP_AVAILABLE
        self.hf_model, self.embedding_dim = self.models[variant]
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            model_name=self.hf_model,
            pretrained="mscoco_finetuned_laion2B-s13B-b90k",
            device=device,
        )

    def __call__(self, images):
        return self.model.encode_image(images)


class CLIPEncoder:
    models = {"B": ("openai/clip-vit-base-patch16", 768), "L": ("openai/clip-vit-large-patch14", 768)}

    def __init__(self, variant, device):
        self.hf_model, self.embedding_dim = self.models[variant]
        self.device = device
        self.clip_encoder = AutoModelForZeroShotImageClassification.from_pretrained(self.hf_model)
        self.clip_encoder = self.clip_encoder.to(device)

    def __call__(self, images):
        embeddings = self.clip_encoder.vision_model(images.to(self.device)).pooler_output
        return self.clip_encoder.visual_projection(embeddings)


class DINOEncoder:
    models = {
        "B": ("facebook/dino-vitb8", 768),
        "v2B": ("facebook/dinov2-base", 768),
        "v2L": ("facebook/dinov2-large", 1024),
        "v3B": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
        "v3L": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
    }

    def __init__(self, variant, device):
        self.hf_model, self.embedding_dim = self.models[variant]
        self.device = device
        self.encoder = AutoModel.from_pretrained(self.hf_model)
        self.encoder = self.encoder.to(device)

    def __call__(self, images):
        return self.encoder(images.to(self.device)).pooler_output
