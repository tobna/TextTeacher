import torch
from timm.models import register_model
from timm.models.resnet import BasicBlock, Bottleneck
from timm.models.resnet import ResNet as ResNetTimm
from torch import nn
from torch.nn import functional as F

from resizing_interface import ResizingInterface


class ResNet(ResNetTimm, ResizingInterface):
    """The popular ResNet model with a ResizingInterface."""

    def __init__(self, *args, text_embed_dim=1024, global_pool="avg", **kwargs):
        """Create ResNet model with resizing capabilities.

        Args:
            *args: Arguments for the ResNet model.
            global_pool (str, optional): _description_. Defaults to "avg".
            **kwargs: Keyword arguments for the ResNet model (from Timm).

        Keyword Args:
            block, layers, num_classes, in_chans, output_stride, cardinality, base_width, stem_width, stem_type, replace_stem_pool, block_reduce_first, down_kernel_size, avg_down, act_layer, norm_layer, aa_layer, drop_rate, drop_path_rate, drop_block_rate, zero_init_last, block_args

        """
        admissible_kwargs = [
            "block",
            "layers",
            "num_classes",
            "in_chans",
            "output_stride",
            "cardinality",
            "base_width",
            "stem_width",
            "stem_type",
            "replace_stem_pool",
            "block_reduce_first",
            "down_kernel_size",
            "avg_down",
            "act_layer",
            "norm_layer",
            "aa_layer",
            "drop_rate",
            "drop_path_rate",
            "drop_block_rate",
            "zero_init_last",
            "block_args",
        ]
        for key in list(kwargs.keys()):
            if key not in admissible_kwargs:
                kwargs.pop(key)
        super().__init__(*args, global_pool=global_pool, **kwargs)
        self.global_pool_str = global_pool
        self.text_head = nn.Linear(self.fc.in_features, text_embed_dim) if text_embed_dim > 0 else nn.Identity()
        self.text_embed_dim = text_embed_dim

    def remove_text_head(self):
        if self.text_embed_dim > 0:
            self.text_head = nn.Identity()
            self.text_embed_dim = -1

    def set_image_res(self, res):
        # resizing not needed for CNNs with pooling
        return

    def set_num_classes(self, n_classes):
        if self.num_classes == n_classes:
            return
        if n_classes > 0:
            self.reset_classifier(num_classes=n_classes, global_pool=self.global_pool_str)
        else:
            self.fc = nn.Identity()

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor, return_all_vals: bool = False):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if not return_all_vals:
            return self.forward_head(x)
        return x, self.forward_head(x), self.text_head(x)


@register_model
def resnet18(pretrained=False, **kwargs):
    """Construct a ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return ResNet(**model_args, **kwargs)


@register_model
def resnet34(pretrained=False, **kwargs):
    """Construct a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return ResNet(**model_args, **kwargs)


@register_model
def resnet26(pretrained=False, **kwargs):
    """Construct a ResNet-26 model."""
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return ResNet(**model_args, **kwargs)


@register_model
def resnet50(pretrained=False, **kwargs):
    """Construct a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    return ResNet(**model_args, **kwargs)


@register_model
def resnet101(pretrained=False, **kwargs):
    """Construct a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return ResNet(**model_args, **kwargs)


@register_model
def resnet152(pretrained=False, **kwargs):
    """Construct a ResNet-152 model."""
    return ResNet(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)


@register_model
def wide_resnet50_2(pretrained=False, **kwargs):
    """Construct a Wide ResNet-50-2 model.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128)
    return ResNet(**model_args, **kwargs)
