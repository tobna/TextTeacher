from functools import partial

import torch
from loguru import logger
from timm.models import register_model
from timm.models.vision_transformer import (
    Attention,
    Block,
    PatchEmbed,
    VisionTransformer,
)
from torch import nn

from resizing_interface import ResizingInterface, vit_sizes


class _MatrixSaveAttn(Attention):
    attn_mat = None

    @classmethod
    def cast(cls, attn: Attention):
        assert isinstance(attn, Attention), "Can only save attention from Timms attention class"
        attn.__class__ = cls
        assert isinstance(attn, _MatrixSaveAttn)
        return attn

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_mat = attn.detach()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


def _index_picker(tensor, idx=-1):
    """Pick a specific index from a tensor.

    tensor: B x N x D -> B x D, by picking idx from N.

    Args:
        tensor (toch.tensor): tensor to pick from.
        idx (int, optional): index to pick. Defaults to -1.

    Returns:
        torch.tensor: index from tensor

    """
    return tensor[..., idx, :]  # B x N x D -> B x D


class TimmViT(VisionTransformer, ResizingInterface):
    """Wrapper for *VisionTransformer* from *timm* library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        text_embed_dim=1024,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        init_values=None,
        class_token=True,
        no_embed_class=True,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        reg_tokens=0,
        save_attention_maps=False,
        fused_attn=True,
        freeze_text_head=False,
        **kwargs,
    ):
        """Create a Vision Transformer model.

        Parameters
        ----------
        img_size : int
            image dimensions -> img_size x img_size
        patch_size : int
            patch_size
        in_chans : int
            number of image channels
        num_classes : int
            number of classes for classification head
        global_pool : str,int
            type of global pooling for final sequence (default: 'token'), or index for token to be taken
        embed_dim : int
            embedding dimension
        depth : int
            number of transformer layers
        num_heads : int
            number of transformer heads
        mlp_ratio : float
            ratio of feed forward (mlp) hidden dimension to embedding dimension
        qkv_bias : bool
            enable bias for query, key, and value (qkv) embeddings
        qk_norm : bool
            normalize query and key embeddings
        init_values : float
            layer scale initial values
        class_token : bool
            use a class token [CLS]
        no_embed_class : bool
            no positional embedding for the class token
        pre_norm : bool
            use pre-norm architecture (norm before the blocks, not after)
        fc_norm : bool
            norm after pool (used when global_pool == 'avg')
        drop_rate : float
            dropout rate
        attn_drop_rate : float
            dropout rate in the attention module
        drop_path_rate : float
            drop path rate (stochastic depth)
        weight_init : str
            scheme for weight initialization
        embed_layer : nn.Module
            patch embedding layer
        norm_layer : nn.Module
            normalization layer
        act_layer : nn.Module
            activation function
        block_fn : nn.Module
            which block structure to use; for parallel attention layers, ...
        save_attention_maps : bool
            save attention maps for each block
        fused_attn : bool
            use fused attention
        kwargs : dict
            additional arguments (will be ignored)

        """
        init_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool if isinstance(global_pool, str) else "avg",
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            reg_tokens=reg_tokens,
            block_fn=block_fn,
        )

        self.new_version = False  # TODO: check based on the timm version
        self.global_pool = global_pool
        if isinstance(global_pool, int):
            self.attn_pool = partial(_index_picker, idx=global_pool)

        if self.new_version:
            init_kwargs["qk_norm"] = qk_norm
            init_kwargs["proj_drop_rate"] = drop_rate
        super(TimmViT, self).__init__(**init_kwargs)
        self.embed_layer = embed_layer
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.pre_norm = pre_norm
        self.class_token = class_token
        self.no_embed_class = no_embed_class
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.depth = depth
        self.save_attention_maps = save_attention_maps
        self.text_head = nn.Linear(self.embed_dim, text_embed_dim) if text_embed_dim > 0 else nn.Identity()
        self.text_embed_dim = text_embed_dim

        if save_attention_maps:
            self.do_save_attention_maps()
        try:
            for block in self.blocks:
                block.attn.fused_attn = fused_attn and self.blocks[0].attn.fused_attn
            use_fused = self.blocks[0].attn.fused_attn
            logger.info(f"Use fused attention: {use_fused}")
        except:  # I'm lazy for now  # noqa: E722
            pass

        if freeze_text_head and text_embed_dim > 0:
            self.text_head.weight.requires_grad = False
            # self.text_head.bias = 0
            self.text_head.bias.requires_grad = False

    def remove_text_head(self):
        self.text_head = nn.Identity()
        self.text_embed_dim = -1

    def do_save_attention_maps(self):
        self.save_attention_maps = True
        for block in self.blocks:
            block.attn = _MatrixSaveAttn.cast(block.attn)

    def attention_maps(self):
        assert self.save_attention_maps, "Have to save attention maps first"
        return [getattr(block.attn, "attn_mat", None) for block in self.blocks]

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        if self.new_version:
            x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        return self.norm(x)

    def forward(self, x, return_all_vals=False):
        x = self.forward_features(x)
        if return_all_vals:
            return x, self.forward_head(x), self.forward_text_head(x)
        return self.forward_head(x)

    def forward_text_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.text_head(x)


@register_model
def mean_vit_s_p16(img_size=224, **kwargs):
    size = vit_sizes["S"]
    return TimmViT(img_size, patch_size=16, in_chans=3, global_pool="avg", **{**size, **kwargs})
