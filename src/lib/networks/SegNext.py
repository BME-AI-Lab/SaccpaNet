import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from ..modules.core.sampler import generate_regnet_full
from .parameter_init import *


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        # self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        # attn = self.conv3(attn)

        return attn * u


class UAttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0_1 = nn.Conv2d(
            dim, dim, (1, 3), padding="same", dilation=1, groups=dim
        )
        self.conv0_2 = nn.Conv2d(
            dim, dim, (3, 1), padding="same", dilation=1, groups=dim
        )
        # self.conv0_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)

        self.conv1_1 = nn.Conv2d(
            dim, dim, (1, 3), padding="same", dilation=2, groups=dim
        )
        self.conv1_2 = nn.Conv2d(
            dim, dim, (3, 1), padding="same", dilation=2, groups=dim
        )
        # self.conv1_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 3), padding="same", dilation=5, groups=dim
        )
        self.conv2_2 = nn.Conv2d(
            dim, dim, (3, 1), padding="same", dilation=5, groups=dim
        )
        # self.conv2_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)

        self.conv3_1 = nn.Conv2d(
            dim, dim, (1, 3), padding="same", dilation=7, groups=dim
        )
        self.conv3_2 = nn.Conv2d(
            dim, dim, (3, 1), padding="same", dilation=7, groups=dim
        )
        # self.conv3_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)

        # self.conv1x1 = nn.Conv2d(dim, dim, (1, 1), padding="same", groups=dim)
        # self.down = nn.Conv2d(dim, dim, (1, 3), padding="same", groups=dim)
        # self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), padding="same", groups=dim)

    def forward(self, x):
        attn = x.clone()
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn = attn + attn_0

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn = attn + attn_1

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_2

        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)
        attn = attn + attn_3

        # attn = self.conv1x1(attn)

        # attn = attn + attn_0 + attn_1 + attn_2
        return attn


class UAttention(nn.Module):
    def __init__(self, dim, reduction_ratio=[2, 2]):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.down1_0 = UAttentionLayer(dim)
        self.down1_1 = nn.MaxPool2d(reduction_ratio[0])
        self.down2_0 = UAttentionLayer(dim)
        self.down2_1 = nn.MaxPool2d(reduction_ratio[1])
        self.down3_0 = UAttentionLayer(dim)
        self.conv3 = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = x  # self.conv0(x)

        attn_0 = self.down1_0(attn)
        x = self.down1_1(attn_0.clone())

        attn_1 = self.down2_0(x)
        x = self.down2_1(attn_1.clone())

        attn_2 = self.down3_0(x)
        # attn_2 = self.down3_1(attn_2)
        attn_1 = nn.functional.interpolate(attn_1.clone(), u.shape[2:], mode="bilinear")
        attn_2 = nn.functional.interpolate(attn_2.clone(), u.shape[2:], mode="bilinear")
        # attn = attn + attn_0 + attn_1 + attn_2
        attn = torch.concat([attn_0, attn_1, attn_2], axis=1)

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = UAttention(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = nn.Identity()  # DropPath(
        #    drop_path) if drop_path > 0. else nn.Identity()
        # patch DropPath
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=7,
        stride=4,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


# @ BACKBONES.register_module()
class SACCPA(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.0,
        depths=[3, 4, 6, 3],
        num_stages=4,
    ):
        super(SACCPA, self).__init__()
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


# hamburger -> not used


import torch.nn.functional as F


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault("SPATIAL", True)

        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)

        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)

        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)

        self.rand_init = args.setdefault("RAND_INIT", True)

        print("spatial", self.spatial)
        print("S", self.S)
        print("D", self.D)
        print("R", self.R)
        print("train_steps", self.train_steps)
        print("eval_steps", self.eval_steps)
        print("inv_t", self.inv_t)
        print("eta", self.eta)
        print("rand_init", self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer("bases", bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__()

        self.ham_in = nn.Conv2d(
            ham_channels,
            ham_channels,
            1,
        )

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = nn.Conv2d(
            ham_channels,
            ham_channels,
            1,
        )

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


from torchvision.transforms.functional import resize as _resize


def resize(level, size, mode="bilinear", align_corners=None):
    level = _resize(level, size)
    return level


class LightHamHead(nn.Module):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.
    """

    def __init__(
        self,
        in_channels,
        in_index=[1, 2, 3],
        ham_channels=512,
        ham_kwargs=dict(),
        num_classes=18,
        channels=18,
        dropout_ratio=0,
        **kwargs,
    ):
        super(LightHamHead, self).__init__(**kwargs)
        self.ham_channels = ham_channels
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index

        self.squeeze = nn.Conv2d(
            sum(self.in_channels),
            self.ham_channels,
            1,
        )

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = nn.Conv2d(
            self.ham_channels,
            self.channels,
            1,
        )
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    # patch
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def _transform_inputs(self, inputs):
        inputs = [inputs[i] for i in self.in_index]
        return inputs

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        size = inputs[0].shape[2:]

        inputs = [
            resize(
                level,
                size=size,
                mode="bilinear",
                # align_corners=self.align_corners
            )
            for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output


class SaccpaNet(nn.Module):
    def __init__(self, params={}, num_joints=18):
        """SCAPPA based regression Network.


        Args:
            params (dict, optional): The generated search parameters from the Sampling process.
            num_joints (int, optional): The number of joints. Defaults to 18.
        """
        super().__init__()
        assert len(params) > 0  # check if params is empty
        self.params = params
        ws, ds, _, _, _ = generate_regnet_full(params)
        self.ws, self.ds = ws, ds
        self.head_input = sum(ws[1:4])
        self.backbone = SACCPA(
            in_chans=3,  # in_chans is fixed at 3 to maintain compatibility with coco pretraining
            embed_dims=ws,  # [64, 128, 320, 512],
            depths=ds,  # [2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],  # mlp ratio need
            drop_rate=0.0,
            drop_path_rate=0.1,
        )
        head_in_index = [1, 2, 3]
        in_channels = [ws[i] for i in head_in_index]

        channels = 1024
        ham_channels = 1024
        ham_dropout_ratio = 0.1
        self.head = LightHamHead(
            in_channels=in_channels,  # [self.head_input, 320, 512],
            in_index=head_in_index,
            channels=channels,
            ham_channels=ham_channels,
            dropout_ratio=ham_dropout_ratio,
            num_classes=num_joints,
        )
        # self.in_index = [1,2,3]

    def forward(self, input):
        """Forward fuction

        Args:
            input (Tensor): input tensor of shape (N, C, H, W)

        Returns:
            Tensor: (N,C,H/8, W/8)
        """
        x = input
        x = self.backbone(x)
        x = self.head(x)
        x = F.sigmoid(x)
        return x
