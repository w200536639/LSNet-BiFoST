# nets/lsnet_bifost_segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import init


# ============================================================
# Helper identity modules for ablation
# ============================================================
class MulOnes(nn.Module):
    """Return ones with the same shape as input so that x * ones(x) == x."""

    def forward(self, x):
        return torch.ones_like(x)


class AddZero(nn.Module):
    """Return zeros with the same shape as input so that x + zeros(x) == x."""

    def forward(self, x):
        return torch.zeros_like(x)


# ============================================================
# Basic convolution wrapper
# ConvModule is retained for compatibility and future extension.
# ============================================================
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ):
        super().__init__()

        auto_padding = (kernel_size - 1) // 2 if padding is None else padding
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=auto_padding,
                groups=groups,
                bias=(norm_cfg is None),
            )
        ]

        if norm_cfg:
            norm_type = norm_cfg["type"]
            if norm_type == "BN":
                layers.append(
                    nn.BatchNorm2d(
                        out_channels,
                        momentum=norm_cfg.get("momentum", 0.1),
                        eps=norm_cfg.get("eps", 1e-5),
                    )
                )
            else:
                raise NotImplementedError(f"Norm type {norm_type} is not implemented.")

        if act_cfg:
            act_type = act_cfg["type"]
            if act_type == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif act_type == "SiLU":
                layers.append(nn.SiLU(inplace=True))
            else:
                raise NotImplementedError(f"Activation type {act_type} is not implemented.")

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# BIE-related components
# ============================================================
def _initialize_weights(modules, scale=0.1):
    """Initialize convolution, linear, and batch-normalization layers."""
    if not isinstance(modules, list):
        modules = [modules]

    for module_group in modules:
        for layer in module_group.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()

            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()

            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)


class _LayerNorm2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        batch_size, channels, height, width = x.size()

        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / (variance + eps).sqrt()

        ctx.save_for_backward(normalized, variance, weight)

        output = weight.view(1, channels, 1, 1) * normalized + bias.view(
            1, channels, 1, 1
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        normalized, variance, weight = ctx.saved_variables

        scaled_grad = grad_output * weight.view(1, -1, 1, 1)
        mean_grad = scaled_grad.mean(dim=1, keepdim=True)
        mean_grad_normalized = (scaled_grad * normalized).mean(dim=1, keepdim=True)

        grad_input = (1.0 / torch.sqrt(variance + eps)) * (
            scaled_grad - normalized * mean_grad_normalized - mean_grad
        )
        grad_weight = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None


class _LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return _LayerNorm2dFn.apply(x, self.weight, self.bias, self.eps)


class _ResBlkNoBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.c1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        _initialize_weights([self.c1, self.c2], scale=0.1)

    def forward(self, x):
        residual = F.relu(self.c1(x), inplace=True)
        residual = self.c2(residual)
        return x + residual


class BIE(nn.Module):
    """Boundary Interaction Enhancement module."""

    def __init__(self, nf=64):
        super().__init__()

        self.conv1 = _ResBlkNoBN(nf)
        self.conv2 = self.conv1  # Shared weights.

        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, 0)
        self.convf2 = self.convf1  # Shared weights.

        self.scale = nf ** -0.5
        self.norm_s = _LayerNorm2d(nf)

        self.clustering = nn.Conv2d(nf, nf, 1, 1, 0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, 1, 0)

        self.v1 = nn.Conv2d(nf, nf, 1, 1, 0)
        self.v2 = nn.Conv2d(nf, nf, 1, 1, 0)

        _initialize_weights(
            [
                self.convf1,
                self.convf2,
                self.clustering,
                self.unclustering,
                self.v1,
                self.v2,
            ],
            scale=0.1,
        )

    def forward(self, x_1, x_2, x_s):
        batch_size, channels, height, width = x_1.shape

        feature_1 = x_1
        feature_2 = x_2
        shared_feature = x_s

        enhanced_feature_1 = self.conv1(feature_1)
        enhanced_feature_2 = self.conv2(feature_2)

        cluster_1 = self.clustering(
            self.norm_s(self.convf1(torch.cat([shared_feature, feature_2], dim=1)))
        ).view(batch_size, channels, -1)

        cluster_2 = self.clustering(
            self.norm_s(self.convf2(torch.cat([shared_feature, feature_1], dim=1)))
        ).view(batch_size, channels, -1)

        value_1 = self.v1(feature_1).view(batch_size, channels, -1).permute(0, 2, 1)
        value_2 = self.v2(feature_2).view(batch_size, channels, -1).permute(0, 2, 1)

        attention_1 = torch.bmm(cluster_1, value_1) * self.scale
        attention_2 = torch.bmm(cluster_2, value_2) * self.scale

        output_1 = torch.bmm(
            torch.softmax(attention_1, dim=-1),
            value_1.permute(0, 2, 1),
        ).view(batch_size, channels, height, width)

        output_2 = torch.bmm(
            torch.softmax(attention_2, dim=-1),
            value_2.permute(0, 2, 1),
        ).view(batch_size, channels, height, width)

        shared_output = self.unclustering(
            torch.cat(
                [
                    cluster_1.view(batch_size, channels, height, width),
                    cluster_2.view(batch_size, channels, height, width),
                ],
                dim=1,
            )
        ) + shared_feature

        return output_1 + enhanced_feature_2, output_2 + enhanced_feature_1, shared_output


class HPA(nn.Module):
    """Hybrid Pooling Attention module."""

    def __init__(self, channels, factor=32):
        super().__init__()

        self.groups = factor
        assert channels % self.groups == 0, (
            f"HPA: channels {channels} must be divisible by factor {factor}"
        )

        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_w = nn.AdaptiveMaxPool2d((1, None))

        channels_per_group = channels // self.groups

        self.gn = nn.GroupNorm(
            num_groups=channels_per_group,
            num_channels=channels_per_group,
        )
        self.conv1x1 = nn.Conv2d(channels_per_group, channels_per_group, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        groups = self.groups

        grouped_feature = x.reshape(batch_size * groups, -1, height, width)

        # Average-pooling branch.
        avg_h = self.pool_h(grouped_feature)
        avg_w = self.pool_w(grouped_feature).permute(0, 1, 3, 2)

        avg_hw = self.conv1x1(torch.cat([avg_h, avg_w], dim=2))
        avg_h, avg_w = torch.split(avg_hw, [height, width], dim=2)

        avg_feature = self.gn(
            grouped_feature * avg_h.sigmoid() * avg_w.permute(0, 1, 3, 2).sigmoid()
        )

        # Max-pooling branch.
        max_h = self.max_h(grouped_feature)
        max_w = self.max_w(grouped_feature).permute(0, 1, 3, 2)

        max_hw = self.conv1x1(torch.cat([max_h, max_w], dim=2))
        max_h, max_w = torch.split(max_hw, [height, width], dim=2)

        max_feature = self.gn(
            grouped_feature * max_h.sigmoid() * max_w.permute(0, 1, 3, 2).sigmoid()
        )

        # Branch fusion.
        avg_spatial_feature = avg_feature.reshape(
            batch_size * groups,
            -1,
            height * width,
        )
        avg_channel_weight = self.softmax(
            self.agp(avg_feature).reshape(batch_size * groups, -1, 1).permute(0, 2, 1)
        )

        max_spatial_feature = max_feature.reshape(
            batch_size * groups,
            -1,
            height * width,
        )
        max_channel_weight = self.softmax(
            self.map(max_feature).reshape(batch_size * groups, -1, 1).permute(0, 2, 1)
        )

        spatial_weight = (
            torch.matmul(avg_channel_weight, max_spatial_feature)
            + torch.matmul(max_channel_weight, avg_spatial_feature)
        ).reshape(batch_size * groups, 1, height, width)

        return (grouped_feature * spatial_weight.sigmoid()).reshape(
            batch_size,
            channels,
            height,
            width,
        )


# ============================================================
# LSNet encoder
# ============================================================
class Conv2d_BN(nn.Sequential):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, ks, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )


class LSBasicBlock(nn.Module):
    def __init__(self, channels, expand=2.0):
        super().__init__()

        hidden_channels = int(channels * expand)

        self.branch = nn.Sequential(
            Conv2d_BN(channels, channels, ks=3, stride=1, pad=1, groups=channels),
            nn.ReLU(inplace=True),
            Conv2d_BN(channels, hidden_channels, ks=1),
            nn.ReLU(inplace=True),
            Conv2d_BN(hidden_channels, channels, ks=1),
        )

    def forward(self, x):
        return x + self.branch(x)


class LSDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.op = nn.Sequential(
            Conv2d_BN(in_ch, in_ch, ks=3, stride=2, pad=1, groups=in_ch),
            nn.ReLU(inplace=True),
            Conv2d_BN(in_ch, out_ch, ks=1),
        )

    def forward(self, x):
        return self.op(x)


class LSStem(nn.Module):
    def __init__(self, out_c):
        super().__init__()

        c1, c2, c3 = out_c // 4, out_c // 2, out_c

        self.s1 = nn.Sequential(
            Conv2d_BN(3, c1, 3, 2, 1),
            nn.ReLU(inplace=True),
        )  # /2

        self.s2 = nn.Sequential(
            Conv2d_BN(c1, c2, 3, 2, 1),
            nn.ReLU(inplace=True),
        )  # /4

        self.s3 = nn.Sequential(
            Conv2d_BN(c2, c3, 3, 2, 1),
        )  # /8

        self.out_c = (c1, c2, c3)

    def forward(self, x):
        f2 = self.s1(x)
        f4 = self.s2(f2)
        f8 = self.s3(f4)

        return f2, f4, f8


class LSNetEncoder(nn.Module):
    PRESETS = {
        "lsnet_t": dict(embed_dim=[64, 128, 256, 384], depth=[0, 2, 8, 10]),
        "lsnet_s": dict(embed_dim=[96, 192, 320, 448], depth=[1, 2, 8, 10]),
        "lsnet_b": dict(embed_dim=[128, 256, 384, 512], depth=[4, 6, 8, 10]),
    }

    def __init__(self, variant="lsnet_b"):
        super().__init__()

        variant = variant.lower()
        assert variant in self.PRESETS, f"Unsupported LSNet variant: {variant}"

        config = self.PRESETS[variant]
        embed_dims = config["embed_dim"]
        depths = config["depth"]

        self.stem = LSStem(embed_dims[0])  # -> /8

        self.stage1 = nn.Sequential(
            *[LSBasicBlock(embed_dims[0]) for _ in range(depths[0])]
        )

        self.down12 = LSDown(embed_dims[0], embed_dims[1])
        self.stage2 = nn.Sequential(
            *[LSBasicBlock(embed_dims[1]) for _ in range(depths[1])]
        )  # /16

        self.down23 = LSDown(embed_dims[1], embed_dims[2])
        self.stage3 = nn.Sequential(
            *[LSBasicBlock(embed_dims[2]) for _ in range(depths[2])]
        )  # /32

        self.down34 = LSDown(embed_dims[2], embed_dims[3])
        self.stage4 = nn.Sequential(
            *[LSBasicBlock(embed_dims[3]) for _ in range(depths[3])]
        )  # /64

        # Feature channels for /2, /4, /8, /16, and /32 outputs.
        self.channels = (
            embed_dims[0] // 4,
            embed_dims[0] // 2,
            embed_dims[0],
            embed_dims[1],
            embed_dims[2],
        )

    @torch.no_grad()
    def out_channels(self):
        return self.channels

    def forward(self, x):
        f2, f4, f8 = self.stem(x)

        f8 = self.stage1(f8)
        f16 = self.stage2(self.down12(f8))
        f32 = self.stage3(self.down23(f16))

        # Keep stage4 for architectural completeness, although it is not returned.
        _ = self.stage4(self.down34(f32))

        return [f2, f4, f8, f16, f32]


# ============================================================
# Decoder blocks with optional BIE
# ============================================================
class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip, lowres):
        upsampled_feature = self.up(lowres)
        fused_feature = torch.cat([skip, upsampled_feature], dim=1)

        return self.conv(fused_feature)


class UpBIE(nn.Module):
    """Upsampling block with optional BIE; falls back to cat+conv when use_bie=False."""

    def __init__(self, c_skip, c_low, out_ch, use_bie: bool = True):
        super().__init__()

        self.use_bie = use_bie
        self.align_low = nn.Conv2d(c_low, c_skip, 1, bias=False)

        if use_bie:
            self.bie = BIE(nf=c_skip)
        else:
            self.bie = None

        self.conv = nn.Sequential(
            nn.Conv2d(2 * c_skip, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip, lowres):
        lowres_aligned = F.interpolate(
            lowres,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        lowres_aligned = self.align_low(lowres_aligned)

        if self.use_bie:
            bie_output_1, bie_output_2, _ = self.bie(skip, lowres_aligned, skip)
            fused_feature = torch.cat([bie_output_1, bie_output_2], dim=1)
        else:
            fused_feature = torch.cat([skip, lowres_aligned], dim=1)

        return self.conv(fused_feature)


# ============================================================
# LSNet-BiFoST segmentation branch
# ============================================================
class LSNetBiFoSTSegmentation(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=False,
        backbone="lsnet_b",
        use_bie: bool = True,
        use_hpa: bool = True,
        **kwargs,
    ):
        """
        LSNet-BiFoST segmentation branch.

        This class implements the crown-segmentation branch of LSNet-BiFoST,
        using an LSNet encoder, HPA-enhanced high-level features, and optional
        BIE-based decoder fusion.

        Ablation switches:
            - use_bie: enable BIE in decoder fusion blocks.
            - use_hpa: enable HPA on high-level encoder features f16 and f32.

        Notes:
            - The argument `pretrained` is retained for compatibility with old scripts.
            - CAA and MSFF have been removed from the network.
            - Extra legacy arguments are accepted and ignored for script compatibility.
        """
        super().__init__()

        self.backbone_name = backbone.lower()
        assert self.backbone_name in LSNetEncoder.PRESETS, (
            f"Unsupported backbone `{backbone}`"
        )

        # Keep this argument for old training scripts, although it is not used here.
        self.pretrained = pretrained

        # Accept and ignore legacy CAA/MSFF arguments for compatibility.
        if len(kwargs) > 0:
            for arg_name in kwargs.keys():
                print(
                    f"[Warning] LSNetBiFoSTSegmentation: argument `{arg_name}` "
                    "is ignored (CAA/MSFF have been removed)."
                )

        self.use_bie = use_bie
        self.use_hpa = use_hpa

        # Encoder
        self.encoder = LSNetEncoder(self.backbone_name)
        c2, c4, c8, c16, c32 = self.encoder.out_channels()  # /2, /4, /8, /16, /32

        # HPA on high-level features
        if use_hpa:
            self.hpa16 = HPA(c16, factor=32)
            self.hpa32 = HPA(c32, factor=32)
        else:
            self.hpa16 = AddZero()
            self.hpa32 = AddZero()

        # Decoder channel settings for /2, /4, /8, and /16 outputs.
        decoder_channels = [64, 128, 256, 512]

        self.up4 = UnetUp(c16 + c32, decoder_channels[3])  # /16
        self.up3 = UpBIE(
            c_skip=c8,
            c_low=decoder_channels[3],
            out_ch=decoder_channels[2],
            use_bie=use_bie,
        )  # /8
        self.up2 = UpBIE(
            c_skip=c4,
            c_low=decoder_channels[2],
            out_ch=decoder_channels[1],
            use_bie=use_bie,
        )  # /4
        self.up1 = UnetUp(c2 + decoder_channels[1], decoder_channels[0])  # /2

        # Output head
        self.out_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(decoder_channels[0], num_classes, 1)

    def forward(self, x):
        # Encoder features at /2, /4, /8, /16, and /32.
        f2, f4, f8, f16, f32 = self.encoder(x)

        # Residual HPA enhancement on high-level features.
        f16 = f16 + self.hpa16(f16)
        f32 = f32 + self.hpa32(f32)

        # Decoder
        u4 = self.up4(f16, f32)  # -> /16
        u3 = self.up3(f8, u4)    # -> /8
        u2 = self.up2(f4, u3)    # -> /4
        u1 = self.up1(f2, u2)    # -> /2

        logits = self.out_head(u1)
        logits = self.final(logits)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


# Backward-compatible alias.
# Old scripts using `from nets.unet import Unet` will still work.
# This alias does not change state_dict keys and should not affect old weights.
Unet = LSNetBiFoSTSegmentation


__all__ = [
    "LSNetBiFoSTSegmentation",
    "Unet",
    "LSNetEncoder",
    "HPA",
    "BIE",
    "UpBIE",
    "UnetUp",
]