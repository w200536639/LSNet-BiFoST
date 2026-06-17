import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import init


# ============================================================
# Identity modules for ablation compatibility
# ============================================================
class MulOnes(nn.Module):
    """Return a tensor of ones with the same shape as input."""

    def forward(self, x):
        return torch.ones_like(x)


class AddZero(nn.Module):
    """Return a tensor of zeros with the same shape as input."""

    def forward(self, x):
        return torch.zeros_like(x)


# ============================================================
# Basic convolution module
# ============================================================
class ConvModule(nn.Module):
    """
    Convolution + optional normalization + optional activation.

    This module is retained for compatibility with previous code.
    """

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

        conv_padding = (kernel_size - 1) // 2 if padding is None else padding

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=conv_padding,
                groups=groups,
                bias=(norm_cfg is None),
            )
        ]

        if norm_cfg:
            if norm_cfg["type"] == "BN":
                layers.append(
                    nn.BatchNorm2d(
                        out_channels,
                        momentum=norm_cfg.get("momentum", 0.1),
                        eps=norm_cfg.get("eps", 1e-5),
                    )
                )
            else:
                raise NotImplementedError(f"Norm type {norm_cfg['type']} is not implemented.")

        if act_cfg:
            if act_cfg["type"] == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif act_cfg["type"] == "SiLU":
                layers.append(nn.SiLU(inplace=True))
            else:
                raise NotImplementedError(f"Activation type {act_cfg['type']} is not implemented.")

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# Weight initialization
# ============================================================
def _initialize_weights(net_l, scale=0.1):
    """Kaiming initialization with residual scaling."""

    if not isinstance(net_l, list):
        net_l = [net_l]

    for net in net_l:
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, a=0, mode="fan_in")
                module.weight.data *= scale

                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, a=0, mode="fan_in")
                module.weight.data *= scale

                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1.0)
                init.constant_(module.bias, 0.0)


# ============================================================
# LayerNorm2d
# ============================================================
class _LayerNorm2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps

        n, c, h, w = x.size()

        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)

        y = (x - mean) / torch.sqrt(variance + eps)

        ctx.save_for_backward(y, variance, weight)

        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, variance, weight = ctx.saved_tensors

        grad = grad_output * weight.view(1, -1, 1, 1)

        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_y = (grad * y).mean(dim=1, keepdim=True)

        grad_x = 1.0 / torch.sqrt(variance + eps) * (
            grad - y * mean_grad_y - mean_grad
        )

        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_x, grad_weight, grad_bias, None


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return _LayerNorm2dFn.apply(x, self.weight, self.bias, self.eps)


# ============================================================
# Boundary Interaction Enhancement
# ============================================================
class _ResBlkNoBN(nn.Module):
    """Residual block without BatchNorm, used inside BIE."""

    def __init__(self, nf=64):
        super().__init__()

        self.c1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        _initialize_weights([self.c1, self.c2], 0.1)

    def forward(self, x):
        out = F.relu(self.c1(x), inplace=True)
        out = self.c2(out)
        return x + out


class BIE(nn.Module):
    """
    Boundary Interaction Enhancement.

    This module enhances complementary boundary and semantic information
    between skip features and decoder features.
    """

    def __init__(self, nf=64):
        super().__init__()

        self.conv1 = _ResBlkNoBN(nf)
        self.conv2 = self.conv1

        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, 0)
        self.convf2 = self.convf1

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
            0.1,
        )

    def forward(self, x_1, x_2, x_s):
        batch_size, channels, height, width = x_1.shape

        x_1_refined = self.conv1(x_1)
        x_2_refined = self.conv2(x_2)

        s1 = self.clustering(
            self.norm_s(
                self.convf1(torch.cat([x_s, x_2], dim=1))
            )
        ).view(batch_size, channels, -1)

        s2 = self.clustering(
            self.norm_s(
                self.convf2(torch.cat([x_s, x_1], dim=1))
            )
        ).view(batch_size, channels, -1)

        v1 = self.v1(x_1).view(batch_size, channels, -1).permute(0, 2, 1)
        v2 = self.v2(x_2).view(batch_size, channels, -1).permute(0, 2, 1)

        att1 = torch.bmm(s1, v1) * self.scale
        att2 = torch.bmm(s2, v2) * self.scale

        o1 = torch.bmm(
            torch.softmax(att1, dim=-1),
            v1.permute(0, 2, 1),
        ).view(batch_size, channels, height, width)

        o2 = torch.bmm(
            torch.softmax(att2, dim=-1),
            v2.permute(0, 2, 1),
        ).view(batch_size, channels, height, width)

        xs = self.unclustering(
            torch.cat(
                [
                    s1.view(batch_size, channels, height, width),
                    s2.view(batch_size, channels, height, width),
                ],
                dim=1,
            )
        ) + x_s

        return o1 + x_2_refined, o2 + x_1_refined, xs


# ============================================================
# Hierarchical Pooling Attention
# ============================================================
class HPA(nn.Module):
    """
    Hierarchical Pooling Attention.

    It combines average-pooling and max-pooling responses along spatial
    dimensions and produces spatially adaptive feature reweighting.
    """

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

        c_group = channels // self.groups

        self.gn = nn.GroupNorm(
            num_groups=c_group,
            num_channels=c_group,
        )

        self.conv1x1 = nn.Conv2d(c_group, c_group, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        groups = self.groups
        grouped_x = x.reshape(batch_size * groups, -1, height, width)

        x_h = self.pool_h(grouped_x)
        x_w = self.pool_w(grouped_x).permute(0, 1, 3, 2)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [height, width], dim=2)

        x_att = self.gn(
            grouped_x
            * x_h.sigmoid()
            * x_w.permute(0, 1, 3, 2).sigmoid()
        )

        y_h = self.max_h(grouped_x)
        y_w = self.max_w(grouped_x).permute(0, 1, 3, 2)

        y_hw = self.conv1x1(torch.cat([y_h, y_w], dim=2))
        y_h, y_w = torch.split(y_hw, [height, width], dim=2)

        y_att = self.gn(
            grouped_x
            * y_h.sigmoid()
            * y_w.permute(0, 1, 3, 2).sigmoid()
        )

        x_feature = x_att.reshape(batch_size * groups, -1, height * width)
        x_weight = self.softmax(
            self.agp(x_att).reshape(batch_size * groups, -1, 1).permute(0, 2, 1)
        )

        y_feature = y_att.reshape(batch_size * groups, -1, height * width)
        y_weight = self.softmax(
            self.map(y_att).reshape(batch_size * groups, -1, 1).permute(0, 2, 1)
        )

        weights = (
            torch.matmul(x_weight, y_feature)
            + torch.matmul(y_weight, x_feature)
        ).reshape(batch_size * groups, 1, height, width)

        return (grouped_x * weights.sigmoid()).reshape(batch_size, channels, height, width)


# ============================================================
# LSNet encoder
# ============================================================
class Conv2d_BN(nn.Sequential):
    """Conv2d + BatchNorm2d block. Module name retained for checkpoint compatibility."""

    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, groups=1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                ks,
                stride,
                pad,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        )


class LSBasicBlock(nn.Module):
    """Lightweight spatial block used by LSNet encoder."""

    def __init__(self, channels, expand=2.0):
        super().__init__()

        hidden = int(channels * expand)

        self.branch = nn.Sequential(
            Conv2d_BN(channels, channels, ks=3, stride=1, pad=1, groups=channels),
            nn.ReLU(inplace=True),
            Conv2d_BN(channels, hidden, ks=1),
            nn.ReLU(inplace=True),
            Conv2d_BN(hidden, channels, ks=1),
        )

    def forward(self, x):
        return x + self.branch(x)


class LSDown(nn.Module):
    """Downsampling block used between LSNet stages."""

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
    """
    Multi-band input stem for high-resolution remote-sensing imagery.

    The module name and submodule names are retained for checkpoint compatibility:
        s1, s2, s3
    """

    def __init__(self, out_c, in_channels=3):
        super().__init__()

        c1 = out_c // 4
        c2 = out_c // 2
        c3 = out_c

        self.s1 = nn.Sequential(
            Conv2d_BN(in_channels, c1, ks=3, stride=2, pad=1),
            nn.ReLU(inplace=True),
        )

        self.s2 = nn.Sequential(
            Conv2d_BN(c1, c2, ks=3, stride=2, pad=1),
            nn.ReLU(inplace=True),
        )

        self.s3 = nn.Sequential(
            Conv2d_BN(c2, c3, ks=3, stride=2, pad=1),
        )

        self.out_c = (c1, c2, c3)

    def forward(self, x):
        f2 = self.s1(x)
        f4 = self.s2(f2)
        f8 = self.s3(f4)

        return f2, f4, f8


class LSNetEncoder(nn.Module):
    """
    LSNet encoder for UAV and high-resolution satellite imagery.

    Notes for checkpoint compatibility:
        The following attribute names are retained:
            stem, stage1, down12, stage2, down23, stage3, down34, stage4
    """

    PRESETS = {
        "lsnet_t": dict(embed_dim=[64, 128, 256, 384], depth=[0, 2, 8, 10]),
        "lsnet_s": dict(embed_dim=[96, 192, 320, 448], depth=[1, 2, 8, 10]),
        "lsnet_b": dict(embed_dim=[128, 256, 384, 512], depth=[4, 6, 8, 10]),
    }

    def __init__(self, variant="lsnet_b", in_channels=3):
        super().__init__()

        variant = variant.lower()

        assert variant in self.PRESETS, f"Unsupported LSNet variant: {variant}"

        cfg = self.PRESETS[variant]
        embed_dims = cfg["embed_dim"]
        depths = cfg["depth"]

        self.stem = LSStem(embed_dims[0], in_channels=in_channels)

        self.stage1 = nn.Sequential(
            *[LSBasicBlock(embed_dims[0]) for _ in range(depths[0])]
        )

        self.down12 = LSDown(embed_dims[0], embed_dims[1])

        self.stage2 = nn.Sequential(
            *[LSBasicBlock(embed_dims[1]) for _ in range(depths[1])]
        )

        self.down23 = LSDown(embed_dims[1], embed_dims[2])

        self.stage3 = nn.Sequential(
            *[LSBasicBlock(embed_dims[2]) for _ in range(depths[2])]
        )

        self.down34 = LSDown(embed_dims[2], embed_dims[3])

        self.stage4 = nn.Sequential(
            *[LSBasicBlock(embed_dims[3]) for _ in range(depths[3])]
        )

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

        _ = self.stage4(self.down34(f32))

        return [f2, f4, f8, f16, f32]


# ============================================================
# Decoder blocks
# ============================================================
class UnetUp(nn.Module):
    """
    Standard U-Net upsampling block.

    Note:
        This small block name is retained only because it is an internal
        decoder block name and may appear in old checkpoints.
    """

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
        x = torch.cat([skip, self.up(lowres)], dim=1)
        return self.conv(x)


class UpBIE(nn.Module):
    """
    BIE-guided upsampling block.

    When use_bie=False, it degrades to standard skip-lowres fusion.

    Attribute names retained:
        align_low, bie, conv
    """

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
        low = F.interpolate(
            lowres,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        low = self.align_low(low)

        if self.use_bie:
            o1, o2, _ = self.bie(skip, low, skip)
            fused = torch.cat([o1, o2], dim=1)
        else:
            fused = torch.cat([skip, low], dim=1)

        return self.conv(fused)


# ============================================================
# High-resolution image crown segmentation network
# ============================================================
class HighResolutionCrownSegmentationNet(nn.Module):
    """
    LSNet-BiFoST high-resolution crown segmentation network.

    This class is intended for UAV, Gaofen, WorldView, and other
    high-resolution remote-sensing imagery.

    Checkpoint compatibility:
        The following module attribute names are retained:
            encoder
            hpa16, hpa32
            up4, up3, up2, up1
            out_head
            final

    Args:
        num_classes:
            Number of output classes. For binary crown segmentation, use 2.
        backbone:
            LSNet variant: lsnet_t, lsnet_s, lsnet_b.
        use_bie:
            Enable Boundary Interaction Enhancement in decoder.
        use_hpa:
            Enable Hierarchical Pooling Attention on f16 and f32.
        in_channels:
            Input channels. Use 3 for RGB and 4/6/8 for multispectral imagery.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=False,
        backbone="lsnet_b",
        use_bie: bool = True,
        use_hpa: bool = True,
        in_channels: int = 3,
        **kwargs,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Backward-compatible input-channel aliases
        # ----------------------------------------------------
        if "input_channels" in kwargs:
            in_channels = int(kwargs.pop("input_channels"))

        if "in_chans" in kwargs:
            in_channels = int(kwargs.pop("in_chans"))

        if "input_bands" in kwargs:
            in_channels = int(kwargs.pop("input_bands"))

        # Legacy arguments from previous experiments are ignored safely.
        ignored_legacy_keys = [
            "use_caa",
            "use_msff",
            "caa",
            "msff",
            "pretrained_path",
        ]

        for key in ignored_legacy_keys:
            if key in kwargs:
                kwargs.pop(key)

        if len(kwargs) > 0:
            for key in kwargs.keys():
                print(
                    f"[Warning] HighResolutionCrownSegmentationNet: "
                    f"argument `{key}` is ignored."
                )

        self.backbone_name = backbone.lower()

        assert self.backbone_name in LSNetEncoder.PRESETS, (
            f"Unsupported backbone `{backbone}`. "
            f"Available options: {list(LSNetEncoder.PRESETS.keys())}"
        )

        self.use_bie = bool(use_bie)
        self.use_hpa = bool(use_hpa)
        self.in_channels = int(in_channels)

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------
        self.encoder = LSNetEncoder(
            variant=self.backbone_name,
            in_channels=self.in_channels,
        )

        c2, c4, c8, c16, c32 = self.encoder.out_channels()

        # ----------------------------------------------------
        # High-level attention
        # ----------------------------------------------------
        if self.use_hpa:
            self.hpa16 = HPA(c16, factor=32)
            self.hpa32 = HPA(c32, factor=32)
        else:
            self.hpa16 = AddZero()
            self.hpa32 = AddZero()

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------
        decoder_channels = [64, 128, 256, 512]

        self.up4 = UnetUp(c16 + c32, decoder_channels[3])

        self.up3 = UpBIE(
            c_skip=c8,
            c_low=decoder_channels[3],
            out_ch=decoder_channels[2],
            use_bie=self.use_bie,
        )

        self.up2 = UpBIE(
            c_skip=c4,
            c_low=decoder_channels[2],
            out_ch=decoder_channels[1],
            use_bie=self.use_bie,
        )

        self.up1 = UnetUp(c2 + decoder_channels[1], decoder_channels[0])

        # ----------------------------------------------------
        # Segmentation head
        # ----------------------------------------------------
        self.out_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(decoder_channels[0], num_classes, 1)

    def forward(self, x):
        f2, f4, f8, f16, f32 = self.encoder(x)

        f16 = f16 + self.hpa16(f16)
        f32 = f32 + self.hpa32(f32)

        u4 = self.up4(f16, f32)
        u3 = self.up3(f8, u4)
        u2 = self.up2(f4, u3)
        u1 = self.up1(f2, u2)

        output = self.out_head(u1)
        output = self.final(output)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(
                output,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return output

    def freeze_backbone(self):
        """Freeze LSNet encoder parameters."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze LSNet encoder parameters."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def get_model_profile(self):
        """Return a lightweight model configuration dictionary."""
        return {
            "network": "HighResolutionCrownSegmentationNet",
            "backbone": self.backbone_name,
            "in_channels": self.in_channels,
            "use_bie": self.use_bie,
            "use_hpa": self.use_hpa,
        }


# ============================================================
# Professional aliases
# ============================================================
# These aliases do not affect state_dict loading.
# Class name is no longer `Unet`.
GFImageCrownSegmentationNet = HighResolutionCrownSegmentationNet
GaofenCrownSegmentationNet = HighResolutionCrownSegmentationNet
HighResolutionRemoteSensingCrownNet = HighResolutionCrownSegmentationNet
LSNetBiFoSTHighResolutionSegmentation = HighResolutionCrownSegmentationNet