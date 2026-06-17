import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Conv + BN
# ============================================================
class Conv2d_BN(nn.Sequential):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, ks, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )


# ============================================================
# FocalNet: FocalModulation + FocalBlock2d + FocalNetEncoder
# ============================================================
class FocalModulation(nn.Module):
    """
    Focal modulation module.

    Input:
        x: B x H x W x C

    Output:
        x_out: B x H x W x C
    """

    def __init__(
        self,
        dim,
        focal_window,
        focal_level,
        focal_factor=2,
        bias=True,
        proj_drop=0.0,
        use_postln_in_modulation=False,
        normalize_modulator=False,
    ):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + self.focal_level + 1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []

        for level in range(self.focal_level):
            kernel_size = self.focal_factor * level + self.focal_window

            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=dim,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )

            self.kernel_sizes.append(kernel_size)

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        channels = x.shape[-1]

        x = self.f(x).permute(0, 3, 1, 2).contiguous()

        q, ctx, gates = torch.split(
            x,
            (channels, channels, self.focal_level + 1),
            dim=1,
        )

        ctx_all = 0

        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * gates[:, level: level + 1]

        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        modulator = self.h(ctx_all)

        x_out = q * modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()

        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class FocalBlock2d(nn.Module):
    """
    2D FocalNet block.

    Input:
        x: B x C x H x W

    Output:
        x_out: B x C x H x W
    """

    def __init__(
        self,
        dim,
        focal_window=3,
        focal_level=2,
        mlp_ratio=4.0,
        use_postln_in_modulation=False,
        normalize_modulator=False,
    ):
        super().__init__()

        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)

        self.focal = FocalModulation(
            dim=dim,
            focal_window=focal_window,
            focal_level=focal_level,
            proj_drop=0.0,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x_hw = x.permute(0, 2, 3, 1)

        shortcut = x_hw
        x1 = self.norm1(x_hw)
        x1 = self.focal(x1)
        x_hw = shortcut + x1

        shortcut = x_hw
        x2 = self.norm2(x_hw)
        x2 = self.mlp(x2)
        x_hw = shortcut + x2

        x_out = x_hw.permute(0, 3, 1, 2).contiguous()

        return x_out


class FocalNetEncoder(nn.Module):
    """
    FocalNet encoder.

    Output:
        [f2, f4, f8, f16, f32]
    """

    PRESETS = {
        "focal_t": dict(
            embed_dim=[64, 128, 256, 512],
            depth=[1, 1, 2, 1],
            focal_window=3,
            focal_level=2,
        ),
        "focal_s": dict(
            embed_dim=[96, 192, 384, 512],
            depth=[1, 1, 3, 1],
            focal_window=3,
            focal_level=2,
        ),
        "focal_b": dict(
            embed_dim=[128, 256, 512, 512],
            depth=[2, 2, 4, 2],
            focal_window=3,
            focal_level=2,
        ),
    }

    def __init__(self, variant="focal_t", in_ch=3):
        super().__init__()

        variant = variant.lower()

        if variant not in self.PRESETS:
            print(f"[Warning] backbone '{variant}' is not supported. Fallback to 'focal_t'.")
            variant = "focal_t"

        cfg = self.PRESETS[variant]

        embed_dims = cfg["embed_dim"]
        depths = cfg["depth"]
        focal_window = cfg["focal_window"]
        focal_level = cfg["focal_level"]

        self.variant = variant

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, embed_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )

        self.stage0 = nn.Sequential(
            *[
                FocalBlock2d(
                    embed_dims[0],
                    focal_window=focal_window,
                    focal_level=focal_level,
                )
                for _ in range(depths[0])
            ]
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            *[
                FocalBlock2d(
                    embed_dims[1],
                    focal_window=focal_window,
                    focal_level=focal_level,
                )
                for _ in range(depths[1])
            ]
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.Sequential(
            *[
                FocalBlock2d(
                    embed_dims[2],
                    focal_window=focal_window,
                    focal_level=focal_level,
                )
                for _ in range(depths[2])
            ]
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.ReLU(inplace=True),
        )

        self.stage3 = nn.Sequential(
            *[
                FocalBlock2d(
                    embed_dims[3],
                    focal_window=focal_window,
                    focal_level=focal_level,
                )
                for _ in range(depths[3])
            ]
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.ReLU(inplace=True),
        )

        self.stage4 = FocalBlock2d(
            embed_dims[3],
            focal_window=focal_window,
            focal_level=focal_level,
        )

        self.channels = (
            embed_dims[0],
            embed_dims[1],
            embed_dims[2],
            embed_dims[3],
            embed_dims[3],
        )

    @torch.no_grad()
    def out_channels(self):
        return self.channels

    def forward(self, x):
        x = self.stem(x)
        f2 = self.stage0(x)

        x = self.down1(f2)
        f4 = self.stage1(x)

        x = self.down2(f4)
        f8 = self.stage2(x)

        x = self.down3(f8)
        f16 = self.stage3(x)

        x = self.down4(f16)
        f32 = self.stage4(x)

        return [f2, f4, f8, f16, f32]


# ============================================================
# Decoder
# ============================================================
class DecoderUpBlock(nn.Module):
    """
    Standard decoder upsampling block.

    This replaces the previous BIE-based fusion block.
    """

    def __init__(self, in_size, out_size):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip, lowres):
        lowres_up = self.up(lowres)

        if lowres_up.shape[-2:] != skip.shape[-2:]:
            lowres_up = F.interpolate(
                lowres_up,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, lowres_up], dim=1)

        return self.conv(x)


# ============================================================
# SVIT: Unfold / Fold / Attention / StokenAttention
# ============================================================
class Unfold(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        return F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=1,
        )


class Fold(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        kernel = self.kernel_size

        assert channels == kernel * kernel, (
            f"Fold expects channel = kernel_size * kernel_size = {kernel * kernel}, "
            f"but got {channels}."
        )

        x = x.reshape(batch_size, channels, height * width)

        out = F.fold(
            x,
            output_size=(height, width),
            kernel_size=kernel,
            padding=self.padding,
            stride=1,
        )

        return out


class Attention(nn.Module):
    """
    Multi-head self-attention.

    Input:
        x: B x C x H x W

    Output:
        x_out: B x C x H x W
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim={dim} must be divisible by num_heads={num_heads}."
        )

        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x_flat = x.flatten(2).transpose(1, 2)
        num_tokens = x_flat.shape[1]

        qkv = self.qkv(x_flat).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x_out = (attention @ v).transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            channels,
        )

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        x_out = x_out.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return x_out


class StokenAttention(nn.Module):
    """
    Super-token attention module.
    """

    def __init__(
        self,
        dim,
        stoken_size,
        n_iter=1,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.scale = dim ** -0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def stoken_forward(self, x):
        batch_size, channels, original_h, original_w = x.shape

        stoken_h, stoken_w = self.stoken_size

        pad_l = 0
        pad_t = 0
        pad_r = (stoken_w - original_w % stoken_w) % stoken_w
        pad_b = (stoken_h - original_h % stoken_h) % stoken_h

        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, height, width = x.shape

        super_h = height // stoken_h
        super_w = width // stoken_w

        stoken_features = F.adaptive_avg_pool2d(x, (super_h, super_w))

        pixel_features = x.reshape(
            batch_size,
            channels,
            super_h,
            stoken_h,
            super_w,
            stoken_w,
        ).permute(0, 2, 4, 3, 5, 1).reshape(
            batch_size,
            super_h * super_w,
            stoken_h * stoken_w,
            channels,
        )

        with torch.no_grad():
            for iteration in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)
                stoken_features = stoken_features.transpose(1, 2).reshape(
                    batch_size,
                    super_h * super_w,
                    channels,
                    9,
                )

                affinity_matrix = pixel_features @ stoken_features * self.scale
                affinity_matrix = affinity_matrix.softmax(-1)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(
                    batch_size,
                    9,
                    super_h,
                    super_w,
                )

                affinity_matrix_sum = self.fold(affinity_matrix_sum)

                if iteration < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix

                    stoken_features = self.fold(
                        stoken_features.permute(0, 2, 3, 1).reshape(
                            batch_size * channels,
                            9,
                            super_h,
                            super_w,
                        )
                    ).reshape(batch_size, channels, super_h, super_w)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix

        stoken_features = self.fold(
            stoken_features.permute(0, 2, 3, 1).reshape(
                batch_size * channels,
                9,
                super_h,
                super_w,
            )
        ).reshape(batch_size, channels, super_h, super_w)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)
        stoken_features = stoken_features.transpose(1, 2).reshape(
            batch_size,
            super_h * super_w,
            channels,
            9,
        )

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)

        pixel_features = pixel_features.reshape(
            batch_size,
            super_h,
            super_w,
            channels,
            stoken_h,
            stoken_w,
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :original_h, :original_w]

        return pixel_features

    def direct_forward(self, x):
        return self.stoken_refine(x)

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)

        return self.direct_forward(x)


# ============================================================
# FocalNet + SVIT Crown Segmentation Network
# ============================================================
class FocalSVITCrownSegmentationNet(nn.Module):
    """
    FocalNet-SVIT crown segmentation network without BIE.

    The BIE branch has been completely removed.
    Decoder fusion is implemented by standard U-Net style
    upsampling and skip concatenation.

    Args:
        num_classes:
            Number of output classes. For binary crown segmentation, use 2.
        backbone:
            focal_t / focal_s / focal_b.
        in_channels:
            Number of image channels. For RGB use 3; for WV3/GF multispectral use 8.
        use_svit:
            Whether to enable SVIT globally.
        svit_on_f16:
            Whether to use SVIT on f16.
        svit_on_f32:
            Whether to use SVIT on f32.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        backbone: str = "focal_t",
        in_channels: int = 3,
        use_svit: bool = True,
        svit_on_f16: bool = True,
        svit_on_f32: bool = True,
        svit_stoken_size=(4, 4),
        svit_heads: int = 8,
        svit_n_iter: int = 1,
        use_bie: bool = False,
        **kwargs,
    ):
        super().__init__()

        if use_bie:
            print("[Warning] use_bie=True is ignored because BIE has been removed.")

        if len(kwargs) > 0:
            for key in kwargs.keys():
                print(f"[Warning] FocalSVITCrownSegmentationNet: argument `{key}` is ignored.")

        self.backbone_name = backbone.lower()

        if self.backbone_name not in FocalNetEncoder.PRESETS:
            print(
                f"[Warning] backbone '{backbone}' is not in FocalNetEncoder.PRESETS. "
                f"Fallback to 'focal_t'."
            )
            self.backbone_name = "focal_t"

        self.in_channels = int(in_channels)

        self.use_svit = bool(use_svit)
        self.svit_on_f16 = bool(use_svit and svit_on_f16)
        self.svit_on_f32 = bool(use_svit and svit_on_f32)

        self.encoder = FocalNetEncoder(
            variant=self.backbone_name,
            in_ch=self.in_channels,
        )

        c2, c4, c8, c16, c32 = self.encoder.out_channels()

        if self.svit_on_f16:
            self.svit16 = StokenAttention(
                dim=c16,
                stoken_size=svit_stoken_size,
                n_iter=svit_n_iter,
                num_heads=svit_heads,
            )
        else:
            self.svit16 = None

        if self.svit_on_f32:
            self.svit32 = StokenAttention(
                dim=c32,
                stoken_size=svit_stoken_size,
                n_iter=svit_n_iter,
                num_heads=svit_heads,
            )
        else:
            self.svit32 = None

        decoder_channels = [64, 128, 256, 512]

        self.up4 = DecoderUpBlock(c16 + c32, decoder_channels[3])
        self.up3 = DecoderUpBlock(c8 + decoder_channels[3], decoder_channels[2])
        self.up2 = DecoderUpBlock(c4 + decoder_channels[2], decoder_channels[1])
        self.up1 = DecoderUpBlock(c2 + decoder_channels[1], decoder_channels[0])

        self.out_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        f2, f4, f8, f16, f32 = self.encoder(x)

        if self.svit16 is not None:
            f16 = self.svit16(f16)

        if self.svit32 is not None:
            f32 = self.svit32(f32)

        u4 = self.up4(f16, f32)
        u3 = self.up3(f8, u4)
        u2 = self.up2(f4, u3)
        u1 = self.up1(f2, u2)

        y = self.out_head(u1)
        y = self.final(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(
                y,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return y

    def freeze_backbone(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def get_ablation_config(self):
        return {
            "network": "FocalSVITCrownSegmentationNet",
            "backbone": self.backbone_name,
            "in_channels": self.in_channels,
            "use_bie": False,
            "use_svit": self.use_svit,
            "svit_on_f16": self.svit_on_f16,
            "svit_on_f32": self.svit_on_f32,
        }

    def get_model_profile(self):
        return self.get_ablation_config()


# Professional aliases
FocalSVITSegmentationNet = FocalSVITCrownSegmentationNet
HighResolutionFocalSVITCrownNet = FocalSVITCrownSegmentationNet