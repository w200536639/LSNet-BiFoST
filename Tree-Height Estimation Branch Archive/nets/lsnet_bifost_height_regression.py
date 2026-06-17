import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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

        batch_size, channels, height, width = grad_output.size()

        scaled_grad = grad_output * weight.view(1, channels, 1, 1)
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
        self.conv2 = self.conv1  # shared weights

        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, 0)
        self.convf2 = self.convf1  # shared weights

        self.scale = nf**-0.5
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

        enhanced_feature_1 = self.conv1(x_1)
        enhanced_feature_2 = self.conv2(x_2)

        cluster_1 = self.clustering(
            self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))
        ).view(batch_size, channels, -1)

        cluster_2 = self.clustering(
            self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))
        ).view(batch_size, channels, -1)

        value_1 = self.v1(x_1).view(batch_size, channels, -1).permute(0, 2, 1)
        value_2 = self.v2(x_2).view(batch_size, channels, -1).permute(0, 2, 1)

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
        ) + x_s

        return output_1 + enhanced_feature_2, output_2 + enhanced_feature_1, shared_output


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
# FocalNet encoder
# ============================================================
class FocalModulation(nn.Module):
    """
    Focal modulation module.

    Input shape:
        (B, H, W, C)

    Output shape:
        (B, H, W, C)
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

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
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

        projected = self.f(x).permute(0, 3, 1, 2).contiguous()
        query, context, gates = torch.split(
            projected,
            (channels, channels, self.focal_level + 1),
            dim=1,
        )

        aggregated_context = 0

        for level in range(self.focal_level):
            context = self.focal_layers[level](context)
            aggregated_context = aggregated_context + context * gates[:, level: level + 1]

        global_context = self.act(context.mean(2, keepdim=True).mean(3, keepdim=True))
        aggregated_context = aggregated_context + global_context * gates[:, self.focal_level:]

        if self.normalize_modulator:
            aggregated_context = aggregated_context / (self.focal_level + 1)

        modulator = self.h(aggregated_context)

        output = query * modulator
        output = output.permute(0, 2, 3, 1).contiguous()

        if self.use_postln_in_modulation:
            output = self.ln(output)

        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class FocalBlock2d(nn.Module):
    """
    2D FocalNet block.

    Input shape:
        (B, C, H, W)

    Output shape:
        (B, C, H, W)
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

        feature_hw = x.permute(0, 2, 3, 1)

        shortcut = feature_hw
        focal_input = self.norm1(feature_hw)
        focal_output = self.focal(focal_input)
        feature_hw = shortcut + focal_output

        shortcut = feature_hw
        mlp_input = self.norm2(feature_hw)
        mlp_output = self.mlp(mlp_input)
        feature_hw = shortcut + mlp_output

        output = feature_hw.permute(0, 3, 1, 2)

        return output


class FocalNetEncoder(nn.Module):
    """
    FocalNet-style encoder.

    Returned features:
        f2, f4, f8, f16, f32

    Their spatial scales are:
        /2, /4, /8, /16, /32
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
        assert variant in self.PRESETS, f"Unsupported FocalNet variant: {variant}"

        config = self.PRESETS[variant]
        embed_dims = config["embed_dim"]
        depths = config["depth"]
        focal_window = config["focal_window"]
        focal_level = config["focal_level"]

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
# Decoder blocks
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
    def __init__(self, c_skip, c_low, out_ch):
        super().__init__()

        self.align_low = nn.Conv2d(c_low, c_skip, 1, bias=False)
        self.bie = BIE(nf=c_skip)

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

        output_1, output_2, _ = self.bie(skip, lowres_aligned, skip)
        fused_feature = torch.cat([output_1, output_2], dim=1)

        return self.conv(fused_feature)


# ============================================================
# SVIT: Unfold / Fold / Attention / StokenAttention
# ============================================================
class Unfold(nn.Module):
    """Wrapper of F.unfold."""

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
    """Wrapper of F.fold used by StokenAttention."""

    def __init__(self, kernel_size: int):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        kernel_size = self.kernel_size
        assert channels == kernel_size * kernel_size, (
            f"Fold expects channel = kernel_size * kernel_size = {kernel_size * kernel_size}, "
            f"but got {channels}."
        )

        x = x.reshape(batch_size, channels, height * width)

        output = F.fold(
            x,
            output_size=(height, width),
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1,
        )

        return output


class Attention(nn.Module):
    """Multi-head self-attention for feature maps."""

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

        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        num_tokens = tokens.shape[1]

        qkv = self.qkv(tokens).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        output = (attention @ value).transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            channels,
        )

        output = self.proj(output)
        output = self.proj_drop(output)

        output = output.transpose(1, 2).reshape(batch_size, channels, height, width)

        return output


class StokenAttention(nn.Module):
    """Super-token attention module."""

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
        self.scale = dim**-0.5

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

        pad_left = 0
        pad_top = 0
        pad_right = (stoken_w - original_w % stoken_w) % stoken_w
        pad_bottom = (stoken_h - original_h % stoken_h) % stoken_h

        if pad_right > 0 or pad_bottom > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        _, _, padded_h, padded_w = x.shape

        num_stoken_h = padded_h // stoken_h
        num_stoken_w = padded_w // stoken_w

        stoken_features = F.adaptive_avg_pool2d(x, (num_stoken_h, num_stoken_w))

        pixel_features = x.reshape(
            batch_size,
            channels,
            num_stoken_h,
            stoken_h,
            num_stoken_w,
            stoken_w,
        ).permute(0, 2, 4, 3, 5, 1).reshape(
            batch_size,
            num_stoken_h * num_stoken_w,
            stoken_h * stoken_w,
            channels,
        )

        with torch.no_grad():
            for iteration in range(self.n_iter):
                unfolded_stokens = self.unfold(stoken_features)
                unfolded_stokens = unfolded_stokens.transpose(1, 2).reshape(
                    batch_size,
                    num_stoken_h * num_stoken_w,
                    channels,
                    9,
                )

                affinity_matrix = pixel_features @ unfolded_stokens * self.scale
                affinity_matrix = affinity_matrix.softmax(dim=-1)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(
                    batch_size,
                    9,
                    num_stoken_h,
                    num_stoken_w,
                )
                affinity_matrix_sum = self.fold(affinity_matrix_sum)

                if iteration < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix
                    stoken_features = self.fold(
                        stoken_features.permute(0, 2, 3, 1).reshape(
                            batch_size * channels,
                            9,
                            num_stoken_h,
                            num_stoken_w,
                        )
                    ).reshape(batch_size, channels, num_stoken_h, num_stoken_w)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix
        stoken_features = self.fold(
            stoken_features.permute(0, 2, 3, 1).reshape(
                batch_size * channels,
                9,
                num_stoken_h,
                num_stoken_w,
            )
        ).reshape(batch_size, channels, num_stoken_h, num_stoken_w)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)

        stoken_features = self.stoken_refine(stoken_features)

        unfolded_stokens = self.unfold(stoken_features)
        unfolded_stokens = unfolded_stokens.transpose(1, 2).reshape(
            batch_size,
            num_stoken_h * num_stoken_w,
            channels,
            9,
        )

        pixel_features = unfolded_stokens @ affinity_matrix.transpose(-1, -2)
        pixel_features = pixel_features.reshape(
            batch_size,
            num_stoken_h,
            num_stoken_w,
            channels,
            stoken_h,
            stoken_w,
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size,
            channels,
            padded_h,
            padded_w,
        )

        if pad_right > 0 or pad_bottom > 0:
            pixel_features = pixel_features[:, :, :original_h, :original_w]

        return pixel_features

    def direct_forward(self, x):
        return self.stoken_refine(x)

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)

        return self.direct_forward(x)


# ============================================================
# LSNet-BiFoST height-regression branch
# ============================================================
class LSNetBiFoSTHeightRegression(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = False,
        backbone: str = "focal_t",
        in_channels: int = 3,
        use_bie: bool = True,
        use_svit: bool = True,
        svit_on_f16: bool = True,
        svit_on_f32: bool = True,
        svit_stoken_size=(4, 4),
        svit_heads: int = 8,
        svit_n_iter: int = 1,
        out_channels: int = None,
        **kwargs,
    ):
        """
        LSNet-BiFoST height-regression branch.

        The historical argument name `num_classes` is retained for compatibility.
        For height regression, the recommended output channel number is 1.

        Args:
            num_classes: Kept for compatibility. Used as output channels if out_channels is None.
            pretrained: Kept for compatibility with old scripts.
            backbone: focal_t / focal_s / focal_b.
            in_channels: Number of input image channels.
            use_bie: Whether to use BIE-based decoder fusion.
            use_svit: Global switch for SVIT.
            svit_on_f16: Whether to apply SVIT to f16.
            svit_on_f32: Whether to apply SVIT to f32.
            svit_stoken_size: Super-token size.
            svit_heads: Number of attention heads in SVIT.
            svit_n_iter: Number of super-token iterations.
            out_channels: Output channels. If None, use num_classes.
        """
        super().__init__()

        if len(kwargs) > 0:
            for arg_name in kwargs.keys():
                print(f"[Warning] LSNetBiFoSTHeightRegression: argument `{arg_name}` is ignored.")

        self.backbone_name = backbone.lower()
        if self.backbone_name not in FocalNetEncoder.PRESETS:
            print(
                f"[Warning] backbone `{backbone}` is not in FocalNetEncoder.PRESETS; "
                "fallback to `focal_t`."
            )
            self.backbone_name = "focal_t"

        self.pretrained = pretrained

        output_channels = num_classes if out_channels is None else out_channels

        # Encoder
        self.encoder = FocalNetEncoder(self.backbone_name, in_ch=in_channels)
        c2, c4, c8, c16, c32 = self.encoder.out_channels()

        # Ablation states
        self.use_bie = use_bie
        self.use_svit = use_svit
        self.svit_on_f16 = use_svit and svit_on_f16
        self.svit_on_f32 = use_svit and svit_on_f32

        # High-level SVIT modules
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

        # Decoder channel settings for /2, /4, /8, and /16 outputs.
        decoder_channels = [64, 128, 256, 512]

        self.up4 = UnetUp(c16 + c32, decoder_channels[3])

        if self.use_bie:
            self.up3 = UpBIE(
                c_skip=c8,
                c_low=decoder_channels[3],
                out_ch=decoder_channels[2],
            )
            self.up2 = UpBIE(
                c_skip=c4,
                c_low=decoder_channels[2],
                out_ch=decoder_channels[1],
            )
        else:
            self.up3 = UnetUp(c8 + decoder_channels[3], decoder_channels[2])
            self.up2 = UnetUp(c4 + decoder_channels[2], decoder_channels[1])

        self.up1 = UnetUp(c2 + decoder_channels[1], decoder_channels[0])

        self.out_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Keep the name `final` unchanged for checkpoint compatibility.
        self.final = nn.Conv2d(decoder_channels[0], output_channels, 1)

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
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_ablation_config(self):
        return {
            "backbone": self.backbone_name,
            "use_bie": self.use_bie,
            "use_svit": self.use_svit,
            "svit_on_f16": self.svit_on_f16,
            "svit_on_f32": self.svit_on_f32,
        }


# Backward-compatible aliases.
# These aliases do not change state_dict keys and should not affect old weights.
Unet = LSNetBiFoSTHeightRegression
HeightRegressionNet = LSNetBiFoSTHeightRegression


__all__ = [
    "LSNetBiFoSTHeightRegression",
    "HeightRegressionNet",
    "Unet",
    "FocalNetEncoder",
    "FocalBlock2d",
    "FocalModulation",
    "StokenAttention",
    "Attention",
    "Unfold",
    "Fold",
    "BIE",
    "UpBIE",
    "UnetUp",
]