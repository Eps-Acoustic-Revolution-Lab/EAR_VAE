import torch
import math

from torch import nn, pow
from alias_free_torch import Activation1d
from dac.nn.layers import WNConv1d, WNConvTranspose1d
from typing import Literal

def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)

class SnakeBeta(nn.Module):
    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def get_activation(
    activation: Literal["elu", "snake", "none"], antialias=False, channels=None
) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation,
        use_snake=False,
        antialias_activation=False,
        bias=True,
    ):
        super().__init__()

        self.dilation = dilation

        act = get_activation(
            "snake" if use_snake else "elu",
            antialias=antialias_activation,
            channels=out_channels,
        )

        padding = (dilation * (7 - 1)) // 2

        self.layers = nn.Sequential(
            act,
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
                bias=bias,
            ),
            act,
            WNConv1d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=bias
            ),
        )

    def forward(self, x):
        res = x

        # x = checkpoint(self.layers, x)
        x = self.layers(x)

        return x + res


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
        bias=True,
    ):
        super().__init__()

        act = get_activation(
            "snake" if use_snake else "elu",
            antialias=antialias_activation,
            channels=in_channels,
        )

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=1,
                use_snake=use_snake,
                bias=bias,
            ),
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=3,
                use_snake=use_snake,
                bias=bias,
            ),
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=9,
                use_snake=use_snake,
                bias=bias,
            ),
            act,
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                bias=bias,
            ),
        )
        
    def forward(self, x):
        return self.layers(x)


class AntiAliasUpsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, bias=True):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

        self.conv = WNConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            bias=bias,
            padding="same",
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        bias=True,
    ):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = AntiAliasUpsamplerBlock(
                in_channels=in_channels, out_channels=out_channels, stride=stride, bias=bias
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                bias=bias,
            )

        act = get_activation(
            "snake" if use_snake else "elu",
            antialias=antialias_activation,
            channels=in_channels,
        )

        self.layers = nn.Sequential(
            act,
            upsample_layer,
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=1,
                use_snake=use_snake,
                bias=bias,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=3,
                use_snake=use_snake,
                bias=bias,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=9,
                use_snake=use_snake,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        bias=True,
    ):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
                bias=bias,
            )
        ]

        for i in range(self.depth - 1):
            layers += [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                    bias=bias,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
        bias=True,
    ):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
                bias=bias,
            ),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                    bias=bias,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
