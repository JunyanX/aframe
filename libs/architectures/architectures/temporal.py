from typing import Literal

from torch import Tensor, nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, 3, dilation=dilation, padding=dilation
        )
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv1d(channels, channels, 1)
        # start each block as the identity on its (nonnegative) input
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.final_conv(self.relu(self.conv(x))))


class AttentionPool(nn.Module):
    """Softmax-weighted average over time, scored per step by a 1x1 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # no bias: softmax ignores a constant shift, so it would never train
        self.score = nn.Conv1d(channels, 1, 1, bias=False)
        # zero scores give uniform weights, i.e. plain average pooling
        nn.init.zeros_(self.score.weight)

    def forward(self, x: Tensor) -> Tensor:
        weights = self.score(x).softmax(dim=-1)
        return (x * weights).sum(-1, keepdim=True)


class TemporalHead(nn.Module):
    """
    Project ResNet features to fewer channels, mix them across
    time with dilated residual convolutions, then pool over time.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        dilations: list[int],
        pooling: Literal["avg", "attention"] = "avg",
    ) -> None:
        super().__init__()
        self.projection = nn.Conv1d(in_channels, channels, 1)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            *(TemporalResidualBlock(channels, d) for d in dilations)
        )
        if pooling == "attention":
            self.pool = AttentionPool(channels)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.blocks(self.relu(self.projection(x))))
