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


class TemporalHead(nn.Module):
    """
    Project ResNet features to fewer channels, mix them across
    time with dilated residual convolutions, then average over time.
    """

    def __init__(
        self, in_channels: int, channels: int, dilations: list[int]
    ) -> None:
        super().__init__()
        self.projection = nn.Conv1d(in_channels, channels, 1)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            *(TemporalResidualBlock(channels, d) for d in dilations)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.blocks(self.relu(self.projection(x))))
