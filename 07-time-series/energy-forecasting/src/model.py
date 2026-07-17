import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal convolution — output at time t only sees inputs up to t."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Single TCN residual block with two dilated causal convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1   = nn.LayerNorm(out_channels)
        self.norm2   = nn.LayerNorm(out_channels)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, channels, time)
        residual = self.residual(x)
        out = self.relu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out).transpose(1,2)).transpose(1,2))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for energy price forecasting.

    Architecture:
        Input  : (batch, input_window, n_features)
        TCN blocks with exponentially growing dilation
        Output : (batch, forecast_horizon)
    """
    def __init__(
        self,
        n_features: int = 9,
        num_channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        input_window: int = 168,
        forecast_horizon: int = 24,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 128, 64]

        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], forecast_horizon)

    def forward(self, x):
        # x: (batch, time, features)
        out = x.permute(0, 2, 1)          # (batch, features, time)
        out = self.network(out)            # (batch, channels, time)
        out = out[:, :, -1]               # last time step
        return self.fc(out)               # (batch, forecast_horizon)
