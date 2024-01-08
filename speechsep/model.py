import julius
import torch
import math
import torch.nn.functional as F
from torch import nn

from speechsep.cli import Args
from speechsep.util import center_trim


class Demucs(nn.Module):
    def __init__(self, args: Args):
        super().__init__()

        context = args.model_args["context"]
        dropout_p = args.model_args["dropout_p"]
        lstm_layers = args.model_args["lstm_layers"]

        self.should_normalize = args.model_args["should_normalize"]
        self.should_upsample = args.model_args["should_upsample"]
        if self.should_upsample:
            self.upsample = julius.resample.ResampleFrac(1, 2)
            self.downsample = julius.resample.ResampleFrac(2, 1)

        self.encoders = nn.ModuleList(
            [
                DemucsEncoder(1, 64, dropout_p),
                DemucsEncoder(64, 128, dropout_p),
                DemucsEncoder(128, 256, dropout_p),
                DemucsEncoder(256, 512, dropout_p),
                DemucsEncoder(512, 1024, dropout_p),
                DemucsEncoder(1024, 2048, dropout_p),
            ]
        )
        self.lstm = DemucsLSTM(lstm_layers, dropout_p)
        self.decoders = nn.ModuleList(
            [
                DemucsDecoder(2048, 1024, context, dropout_p),
                DemucsDecoder(1024, 512, context, dropout_p),
                DemucsDecoder(512, 256, context, dropout_p),
                DemucsDecoder(256, 128, context, dropout_p),
                DemucsDecoder(128, 64, context, dropout_p),
                DemucsDecoder(64, 2, context, dropout_p, use_activation=False),
            ]
        )

        # Rescale initial weights
        for sub in self.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
                rescale_conv(sub)

    def forward(self, x):
        """
        Forward-pass of the Demucs model. Only n_channels = 1 is supported.

        Args:
            x: input signal, shape (n_batch, n_channels, n_samples)

        Returns:
            Separated signal for both speakers, shape (n_batch, 2, n_samples)
        """
        skip_activations: list[torch.Tensor] = []

        if self.should_normalize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-5
            x = (x - mean) / std

        if self.should_upsample:
            x = self.upsample(x)

        for encoder in self.encoders:
            x = encoder(x)
            skip_activations.append(x)

        x = self.lstm(x)

        for decoder in self.decoders:
            skip_activation = center_trim(skip_activations.pop(), target=x)

            # x = torch.cat([x, skip_activation], dim=1)
            # Demucs adds instead of concatenates the skip activations, contrary to U-net
            x = x + skip_activation
            x = decoder(x)

        if self.should_upsample:
            x = self.downsample(x)

        if self.should_normalize:
            x = x * std + mean

        return x


class DemucsEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()

        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=4)
        self.dropout_1 = nn.Dropout(dropout_p)
        self.conv_2 = nn.Conv1d(out_channels, 2 * out_channels, kernel_size=1, stride=1)
        self.dropout_2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv_1(x)
        # Dropout should be placed before ReLU but after any other activation function
        # Source: https://sebastianraschka.com/faq/docs/dropout-activation.html
        x = self.dropout_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.glu(x, dim=1)  # split in channel dimension
        x = self.dropout_2(x)
        return x


class DemucsLSTM(nn.Module):
    def __init__(self, lstm_layers: int, dropout_p: float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=2048, hidden_size=2048, num_layers=lstm_layers, bidirectional=True
        )
        self.linear = nn.Linear(4096, 2048)
        self.dropout_1 = nn.Dropout(dropout_p)
        self.dropout_2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # move sequence first
        x, _ = self.lstm(x)
        x = self.dropout_1(x)
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        x = self.dropout_2(x)
        return x


class DemucsDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context: int,
        dropout_p: float,
        use_activation=True,
    ):
        super().__init__()

        self.conv_1 = nn.Conv1d(in_channels, 2 * in_channels, kernel_size=context, stride=1)
        self.dropout_1 = nn.Dropout(dropout_p)
        self.conv_2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=8, stride=4)
        self.dropout_2 = nn.Dropout(dropout_p)
        self.use_activation = use_activation

    def forward(self, x):
        x = self.conv_1(x)
        x = F.glu(x, dim=1)  # split in channel dimension
        x = self.dropout_1(x)
        x = self.conv_2(x)
        if self.use_activation:
            x = self.dropout_2(x)
            x = F.relu(x)
        return x


# From https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py#L29
def rescale_conv(conv, reference=0.1):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


# From https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py#L145
def valid_n_samples(n_samples, context):
    """
    Return the nearest valid number of samples to use with the model so that
    there is no time steps left over in a convolutions, e.g. for all
    layers, size of the input - kernel_size % stride = 0.

    If the mixture has a valid number of samples, the estimated sources can
    be center trimmed to match.

    For training, extracts should have a valid number of samples. For
    evaluation on full signals we recommend padding with zeros.

    Args:
        n_samples: the original number of samples
        context: width of kernel in decoder

    Returns:
        The nearest valid number of samples
    """
    resample = False
    depth = 6
    kernel_size = 8
    stride = 4

    if resample:
        n_samples *= 2
    for _ in range(depth):
        n_samples = math.ceil((n_samples - kernel_size) / stride) + 1
        n_samples = max(1, n_samples)
        n_samples += context - 1
    for _ in range(depth):
        n_samples = (n_samples - 1) * stride + kernel_size

    if resample:
        n_samples = math.ceil(n_samples / 2)
    return int(n_samples)


if __name__ == "__main__":
    from speechsep.mock_dataset import SinusoidDataset
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        SinusoidDataset(1000, example_length=1, extend_to_valid=True), batch_size=4
    )
    x, y_true = next(iter(train_dataloader))
    model = Demucs()

    y_pred = model.forward(x)
    y_true = center_trim(y_true, target=y_pred)
    assert y_true.shape == y_pred.shape
    print(y_true.shape)

    loss = F.mse_loss(y_pred, y_true)
    print(loss)
    loss.backward()
