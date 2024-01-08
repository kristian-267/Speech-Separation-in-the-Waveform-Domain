import math

import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from speechsep.cli import Args
from speechsep.model import valid_n_samples
from speechsep.util import pad


class SinusoidDataset(Dataset):
    """
    A dataset of sinusoids with random amplitude, frequency and phase.

    Examples can be padded or extended to the next valid number of samples
    that is compatible with the encoder-decoder structure of the Demucs model.
    For a more detailed description, see :func:`speechsep.model.valid_n_samples`.

    For two sinusoid datasets of differing lengths, the first n examples will be
    identical.
    """

    def __init__(
        self,
        n,
        example_length=8,
        sample_rate=8e3,
        context=3,
        pad_to_valid=False,
        extend_to_valid=False,
        seed=42,
    ):
        """
        Initialize a sinusoid dataset.

        If neither `pad_to_valid` or `extend_to_valid` is given, the number of
        samples may be invalid for Demucs.

        Args:
            n: number of examples
            example_length: length of each example [s]
            sample_rate: sample rate [Hz]
            context: width of kernel in decoder
            pad_to_valid: pad with 0s to valid number of samples (for evaluation)
            extend_to_valid: extend sinusoid to valid number of samples(for training)
            seed: random seed for amplitude, frequency and phase
        """
        self.n = n

        if pad_to_valid and extend_to_valid:
            raise "Cannot use both pad_to_valid and extend_to_valid"
        self.pad_to_valid = pad_to_valid
        self.extend_to_valid = extend_to_valid

        self.n_samples = example_length * int(sample_rate)
        if pad_to_valid or extend_to_valid:
            self.n_samples_valid = valid_n_samples(self.n_samples, context)
        else:
            self.n_samples_valid = self.n_samples

        self.ts = np.arange(0, self.n_samples_valid / sample_rate, 1 / sample_rate)
        self._ts_unpadded = np.arange(0, self.n_samples / sample_rate, 1 / sample_rate)

        # Amplitude
        self.amps = 1 + default_rng(seed).random((n, 2)) * 2
        # Angular frequency
        self.omegas = 1 + default_rng(seed + 1).random((n, 2)) * 30
        # Initial phase
        self.phis = default_rng(seed + 2).random((n, 2)) * 2 * np.pi

        # Ensure that sinusoids are below Nyquist frequency
        assert self.omegas.max().max() / (2 * np.pi) < sample_rate / 2

    @classmethod
    def from_args(cls, args: Args, split: str):
        assert (
            args.dataset_args["dataset"] == "sinusoid"
        ), "Cannot create SinusoidDataset from given arguments"

        seed = args.dataset_args["sinusoid_seed"]
        n_examples = args.dataset_args["sinusoid_n_examples"]
        if split == "train":
            seed += 0
        elif split == "val":
            seed += 1
            n_examples = math.ceil(n_examples / 8)
        elif split == "test":
            seed += 2
            n_examples = math.ceil(n_examples / 8)
        else:
            raise ValueError(f"Invalid split {split}")

        return cls(
            n_examples,
            args.dataset_args["example_length"],
            args.dataset_args["sinusoid_sample_rate"],
            args.model_args["context"],
            args.dataset_args["pad_to_valid"],
            args.dataset_args["extend_to_valid"],
            seed,
        )

    def __len__(self):
        return self.n

    def _generate_sinusoid(self, idx, speaker, ts):
        amp = self.amps[idx, speaker]
        omega = self.omegas[idx, speaker]
        phi = self.phis[idx, speaker]
        return amp * np.sin(omega * ts + phi)

    def __getitem__(self, idx):
        if self.pad_to_valid:
            speaker1 = pad(self._generate_sinusoid(idx, 0, self._ts_unpadded), self.n_samples_valid)
            speaker2 = pad(self._generate_sinusoid(idx, 1, self._ts_unpadded), self.n_samples_valid)
            assert speaker1.shape[-1] == self.n_samples_valid
            assert speaker2.shape[-1] == self.n_samples_valid
        else:
            speaker1 = self._generate_sinusoid(idx, 0, self.ts)
            speaker2 = self._generate_sinusoid(idx, 1, self.ts)

        speaker1 = torch.from_numpy(speaker1).view(1, -1).float()
        speaker2 = torch.from_numpy(speaker2).view(1, -1).float()

        # Mix speaker signals
        x = speaker1 + speaker2
        # Concatenate speaker signals
        y_true = torch.cat([speaker1, speaker2], dim=0)

        return x, y_true


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SinusoidDataset(10, example_length=1, pad_to_valid=True)

    x, y = dataset[0]
    print(x.shape)
    print(y.shape)

    plt.plot(dataset.ts, x[0])
    plt.plot(dataset.ts, y[0])
    plt.plot(dataset.ts, y[1])
    plt.show()
