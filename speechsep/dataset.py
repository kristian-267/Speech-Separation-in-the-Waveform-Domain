import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from speechsep.cli import Args
from speechsep.model import valid_n_samples
from speechsep.util import pad


class LibrimixDataset(Dataset):
    def __init__(
        self, metadata_file: str, example_length=8, context=3, pad_to_valid=False, limit=None
    ):
        self.metadata = pd.read_csv(metadata_file)
        self.pad_to_valid = pad_to_valid

        if len(self) > 0:
            _, self.sample_rate = torchaudio.load(self.metadata.iloc[0]["mixture_path"])
        else:
            raise "Empty dataset for LibriMix"

        # Remove examples that are shorter than given example length
        self.metadata = self.metadata[self.metadata["length"] / self.sample_rate >= example_length]
        if len(self) == 0:
            raise f"No examples longer than {example_length} s in LibriMix dataset"

        if limit is not None:
            self.metadata = self.metadata.iloc[:limit]

        self.n_samples = int(example_length * self.sample_rate)
        if pad_to_valid:
            self.n_samples_valid = valid_n_samples(self.n_samples, context)
        else:
            self.n_samples_valid = self.n_samples
        self.ts = np.arange(0, self.n_samples_valid / self.sample_rate, 1 / self.sample_rate)

    @classmethod
    def from_args(cls, args: Args, split: str):
        assert (
            args.dataset_args["dataset"] == "librimix"
        ), "Cannot create LibrimixDataset from given arguments"

        if split == "train":
            metadata_files = args.dataset_args["librimix_train_metadata"]
        elif split == "val":
            metadata_files = [args.dataset_args["librimix_val_metadata"]]
        elif split == "test":
            metadata_files = [args.dataset_args["librimix_test_metadata"]]
        else:
            raise ValueError(f"Invalid split {split}")

        datasets = [
            cls(
                metadata_file,
                args.dataset_args["example_length"],
                args.model_args["context"],
                args.dataset_args["pad_to_valid"],
                args.dataset_args["librimix_limit"],
            )
            for metadata_file in metadata_files
        ]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = torch.utils.data.ConcatDataset(datasets)

        print(f"Loaded LibriMix dataset ({len(dataset)} examples, {split})")

        return dataset

    def __len__(self):
        return len(self.metadata.index)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        speaker1, _ = torchaudio.load(row["source_1_path"])
        speaker2, _ = torchaudio.load(row["source_2_path"])
        # Concatenate speaker signals
        y_true = torch.cat([speaker1, speaker2], dim=0)[..., : self.n_samples]

        x, _ = torchaudio.load(row["mixture_path"])
        x = x[..., : self.n_samples]

        if self.pad_to_valid:
            x = pad(x, self.n_samples_valid)
            y_true = pad(y_true, self.n_samples_valid)

        return x, y_true


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = LibrimixDataset("data/datasets/mini/mixture_mini_mix_both.csv", pad_to_valid=True)

    x, y = dataset[0]
    print(x.shape)
    print(y.shape)
    print(len(dataset))
    print(dataset.sample_rate)

    plt.plot(dataset.ts, x[0])
    plt.plot(dataset.ts, y[0])
    plt.plot(dataset.ts, y[1])
    plt.show()
