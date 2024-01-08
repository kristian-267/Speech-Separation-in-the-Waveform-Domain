"""Parsing of command-line arguments"""
import json
import os.path
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any

import torch.cuda


@dataclass
class Args:
    model_args: dict[str, Any]
    dataset_args: dict[str, Any]
    dataloader_args: dict[str, Any]
    trainer_args: dict[str, Any]
    all: dict[str, Any]

    @classmethod
    def from_dict(cls, args: dict[str, Any]) -> "Args":
        if args["num_workers"] is None:
            # Use two dataloader worker per GPU
            args["num_workers"] = 2 * args["devices"]

        model_args = {
            "should_upsample": not args["skip_upsampling"],
            "should_normalize": not args["skip_normalization"],
            "context": args["context"],
            "dropout_p": args["dropout_p"],
            "lstm_layers": args["lstm_layers"],
        }
        dataset_args = {
            "dataset": args["dataset"],
            "pad_to_valid": args["valid_length"] == "pad",
            "extend_to_valid": args["valid_length"] == "extend",
            "example_length": args["example_length"],
            "librimix_train_metadata": args["librimix_train_metadata"],
            "librimix_val_metadata": args["librimix_val_metadata"],
            "librimix_test_metadata": args["librimix_test_metadata"],
            "librimix_limit": args["librimix_limit"],
            "sinusoid_n_examples": args["sinusoid_n_examples"],
            "sinusoid_sample_rate": args["sinusoid_sample_rate"],
            "sinusoid_seed": args["sinusoid_seed"],
        }
        dataloader_args = {
            "batch_size": args["batch_size"] if args["mode"] == "train" else 1,
            "num_workers": args["num_workers"],
        }

        if args["mode"] == "train":
            limit_val_batches = args["limit_val_batches"]
            if limit_val_batches and limit_val_batches > 1.0:
                limit_val_batches = int(limit_val_batches)

            trainer_args = {
                "max_epochs": args["max_epochs"],
                "log_every_n_steps": args["log_every_n_steps"],
                "accelerator": "gpu" if (args["gpu"] and torch.cuda.is_available()) else None,
                "devices": args["devices"],
                "limit_val_batches": limit_val_batches,
                "val_check_interval": args["val_check_interval"],
            }
        else:
            trainer_args = None

        if "LSB_JOBID" in os.environ:
            args["lsf_job"] = os.getenv("LSB_JOBID")

        return Args(model_args, dataset_args, dataloader_args, trainer_args, args)

    def save_to_json(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "args.json"), "w") as f:
            json.dump(
                {
                    "model_args": self.model_args,
                    "dataset_args": self.dataset_args,
                    "dataloader_args": self.dataloader_args,
                    "trainer_args": self.trainer_args,
                    "all": self.all,
                },
                f,
                indent=3,
                sort_keys=True,
            )

    def __getitem__(self, item):
        return self.all[item]

    @property
    def mode(self):
        return self["mode"]

    @property
    def dataset(self):
        return self["dataset"]


def parse_cli_args() -> Args:
    parser = ArgumentParser(
        prog="python -m speechsep",
        description="A tool for separating speakers in a conversation. "
        "The tool has 2 modes: training and prediction. "
        "See https://github.com/DominikStiller/dtu-speechsep for more documentation.",
    )
    subparsers = parser.add_subparsers()

    # Parent parser for common parameters
    parser_params = ArgumentParser(add_help=False)

    parser_params.add_argument("--dataset", choices=["librimix", "sinusoid"], required=True)
    parser_params.add_argument("--context", type=int, default=3)
    parser_params.add_argument("--dropout-p", type=float, default=0)
    parser_params.add_argument("--lstm-layers", type=int, default=2)
    parser_params.add_argument("--skip-upsampling", action="store_true")
    parser_params.add_argument("--skip-normalization", action="store_true")
    parser_params.add_argument("--valid-length", choices=["pad", "extend", "none"], default="pad")
    parser_params.add_argument("--example-length", type=float, default=1)
    parser_params.add_argument("--librimix-train-metadata", action="extend", nargs="+", type=str)
    parser_params.add_argument("--librimix-val-metadata", type=str)
    parser_params.add_argument("--librimix-test-metadata", type=str)
    parser_params.add_argument("--librimix-limit", type=int)
    parser_params.add_argument("--sinusoid-n-examples", type=int, default=2**15)
    parser_params.add_argument("--sinusoid-sample-rate", type=int, default=int(8e3))
    parser_params.add_argument("--sinusoid-seed", type=int, default=42)
    parser_params.add_argument("--checkpoint-path", type=str)
    parser_params.add_argument("--num-workers", type=int)
    parser_params.add_argument("--gpu", action="store_true")
    parser_params.add_argument("--devices", type=int, default=1)

    # Training mode
    parser_training = subparsers.add_parser("train", parents=[parser_params])
    parser_training.set_defaults(mode="train")

    parser_training.add_argument("--batch-size", type=int, default=32)
    parser_training.add_argument("--max-epochs", type=int, default=500)
    parser_training.add_argument("--weight-decay", type=float, default=0.0)
    parser_training.add_argument("--log-every-n-steps", type=int, default=10)
    parser_training.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    parser_training.add_argument("--limit-val-batches", type=float)
    parser_training.add_argument("--val-check-interval", type=float, default=1.0)
    parser_training.add_argument("--cosine-anneal-period", type=int)
    parser_training.add_argument("--reduce-on-plateau-metric", type=str)

    # Prediction mode
    parser_prediction = subparsers.add_parser("predict", parents=[parser_params])
    parser_prediction.set_defaults(mode="predict")

    parser_prediction.add_argument("--item", type=int, default=1)
    parser_prediction.add_argument("--hp-filter", action="store_true")

    args = vars(parser.parse_args())
    if "mode" not in args:
        parser.print_help()
        exit(1)

    return Args.from_dict(args)
